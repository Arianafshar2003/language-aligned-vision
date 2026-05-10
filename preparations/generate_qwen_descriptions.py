# generate_qwen_descriptions.py
"""
Generate structured scene descriptions and extract embeddings using Qwen2.5-VL.

For each image in a subject's image_paths file:
1. Generate a narrative description, object list, verb list, and adjective list
   in a structured JSON format using a constrained prompt.
2. Extract the final hidden state immediately preceding the EOS token
   during generation as a multimodal embedding.
3. Save embeddings (.npy) and descriptions (.npy) per subject.

Dependencies: transformers, PIL, torch, numpy, tqdm
"""

import textwrap
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import warnings
import sys

warnings.filterwarnings("ignore")

# CONFIG
NSD_ROOT = Path("/path/to/NSD")     # CHANGE THIS TO YOUR NSD ROOT
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
BATCH_SIZE = 6

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL LOADING
print("Loading model and processor...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)

# Ensure proper EOS token
if processor.tokenizer.eos_token_id is None:
    processor.tokenizer.eos_token_id = processor.tokenizer.convert_tokens_to_ids('<|endoftext|>')

print("Model loaded.")

# HOOK FOR HIDDEN STATE CAPTURE
generation_features = {}

def generation_hook_fn(module, inp, out):
    """
    Forward hook that captures hidden states during model.generate().
    Stores each layer's output under 'all_hidden_states' for later
    extraction of the pre-EOS representation.
    """
    if isinstance(out, tuple):
        hidden_states = out[0]
    else:
        hidden_states = out

    if "all_hidden_states" not in generation_features:
        generation_features["all_hidden_states"] = []

    generation_features["all_hidden_states"].append(hidden_states.clone())

final_layer = model.language_model.layers[-1]
hook_handle = final_layer.register_forward_hook(generation_hook_fn)


# PROMPT TEMPLATE
PROMPT_TEXT = (
    "You are a precise vision-language assistant. "
    "Given the image, first create a concise English description (no more than 40 words). "
    "Then return your entire answer strictly as valid JSON -- do not include extra text or Markdown.\n"
    "Use this *exact* structure (keys and brackets included):\n"
    "{\n"
    '  "description": "<Scene Description, Max 40 words>",\n'
    '  "objects": ["object1", "object2", "object3"],\n'
    '  "verbs": ["verb1", "verb2", "verb3"],\n'
    '  "adjectives": ["adjective1", "adjective2", "adjective3"]\n'
    "}\n"
    "Rules:\n"
    "- Output nothing before or after the JSON.\n"
    "- The description must be fluent natural English, not list form.\n"
    "- Each list must have at least two elements.\n"
    "- Use double quotes for all strings to keep JSON valid."
)


# SUBJECT PROCESSING
def process_subject(subj):
    """
    Run Qwen2.5-VL description generation and embedding extraction
    for a single subject.

    Args:
        subj: Subject number (1-8)
    """
    print("\n" + "=" * 50)
    print(f"Processing Subject {subj}")
    print("=" * 50)

    # Paths
    IMAGE_PATHS_FILE = NSD_ROOT / f"subj{subj:02d}_all" / "outputs2" / f"image_paths_final_{subj}.npy"
    output_dir = NSD_ROOT / f"subj{subj:02d}_all" / "outputs2"

    if not IMAGE_PATHS_FILE.exists():
        print(f"Error: Image paths file not found: {IMAGE_PATHS_FILE}")
        print("   Run prepare_betas.py first.")
        return

    image_paths = np.load(IMAGE_PATHS_FILE, allow_pickle=True).tolist()
    num_images = len(image_paths)
    print(f"Total images: {num_images}")

    # Storage arrays
    multimodal_embeddings_matrix = np.zeros((num_images, 3584), dtype=np.float32)
    descriptions_list = []

    # Batch processing
    num_batches = (num_images + BATCH_SIZE - 1) // BATCH_SIZE
    print("Extracting multimodal embeddings from generation...")

    for batch_idx in tqdm(range(num_batches), desc="Batches"):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, num_images)

        batch_paths = image_paths[start:end]
        batch_images = []
        batch_messages = []

        # Prepare image and prompt pairs
        for img_path in batch_paths:
            if not img_path or not Path(img_path).is_file():
                batch_images.append(None)
                batch_messages.append(None)
                continue
            try:
                image = Image.open(img_path).convert("RGB")
                batch_images.append(image)
                batch_messages.append([{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": PROMPT_TEXT}
                    ],
                }])
            except Exception:
                batch_images.append(None)
                batch_messages.append(None)

        # Identify valid entries in batch
        valid_indices = [i for i, img in enumerate(batch_images) if img is not None]
        if not valid_indices:
            for global_i in range(start, end):
                multimodal_embeddings_matrix[global_i, :] = np.nan
                descriptions_list.append("")
            continue

        chat_texts = [
            processor.apply_chat_template(batch_messages[i], tokenize=False, add_generation_prompt=True)
            for i in valid_indices
        ]
        imgs_for_proc = [batch_images[i] for i in valid_indices]

        inputs = processor(
            text=chat_texts,
            images=imgs_for_proc,
            return_tensors="pt"
        ).to(DEVICE, non_blocking=True)

        generation_features.clear()

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=250,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                output_hidden_states=False,
                return_dict_in_generate=False
            )

        # Decode generated text
        generated_ids_trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        descriptions_batch = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

        # Extract pre-EOS hidden states
        if "all_hidden_states" in generation_features and len(generation_features["all_hidden_states"]) > 0:
            all_hidden = torch.cat(generation_features["all_hidden_states"], dim=1)
            pre_eos_embeddings = []

            for batch_item in range(len(valid_indices)):
                if batch_item < all_hidden.size(0):
                    full_seq = generated_ids[batch_item]
                    input_len = inputs.input_ids[batch_item].size(0)

                    eos_token_id = processor.tokenizer.eos_token_id
                    all_eos_positions = (full_seq == eos_token_id).nonzero(as_tuple=True)[0]
                    eos_positions_in_generated = all_eos_positions[all_eos_positions >= input_len]

                    if len(eos_positions_in_generated) > 0:
                        first_eos_in_generated = eos_positions_in_generated[0].item()
                        pre_eos_pos = first_eos_in_generated - 1

                        if pre_eos_pos >= input_len and pre_eos_pos < all_hidden.size(1):
                            pre_eos_embedding = all_hidden[batch_item, pre_eos_pos, :]
                        else:
                            pre_eos_embedding = all_hidden[batch_item, -1, :]
                    else:
                        pre_eos_embedding = all_hidden[batch_item, -1, :]

                    pre_eos_embeddings.append(pre_eos_embedding)
                else:
                    pre_eos_embeddings.append(torch.zeros(3584, device=DEVICE))

            pre_eos_embeddings = torch.stack(pre_eos_embeddings).to(torch.float32).cpu().numpy()

            for idx_in_batch, valid_i in enumerate(valid_indices):
                global_i = start + valid_i
                if idx_in_batch < len(pre_eos_embeddings):
                    multimodal_embeddings_matrix[global_i, :] = pre_eos_embeddings[idx_in_batch]
                    descriptions_list.append(descriptions_batch[idx_in_batch])
                else:
                    multimodal_embeddings_matrix[global_i, :] = np.nan
                    descriptions_list.append("")
        else:
            for valid_i in valid_indices:
                global_i = start + valid_i
                multimodal_embeddings_matrix[global_i, :] = np.nan
                descriptions_list.append("")

    # Cleanup
    hook_handle.remove()

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    multimodal_embeddings_file = output_dir / f"features_qwen2_5vl_semantic_pre_eos_{subj}.npy"
    np.save(multimodal_embeddings_file, multimodal_embeddings_matrix)

    descriptions_file = output_dir / f"qwen_semantic_descriptions_{subj}.npy"
    np.save(descriptions_file, np.array(descriptions_list, dtype=object))

    print(f"\nSubject {subj} complete.")
    print(f"Processed {len(descriptions_list)} images")
    print(f"Embedding dimensions: {multimodal_embeddings_matrix.shape}")
    print(f"Output saved to:")
    print(f"  Embeddings: {multimodal_embeddings_file}")
    print(f"  Descriptions: {descriptions_file}")


# ENTRY POINT
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_qwen_descriptions.py <subject_number>")
        print("Example: python generate_qwen_descriptions.py 1")
        sys.exit(1)

    subject = int(sys.argv[1])
    if subject < 1 or subject > 8:
        print("Error: Subject number must be between 1 and 8")
        sys.exit(1)

    process_subject(subject)