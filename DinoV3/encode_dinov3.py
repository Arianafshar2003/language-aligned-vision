# encode_dinov3.py
"""
Extract DINOv3 ViT-L/16 visual features for NSD images.

Uses official DINOv3 evaluation protocol:
- Resize shortest edge to 256
- CenterCrop to 224x224
- ImageNet normalization
- L2 normalization on CLS token

Requires local weights file: dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth

Usage:
  python encode_dinov3.py <subject_number>

Output:
  features_dinov3_large2.npy  (n_images x 1024)
"""

import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
import sys

# CONFIG
NSD_ROOT = Path("/path/to/NSD")     # CHANGE THIS TO YOUR NSD ROOT
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EMBED_DIM = 1024  # ViT-L/16

# Path to local DINOv3 weights
LOCAL_WEIGHTS_NAME = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"


def process_subject(subj):
    """
    Extract DINOv3 features for a single subject.

    Args:
        subj: Subject number (1-8)
    """
    output_dir = NSD_ROOT / f"subj{subj:02d}_all" / "outputs2"

    IMAGE_PATHS_FILE = output_dir / f"image_paths_final_{subj}.npy"
    OUTPUT_FEATURES_FILE = output_dir / "features_dinov3_large2.npy"
    LOCAL_WEIGHTS_PATH = NSD_ROOT / LOCAL_WEIGHTS_NAME

    # Hub cache setup
    HUB_CACHE_DIR = NSD_ROOT / "torch_hub_cache"
    HUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(HUB_CACHE_DIR))

    # Validate inputs
    if not IMAGE_PATHS_FILE.exists():
        print(f"Error: Image paths file not found: {IMAGE_PATHS_FILE}")
        return
    if not LOCAL_WEIGHTS_PATH.exists():
        print(f"Error: DINOv3 weights not found at {LOCAL_WEIGHTS_PATH}")
        return

    # Load image paths
    print(f"Subject {subj}: Loading image paths...")
    image_paths = np.load(IMAGE_PATHS_FILE, allow_pickle=True)
    num_images = len(image_paths)
    print(f"  Loaded {num_images} image paths.")

    # Transforms (official DINOv3 evaluation protocol)
    transform_pipeline = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load model architecture from torch hub
    print("  Loading DINOv3 architecture from torch hub...")
    model = torch.hub.load(
        'facebookresearch/dinov3',
        'dinov3_vitl16',
        pretrained=False,
        force_reload=False
    )

    # Load local weights
    print(f"  Loading local weights from {LOCAL_WEIGHTS_PATH}...")
    state_dict = torch.load(LOCAL_WEIGHTS_PATH, map_location="cpu")

    # Handle common checkpoint formats
    if "teacher" in state_dict:
        print("  Using teacher weights.")
        state_dict = state_dict["teacher"]
    elif "model" in state_dict:
        print("  Using model weights.")
        state_dict = state_dict["model"]

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"  Weights loaded: {msg}")

    model.to(DEVICE)
    model.eval()

    # Extraction loop
    features_dinov3 = np.zeros((num_images, EMBED_DIM), dtype=np.float32)
    num_batches = (num_images + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"  Encoding {num_images} images (batch size {BATCH_SIZE})...")
    for i in tqdm(range(num_batches), desc="  Encoding DINOv3"):
        start_idx = i * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, num_images)

        batch_paths = image_paths[start_idx:end_idx]
        batch_tensors = []
        valid_indices = []

        # Load and preprocess images
        for local_idx, path_str in enumerate(batch_paths):
            try:
                img = Image.open(path_str).convert("RGB")
                tensor = transform_pipeline(img)
                batch_tensors.append(tensor)
                valid_indices.append(local_idx)
            except (FileNotFoundError, UnidentifiedImageError, OSError):
                continue

        if not batch_tensors:
            continue

        # Encode batch
        try:
            input_batch = torch.stack(batch_tensors).to(DEVICE)
            with torch.no_grad():
                outputs = model(input_batch)

                # Extract CLS token
                if isinstance(outputs, dict):
                    embeddings = outputs.get('x_norm_clstoken', outputs.get('x_norm_cls'))
                elif isinstance(outputs, torch.Tensor):
                    embeddings = outputs
                else:
                    embeddings = outputs[0]

                # L2 normalize
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                embeddings = embeddings.cpu().numpy()

            # Store results
            for k, local_valid_idx in enumerate(valid_indices):
                global_idx = start_idx + local_valid_idx
                features_dinov3[global_idx] = embeddings[k]

        except RuntimeError as e:
            print(f"\n  Warning: Batch {i} error: {e}")
            continue

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(OUTPUT_FEATURES_FILE), features_dinov3)
    print(f"  Saved: {OUTPUT_FEATURES_FILE}")
    print(f"  Shape: {features_dinov3.shape}")
    print(f"Subject {subj} complete.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python encode_dinov3.py <subject_number>")
        sys.exit(1)

    subject = int(sys.argv[1])
    if subject < 1 or subject > 8:
        print("Error: Subject must be 1-8")
        sys.exit(1)

    process_subject(subject)