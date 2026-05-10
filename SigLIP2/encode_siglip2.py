# encode_siglip2.py
"""
Extract SigLIP2 image features for NSD images.

Uses google/siglip2-large-patch16-256 with the HuggingFace
processor for image preprocessing and get_image_features()
for extraction. Outputs L2-normalized embeddings.

Usage:
  python encode_siglip2.py <subject_number>

Output:
  features_siglip2_large.npy
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from transformers import AutoModel, AutoProcessor
import sys

# CONFIG
NSD_ROOT = Path("/path/to/NSD")     # CHANGE THIS TO YOUR NSD ROOT
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
MODEL_ID = "google/siglip2-large-patch16-256"


def process_subject(subj):
    """
    Extract SigLIP2 features for a single subject.

    Args:
        subj: Subject number (1-8)
    """
    output_dir = NSD_ROOT / f"subj{subj:02d}_all" / "outputs2"

    IMAGE_PATHS_FILE = output_dir / f"image_paths_final_{subj}.npy"
    OUTPUT_FEATURES_FILE = output_dir / "features_siglip2_large.npy"

    # Validate input
    if not IMAGE_PATHS_FILE.exists():
        print(f"Error: Image paths file not found: {IMAGE_PATHS_FILE}")
        return

    # Load image paths
    print(f"Subject {subj}: Loading image paths...")
    image_paths = np.load(IMAGE_PATHS_FILE, allow_pickle=True)
    num_images = len(image_paths)
    print(f"  Loaded {num_images} image paths.")

    # Load model
    print(f"  Loading {MODEL_ID}...")
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()

    embed_dim = model.config.vision_config.hidden_size
    print(f"  Model loaded. Embedding dimension: {embed_dim}")

    # Extraction loop
    features_image = np.zeros((num_images, embed_dim), dtype=np.float32)
    num_batches = (num_images + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"  Encoding {num_images} images (batch size {BATCH_SIZE})...")
    for i in tqdm(range(num_batches), desc="  Encoding SigLIP2"):
        start_idx = i * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, num_images)

        batch_paths = image_paths[start_idx:end_idx]
        batch_images = []
        valid_indices = []

        # Load images
        for local_idx, path_str in enumerate(batch_paths):
            try:
                img = Image.open(path_str).convert("RGB")
                batch_images.append(img)
                valid_indices.append(local_idx)
            except (FileNotFoundError, UnidentifiedImageError, OSError):
                continue

        if not batch_images:
            continue

        # Preprocess and encode
        inputs = processor(images=batch_images, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = F.normalize(image_features, p=2, dim=-1)
            image_features = image_features.cpu().numpy()

        # Store results
        for k, local_valid_idx in enumerate(valid_indices):
            global_idx = start_idx + local_valid_idx
            features_image[global_idx] = image_features[k]

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(OUTPUT_FEATURES_FILE), features_image)
    print(f"  Saved: {OUTPUT_FEATURES_FILE}")
    print(f"  Shape: {features_image.shape}")
    print(f"Subject {subj} complete.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python encode_siglip2.py <subject_number>")
        sys.exit(1)

    subject = int(sys.argv[1])
    if subject < 1 or subject > 8:
        print("Error: Subject must be 1-8")
        sys.exit(1)

    process_subject(subject)