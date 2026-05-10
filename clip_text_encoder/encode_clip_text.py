# encode_clip_text.py
"""
Extract CLIP ViT-L/14 text embeddings for linguistic feature spaces.

Processes one feature type per run. Supported types:
  descriptions, descriptions_shuffled, bag, bag_shuffled,
  objects, verbs, adjectives

Each input is tokenized, encoded via CLIP text encoder, L2-normalized,
and saved as raw embeddings. PCA reduction and voxelwise encoding are
handled by separate scripts.

Usage:
  python encode_clip_text.py <subject_number> <feature_type>

Example:
  python encode_clip_text.py 1 descriptions
  python encode_clip_text.py 1 bag
"""

import numpy as np
import torch
import clip
import sys
from pathlib import Path
from tqdm import tqdm

# CONFIG
NSD_ROOT = Path("/path/to/NSD")     # CHANGE THIS TO YOUR NSD ROOT
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Map feature types to input/output filenames
FEATURE_CONFIG = {
    "descriptions": {
        "input": "descriptions.npy",
        "output": "features_clip_text_vitl14_descriptions.npy",
        "is_list": False,
    },
    "descriptions_shuffled": {
        "input": "descriptions_shuffled.npy",
        "output": "features_clip_text_vitl14_descriptions_shuffled.npy",
        "is_list": False,
    },
    "bag": {
        "input": "bag.npy",
        "output": "features_clip_text_vitl14_bag.npy",
        "is_list": True,
    },
    "bag_shuffled": {
        "input": "bag_shuffled.npy",
        "output": "features_clip_text_vitl14_bag_shuffled.npy",
        "is_list": True,
    },
    "objects": {
        "input": "objects.npy",
        "output": "features_clip_text_vitl14_objects.npy",
        "is_list": True,
    },
    "verbs": {
        "input": "verbs.npy",
        "output": "features_clip_text_vitl14_verbs.npy",
        "is_list": True,
    },
    "adjectives": {
        "input": "adjectives.npy",
        "output": "features_clip_text_vitl14_adjectives.npy",
        "is_list": True,
    },
}


def list_to_text(item):
    """
    Convert a list of words to a space-separated string.
    Handles empty lists and non-list inputs gracefully.
    """
    if isinstance(item, (list, np.ndarray)):
        return " ".join(str(w) for w in item if w)
    return str(item).strip()


def process_subject(subj, feature_type):
    """
    Extract CLIP text embeddings for a subject and feature type.

    Args:
        subj: Subject number (1-8)
        feature_type: Key from FEATURE_CONFIG
    """
    output_dir = NSD_ROOT / f"subj{subj:02d}_all" / "outputs2"
    paths_file = output_dir / f"image_paths_final_{subj}.npy"

    config = FEATURE_CONFIG[feature_type]
    input_file = output_dir / config["input"]
    output_file = output_dir / config["output"]
    is_list = config["is_list"]

    # Validate inputs
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return
    if not paths_file.exists():
        print(f"Error: Image paths file not found: {paths_file}")
        return

    # Load data
    print(f"Subject {subj}, feature='{feature_type}': Loading data...")
    data = np.load(input_file, allow_pickle=True)
    image_paths = np.load(paths_file, allow_pickle=True)

    if len(data) != len(image_paths):
        print(f"  Warning: Length mismatch ({len(data)} vs {len(image_paths)}).")
    print(f"  Loaded {len(data)} items.")

    # Load CLIP model
    print("  Loading CLIP ViT-L/14...")
    model, _ = clip.load("ViT-L/14", device=DEVICE)
    model.eval()

    # Extract features
    features = np.empty((len(data), 768), dtype=np.float32)

    for idx in tqdm(range(len(data)), desc=f"  Encoding {feature_type}"):
        item = data[idx]

        # Handle missing/empty data
        if item is None:
            features[idx] = np.nan
            continue

        # Convert to text string
        if is_list:
            text = list_to_text(item)
        else:
            text = str(item).strip()

        if text == "":
            features[idx] = np.nan
            continue

        # Tokenize and encode
        tokens = clip.tokenize([text]).to(DEVICE)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)  # L2 normalize

        features[idx] = emb.cpu().numpy()

    # Save
    np.save(str(output_file), features)
    print(f"  Saved: {output_file}")
    print(f"  Shape: {features.shape}")
    print(f"  NaN count: {np.isnan(features).any(axis=1).sum()}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python encode_clip_text.py <subject_number> <feature_type>")
        print("Feature types:", ", ".join(FEATURE_CONFIG.keys()))
        sys.exit(1)

    subject = int(sys.argv[1])
    feature = sys.argv[2]

    if subject < 1 or subject > 8:
        print("Error: Subject must be 1-8")
        sys.exit(1)
    if feature not in FEATURE_CONFIG:
        print(f"Error: Unknown feature type '{feature}'")
        print("  Choose from:", ", ".join(FEATURE_CONFIG.keys()))
        sys.exit(1)

    process_subject(subject, feature)