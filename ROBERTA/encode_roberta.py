# encode_roberta.py
"""
Extract RoBERTa-base text embeddings for linguistic feature spaces.

Uses the <s> token (CLS equivalent) representation, L2-normalized.
Processes one feature type per run. Supported types:
  descriptions, descriptions_shuffled, bag, bag_shuffled,
  objects, verbs, adjectives

Usage:
  python encode_roberta.py <subject_number> <feature_type>

Example:
  python encode_roberta.py 1 descriptions
  python encode_roberta.py 1 adjectives
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from pathlib import Path
import sys

# CONFIG
NSD_ROOT = Path("/path/to/NSD")     # CHANGE THIS TO YOUR NSD ROOT
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Map feature types to input/output filenames
FEATURE_CONFIG = {
    "descriptions": {
        "input": "descriptions.npy",
        "output": "features_roberta_base_descriptions.npy",
        "is_list": False,
    },
    "descriptions_shuffled": {
        "input": "descriptions_shuffled.npy",
        "output": "features_roberta_base_descriptions_shuffled.npy",
        "is_list": False,
    },
    "bag": {
        "input": "bag.npy",
        "output": "features_roberta_base_bag.npy",
        "is_list": True,
    },
    "bag_shuffled": {
        "input": "bag_shuffled.npy",
        "output": "features_roberta_base_bag_shuffled.npy",
        "is_list": True,
    },
    "objects": {
        "input": "objects.npy",
        "output": "features_roberta_base_objects.npy",
        "is_list": True,
    },
    "verbs": {
        "input": "verbs.npy",
        "output": "features_roberta_base_verbs.npy",
        "is_list": True,
    },
    "adjectives": {
        "input": "adjectives.npy",
        "output": "features_roberta_base_adjectives.npy",
        "is_list": True,
    },
}


def list_to_text(item):
    """Convert a list of words to a space-separated string."""
    if isinstance(item, (list, np.ndarray)):
        return " ".join(str(w) for w in item if w)
    return str(item).strip()


def process_subject(subj, feature_type):
    """
    Extract RoBERTa embeddings for a subject and feature type.

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

    # Load RoBERTa model
    print("  Loading RoBERTa-base...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModel.from_pretrained("roberta-base").to(DEVICE)
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
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            # <s> token representation (index 0)
            emb = outputs.last_hidden_state[:, 0, :]
            emb = emb / emb.norm(dim=-1, keepdim=True)  # L2 normalize

        features[idx] = emb.cpu().numpy()

    # Save
    np.save(str(output_file), features)
    print(f"  Saved: {output_file}")
    print(f"  Shape: {features.shape}")
    print(f"  NaN count: {np.isnan(features).any(axis=1).sum()}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python encode_roberta.py <subject_number> <feature_type>")
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