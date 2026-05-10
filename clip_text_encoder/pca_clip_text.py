# pca_clip_text.py
"""
Apply PCA dimensionality reduction (97% variance retained) to CLIP text features.

Reads raw CLIP text embeddings and saves PCA-transformed versions.
Matched to the same PCA method used for visual features.

Usage:
  python pca_clip_text.py <subject_number> <feature_type>

Example:
  python pca_clip_text.py 1 descriptions
"""

import numpy as np
from sklearn.decomposition import PCA
import sys
from pathlib import Path

# CONFIG
NSD_ROOT = Path("/path/to/NSD")     # CHANGE THIS TO YOUR NSD ROOT

# Input/output mapping (must match encode_clip_text.py outputs)
PCA_CONFIG = {
    "descriptions": {
        "input": "features_clip_text_vitl14_descriptions.npy",
        "output": "features_clip_text_vitl14_pca_descriptions.npy",
    },
    "descriptions_shuffled": {
        "input": "features_clip_text_vitl14_descriptions_shuffled.npy",
        "output": "features_clip_text_vitl14_pca_descriptions_shuffled.npy",
    },
    "bag": {
        "input": "features_clip_text_vitl14_bag.npy",
        "output": "features_clip_text_vitl14_pca_bag.npy",
    },
    "bag_shuffled": {
        "input": "features_clip_text_vitl14_bag_shuffled.npy",
        "output": "features_clip_text_vitl14_pca_bag_shuffled.npy",
    },
    "objects": {
        "input": "features_clip_text_vitl14_objects.npy",
        "output": "features_clip_text_vitl14_pca_objects.npy",
    },
    "verbs": {
        "input": "features_clip_text_vitl14_verbs.npy",
        "output": "features_clip_text_vitl14_pca_verbs.npy",
    },
    "adjectives": {
        "input": "features_clip_text_vitl14_adjectives.npy",
        "output": "features_clip_text_vitl14_pca_adjectives.npy",
    },
}


def process_subject(subj, feature_type):
    """
    Apply PCA reduction for a subject and feature type.

    Args:
        subj: Subject number (1-8)
        feature_type: Key from PCA_CONFIG
    """
    output_dir = NSD_ROOT / f"subj{subj:02d}_all" / "outputs2"

    config = PCA_CONFIG[feature_type]
    input_file = output_dir / config["input"]
    output_file = output_dir / config["output"]

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        print("   Run encode_clip_text.py first.")
        return

    print(f"Subject {subj}, feature='{feature_type}': Loading embeddings...")
    X = np.load(input_file)
    print(f"  Original shape: {X.shape}")

    # Remove NaN rows for PCA fitting (if any)
    valid_mask = ~np.isnan(X).any(axis=1)
    X_valid = X[valid_mask]
    print(f"  Valid samples: {X_valid.shape[0]} / {X.shape[0]}")

    # Fit PCA on valid data
    print("  Running PCA (97% variance retained)...")
    pca = PCA(n_components=0.97, svd_solver='full')
    X_pca_valid = pca.fit_transform(X_valid)

    # Reconstruct full array with NaN for invalid rows
    X_pca = np.full((X.shape[0], X_pca_valid.shape[1]), np.nan, dtype=np.float32)
    X_pca[valid_mask] = X_pca_valid

    # Save
    np.save(str(output_file), X_pca)
    print(f"  PCA-reduced shape: {X_pca.shape}")
    print(f"  Components retained: {X_pca_valid.shape[1]}")
    print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"  Saved to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pca_clip_text.py <subject_number> <feature_type>")
        print("Feature types:", ", ".join(PCA_CONFIG.keys()))
        sys.exit(1)

    subject = int(sys.argv[1])
    feature = sys.argv[2]

    if subject < 1 or subject > 8:
        print("Error: Subject must be 1-8")
        sys.exit(1)
    if feature not in PCA_CONFIG:
        print(f"Error: Unknown feature type '{feature}'")
        sys.exit(1)

    process_subject(subject, feature)