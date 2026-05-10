# pca_dinov3.py
"""
Apply PCA dimensionality reduction (97% variance retained) to DINOv3 features.

Usage:
  python pca_dinov3.py <subject_number>

Output:
  features_dinov3_large2_pca.npy
"""

import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
import sys

# CONFIG
NSD_ROOT = Path("/path/to/NSD")     # CHANGE THIS TO YOUR NSD ROOT


def process_subject(subj):
    """
    Apply PCA to DINOv3 features for a single subject.

    Args:
        subj: Subject number (1-8)
    """
    output_dir = NSD_ROOT / f"subj{subj:02d}_all" / "outputs2"

    INPUT_FILE = output_dir / "features_dinov3_large2.npy"
    OUTPUT_FILE = output_dir / "features_dinov3_large2_pca.npy"

    if not INPUT_FILE.exists():
        print(f"Error: DINOv3 features not found: {INPUT_FILE}")
        print("   Run encode_dinov3.py first.")
        return

    print(f"Subject {subj}: Loading DINOv3 features...")
    X = np.load(INPUT_FILE)
    print(f"  Original shape: {X.shape}")

    print("  Running PCA (97% variance retained)...")
    pca = PCA(n_components=0.97, svd_solver='full')
    X_pca = pca.fit_transform(X)

    np.save(str(OUTPUT_FILE), X_pca)
    print(f"  PCA-reduced shape: {X_pca.shape}")
    print(f"  Components retained: {X_pca.shape[1]}")
    print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"  Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pca_dinov3.py <subject_number>")
        sys.exit(1)

    subject = int(sys.argv[1])
    if subject < 1 or subject > 8:
        print("Error: Subject must be 1-8")
        sys.exit(1)

    process_subject(subject)