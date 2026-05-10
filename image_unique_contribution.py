# image_unique_contribution.py
"""
Stimulus-level unique contribution analysis for variance partitioning.

Identifies images where a feature space uniquely contributes to
predicting cortical responses. For each test image, computes the
alignment between observed neural pattern and the unique predictive
component of feature A beyond feature B.

Outputs a ranked list of images driving unique variance, displayed
as top-positive and near-zero examples.

Usage:
  python image_unique_contribution.py <subject_number> <feature_A> <feature_B>

Feature options: descriptions, dinov3, siglip2, bag, bag_shuffled,
                  shuffled, objects, verbs, adjectives

Example:
  python image_unique_contribution.py 5 siglip2 dinov3
"""

import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# CONFIG
NSD_ROOT = Path("/path/to/NSD")     # CHANGE THIS TO YOUR NSD ROOT
N_DISPLAY = 20  # Number of top/bottom images to display

# Single-feature ridge predictions
SINGLE_PREDS = {
    "descriptions":          "Y_pred_clip_text_descriptions.npy",
    "dinov3":                "Y_pred2_dino3.npy",
    "siglip2":               "Y_pred_siglip2.npy",
    "bag":                   "Y_pred_clip_text_bag.npy",
    "bag_shuffled":          "Y_pred_clip_text_bag_shuffled.npy",
    "shuffled":              "Y_pred_clip_text_descriptions_shuffled.npy",
    "objects":               "Y_pred_clip_text_objects.npy",
    "verbs":                 "Y_pred_clip_text_verbs.npy",
    "adjectives":            "Y_pred_clip_text_adjectives.npy",
}

# Joint predictions need specific naming conventions
# Format: pred_A_source + pred_B_source -> joint file
# You may need to adjust this mapping based on your actual filenames
JOINT_PRED_MAP = {
    ("siglip2", "dinov3"):              "Y_pred_dino3_siglip_joint2.npy",
    ("dinov3", "siglip2"):              "Y_pred_dino3_siglip_joint2.npy",
    ("descriptions", "dinov3"):         "Y_pred_dino3_descriptions_joint2.npy",
    ("dinov3", "descriptions"):         "Y_pred_dino3_descriptions_joint2.npy",
    ("descriptions", "siglip2"):        "Y_pred_siglip_descriptions_joint.npy",
    ("siglip2", "descriptions"):        "Y_pred_siglip_descriptions_joint.npy",
    ("descriptions", "bag"):            "Y_pred_clip_text_descriptions_joint_bag.npy",
    ("descriptions", "bag_shuffled"):   "Y_pred_clip_text_descriptions_joint_bag_shuffled.npy",
    ("descriptions", "shuffled"):       "Y_pred_clip_text_descriptions_joint_shuffled.npy",
    ("descriptions", "objects"):        "Y_pred_clip_text_descriptions_joint_objects.npy",
    ("descriptions", "verbs"):          "Y_pred_clip_text_descriptions_joint_verbs.npy",
    ("descriptions", "adjectives"):     "Y_pred_clip_text_descriptions_joint_adjectives.npy",
}

# Unique variance mask files
UNIQUE_MASK_MAP = {
    ("siglip2", "dinov3"):              "mask_significant_A_Dino3VsSiglip.npy",
    ("dinov3", "siglip2"):              "mask_significant_A_SiglipVsDino3.npy",
    ("descriptions", "dinov3"):         "mask_significant_A_Dino3VsDescription.npy",
    ("dinov3", "descriptions"):         "mask_significant_A_descriptionVsDino3.npy",
    ("descriptions", "siglip2"):        "mask_significant_A_SiglipVsDescription.npy",
    ("siglip2", "descriptions"):        "mask_significant_A_descriptionVsSiglip.npy",
}


def load_and_align_prediction(filepath, expected_samples):
    """
    Load a prediction file and align orientation to (n_samples, n_voxels).
    """
    data = np.load(filepath).astype(np.float32)

    if data.shape[0] > data.shape[1]:
        # Appears transposed (voxels, samples) -> transpose
        data = data.T

    # Truncate or pad if needed
    if data.shape[0] != expected_samples:
        min_len = min(data.shape[0], expected_samples)
        data = data[:min_len]

    return data


def compute_image_scores(Y_true, Y_B, Y_AB, mask):
    """
    Compute stimulus-level unique contribution scores.

    Score formula (from Methods section):
      S_i = sum_v [ y_true(i,v) * (y_AB(i,v) - y_B(i,v)) ]

    Only voxels in the significance mask are included.

    Args:
        Y_true: Ground truth responses (n_samples, n_voxels)
        Y_B: Predictions from baseline model
        Y_AB: Predictions from joint model
        mask: Boolean array indicating significant voxels

    Returns:
        Array of scores (n_samples,)
    """
    # Restrict to significant voxels
    Y_true_roi = Y_true[:, mask]
    Y_B_roi = Y_B[:, mask]
    Y_AB_roi = Y_AB[:, mask]

    # Unique predictive component
    unique_diff = Y_AB_roi - Y_B_roi

    # Per-image score: sum over voxels of y_true * unique_diff
    scores = np.sum(Y_true_roi * unique_diff, axis=1)

    return scores


def display_images(indices, scores, test_paths, title):
    """
    Display a grid of images with their scores.

    Args:
        indices: Array of image indices to display
        scores: Corresponding score values
        test_paths: File paths for test images
        title: Plot title
    """
    n_rows = 4
    n_cols = 5
    n_images = min(len(indices), n_rows * n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    axes = axes.ravel()

    for i in range(n_images):
        idx = indices[i]
        path = test_paths[idx]
        try:
            img = mpimg.imread(path)
            axes[i].imshow(img)
        except (FileNotFoundError, OSError):
            axes[i].text(0.5, 0.5, "Image not found",
                        ha='center', va='center', transform=axes[i].transAxes)

        axes[i].axis('off')

    # Hide remaining axes
    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    plt.suptitle(title, fontsize=14)
    plt.subplots_adjust(hspace=0.03, wspace=0.03)
    plt.tight_layout(pad=0.5)
    plt.show()


def process_subject(subj, feature_A, feature_B):
    """
    Run stimulus-level unique contribution analysis.

    Args:
        subj: Subject number (1-8)
        feature_A: Key for feature whose unique contribution is analyzed
        feature_B: Key for baseline feature
    """
    output_dir = NSD_ROOT / f"subj{subj:02d}_all" / "outputs2"
    paths_file = output_dir / f"image_paths_final_{subj}.npy"
    beta_file = output_dir / f"betas_averaged_final_{subj}.npy"

    pair = (feature_A, feature_B)
    joint_pred_file = JOINT_PRED_MAP.get(pair)
    unique_mask_file = UNIQUE_MASK_MAP.get(pair)

    print(f"\n{'='*60}")
    print(f"Subject {subj}: Image-wise unique contribution")
    print(f"Feature A: {feature_A} (tested)")
    print(f"Feature B: {feature_B} (baseline)")
    print(f"{'='*60}")

    # Validate files
    if joint_pred_file is None:
        print(f"Error: No joint prediction mapping for pair {pair}")
        print("  Add entry to JOINT_PRED_MAP in the script.")
        return
    if unique_mask_file is None:
        print(f"Error: No unique mask mapping for pair {pair}")
        print("  Add entry to UNIQUE_MASK_MAP in the script.")
        return

    pred_A_file = output_dir / SINGLE_PREDS.get(feature_A, "")
    pred_B_file = output_dir / SINGLE_PREDS.get(feature_B, "")
    pred_AB_file = output_dir / joint_pred_file
    mask_file = output_dir / unique_mask_file

    missing = []
    for f, name in [
        (beta_file, "Betas"),
        (paths_file, "Image paths"),
        (pred_AB_file, f"Joint pred ({joint_pred_file})"),
        (mask_file, f"Unique mask ({unique_mask_file})"),
    ]:
        if not f.exists():
            missing.append(f"  - {name}: {f}")

    if missing:
        print("Error: Missing files:")
        for m in missing:
            print(m)
        return

    # Load data
    print("\nLoading data...")
    Y_true_full = np.load(beta_file).astype(np.float32)
    split_point = int(Y_true_full.shape[0] * 0.8)
    Y_test = Y_true_full[split_point:]

    Y_AB = load_and_align_prediction(pred_AB_file, Y_test.shape[0])
    Y_B = np.load(pred_B_file).astype(np.float32)

    if Y_B.shape[0] > Y_B.shape[1]:
        Y_B = Y_B.T
    Y_B = Y_B[:Y_test.shape[0]]

    # Load image paths
    all_paths = np.load(paths_file, allow_pickle=True)
    test_paths = all_paths[split_point:]
    test_paths = test_paths[:Y_test.shape[0]]

    n_samples, n_voxels = Y_test.shape
    print(f"  Test samples: {n_samples}")
    print(f"  Voxels: {n_voxels}")

    # Load significance mask
    print(f"\nLoading significance mask: {mask_file.name}")
    sig_data = np.load(mask_file)
    if sig_data.dtype == np.uint8:
        mask = sig_data.astype(bool)
    else:
        mask = sig_data > 0

    n_sig = np.sum(mask)
    print(f"  Significant voxels: {n_sig} / {len(mask)}")

    if n_sig < 5:
        print("Error: Not enough significant voxels for meaningful analysis.")
        return

    # Compute scores
    print("\nComputing stimulus-level scores...")
    scores = compute_image_scores(Y_test, Y_B, Y_AB, mask)

    # Rank images
    sorted_abs_indices = np.argsort(np.abs(scores))

    top_indices = sorted_abs_indices[-N_DISPLAY:][::-1]
    bottom_indices = sorted_abs_indices[:N_DISPLAY]

    print(f"\n  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Top {N_DISPLAY} mean score: {scores[top_indices].mean():.4f}")
    print(f"  Bottom {N_DISPLAY} mean score: {scores[bottom_indices].mean():.4f}")

    # Display
    print("\nDisplaying images with highest unique contribution...")
    display_images(
        top_indices, scores[top_indices], test_paths,
        f"Subject {subj}: Top {feature_A} unique contribution beyond {feature_B}"
    )

    print("\nDisplaying images with near-zero unique contribution...")
    display_images(
        bottom_indices, scores[bottom_indices], test_paths,
        f"Subject {subj}: Near-zero {feature_A} unique contribution beyond {feature_B}"
    )

    # Save scores for later use
    scores_file = output_dir / f"image_scores_unique_{feature_A}_vs_{feature_B}.npy"
    np.save(str(scores_file), scores)
    print(f"\nScores saved to: {scores_file}")
    print(f"Subject {subj} complete.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python image_unique_contribution.py <subject> <feature_A> <feature_B>")
        print("\nFeature options:")
        print("  ", ", ".join(SINGLE_PREDS.keys()))
        print("\nExample comparisons:")
        print("  python image_unique_contribution.py 5 siglip2 dinov3")
        print("  python image_unique_contribution.py 5 descriptions dinov3")
        print("  python image_unique_contribution.py 5 descriptions bag")
        sys.exit(1)

    subject = int(sys.argv[1])
    feat_A = sys.argv[2]
    feat_B = sys.argv[3]

    if subject < 1 or subject > 8:
        print("Error: Subject must be 1-8")
        sys.exit(1)
    if feat_A not in SINGLE_PREDS:
        print(f"Error: Unknown feature '{feat_A}'")
        sys.exit(1)
    if feat_B not in SINGLE_PREDS:
        print(f"Error: Unknown feature '{feat_B}'")
        sys.exit(1)

    process_subject(subject, feat_A, feat_B)