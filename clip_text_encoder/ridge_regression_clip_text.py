# ridge_regression_clip_text.py
"""
Voxelwise ridge regression encoding model for CLIP text features.

For a given feature type:
1. Loads PCA-reduced text features (X) and averaged betas (Y).
2. Splits into 80/20 train/test.
3. Standardizes features and voxel responses.
4. Selects ridge alpha per voxel via 7-fold CV on training set.
5. Fits final ridge model and predicts on held-out test set.
6. Saves predicted Y values and best alpha per voxel.

Usage:
  python ridge_regression_clip_text.py <subject_number> <feature_type>

Example:
  python ridge_regression_clip_text.py 1 descriptions
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# CONFIG
NSD_ROOT = Path("/path/to/NSD")     # CHANGE THIS TO YOUR NSD ROOT
BATCH_SIZE = 60000
N_SPLITS = 7
DEVICE = torch.device("cuda")

# Feature configuration
RIDGE_CONFIG = {
    "descriptions": {
        "input": "features_clip_text_vitl14_pca_descriptions.npy",
        "alphas_out": "best_alphas_clip_text_descriptions.npy",
        "pred_out": "Y_pred_clip_text_descriptions.npy",
    },
    "descriptions_shuffled": {
        "input": "features_clip_text_vitl14_pca_descriptions_shuffled.npy",
        "alphas_out": "best_alphas_clip_text_descriptions_shuffled.npy",
        "pred_out": "Y_pred_clip_text_descriptions_shuffled.npy",
    },
    "bag": {
        "input": "features_clip_text_vitl14_pca_bag.npy",
        "alphas_out": "best_alphas_clip_text_bag.npy",
        "pred_out": "Y_pred_clip_text_bag.npy",
    },
    "bag_shuffled": {
        "input": "features_clip_text_vitl14_pca_bag_shuffled.npy",
        "alphas_out": "best_alphas_clip_text_bag_shuffled.npy",
        "pred_out": "Y_pred_clip_text_bag_shuffled.npy",
    },
    "objects": {
        "input": "features_clip_text_vitl14_pca_objects.npy",
        "alphas_out": "best_alphas_clip_text_objects.npy",
        "pred_out": "Y_pred_clip_text_objects.npy",
    },
    "verbs": {
        "input": "features_clip_text_vitl14_pca_verbs.npy",
        "alphas_out": "best_alphas_clip_text_verbs.npy",
        "pred_out": "Y_pred_clip_text_verbs.npy",
    },
    "adjectives": {
        "input": "features_clip_text_vitl14_pca_adjectives.npy",
        "alphas_out": "best_alphas_clip_text_adjectives.npy",
        "pred_out": "Y_pred_clip_text_adjectives.npy",
    },
}


def process_subject(subj, feature_type):
    """
    Run voxelwise ridge regression for a subject and feature type.

    Args:
        subj: Subject number (1-8)
        feature_type: Key from RIDGE_CONFIG
    """
    output_dir = NSD_ROOT / f"subj{subj:02d}_all" / "outputs2"

    config = RIDGE_CONFIG[feature_type]
    feature_file = output_dir / config["input"]
    beta_file = output_dir / f"betas_averaged_final_{subj}.npy"
    alphas_file = output_dir / config["alphas_out"]
    pred_file = output_dir / config["pred_out"]

    # Validate inputs
    if not feature_file.exists():
        print(f"Error: Feature file not found: {feature_file}")
        print("   Run pca_clip_text.py first.")
        return
    if not beta_file.exists():
        print(f"Error: Beta file not found: {beta_file}")
        print("   Run prepare_betas.py first.")
        return

    # Load data
    print(f"Subject {subj}, feature='{feature_type}': Loading data...")
    X = np.load(feature_file).astype(np.float32)
    Y = np.load(beta_file).astype(np.float32)

    n_samples, n_features = X.shape[0], X.shape[1]
    n_voxels = Y.shape[1]
    print(f"  X: {X.shape}, Y: {Y.shape}")

    # Handle NaN in features (fill with column mean from training split)
    split_point = int(n_samples * 0.8)
    X_train_raw, X_test_raw = X[:split_point], X[split_point:]
    Y_train_raw, Y_test_raw = Y[:split_point], Y[split_point:]

    # Fill NaN with training set column means
    col_means = np.nanmean(X_train_raw, axis=0)
    nan_mask_train = np.isnan(X_train_raw)
    nan_mask_test = np.isnan(X_test_raw)
    for col in range(n_features):
        X_train_raw[nan_mask_train[:, col], col] = col_means[col]
        X_test_raw[nan_mask_test[:, col], col] = col_means[col]

    del X, Y

    # Standardization
    print("  Standardizing...")
    scaler_X = StandardScaler().fit(X_train_raw)
    X_train = scaler_X.transform(X_train_raw).astype(np.float32)
    X_test = scaler_X.transform(X_test_raw).astype(np.float32)

    y_mean = Y_train_raw.mean(axis=0, dtype=np.float32)
    y_std = Y_train_raw.std(axis=0, dtype=np.float32)
    y_std[y_std == 0] = 1.0
    Y_train = ((Y_train_raw - y_mean) / y_std).astype(np.float32)
    Y_test = ((Y_test_raw - y_mean) / y_std).astype(np.float32)

    # Send to GPU
    X_train_t = torch.tensor(X_train, device=DEVICE)
    X_test_t = torch.tensor(X_test, device=DEVICE)
    del X_train_raw, X_test_raw, X_train, X_test

    # Cross-validation folds
    kf = KFold(n_splits=N_SPLITS)
    fold_indices = [
        (torch.tensor(trn, device=DEVICE), torch.tensor(val, device=DEVICE))
        for trn, val in kf.split(np.arange(split_point))
    ]

    alphas = torch.logspace(-8, 10, 100, device=DEVICE)
    best_alphas = torch.empty(n_voxels, dtype=torch.float32, device="cpu")
    Y_pred_all = np.empty((Y_test.shape[0], n_voxels), dtype=np.float32)

    # Voxelwise loop
    print(f"  Running ridge regression ({n_voxels} voxels)...")
    for start in range(0, n_voxels, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_voxels)
        print(f"    Voxels {start}-{end}")

        Yb_train_t = torch.tensor(Y_train[:, start:end], device=DEVICE)

        # Cross-validate alpha
        r2_alpha_sum = torch.zeros(len(alphas), end - start, device=DEVICE)
        for train_idx, val_idx in fold_indices:
            Xtr, Xval = X_train_t[train_idx], X_train_t[val_idx]
            Ytr, Yval = Yb_train_t[train_idx], Yb_train_t[val_idx]
            XtX = Xtr.T @ Xtr
            Xty = Xtr.T @ Ytr
            eye_d = torch.eye(XtX.shape[0], device=DEVICE)
            for ai, alpha in enumerate(alphas):
                w = torch.linalg.solve(XtX + eye_d * alpha, Xty)
                Ypred = Xval @ w
                ss_res = torch.sum((Yval - Ypred) ** 2, dim=0)
                ss_tot = torch.sum((Yval - torch.mean(Yval, dim=0)) ** 2, dim=0)
                r2_alpha_sum[ai] += 1 - ss_res / (ss_tot + 1e-12)

        mean_r2_alpha = r2_alpha_sum / N_SPLITS
        best_ai = torch.argmax(mean_r2_alpha, dim=0)
        best_alpha_batch = alphas[best_ai]
        best_alphas[start:end] = best_alpha_batch.cpu()

        # Fit final model
        XtX_full = X_train_t.T @ X_train_t
        Xty_full = X_train_t.T @ Yb_train_t

        for i in range(end - start):
            alpha = best_alpha_batch[i]
            w = torch.linalg.solve(
                XtX_full + torch.eye(XtX_full.shape[0], device=DEVICE) * alpha,
                Xty_full[:, i],
            )
            Y_pred_all[:, start + i] = (X_test_t @ w).cpu().numpy()

        del Yb_train_t
        torch.cuda.empty_cache()

    # Save outputs
    np.save(str(alphas_file), best_alphas.numpy())
    np.save(str(pred_file), Y_pred_all)
    print(f"  Saved alphas: {alphas_file}")
    print(f"  Saved predictions: {pred_file}")
    print(f"  Predicted Y shape: {Y_pred_all.shape}")
    print("  Complete.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ridge_regression_clip_text.py <subject_number> <feature_type>")
        print("Feature types:", ", ".join(RIDGE_CONFIG.keys()))
        sys.exit(1)

    subject = int(sys.argv[1])
    feature = sys.argv[2]

    if subject < 1 or subject > 8:
        print("Error: Subject must be 1-8")
        sys.exit(1)
    if feature not in RIDGE_CONFIG:
        print(f"Error: Unknown feature type '{feature}'")
        sys.exit(1)

    process_subject(subject, feature)