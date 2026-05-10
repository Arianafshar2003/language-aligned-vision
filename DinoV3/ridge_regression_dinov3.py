# ridge_regression_dinov3.py
"""
Voxelwise ridge regression encoding model for DINOv3 PCA features.

Splits data 80/20, standardizes, selects alpha per voxel via 7-fold CV,
and predicts held-out responses.

Usage:
  python ridge_regression_dinov3.py <subject_number>

Output:
  best_alphas_dinov3.npy
  Y_pred_dinov3.npy
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
BATCH_SIZE = 80000
N_SPLITS = 7
DEVICE = torch.device("cuda")


def process_subject(subj):
    """
    Fit voxelwise ridge regression for DINOv3 features.

    Args:
        subj: Subject number (1-8)
    """
    output_dir = NSD_ROOT / f"subj{subj:02d}_all" / "outputs2"

    FEATURE_FILE = output_dir / "features_dinov3_large2_pca.npy"
    BETA_FILE = output_dir / f"betas_averaged_final_{subj}.npy"
    ALPHAS_FILE = output_dir / "best_alphas_dinov3.npy"
    PRED_FILE = output_dir / "Y_pred_dinov3.npy"

    # Validate inputs
    if not FEATURE_FILE.exists():
        print(f"Error: PCA features not found: {FEATURE_FILE}")
        print("   Run pca_dinov3.py first.")
        return
    if not BETA_FILE.exists():
        print(f"Error: Beta file not found: {BETA_FILE}")
        print("   Run prepare_betas.py first.")
        return

    # Load data
    print(f"Subject {subj}: Loading data...")
    X = np.load(FEATURE_FILE).astype(np.float32)
    Y = np.load(BETA_FILE).astype(np.float32)
    n_samples, n_voxels = X.shape[0], Y.shape[1]
    print(f"  X: {X.shape}, Y: {Y.shape}")

    # Train/test split (80/20)
    split_point = int(n_samples * 0.8)
    X_train, X_test = X[:split_point], X[split_point:]
    Y_train, Y_test = Y[:split_point], Y[split_point:]
    del X

    # Standardization
    print("  Standardizing...")
    scaler_X = StandardScaler().fit(X_train)
    X_train = scaler_X.transform(X_train).astype(np.float32)
    X_test = scaler_X.transform(X_test).astype(np.float32)

    y_mean = Y_train.mean(axis=0, dtype=np.float32)
    y_std = Y_train.std(axis=0, dtype=np.float32)
    y_std[y_std == 0] = 1.0
    Y_train = ((Y_train - y_mean) / y_std).astype(np.float32)
    Y_test = ((Y_test - y_mean) / y_std).astype(np.float32)

    # To GPU
    X_train_t = torch.tensor(X_train, device=DEVICE)
    X_test_t = torch.tensor(X_test, device=DEVICE)
    del X_train, X_test

    # Cross-validation setup
    alphas = torch.logspace(-8, 10, 100, device=DEVICE)
    kf = KFold(n_splits=N_SPLITS)
    fold_indices = [
        (torch.tensor(trn, device=DEVICE), torch.tensor(val, device=DEVICE))
        for trn, val in kf.split(np.arange(split_point))
    ]

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

    # Save
    np.save(str(ALPHAS_FILE), best_alphas.numpy())
    np.save(str(PRED_FILE), Y_pred_all)
    print(f"  Saved alphas: {ALPHAS_FILE}")
    print(f"  Saved predictions: {PRED_FILE}")
    print(f"  Y_pred shape: {Y_pred_all.shape}")
    print(f"Subject {subj} complete.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ridge_regression_dinov3.py <subject_number>")
        sys.exit(1)

    subject = int(sys.argv[1])
    if subject < 1 or subject > 8:
        print("Error: Subject must be 1-8")
        sys.exit(1)

    process_subject(subject)