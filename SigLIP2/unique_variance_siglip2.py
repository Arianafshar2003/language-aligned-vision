# unique_variance_siglip2.py
"""
Variance partitioning analysis for SigLIP2 vs CLIP text descriptions.

Computes unique variance in both directions:
  - What does CLIP descriptions add beyond SigLIP2?
  - What does SigLIP2 add beyond CLIP descriptions?

For each comparison:
1. Concatenates RAW (non-PCA) features.
2. Applies joint PCA (97% variance retained).
3. Fits voxelwise ridge regression on the joint PCA space.
4. Computes unique variance with bootstrap + FDR correction.

Usage:
  python unique_variance_siglip2.py <subject_number> <direction>

Direction options:
  desc_over_siglip  -- Unique variance of descriptions beyond SigLIP2
  siglip_over_desc  -- Unique variance of SigLIP2 beyond descriptions
  both              -- Run both directions

Example:
  python unique_variance_siglip2.py 1 both
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import fdrcorrection
import sys
import time
import gc
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# CONFIG
NSD_ROOT = Path("/path/to/NSD")     # CHANGE THIS TO YOUR NSD ROOT
BATCH_SIZE = 80000
N_SPLITS = 7
N_BOOTSTRAPS = 2000
BOOT_BATCH_SIZE = 4
CHUNK_SIZE = 5000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RAW (non-PCA) feature files
RAW_FEATURES = {
    "descriptions": "features_clip_text_vitl14_descriptions.npy",
    "siglip2":      "features_siglip2_large.npy",
}

# Single-feature ridge predictions
SINGLE_PREDS = {
    "descriptions": "Y_pred_clip_text_descriptions.npy",
    "siglip2":      "Y_pred_siglip2.npy",
}

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def compute_r2(y_true, y_pred):
    """Compute R2 per voxel."""
    mean_true = torch.mean(y_true, dim=-1, keepdim=True)
    ss_res = torch.sum((y_true - y_pred) ** 2, dim=-1)
    ss_tot = torch.sum((y_true - mean_true) ** 2, dim=-1)
    return 1 - ss_res / (ss_tot + 1e-12)


def compute_r2_chunked(y_true_full, y_pred_full, chunk_size=5000):
    """Compute R2 in chunks to avoid GPU OOM."""
    n_vox = y_true_full.shape[0]
    r2_list = []
    for i in range(0, n_vox, chunk_size):
        end = min(i + chunk_size, n_vox)
        y_t_chunk = y_true_full[i:end].float()
        y_p_chunk = y_pred_full[i:end].float()
        r2_list.append(compute_r2(y_t_chunk, y_p_chunk))
    return torch.cat(r2_list)


def fit_ridge_joint(X_train, X_test, Y_train, Y_test, n_voxels):
    """
    Fit voxelwise ridge regression on joint PCA feature space.
    """
    scaler_X = StandardScaler().fit(X_train)
    X_train_s = scaler_X.transform(X_train).astype(np.float32)
    X_test_s = scaler_X.transform(X_test).astype(np.float32)

    y_mean = Y_train.mean(axis=0, dtype=np.float32)
    y_std = Y_train.std(axis=0, dtype=np.float32)
    y_std[y_std == 0] = 1.0
    Y_train_s = ((Y_train - y_mean) / y_std).astype(np.float32)
    Y_test_s = ((Y_test - y_mean) / y_std).astype(np.float32)

    X_train_t = torch.tensor(X_train_s, device=DEVICE)
    X_test_t = torch.tensor(X_test_s, device=DEVICE)
    split_point = X_train_s.shape[0]

    kf = KFold(n_splits=N_SPLITS)
    fold_indices = [
        (torch.tensor(trn, device=DEVICE), torch.tensor(val, device=DEVICE))
        for trn, val in kf.split(np.arange(split_point))
    ]

    alphas = torch.logspace(-8, 10, 100, device=DEVICE)
    best_alphas = torch.empty(n_voxels, dtype=torch.float32, device="cpu")
    Y_pred_all = np.empty((Y_test_s.shape[0], n_voxels), dtype=np.float32)

    for start in range(0, n_voxels, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_voxels)
        print(f"    Voxels {start}-{end}")

        Yb_train_t = torch.tensor(Y_train_s[:, start:end], device=DEVICE)

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

        XtX_full = X_train_t.T @ X_train_t
        Xty_full = X_train_t.T @ Yb_train_t
        for i in range(end - start):
            w = torch.linalg.solve(
                XtX_full + torch.eye(XtX_full.shape[0], device=DEVICE) * best_alpha_batch[i],
                Xty_full[:, i],
            )
            Y_pred_all[:, start + i] = (X_test_t @ w).cpu().numpy()

        del Yb_train_t
        torch.cuda.empty_cache()

    return Y_pred_all


def run_unique_variance(output_dir, feature_A, feature_B, Y, split_point, 
                        output_suffix):
    """
    Compute unique variance of feature_A beyond feature_B.

    Args:
        output_dir: Output directory path
        feature_A: Key for the feature whose unique contribution is tested
        feature_B: Key for the baseline feature
        Y: Full beta matrix
        split_point: Train/test split index
        output_suffix: String suffix for output filenames
    """
    print(f"\n  Unique variance: {feature_A} beyond {feature_B}")
    print(f"  Output suffix: {output_suffix}")

    n_voxels = Y.shape[1]
    Y_train = Y[:split_point]
    Y_test = Y[split_point:]

    # Validate files
    feat_A_raw = output_dir / RAW_FEATURES[feature_A]
    feat_B_raw = output_dir / RAW_FEATURES[feature_B]
    pred_A_file = output_dir / SINGLE_PREDS[feature_A]
    pred_B_file = output_dir / SINGLE_PREDS[feature_B]

    for f, name in [(feat_A_raw, f"Raw {feature_A}"), 
                     (feat_B_raw, f"Raw {feature_B}"),
                     (pred_A_file, f"Pred {feature_A}"),
                     (pred_B_file, f"Pred {feature_B}")]:
        if not f.exists():
            print(f"    Error: {name} not found: {f}")
            return

    # Step 1: Concatenate RAW features
    print("    [1] Concatenating raw features...")
    X_A = np.load(feat_A_raw).astype(np.float32)
    X_B = np.load(feat_B_raw).astype(np.float32)
    X_A = np.nan_to_num(X_A, nan=0.0)
    X_B = np.nan_to_num(X_B, nan=0.0)
    X_joint_raw = np.concatenate([X_A, X_B], axis=1)
    print(f"        Shapes: A={X_A.shape}, B={X_B.shape}, Joint={X_joint_raw.shape}")

    # Step 2: Joint PCA
    print("    [2] Joint PCA (97% variance)...")
    pca = PCA(n_components=0.97, svd_solver="full")
    X_joint_pca = pca.fit_transform(X_joint_raw)
    print(f"        PCA shape: {X_joint_pca.shape}")

    # Step 3: Fit joint ridge
    print("    [3] Fitting joint ridge regression...")
    X_train_joint = X_joint_pca[:split_point]
    X_test_joint = X_joint_pca[split_point:]
    Y_pred_joint = fit_ridge_joint(
        X_train_joint, X_test_joint, Y_train, Y_test, n_voxels
    )

    # Step 4: Load single predictions
    print("    [4] Loading single predictions...")
    y_pred_A = np.load(pred_A_file).astype(np.float32)
    y_pred_B = np.load(pred_B_file).astype(np.float32)

    # Build standardized y_true
    y_mean = Y_train.mean(axis=0, dtype=np.float32)
    y_std = Y_train.std(axis=0, dtype=np.float32)
    y_std[y_std == 0] = 1.0
    y_true_std = ((Y_test - y_mean) / y_std).astype(np.float32)

    # Fix orientation if needed
    if y_true_std.shape[0] < y_true_std.shape[1]:
        y_true_std = y_true_std.T
        y_pred_A = y_pred_A.T
        y_pred_B = y_pred_B.T
        Y_pred_joint = Y_pred_joint.T

    n_vox, n_test = y_true_std.shape
    print(f"        n_voxels={n_vox}, n_test={n_test}")

    # Step 5: Compute R2 values
    print("    [5] Computing R2 scores...")
    y_true_t = torch.from_numpy(y_true_std).half().to(DEVICE, non_blocking=True)
    y_pred_A_t = torch.from_numpy(y_pred_A).half().to(DEVICE, non_blocking=True)
    y_pred_B_t = torch.from_numpy(y_pred_B).half().to(DEVICE, non_blocking=True)
    y_pred_AB_t = torch.from_numpy(Y_pred_joint).half().to(DEVICE, non_blocking=True)

    with torch.no_grad():
        R2_A = compute_r2_chunked(y_true_t, y_pred_A_t, CHUNK_SIZE)
        R2_B = compute_r2_chunked(y_true_t, y_pred_B_t, CHUNK_SIZE)
        R2_AB = compute_r2_chunked(y_true_t, y_pred_AB_t, CHUNK_SIZE)
        Unique_A = R2_AB - R2_B

    # Step 6: Bootstrap significance
    print("    [6] Bootstrap test (H0: Unique_A <= 0)...")
    torch.cuda.set_per_process_memory_fraction(0.9)

    start_time = time.time()
    batches = int(np.ceil(N_BOOTSTRAPS / BOOT_BATCH_SIZE))
    p_accum = []

    for b in range(batches):
        torch.cuda.empty_cache()
        gc.collect()

        bs_start = b * BOOT_BATCH_SIZE
        bs_end = min((b + 1) * BOOT_BATCH_SIZE, N_BOOTSTRAPS)
        current_bs = bs_end - bs_start

        boot_idx = torch.randint(0, n_test, (current_bs, n_test), device=DEVICE)
        batch_p_vals = torch.zeros(n_vox, device="cpu")

        for voxel_start in range(0, n_vox, CHUNK_SIZE):
            voxel_end = min(voxel_start + CHUNK_SIZE, n_vox)
            v_range = slice(voxel_start, voxel_end)

            gather_idx = boot_idx.unsqueeze(1).expand(-1, voxel_end - voxel_start, -1)
            y_t_b = y_true_t[v_range].unsqueeze(0).expand(current_bs, -1, -1).gather(2, gather_idx)
            y_pb_b = y_pred_B_t[v_range].unsqueeze(0).expand(current_bs, -1, -1).gather(2, gather_idx)
            y_pab_b = y_pred_AB_t[v_range].unsqueeze(0).expand(current_bs, -1, -1).gather(2, gather_idx)

            mean_t_bs = y_t_b.mean(dim=2, keepdim=True)
            ss_tot = torch.sum((y_t_b - mean_t_bs) ** 2, dim=2)
            R2_B_boot = 1 - torch.sum((y_t_b - y_pb_b) ** 2, dim=2).float() / (ss_tot.float() + 1e-12)
            R2_AB_boot = 1 - torch.sum((y_t_b - y_pab_b) ** 2, dim=2).float() / (ss_tot.float() + 1e-12)
            boot_diff = R2_AB_boot - R2_B_boot

            batch_p_vals[v_range] = torch.mean((boot_diff <= 0).float(), dim=0).cpu()

            del y_t_b, y_pb_b, y_pab_b, boot_diff
            torch.cuda.empty_cache()

        p_accum.append(batch_p_vals)

        elapsed = time.time() - start_time
        if b > 0:
            eta = (elapsed / (b + 1)) * (batches - b - 1)
            print(f"        Batch {b+1}/{batches} | Elapsed: {elapsed/60:.1f} min | ETA: {eta/60:.1f} min")

    p_vals = torch.stack(p_accum).mean(dim=0).numpy()
    reject, _ = fdrcorrection(p_vals, alpha=0.05)
    mask = reject.astype(np.uint8)

    n_sig = np.sum(mask)
    print(f"        Significant voxels: {n_sig} / {n_vox}")

    # Step 7: Save
    print("    [7] Saving results...")
    np.save(str(output_dir / f"R2_A_{output_suffix}.npy"), R2_A.cpu().numpy())
    np.save(str(output_dir / f"R2_B_{output_suffix}.npy"), R2_B.cpu().numpy())
    np.save(str(output_dir / f"R2_AB_{output_suffix}.npy"), R2_AB.cpu().numpy())
    np.save(str(output_dir / f"Unique_A_{output_suffix}.npy"), Unique_A.cpu().numpy())
    np.save(str(output_dir / f"mask_significant_A_{output_suffix}.npy"), mask)

    del y_true_t, y_pred_A_t, y_pred_B_t, y_pred_AB_t, R2_A, R2_B, R2_AB, Unique_A
    torch.cuda.empty_cache()
    gc.collect()


def process_subject(subj, direction):
    """
    Run SigLIP2 vs CLIP descriptions unique variance analysis.

    Args:
        subj: Subject number (1-8)
        direction: 'desc_over_siglip', 'siglip_over_desc', or 'both'
    """
    output_dir = NSD_ROOT / f"subj{subj:02d}_all" / "outputs2"
    beta_file = output_dir / f"betas_averaged_final_{subj}.npy"

    print(f"\n{'='*60}")
    print(f"Subject {subj}: SigLIP2 vs CLIP Descriptions Unique Variance")
    print(f"Direction: {direction}")
    print(f"{'='*60}")

    if not beta_file.exists():
        print(f"Error: Beta file not found: {beta_file}")
        return

    # Load betas
    Y = np.load(beta_file).astype(np.float32)
    split_point = int(Y.shape[0] * 0.8)

    if direction in ["desc_over_siglip", "both"]:
        run_unique_variance(
            output_dir=output_dir,
            feature_A="descriptions",
            feature_B="siglip2",
            Y=Y,
            split_point=split_point,
            output_suffix=f"sub{subj:02d}_descriptionVsSiglip"
        )

    if direction in ["siglip_over_desc", "both"]:
        run_unique_variance(
            output_dir=output_dir,
            feature_A="siglip2",
            feature_B="descriptions",
            Y=Y,
            split_point=split_point,
            output_suffix=f"sub{subj:02d}_SiglipVsDescription"
        )

    print(f"\nSubject {subj} complete.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python unique_variance_siglip2.py <subject> <direction>")
        print("Direction options:")
        print("  desc_over_siglip  -- Unique variance of CLIP descriptions beyond SigLIP2")
        print("  siglip_over_desc  -- Unique variance of SigLIP2 beyond CLIP descriptions")
        print("  both              -- Run both directions")
        sys.exit(1)

    subject = int(sys.argv[1])
    direction = sys.argv[2]

    if subject < 1 or subject > 8:
        print("Error: Subject must be 1-8")
        sys.exit(1)
    if direction not in ["desc_over_siglip", "siglip_over_desc", "both"]:
        print("Error: Invalid direction. Choose: desc_over_siglip, siglip_over_desc, or both")
        sys.exit(1)

    process_subject(subject, direction)