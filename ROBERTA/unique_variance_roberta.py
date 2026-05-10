# unique_variance_roberta.py
"""
Variance partitioning analysis for RoBERTa text feature spaces.

For a pair of feature spaces (A, B):
1. Concatenates RAW (non-PCA) RoBERTa features from A and B.
2. Applies joint PCA (97% variance retained) on concatenated features.
3. Fits voxelwise ridge regression on the joint PCA space.
4. Computes unique variance of A beyond B via bootstrap resampling.
5. Applies FDR correction and saves:
   - R2_A, R2_B, R2_AB, Unique_A (R2_AB - R2_B), significance mask.

Also computes base model R2 and significance for all individual features.

Usage:
  python unique_variance_roberta.py <subject_number> <feature_A> <feature_B>

Example:
  python unique_variance_roberta.py 1 descriptions bag_shuffled
  python unique_variance_roberta.py 1 descriptions shuffled
  python unique_variance_roberta.py 1 bag_shuffled descriptions
  python unique_variance_roberta.py 1 shuffled descriptions
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
BATCH_SIZE = 60000
N_SPLITS = 7
N_BOOTSTRAPS = 2000
BOOT_BATCH_SIZE = 4
CHUNK_SIZE = 5000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RAW (non-PCA) RoBERTa feature file mapping
RAW_FEATURES = {
    "descriptions":          "features_roberta_base_descriptions.npy",
    "shuffled":              "features_roberta_base_descriptions_shuffled.npy",
    "bag_shuffled":          "features_roberta_base_bag_shuffled.npy",
    "bag":                   "features_roberta_base_bag.npy",
    "objects":               "features_roberta_base_objects.npy",
    "verbs":                 "features_roberta_base_verbs.npy",
    "adjectives":            "features_roberta_base_adjectives.npy",
}

# Single-feature ridge predictions
SINGLE_PREDS = {
    "descriptions":          "Y_pred_roberta_base_pca_descriptions.npy",
    "shuffled":              "Y_pred_roberta_base_shuffled_pca.npy",
    "bag_shuffled":          "Y_pred_roberta_base_bag_shuffled_pca.npy",
    "bag":                   "Y_pred_roberta_base_bag_pca.npy",
    "objects":               "Y_pred_roberta_base_objects_pca.npy",
    "verbs":                 "Y_pred_roberta_base_verbs_pca.npy",
    "adjectives":            "Y_pred_roberta_base_adjectives_pca.npy",
}

# Base features for which to compute standalone R2 and significance
BASE_FEATURES = [
    "descriptions", "shuffled", "bag_shuffled",
    "adjectives", "verbs", "objects"
]

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


def bootstrap_significance(y_true_t, y_pred_t, n_voxels, n_test, null_value=0):
    """
    Bootstrap test for R2 > null_value.

    Returns p-values and FDR-corrected significance mask.
    """
    batches = int(np.ceil(N_BOOTSTRAPS / BOOT_BATCH_SIZE))
    p_accum = []

    start_time = time.time()
    for b in range(batches):
        torch.cuda.empty_cache()
        gc.collect()

        bs_start = b * BOOT_BATCH_SIZE
        bs_end = min((b + 1) * BOOT_BATCH_SIZE, N_BOOTSTRAPS)
        current_bs = bs_end - bs_start

        boot_idx = torch.randint(0, n_test, (current_bs, n_test), device=DEVICE)
        batch_p_vals = torch.zeros(n_voxels, device="cpu")

        for voxel_start in range(0, n_voxels, CHUNK_SIZE):
            voxel_end = min(voxel_start + CHUNK_SIZE, n_voxels)
            v_range = slice(voxel_start, voxel_end)

            gather_idx = boot_idx.unsqueeze(1).expand(-1, voxel_end - voxel_start, -1)
            y_t_b = y_true_t[v_range].unsqueeze(0).expand(current_bs, -1, -1).gather(2, gather_idx)
            y_p_b = y_pred_t[v_range].unsqueeze(0).expand(current_bs, -1, -1).gather(2, gather_idx)

            mean_t_bs = y_t_b.mean(dim=2, keepdim=True)
            ss_res = torch.sum((y_t_b - y_p_b) ** 2, dim=2)
            ss_tot = torch.sum((y_t_b - mean_t_bs) ** 2, dim=2)
            R2_boot = 1 - ss_res.float() / (ss_tot.float() + 1e-12)

            batch_p_vals[v_range] = torch.mean((R2_boot <= null_value).float(), dim=0).cpu()

            del y_t_b, y_p_b, R2_boot
            torch.cuda.empty_cache()

        p_accum.append(batch_p_vals)

        elapsed = time.time() - start_time
        if b > 0:
            eta = (elapsed / (b + 1)) * (batches - b - 1)
            print(f"    Bootstrap batch {b+1}/{batches} | "
                  f"Elapsed: {elapsed/60:.1f} min | ETA: {eta/60:.1f} min")

    p_vals = torch.stack(p_accum).mean(dim=0).numpy()
    reject, _ = fdrcorrection(p_vals, alpha=0.05)
    mask = reject.astype(np.uint8)

    return p_vals, mask


def bootstrap_unique_variance(y_true_t, y_pred_B_t, y_pred_AB_t, n_voxels, n_test):
    """
    Bootstrap test for unique variance of A beyond B.

    H0: Unique_A <= 0  (R2_AB - R2_B <= 0)
    """
    batches = int(np.ceil(N_BOOTSTRAPS / BOOT_BATCH_SIZE))
    p_accum = []

    start_time = time.time()
    for b in range(batches):
        torch.cuda.empty_cache()
        gc.collect()

        bs_start = b * BOOT_BATCH_SIZE
        bs_end = min((b + 1) * BOOT_BATCH_SIZE, N_BOOTSTRAPS)
        current_bs = bs_end - bs_start

        boot_idx = torch.randint(0, n_test, (current_bs, n_test), device=DEVICE)
        batch_p_vals = torch.zeros(n_voxels, device="cpu")

        for voxel_start in range(0, n_voxels, CHUNK_SIZE):
            voxel_end = min(voxel_start + CHUNK_SIZE, n_voxels)
            v_range = slice(voxel_start, voxel_end)

            gather_idx = boot_idx.unsqueeze(1).expand(-1, voxel_end - voxel_start, -1)
            y_t_b = y_true_t[v_range].unsqueeze(0).expand(current_bs, -1, -1).gather(2, gather_idx)
            y_pj_b = y_pred_AB_t[v_range].unsqueeze(0).expand(current_bs, -1, -1).gather(2, gather_idx)
            y_pb_b = y_pred_B_t[v_range].unsqueeze(0).expand(current_bs, -1, -1).gather(2, gather_idx)

            mean_t_bs = y_t_b.mean(dim=2, keepdim=True)
            ss_tot = torch.sum((y_t_b - mean_t_bs) ** 2, dim=2)
            ss_res_j = torch.sum((y_t_b - y_pj_b) ** 2, dim=2)
            ss_res_b = torch.sum((y_t_b - y_pb_b) ** 2, dim=2)

            R2_j_boot = 1 - ss_res_j.float() / (ss_tot.float() + 1e-12)
            R2_b_boot = 1 - ss_res_b.float() / (ss_tot.float() + 1e-12)
            boot_unique = R2_j_boot - R2_b_boot

            batch_p_vals[v_range] = torch.mean((boot_unique <= 0).float(), dim=0).cpu()

            del y_t_b, y_pj_b, y_pb_b, boot_unique
            torch.cuda.empty_cache()

        p_accum.append(batch_p_vals)

        elapsed = time.time() - start_time
        if b > 0:
            eta = (elapsed / (b + 1)) * (batches - b - 1)
            print(f"    Bootstrap batch {b+1}/{batches} | "
                  f"Elapsed: {elapsed/60:.1f} min | ETA: {eta/60:.1f} min")

    p_vals = torch.stack(p_accum).mean(dim=0).numpy()
    reject, _ = fdrcorrection(p_vals, alpha=0.05)
    mask = reject.astype(np.uint8)

    return p_vals, mask


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


def process_subject(subj, feature_A, feature_B):
    """
    Run unique variance analysis for a pair of RoBERTa feature spaces.

    Also computes standalone R2 + significance for all base features.

    Args:
        subj: Subject number (1-8)
        feature_A: Key for feature A (e.g., 'descriptions')
        feature_B: Key for feature B (e.g., 'bag_shuffled')
    """
    output_dir = NSD_ROOT / f"subj{subj:02d}_all" / "outputs2"
    beta_file = output_dir / f"betas_averaged_final_{subj}.npy"

    print(f"\n{'='*60}")
    print(f"Subject {subj}: RoBERTa unique variance of {feature_A} beyond {feature_B}")
    print(f"{'='*60}")

    if not beta_file.exists():
        print(f"Error: Beta file not found: {beta_file}")
        return

    # Load betas
    Y = np.load(beta_file).astype(np.float32)
    split_point = int(Y.shape[0] * 0.8)
    Y_train = Y[:split_point]
    Y_test = Y[split_point:]
    n_voxels = Y.shape[1]

    # Standardize y_true for test set
    y_mean = Y_train.mean(axis=0, dtype=np.float32)
    y_std = Y_train.std(axis=0, dtype=np.float32)
    y_std[y_std == 0] = 1.0
    y_true_std = ((Y_test - y_mean) / y_std).astype(np.float32)

    # PART A: Standalone base model R2 and significance
    print("\n[Part A] Computing base model R2 and significance...")
    for base_feat in BASE_FEATURES:
        pred_file = output_dir / SINGLE_PREDS.get(base_feat, "")
        if not pred_file.exists():
            print(f"  Skipping {base_feat}: prediction file not found")
            continue

        print(f"  Processing: {base_feat}")
        y_pred = np.load(pred_file).astype(np.float32)
        if y_pred.shape[0] < y_pred.shape[1]:
            y_pred = y_pred.T

        y_true_t = torch.from_numpy(y_true_std).half().to(DEVICE, non_blocking=True)
        y_pred_t = torch.from_numpy(y_pred).half().to(DEVICE, non_blocking=True)

        n_vox, n_test = y_true_std.shape
        R2_actual = compute_r2_chunked(y_true_t, y_pred_t, CHUNK_SIZE)

        torch.cuda.set_per_process_memory_fraction(0.9)
        _, mask = bootstrap_significance(y_true_t, y_pred_t, n_vox, n_test, null_value=0)

        np.save(str(output_dir / f"R2_{base_feat}_roberta.npy"), R2_actual.cpu().numpy())
        np.save(str(output_dir / f"mask_{base_feat}_roberta.npy"), mask)
        print(f"    Significant voxels: {np.sum(mask)}")

        del y_pred_t, R2_actual, mask
        torch.cuda.empty_cache()
        gc.collect()

    # PART B: Unique variance (A beyond B)
    print(f"\n[Part B] Unique variance: {feature_A} beyond {feature_B}")

    feat_A_raw = output_dir / RAW_FEATURES[feature_A]
    feat_B_raw = output_dir / RAW_FEATURES[feature_B]
    pred_A_file = output_dir / SINGLE_PREDS[feature_A]
    pred_B_file = output_dir / SINGLE_PREDS[feature_B]

    missing = []
    if not feat_A_raw.exists():
        missing.append(f"Raw features A: {feat_A_raw}")
    if not feat_B_raw.exists():
        missing.append(f"Raw features B: {feat_B_raw}")
    if not pred_A_file.exists():
        missing.append(f"Prediction A: {pred_A_file}")
    if not pred_B_file.exists():
        missing.append(f"Prediction B: {pred_B_file}")

    if missing:
        print("  Error: Missing files:")
        for m in missing:
            print(f"    - {m}")
        return

    # Step B1: Concatenate raw features
    print("  [B1] Concatenating raw features...")
    X_A = np.load(feat_A_raw).astype(np.float32)
    X_B = np.load(feat_B_raw).astype(np.float32)
    X_A = np.nan_to_num(X_A, nan=0.0)
    X_B = np.nan_to_num(X_B, nan=0.0)
    X_joint_raw = np.concatenate([X_A, X_B], axis=1)
    print(f"    Joint raw shape: {X_joint_raw.shape}")

    # Step B2: Joint PCA
    print("  [B2] Joint PCA (97% variance)...")
    pca = PCA(n_components=0.97, svd_solver="full")
    X_joint_pca = pca.fit_transform(X_joint_raw)
    print(f"    Joint PCA shape: {X_joint_pca.shape}")

    # Step B3: Fit joint ridge
    print("  [B3] Fitting joint ridge regression...")
    X_train_joint = X_joint_pca[:split_point]
    X_test_joint = X_joint_pca[split_point:]
    Y_pred_joint = fit_ridge_joint(X_train_joint, X_test_joint, Y_train, Y_test, n_voxels)

    # Step B4: Load single predictions
    print("  [B4] Loading single predictions...")
    y_pred_A = np.load(pred_A_file).astype(np.float32)
    y_pred_B = np.load(pred_B_file).astype(np.float32)

    # Fix orientation
    if y_true_std.shape[0] < y_true_std.shape[1]:
        y_true_std = y_true_std.T
        y_pred_A = y_pred_A.T
        y_pred_B = y_pred_B.T
        Y_pred_joint = Y_pred_joint.T

    n_vox, n_test = y_true_std.shape

    # Step B5: Compute R2 and unique variance
    print("  [B5] Computing R2 and unique variance...")
    y_true_t = torch.from_numpy(y_true_std).half().to(DEVICE, non_blocking=True)
    y_pred_A_t = torch.from_numpy(y_pred_A).half().to(DEVICE, non_blocking=True)
    y_pred_B_t = torch.from_numpy(y_pred_B).half().to(DEVICE, non_blocking=True)
    y_pred_AB_t = torch.from_numpy(Y_pred_joint).half().to(DEVICE, non_blocking=True)

    R2_A = compute_r2_chunked(y_true_t, y_pred_A_t, CHUNK_SIZE)
    R2_B = compute_r2_chunked(y_true_t, y_pred_B_t, CHUNK_SIZE)
    R2_AB = compute_r2_chunked(y_true_t, y_pred_AB_t, CHUNK_SIZE)
    Unique_A = R2_AB - R2_B

    # Step B6: Bootstrap unique variance
    print("  [B6] Bootstrap test for unique variance...")
    torch.cuda.set_per_process_memory_fraction(0.9)
    _, mask_unique = bootstrap_unique_variance(y_true_t, y_pred_B_t, y_pred_AB_t, n_vox, n_test)

    n_sig = np.sum(mask_unique)
    print(f"    Significant voxels (Unique_A > 0): {n_sig} / {n_vox}")

    # Step B7: Save
    print("  [B7] Saving results...")
    pair_name = f"{feature_A}_{feature_B}"
    np.save(str(output_dir / f"R2_A_roberta_{pair_name}.npy"), R2_A.cpu().numpy())
    np.save(str(output_dir / f"R2_B_roberta_{pair_name}.npy"), R2_B.cpu().numpy())
    np.save(str(output_dir / f"R2_AB_roberta_{pair_name}.npy"), R2_AB.cpu().numpy())
    np.save(str(output_dir / f"Unique_A_roberta_{pair_name}.npy"), Unique_A.cpu().numpy())
    np.save(str(output_dir / f"mask_unique_A_roberta_{pair_name}.npy"), mask_unique)

    print(f"  Subject {subj}: {feature_A} vs {feature_B} complete.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python unique_variance_roberta.py <subject> <feature_A> <feature_B>")
        print("Feature options:", ", ".join(RAW_FEATURES.keys()))
        print("\nExample comparisons:")
        print("  python unique_variance_roberta.py 1 descriptions bag_shuffled")
        print("  python unique_variance_roberta.py 1 descriptions shuffled")
        print("  python unique_variance_roberta.py 1 bag_shuffled descriptions")
        print("  python unique_variance_roberta.py 1 shuffled descriptions")
        sys.exit(1)

    subject = int(sys.argv[1])
    feat_A = sys.argv[2]
    feat_B = sys.argv[3]

    if subject < 1 or subject > 8:
        print("Error: Subject must be 1-8")
        sys.exit(1)
    if feat_A not in RAW_FEATURES:
        print(f"Error: Unknown feature '{feat_A}'")
        sys.exit(1)
    if feat_B not in RAW_FEATURES:
        print(f"Error: Unknown feature '{feat_B}'")
        sys.exit(1)

    process_subject(subject, feat_A, feat_B)