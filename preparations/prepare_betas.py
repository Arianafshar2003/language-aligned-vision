# prepare_betas.py
"""
Prepare averaged beta values from NSD single-trial estimates.

Loads session-wise beta HDF5 files, applies voxel-wise z-scoring
within each session, accumulates trials across sessions, and
averages repeated image presentations. Outputs are saved as:
  - betas_averaged_final_{subject}.npy  (n_images x n_voxels)
  - image_paths_final_{subject}.npy     (n_images, string paths)
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import gc
import sys

# CONFIGURATION
NSD_ROOT = Path("/path/to/NSD")     # CHANGE THIS TO YOUR NSD ROOT
IMAGES_DIR = NSD_ROOT / "images"
STIM_INFO_FILE = NSD_ROOT / "nsd_stim_info_merged.csv"

# HELPERS
def find_beta_file(base_dir, hemi, sess):
    """
    Locate the beta file for a given hemisphere and session.

    Searches multiple naming conventions and subdirectory structures
    to accommodate variations in NSD file organization across subjects.
    """
    patterns = [
        f"{hemi}.betas_session{sess:02d}.hdf5",
        f"{hemi}.betas_session{sess:02d}.hdf",
        f"{hemi}.betas_session{sess:02d}_2.hdf5",
        f"{hemi}.betas_session{sess:02d}_2.hdf",
        f"{hemi}_betas_session{sess:02d}_2.hdf5",
        f"{hemi}_betas_session{sess:02d}_2.hdf",
        f"{hemi}_rh_betas_session{sess:02d}.hdf5",
        f"{hemi}_rh_betas_session{sess:02d}.hdf",
    ]

    search_dirs = [
        base_dir,
        base_dir / "func1pt8mm" / "betas_fithrf_GLMdenoise_RR",
        base_dir / "betas_fithrf_GLMdenoise_RR"
    ]

    for d in search_dirs:
        if not d.exists():
            continue
        for p in patterns:
            candidate = d / p
            if candidate.exists():
                return candidate

    return None


def load_betas_voxel_first(path):
    """
    Load beta data and transpose to (n_voxels, n_trials).

    Uses chunked loading for memory efficiency.
    """
    with h5py.File(path, 'r') as f:
        key = 'betas' if 'betas' in f.keys() else list(f.keys())[0]
        dset = f[key]
        n_trials, n_vox = dset.shape

        betas = np.empty((n_vox, n_trials), dtype=np.float32)
        chunk_size = 10000
        for start in range(0, n_vox, chunk_size):
            end = min(start + chunk_size, n_vox)
            betas[start:end, :] = dset[:, start:end].T

    return betas


def zscore_axis1_inplace(data, chunk_size=10000):
    """
    Z-score normalize voxel-wise (modifies array in place).

    Each voxel is standardized across trials to have mean 0 and
    standard deviation 1.
    """
    n_vox = data.shape[0]
    for start in range(0, n_vox, chunk_size):
        end = min(start + chunk_size, n_vox)
        block = data[start:end, :]

        mean = block.mean(axis=1, keepdims=True, dtype=np.float32)
        std = block.std(axis=1, keepdims=True, dtype=np.float32)
        std[std == 0] = 1.0

        block -= mean
        block /= std


# SUBJECT PROCESSING
def process_subject(subj):
    """
    Main processing pipeline for a single subject.

    Steps:
    1. Identify subject directory and set output paths
    2. Load stimulus metadata and map trials to sessions
    3. Determine voxel dimensions from session 1
    4. Process each session: load betas, z-score, accumulate
    5. Average across repeated image presentations
    6. Save averaged betas and corresponding image paths
    """
    print("\n" + "=" * 50)
    print(f"Processing Subject {subj}")
    print("=" * 50)

    # -- Path setup --
    BETAS_DIR = NSD_ROOT / f"S{subj}"
    if not BETAS_DIR.exists():
        BETAS_DIR = NSD_ROOT / f"subj{subj:02d}"

    if not BETAS_DIR.exists():
        print(f"Error: Subject {subj} directory not found in {NSD_ROOT}")
        print("   Expected naming: 'S1' or 'subj01'")
        return

    OUTPUT_DIR = NSD_ROOT / f"subj{subj:02d}_all" / "outputs2"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_BETAS = OUTPUT_DIR / f"betas_averaged_final_{subj}.npy"
    OUTPUT_PATHS = OUTPUT_DIR / f"image_paths_final_{subj}.npy"

    if OUTPUT_BETAS.exists():
        print(f"Output file already exists: {OUTPUT_BETAS}")
        print("   Delete to re-run.")
        return

    # -- Load metadata --
    print("Loading stimulus info...")
    if not STIM_INFO_FILE.exists():
        print(f"Error: Stimulus info file missing: {STIM_INFO_FILE}")
        return

    stim_info = pd.read_csv(STIM_INFO_FILE)

    sessions_map = {1: 40, 2: 40, 3: 32, 4: 30, 5: 40, 6: 32, 7: 40, 8: 30}
    max_sessions = sessions_map.get(subj, 40)

    col_name = f"subject{subj}"
    if col_name not in stim_info.columns:
        print(f"Error: Column {col_name} not found in CSV.")
        return

    stim_sub = stim_info[stim_info[col_name] == 1]
    stim_sub_unique = stim_sub.drop_duplicates(subset=["nsdId"]).reset_index(drop=True)
    subject_ids = stim_sub_unique["nsdId"].astype(int).tolist()
    nsd_to_idx_map = {nsd_id: i for i, nsd_id in enumerate(subject_ids)}
    n_images = len(subject_ids)
    print(f"Found {n_images} unique images for Subject {subj}")

    nsd_to_coco = dict(zip(
        stim_sub_unique["nsdId"].astype(int),
        stim_sub_unique["cocoId"].astype(int)
    ))

    # -- Map trials to sessions --
    print("Mapping trials to sessions...")
    rep_cols = [f"subject{subj}_rep{i}" for i in range(3)]
    session_trials = {s: [] for s in range(1, max_sessions + 1)}

    for _, row in stim_sub_unique.iterrows():
        nsd_id = int(row["nsdId"])
        for col in rep_cols:
            tid = row.get(col)
            if pd.isna(tid):
                continue
            tid = int(tid) - 1  # 1-based to 0-based
            sess_id = tid // 750 + 1
            trial_idx_in_sess = tid % 750

            if sess_id <= max_sessions:
                session_trials[sess_id].append({
                    'nsd_id': nsd_id,
                    'trial_idx': trial_idx_in_sess
                })

    # -- Determine voxel dimensions --
    print("Checking voxel dimensions...")
    lh_path = find_beta_file(BETAS_DIR, "lh", 1)
    rh_path = find_beta_file(BETAS_DIR, "rh", 1)

    if lh_path is None or rh_path is None:
        print("Error: Could not find Session 01 beta files.")
        print(f"   Looked in: {BETAS_DIR}")
        return

    with h5py.File(lh_path, 'r') as f:
        key = 'betas' if 'betas' in f.keys() else list(f.keys())[0]
        n_vox_lh = f[key].shape[1]
    with h5py.File(rh_path, 'r') as f:
        key = 'betas' if 'betas' in f.keys() else list(f.keys())[0]
        n_vox_rh = f[key].shape[1]

    n_vox_total = n_vox_lh + n_vox_rh
    print(f"   LH: {n_vox_lh}  RH: {n_vox_rh}  Total: {n_vox_total}")

    # -- Initialize accumulators --
    betas_sum = np.zeros((n_images, n_vox_total), dtype=np.float32)
    repetition_count = np.zeros(n_images, dtype=np.int16)

    # -- Process sessions --
    print(f"Processing {max_sessions} sessions...")

    for sess in range(1, max_sessions + 1):
        trials = session_trials[sess]
        if not trials:
            print(f"   Session {sess:02d}: No trials (skipping)")
            continue

        lh_file = find_beta_file(BETAS_DIR, "lh", sess)
        rh_file = find_beta_file(BETAS_DIR, "rh", sess)

        if lh_file is None or rh_file is None:
            print(f"   Session {sess:02d}: Beta files missing (skipping)")
            continue

        # Left hemisphere
        lh_data = load_betas_voxel_first(lh_file)
        zscore_axis1_inplace(lh_data)

        for tr in trials:
            idx = nsd_to_idx_map[tr['nsd_id']]
            betas_sum[idx, :n_vox_lh] += lh_data[:, tr['trial_idx']]
        del lh_data

        # Right hemisphere
        rh_data = load_betas_voxel_first(rh_file)
        zscore_axis1_inplace(rh_data)

        for tr in trials:
            idx = nsd_to_idx_map[tr['nsd_id']]
            betas_sum[idx, n_vox_lh:] += rh_data[:, tr['trial_idx']]
        del rh_data

        # Update repetition counts
        for tr in trials:
            idx = nsd_to_idx_map[tr['nsd_id']]
            repetition_count[idx] += 1

        gc.collect()
        sys.stdout.write(f"\r   Session {sess:02d} complete")
        sys.stdout.flush()

    # -- Average across repetitions --
    print("\nAveraging repetitions...")
    repetition_count[repetition_count == 0] = 1
    betas_avg = betas_sum / repetition_count[:, np.newaxis]

    del betas_sum
    gc.collect()

    # -- Save betas --
    print(f"Saving betas to {OUTPUT_BETAS}...")
    np.save(OUTPUT_BETAS, betas_avg)

    # -- Save image paths --
    print("Saving image paths...")
    image_paths_list = []
    missing_imgs = 0

    for nsd_id in subject_ids:
        coco_id = nsd_to_coco[nsd_id]
        fname = f"{coco_id}.jpg"
        p = IMAGES_DIR / fname

        if p.exists():
            image_paths_list.append(str(p))
        else:
            image_paths_list.append("")
            missing_imgs += 1

    np.save(OUTPUT_PATHS, np.array(image_paths_list, dtype=object))

    print(f"Subject {subj} complete.")
    print(f"   Betas shape: {betas_avg.shape}")
    print(f"   Missing images: {missing_imgs}")


# ENTRY POINT
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prepare_betas.py <subject_number>")
        print("Example: python prepare_betas.py 1")
        sys.exit(1)

    subject = int(sys.argv[1])
    if subject < 1 or subject > 8:
        print("Error: Subject number must be between 1 and 8")
        sys.exit(1)

    process_subject(subject)