# config/paths_config.py
"""
Central path configuration for all pipelines.

Edit NSD_ROOT to point to your local NSD dataset location.
All other paths are derived from this root.
"""

from pathlib import Path

# CHANGE THIS to your NSD dataset root directory
NSD_ROOT = Path("/path/to/NSD")

# Stimulus metadata (assumed to be in NSD_ROOT)
STIM_INFO_FILE = NSD_ROOT / "nsd_stim_info_merged.csv"
IMAGES_DIR = NSD_ROOT / "images"

# DINOv3 local weights filename
DINOV3_WEIGHTS = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

# Session configuration (NSD standard)
SESSIONS_MAP = {1: 40, 2: 40, 3: 32, 4: 30, 5: 40, 6: 32, 7: 40, 8: 30}


def get_subject_dir(subject_id: int) -> Path:
    """Get the output directory for a subject."""
    return NSD_ROOT / f"subj{subject_id:02d}_all" / "outputs2"