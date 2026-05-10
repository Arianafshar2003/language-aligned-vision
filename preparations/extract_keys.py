# extract_keys.py
"""
Parse structured JSON descriptions from Qwen output and extract components.

Reads the raw Qwen-generated descriptions (.npy), parses each as JSON,
and extracts four structured fields:
  - descriptions: narrative scene description
  - objects: list of object terms
  - verbs: list of action/verb terms
  - adjectives: list of attribute terms

Also validates image path alignment and saves a filtered path file.

Outputs (saved to subject output directory):
  descriptions.npy, objects.npy, verbs.npy, adjectives.npy, paths.npy
"""

import numpy as np
import ast
import os
import random
import sys
from pathlib import Path

# CONFIG
NSD_ROOT = Path("/path/to/NSD")     # CHANGE THIS TO YOUR NSD ROOT


def process_subject(subj):
    """
    Extract structured fields from Qwen descriptions for a single subject.

    Args:
        subj: Subject number (1-8)
    """
    output_dir = NSD_ROOT / f"subj{subj:02d}_all" / "outputs2"

    SOURCE_FILE = output_dir / f"qwen_semantic_descriptions_{subj}.npy"
    PATHS_FILE  = output_dir / f"image_paths_final_{subj}.npy"

    # Check inputs exist
    if not SOURCE_FILE.exists():
        print(f"Error: Source file not found: {SOURCE_FILE}")
        print("   Run generate_qwen_descriptions.py first.")
        return

    if not PATHS_FILE.exists():
        print(f"Error: Image paths file not found: {PATHS_FILE}")
        print("   Run prepare_betas.py first.")
        return

    # Load files
    print(f"Subject {subj}: Loading descriptions and image paths...")
    raw_data = np.load(SOURCE_FILE, allow_pickle=True)
    image_paths = np.load(PATHS_FILE, allow_pickle=True)

    print(f"  Loaded {len(raw_data)} descriptions and {len(image_paths)} image paths.")

    # Align lengths
    n = min(len(raw_data), len(image_paths))
    if len(raw_data) != len(image_paths):
        print(f"  Warning: Length mismatch ({len(raw_data)} vs {len(image_paths)}), truncating to {n}.")
    else:
        print(f"  Counts aligned ({n}).")

    # Parse JSON-like strings
    clean_data = []
    for i, item in enumerate(raw_data[:n]):
        text = str(item).strip().removeprefix("```json").removesuffix("```").strip()
        text = text.replace('\\n', ' ').replace('\r', '').strip()
        try:
            parsed = ast.literal_eval(text)
            clean_data.append(parsed)
        except Exception:
            print(f"  Warning: Skipping index {i}, cannot parse.")

    print(f"  Parsed {len(clean_data)} valid entries out of {n}.")

    # Extract fields
    descriptions = [d.get("description", "") for d in clean_data]
    objects      = [d.get("objects", []) for d in clean_data]
    verbs        = [d.get("verbs", []) for d in clean_data]
    adjectives   = [d.get("adjectives", []) for d in clean_data]
    paths_final  = image_paths[:len(clean_data)]

    # Save outputs
    np.save(str(output_dir / "descriptions.npy"), np.array(descriptions, dtype=object))
    np.save(str(output_dir / "objects.npy"),      np.array(objects, dtype=object))
    np.save(str(output_dir / "verbs.npy"),        np.array(verbs, dtype=object))
    np.save(str(output_dir / "adjectives.npy"),   np.array(adjectives, dtype=object))
    np.save(str(output_dir / "paths.npy"),        np.array(paths_final, dtype=object))

    # Sample check
    sample_ids = random.sample(range(len(clean_data)), min(10, len(clean_data)))
    print("\nSample extracted entries:")
    for i in sample_ids:
        desc = descriptions[i]
        obj  = objects[i]
        vrb  = verbs[i]
        adj  = adjectives[i]
        path = paths_final[i]
        exists = os.path.exists(path)
        status = "[OK]" if exists else "[MISSING]"
        print(f"  [{i:5d}] {status}")
        print(f"  Desc : {desc}")
        print(f"  Objs : {obj}")
        print(f"  Verbs: {vrb}")
        print(f"  Adjs : {adj}")
        print(f"  Path : {path}")
        print("  " + "-" * 46)

    # Summary
    missing_paths = [p for p in paths_final if not os.path.exists(p)]
    if missing_paths:
        print(f"\n  Warning: {len(missing_paths)} image paths missing from disk.")
    else:
        print("\n  All image paths exist and accessible.")

    print(f"\nSubject {subj} complete. Five .npy files saved to {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_keys.py <subject_number>")
        print("Example: python extract_keys.py 1")
        sys.exit(1)

    subject = int(sys.argv[1])
    if subject < 1 or subject > 8:
        print("Error: Subject number must be between 1 and 8")
        sys.exit(1)

    process_subject(subject)