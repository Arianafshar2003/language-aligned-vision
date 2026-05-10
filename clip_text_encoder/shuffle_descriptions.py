# shuffle_descriptions.py
"""
Create word-order-shuffled versions of narrative descriptions.

Tokenizes each description by whitespace, randomly permutes word order,
and rejoins. Used as a syntactic-structure control in encoding analyses.
"""

import numpy as np
import random
import sys
from pathlib import Path

# CONFIG
NSD_ROOT = Path("/path/to/NSD")     # CHANGE THIS TO YOUR NSD ROOT


def process_subject(subj):
    """
    Generate shuffled descriptions for a single subject.

    Args:
        subj: Subject number (1-8)
    """
    output_dir = NSD_ROOT / f"subj{subj:02d}_all" / "outputs2"

    DESCRIPTIONS_FILE = output_dir / "descriptions.npy"
    OUTPUT_FILE = output_dir / "descriptions_shuffled.npy"

    if not DESCRIPTIONS_FILE.exists():
        print(f"Error: Descriptions file not found: {DESCRIPTIONS_FILE}")
        print("   Run extract_keys.py first.")
        return

    descriptions = np.load(DESCRIPTIONS_FILE, allow_pickle=True)
    print(f"Subject {subj}: Loaded {len(descriptions)} descriptions.")

    shuffled_descriptions = []
    for text in descriptions:
        if text is None or str(text).strip() == "":
            shuffled_descriptions.append("")
            continue
        words = str(text).strip().split()
        random.shuffle(words)
        shuffled_descriptions.append(" ".join(words))

    np.save(str(OUTPUT_FILE), np.array(shuffled_descriptions, dtype=object))
    print(f"Saved to {OUTPUT_FILE}")
    print(f"Example original : {descriptions[0]}")
    print(f"Example shuffled : {shuffled_descriptions[0]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python shuffle_descriptions.py <subject_number>")
        sys.exit(1)
    subject = int(sys.argv[1])
    if subject < 1 or subject > 8:
        print("Error: Subject number must be between 1 and 8")
        sys.exit(1)
    process_subject(subject)