# build_bag_of_words.py
"""
Construct bag-of-words and shuffled-word representations from narrative descriptions.

For each scene description:
1. Lowercase, remove punctuation and numbers.
2. Tokenize by whitespace.
3. Remove standard English stopwords and single-character tokens.
4. Deduplicate to create a bag-of-words (unordered set of content words).
5. Shuffle the bag to create a word-order-scrambled control.

Outputs (saved to subject output directory):
  bag.npy            -- bag-of-words per image
  bag_shuffled.npy   -- shuffled bag-of-words per image
"""

import numpy as np
import os
import re
import random
import sys
from pathlib import Path

# CONFIG
NSD_ROOT = Path("/path/to/NSD")     # CHANGE THIS TO YOUR NSD ROOT

# Standard English stopwords plus ordinal artifacts (e.g., "th" from "5th")
STOPWORDS = set([
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "when", "where", "how", "who", "which", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "at", "by", "for", "from", "in", "into", "of", "off", "on", "onto",
    "out", "over", "to", "up", "with", "about", "against", "between",
    "during", "before", "after", "above", "below", "under", "again",
    "further", "then", "once", "here", "there", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "should", "now",
    "th", "nd", "rd", "st"
])


def text_to_bag(text):
    """
    Convert a narrative text to a deduplicated bag of content words.

    Steps: lowercase, strip punctuation/numbers, tokenize, remove
    stopwords and single-character tokens, deduplicate.

    Args:
        text: Input description string

    Returns:
        List of unique content words
    """
    if not isinstance(text, str):
        return []

    # Lowercase
    text = text.lower()

    # Remove punctuation and numbers, keep only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize by whitespace
    tokens = text.split()

    # Filter stopwords and short tokens, deduplicate
    unique_words = set()
    for w in tokens:
        if w not in STOPWORDS and len(w) > 1:
            unique_words.add(w)

    return list(unique_words)


def process_subject(subj):
    """
    Build bag-of-words and shuffled variants for a single subject.

    Args:
        subj: Subject number (1-8)
    """
    output_dir = NSD_ROOT / f"subj{subj:02d}_all" / "outputs2"

    DESCS_FILE = output_dir / "descriptions.npy"
    BAG_FILE   = output_dir / "bag.npy"
    SHUF_FILE  = output_dir / "bag_shuffled.npy"

    # Check input exists
    if not DESCS_FILE.exists():
        print(f"Error: Descriptions file not found: {DESCS_FILE}")
        print("   Run extract_keys.py first.")
        return

    # Load descriptions
    print(f"Subject {subj}: Loading descriptions...")
    descriptions = np.load(DESCS_FILE, allow_pickle=True)
    n = len(descriptions)
    print(f"  Loaded {n} descriptions.")

    # Build bag-of-words
    print("  Building bag-of-words...")
    bags = []
    for desc in descriptions:
        bag = text_to_bag(desc)
        bags.append(bag)

    bags_array = np.array(bags, dtype=object)
    print(f"  Saving to {BAG_FILE}")
    np.save(str(BAG_FILE), bags_array)

    # Build shuffled variant
    print("  Building shuffled bags...")
    shuffled_bags = []
    for bag in bags:
        bag_copy = list(bag)
        random.shuffle(bag_copy)
        shuffled_bags.append(bag_copy)

    shuffled_array = np.array(shuffled_bags, dtype=object)
    print(f"  Saving to {SHUF_FILE}")
    np.save(str(SHUF_FILE), shuffled_array)

    # Verification sample
    idx = random.choice(range(n))
    print(f"\n  Sample verification (index {idx}):")
    print(f"  Original text: {descriptions[idx]}")
    print(f"  Bag-of-words : {bags_array[idx]}")
    print(f"  Shuffled bag : {shuffled_array[idx]}")

    # Statistics
    lengths = [len(b) for b in bags]
    max_len = max(lengths)
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    print(f"\n  Statistics:")
    print(f"    Max bag length: {max_len}")
    print(f"    Mean bag length: {avg_len:.2f}")

    print(f"\nSubject {subj} complete. Bag files saved to {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python build_bag_of_words.py <subject_number>")
        print("Example: python build_bag_of_words.py 1")
        sys.exit(1)

    subject = int(sys.argv[1])
    if subject < 1 or subject > 8:
        print("Error: Subject number must be between 1 and 8")
        sys.exit(1)

    process_subject(subject)