#!/usr/bin/env python3
"""
Multi-label Dataset Splitter
Attempts to preserve multi-label coverage in both train and test sets.

Strategy:
  1. Random initial split (default 80/20)
  2. Per-label coverage repair: if any label has 0 positives in test (or train),
     move one positive sample across the boundary to fix it
  3. Overlap validation: assert no sample appears in both sets
  4. Summary statistics

Usage:
    python split_dataset_odor.py [input.csv]
    python split_dataset_odor.py            # uses DEFAULT_INPUT

Output:
    <input_dir>/train_data.csv
    <input_dir>/test_data.csv
"""

import os
import sys
import random
import pandas as pd
import numpy as np
from typing import List, Tuple

# Default path — override via CLI arg
DEFAULT_INPUT = r"./odor_data_all.csv"

# Columns that are NOT labels
NON_LABEL_COLS = ["SMILES"]

# Split config
TEST_SIZE    = 0.2
RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    """Load CSV and print basic shape info."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded : {path}")
    print(f"Shape  : {df.shape[0]} samples x {df.shape[1]} columns")
    return df


def get_label_cols(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    """Return all columns not in the exclude list (i.e., the label columns)."""
    return [c for c in df.columns if c not in exclude]


def print_label_stats(df: pd.DataFrame, label_cols: List[str]) -> None:
    """Print positive-sample counts per label, flagging zero and rare labels."""
    print(f"\nLabel positive counts ({len(label_cols)} labels):")
    for lbl in label_cols:
        n = int((df[lbl] == 1).sum())
        flag = "  *** ZERO ***" if n == 0 else ("  (rare <5)" if n < 5 else "")
        print(f"  {lbl}: {n}{flag}")


def initial_split(df: pd.DataFrame, test_size: float, seed: int) -> Tuple[List[int], List[int]]:
    """Plain random split, returns positional (iloc) indices."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_test    = max(1, int(len(df) * test_size))
    test_idx  = idx[:n_test].tolist()
    train_idx = idx[n_test:].tolist()
    return train_idx, test_idx


def repair_coverage(
    df: pd.DataFrame,
    label_cols: List[str],
    train_idx: List[int],
    test_idx: List[int],
    seed: int,
) -> Tuple[List[int], List[int]]:
    """
    For each label with 0 positives in test (or train), move one positive
    sample across the boundary.  Uses positional (iloc) indices.
    """
    rng = random.Random(seed)
    train_set = set(train_idx)
    test_set  = set(test_idx)

    moves_to_test  = 0
    moves_to_train = 0

    for lbl in label_cols:
        pos_in_train = [i for i in train_set if df.iloc[i][lbl] == 1]
        pos_in_test  = [i for i in test_set  if df.iloc[i][lbl] == 1]

        # Label absent from test: move one positive from train -> test
        if len(pos_in_test) == 0 and len(pos_in_train) > 0:
            pick = rng.choice(pos_in_train)
            train_set.discard(pick)
            test_set.add(pick)
            moves_to_test += 1

        # Label absent from train: move one positive from test -> train
        elif len(pos_in_train) == 0 and len(pos_in_test) > 0:
            pick = rng.choice(pos_in_test)
            test_set.discard(pick)
            train_set.add(pick)
            moves_to_train += 1

        # Both zero: label has NO positives anywhere, nothing to do

    print(f"\nCoverage repair: {moves_to_test} sample(s) moved to test, "
          f"{moves_to_train} sample(s) moved to train")

    return list(train_set), list(test_set)


def validate_no_overlap(train_idx: List[int], test_idx: List[int]) -> None:
    """Raise RuntimeError if any index appears in both splits."""
    overlap = set(train_idx) & set(test_idx)
    if overlap:
        raise RuntimeError(
            f"Overlap detected: {len(overlap)} sample(s) in both train and test. "
            f"First few: {sorted(overlap)[:10]}"
        )
    print("Overlap check  : PASSED")


def print_split_stats(
    df: pd.DataFrame,
    label_cols: List[str],
    train_idx: List[int],
    test_idx: List[int],
) -> None:
    """Print split sizes and per-label positive counts in each subset."""
    total = len(df)
    n_tr  = len(train_idx)
    n_te  = len(test_idx)
    print(f"\nSplit sizes    : train={n_tr} ({100*n_tr/total:.1f}%)  "
          f"test={n_te} ({100*n_te/total:.1f}%)")

    print(f"\nPer-label coverage in each split:")
    missing = []
    for lbl in label_cols:
        tr = int(df.iloc[train_idx][lbl].sum())
        te = int(df.iloc[test_idx][lbl].sum())
        flag = ""
        if tr == 0 or te == 0:
            flag = "  <<< MISSING"
            missing.append(lbl)
        print(f"  {lbl}: train={tr}  test={te}{flag}")

    print()
    if missing:
        print(f"WARNING: {len(missing)} label(s) still uncovered in one split: {missing}")
        print("  Likely cause: label has only 1 total positive sample.")
    else:
        print(f"All {len(label_cols)} labels have positives in both splits.")


def save_splits(
    df: pd.DataFrame,
    train_idx: List[int],
    test_idx: List[int],
    output_dir: str,
) -> Tuple[str, str]:
    """Write sorted train/test subsets to CSV and return their paths."""
    train_path = os.path.join(output_dir, "train_data.csv")
    test_path  = os.path.join(output_dir, "test_data.csv")
    df.iloc[sorted(train_idx)].to_csv(train_path, index=False)
    df.iloc[sorted(test_idx)].to_csv(test_path,  index=False)
    return train_path, test_path


def main():
    """Entry point: load data, split, repair coverage, validate, and save."""
    file_path  = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_INPUT
    df         = load_data(file_path)
    label_cols = get_label_cols(df, NON_LABEL_COLS)
    print(f"Labels : {len(label_cols)}")

    print_label_stats(df, label_cols)

    # Step 1: random split
    train_idx, test_idx = initial_split(df, TEST_SIZE, RANDOM_STATE)
    print(f"\nInitial split  : train={len(train_idx)}  test={len(test_idx)}")

    # Step 2: per-label coverage repair
    train_idx, test_idx = repair_coverage(df, label_cols, train_idx, test_idx, RANDOM_STATE)

    # Step 3: overlap check
    validate_no_overlap(train_idx, test_idx)

    # Step 4: report
    print_split_stats(df, label_cols, train_idx, test_idx)

    # Step 5: save
    output_dir = os.path.dirname(os.path.abspath(file_path))
    train_path, test_path = save_splits(df, train_idx, test_idx, output_dir)
    print(f"Saved  : {train_path}")
    print(f"Saved  : {test_path}")


if __name__ == "__main__":
    main()
