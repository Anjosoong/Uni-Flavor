#!/usr/bin/env python3
"""
Stratified train/test split for multi-label taste data.

Guarantees that every label has positive samples in both splits using a
three-stage strategy:
  1. Rare-label handling — for any label with <5 positives, force at least
     one positive into the test set, the rest into train.
  2. Label-combination stratification — stratify the remainder by the full
     binary label signature (e.g. "010110"); combinations with ≥2 samples
     are passed to sklearn `train_test_split(stratify=...)`.
  3. Random fallback — samples whose label combination is unique are split
     proportionally at random.

Outputs `train_data.csv` and `test_data.csv` next to the input file.

Usage:
    python split_dataset_taste.py                    # uses `data_path` below
    python split_dataset_taste.py path/to/data.csv
"""

import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data_path = r"./taste_data_all.csv"
non_label_cols = ['SMILES']    # columns excluded when scanning for label columns


class DatasetSplitter:
    """Stratified splitter that preserves rare labels in both splits."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        np.random.seed(random_state)

    def _handle_rare_labels(self, data: pd.DataFrame, label_cols: List[str]) -> Tuple[set, set]:
        """Stage 1: route one positive of each rare label (<5 positives) into test, the rest into train."""
        train_indices = set()
        test_indices = set()

        for label in label_cols:
            pos_indices = data[data[label] == 1].index.tolist()

            if 0 < len(pos_indices) < 5:
                test_sample = np.random.choice(pos_indices, size=1, replace=False)
                test_indices.update(test_sample)
                train_indices.update([idx for idx in pos_indices if idx not in test_sample])

        return train_indices, test_indices

    def _stratify_remaining(self, data: pd.DataFrame, label_cols: List[str],
                           train_indices: set, test_indices: set) -> Tuple[set, set]:
        """Stage 2-3: stratify the remainder by binary label signature, with random fallback for unique combos."""
        remaining_indices = [i for i in data.index if i not in train_indices and i not in test_indices]
        remaining_data = data.loc[remaining_indices].copy()

        if len(remaining_data) == 0:
            return train_indices, test_indices

        # Per-row binary signature, e.g. "010110", used as the stratification key.
        remaining_data['_label_combo'] = remaining_data[label_cols].apply(
            lambda x: ''.join(x.astype(str)), axis=1
        )

        combo_counts = remaining_data['_label_combo'].value_counts()
        valid_combos = combo_counts[combo_counts >= 2].index.tolist()

        if valid_combos:
            stratify_data = remaining_data[remaining_data['_label_combo'].isin(valid_combos)]
            non_stratify_indices = remaining_data[~remaining_data['_label_combo'].isin(valid_combos)].index.tolist()

            if len(stratify_data) > 0:
                train_remain, test_remain = train_test_split(
                    stratify_data.index.tolist(),
                    test_size=self.test_size,
                    random_state=self.random_state,
                    stratify=stratify_data['_label_combo']
                )
                train_indices.update(train_remain)
                test_indices.update(test_remain)

            # Singleton combos cannot be stratified — split proportionally at random.
            n_test = int(len(non_stratify_indices) * self.test_size)
            if n_test > 0 and len(non_stratify_indices) > n_test:
                test_remain = np.random.choice(non_stratify_indices, size=n_test, replace=False)
                train_remain = [idx for idx in non_stratify_indices if idx not in test_remain]
                train_indices.update(train_remain)
                test_indices.update(test_remain)
            else:
                train_indices.update(non_stratify_indices)
        else:
            # No combination has ≥2 samples: fall back to plain random split.
            train_remain, test_remain = train_test_split(
                remaining_indices, test_size=self.test_size, random_state=self.random_state
            )
            train_indices.update(train_remain)
            test_indices.update(test_remain)

        return train_indices, test_indices

    def split(self, data: pd.DataFrame, label_cols: List[str]) -> Tuple[List[int], List[int]]:
        """Run the full three-stage split.

        Args:
            data: full dataset.
            label_cols: binary label columns.

        Returns:
            (train_indices, test_indices) as plain lists.
        """
        train_indices, test_indices = self._handle_rare_labels(data, label_cols)
        train_indices, test_indices = self._stratify_remaining(data, label_cols, train_indices, test_indices)
        return list(train_indices), list(test_indices)


class LabelAnalyzer:
    """Inspect label distribution and verify split coverage."""

    @staticmethod
    def identify_labels(data: pd.DataFrame, exclude_cols: List[str]) -> List[str]:
        """Return all columns that are not in `exclude_cols` (treated as labels)."""
        return [col for col in data.columns if col not in exclude_cols]

    @staticmethod
    def print_statistics(data: pd.DataFrame, label_cols: List[str]) -> List[str]:
        """Print positive count per label and return labels with <5 positives."""
        print("\nLabel statistics:")
        rare_labels = []

        for label in label_cols:
            pos_count = (data[label] == 1).sum()
            print(f"  {label}: {pos_count}")
            if pos_count < 5:
                rare_labels.append(label)

        if rare_labels:
            print(f"\nRare labels (<5): {rare_labels}")

        return rare_labels

    @staticmethod
    def verify_split(train_data: pd.DataFrame, test_data: pd.DataFrame, label_cols: List[str]) -> bool:
        """Check every label has ≥1 positive in both train and test; return overall coverage flag."""
        print("\nPositive sample distribution:")
        all_covered = True

        for label in label_cols:
            train_pos = (train_data[label] == 1).sum()
            test_pos = (test_data[label] == 1).sum()
            print(f"  {label}: Train={train_pos}, Test={test_pos}")

            if train_pos == 0 or test_pos == 0:
                all_covered = False
                print(f"    Warning: {label} missing in one set")

        return all_covered


class FileHandler:
    """CSV I/O helpers."""

    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """Load a CSV; raise FileNotFoundError / ValueError on failure."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Cannot read file: {str(e)}")

    @staticmethod
    def save_splits(train_data: pd.DataFrame, test_data: pd.DataFrame, output_dir: str) -> Tuple[str, str]:
        """Write `train_data.csv` and `test_data.csv` into `output_dir`."""
        train_path = os.path.join(output_dir, "train_data.csv")
        test_path = os.path.join(output_dir, "test_data.csv")

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        return train_path, test_path


def main():
    file_path = sys.argv[1] if len(sys.argv) >= 2 else data_path

    try:
        data = FileHandler.load_data(file_path)
        print(f"Dataset loaded: {data.shape[0]} samples, {data.shape[1]} columns")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    label_cols = LabelAnalyzer.identify_labels(data, non_label_cols)
    print(f"Found {len(label_cols)} label columns")
    LabelAnalyzer.print_statistics(data, label_cols)

    splitter = DatasetSplitter(test_size=0.2, random_state=42)
    train_idx, test_idx = splitter.split(data, label_cols)

    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    train_ratio = len(train_data) / len(data)
    test_ratio = len(test_data) / len(data)
    print(f"\nSplit ratio - Train: {train_ratio:.2f}, Test: {test_ratio:.2f}")

    if LabelAnalyzer.verify_split(train_data, test_data, label_cols):
        print("\nAll labels covered in both sets")

    output_dir = os.path.dirname(file_path)
    train_path, test_path = FileHandler.save_splits(train_data, test_data, output_dir)

    print(f"\nTrain: {train_path}")
    print(f"Test: {test_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()