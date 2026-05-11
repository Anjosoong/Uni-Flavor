#!/usr/bin/env python3
"""
Apply transformers.pkl (selected_features + imputer + scaler) to
the OM Mordred feature array.

Input:
  ../mordred_descriptors/mordred_output/coconut_mordred_descriptors_features.npy      (N, D)  float64
  ../mordred_descriptors/mordred_output/coconut_mordred_descriptors_descriptor_names.txt       D descriptor names
  ../../../Odor/mordred/odor_descriptors/transformers.pkl                                      selected features + imputer + scaler

Output (in output_dir):
  om_mordred_processed.npy              (N, 300)  float32
  om_mordred_processed_feature_names.txt            300 selected feature names
"""

import pickle
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Paths
mordred_feat_path = Path(r"../mordred_descriptors/mordred_output/coconut_mordred_descriptors_features.npy")
descriptor_names  = Path(r"../mordred_descriptors/mordred_output/coconut_mordred_descriptors_descriptor_names.txt")
transformers_pkl  = Path(r"../../../Odor/mordred/odor_descriptors/transformers.pkl")
output_dir        = Path(r"./mordred_processed")

chunk_size = 10_000   # rows processed at once to limit RAM usage


# Reproduce dataclass needed for pickle compat
@dataclass
class ImprovedFastConfig:
    """Stub matching the saved dataclass so pickle can reconstruct config."""
    train_descriptor_path: str = ""
    test_descriptor_path:  str = ""
    train_label_path:      str = ""
    output_dir:            str = ""
    missing_threshold:     float = 0.3
    variance_threshold:    float = 1e-5
    correlation_threshold: float = 0.95
    n_top_features:        int   = 300
    selection_method:      str   = "stability_shap"
    n_bootstrap:           int   = 50
    bootstrap_fraction:    float = 0.8
    stability_threshold:   float = 0.4
    stage1_max_features:   int   = 300
    use_threshold_first:   bool  = True
    l1_C:                  float = 0.05
    use_true_shap:         bool  = True
    shap_sample_size:      int   = 3000
    tree_n_estimators:     int   = 200
    tree_max_depth:        int   = 8
    use_aggregated_label:  bool  = True
    enable_stability_check: bool = True
    stability_check_seeds: List[int] = field(default_factory=lambda: [42, 43])
    random_seed:           int   = 42
    impute_strategy:       str   = "median"
    smiles_column:         str   = "SMILES"


class _CompatUnpickler(pickle.Unpickler):
    """Redirects __main__ class lookups to this module."""
    def find_class(self, module, name):
        if module == "__main__":
            current = sys.modules.get(__name__)
            if current is not None and hasattr(current, name):
                return getattr(current, name)
        return super().find_class(module, name)


def load_transformers(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        transformers = _CompatUnpickler(f).load()
    # Patch sklearn version incompatibility (_fit_dtype → _fill_dtype)
    imputer = transformers.get("imputer")
    if imputer is not None:
        if hasattr(imputer, "_fit_dtype") and not hasattr(imputer, "_fill_dtype"):
            imputer._fill_dtype = imputer._fit_dtype
    return transformers


def main() -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load descriptor column names
    print("Loading descriptor names...")
    with open(descriptor_names) as f:
        all_desc_names = [l.strip() for l in f if l.strip()]
    print(f"  Total descriptor columns: {len(all_desc_names)}")

    # Load transformers
    print("Loading transformers.pkl...")
    transformers     = load_transformers(transformers_pkl)
    selected_features: List[str] = transformers["selected_features"]
    imputer          = transformers["imputer"]
    scaler           = transformers["scaler"]
    print(f"  Selected features: {len(selected_features)}")

    # Map selected feature names to column indices in the full descriptor array
    name2col = {name: i for i, name in enumerate(all_desc_names)}
    missing  = [f for f in selected_features if f not in name2col]
    if missing:
        raise ValueError(
            f"{len(missing)} selected features not found in descriptor names file: {missing[:5]}"
        )
    col_indices = np.array([name2col[f] for f in selected_features], dtype=np.intp)
    print(f"  Column index range: {col_indices.min()} – {col_indices.max()}")

    # Load source features as memmap
    print("Opening source Mordred features (memmap)...")
    src = np.lib.format.open_memmap(str(mordred_feat_path), mode="r")
    n_rows, n_cols_src = src.shape
    print(f"  Source shape: {src.shape}")
    assert n_cols_src == len(all_desc_names), (
        f"Column count mismatch: source={n_cols_src}, names={len(all_desc_names)}"
    )

    # Pre-allocate output memmap
    out_path = output_dir / "om_mordred_processed.npy"
    print(f"Allocating output: {out_path}  ({n_rows} × {len(selected_features)}) float32")
    dst = np.lib.format.open_memmap(
        str(out_path), mode="w+", dtype=np.float32,
        shape=(n_rows, len(selected_features))
    )

    # Process in chunks
    n_chunks = (n_rows + chunk_size - 1) // chunk_size
    print(f"Processing {n_rows:,} rows in {n_chunks} chunks of {chunk_size:,}...")

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end   = min(start + chunk_size, n_rows)

        # Select only the 300 needed columns from the full 1613
        chunk = src[start:end][:, col_indices].astype(np.float64)

        # Replace inf/-inf with NaN before imputation
        chunk = np.where(np.isinf(chunk), np.nan, chunk)

        # Apply imputer (median fill) then scaler (standardize)
        # Wrap in DataFrame to match the feature names the transformers were fitted with
        chunk = imputer.transform(pd.DataFrame(chunk, columns=selected_features))
        chunk = scaler.transform(pd.DataFrame(chunk, columns=selected_features))

        # Clip extreme values caused by Mordred numerical overflow on large molecules
        # (affects <0.3% of rows; ±5 sigma is standard practice for robust inference)
        chunk = np.clip(chunk, -5.0, 5.0)

        dst[start:end] = chunk.astype(np.float32)

        if (chunk_idx + 1) % 10 == 0 or (chunk_idx + 1) == n_chunks:
            print(f"  [{chunk_idx+1:>4}/{n_chunks}]  rows {start:>7}–{end:>7}")

    dst.flush()
    del dst

    # Save feature names
    names_path = output_dir / "om_mordred_processed_feature_names.txt"
    with open(names_path, "w") as f:
        f.write("\n".join(selected_features))

    # Sanity check
    result = np.lib.format.open_memmap(str(out_path), mode="r")
    nan_count = np.isnan(result).sum()
    print("Done!")
    print(f"  Output shape : {result.shape}")
    print(f"  Output dtype : {result.dtype}")
    print(f"  NaN cells    : {nan_count}")
    print(f"  Feature array: {out_path}")
    print(f"  Feature names: {names_path}")
    print(f"  Value range  : [{result.min():.4f}, {result.max():.4f}]")


if __name__ == "__main__":
    main()
