#!/usr/bin/env python3
"""
Align Mordred-processed features with UniMol features by SMILES matching.

Both datasets originate from coconut_calculation.csv.
UniMol processed 669052 of them (3 failed). Order may differ.
Primary alignment key: SMILES string from each dataset's metadata.

Prerequisites:
  Run preprocess_mordred_odor.py first to produce om_mordred_processed.npy.

Input:
  ./mordred_processed/om_mordred_processed.npy                               (N, 300) float32
  ../mordred_descriptors/mordred_output/coconut_mordred_descriptors_metadata.csv      id + SMILES
  ./unimol2_odor_feature/coconut_calculation_molecular_features.npy          (M, 768) float32
  ./unimol2_odor_feature/coconut_calculation_success_indices.npy             (M,)     int64
  ./unimol2_odor_feature/coconut_calculation_filtered_data.csv               SMILES column

Output (in output_dir):
  om_aligned_mordred.npy    (N, 300) float32  — Mordred features, SMILES-matched
  om_aligned_unimol.npy     (N, 768) float32  — UniMol  features, same row order
  om_aligned_smiles.txt     N lines            — SMILES for each aligned row
  om_aligned_metadata.csv   mordred_id + SMILES
  om_unmatched.txt          SMILES that could not be matched (for diagnostics)

  N = number of UniMol successes whose SMILES are found in Mordred metadata
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Paths
mordred_processed   = Path(r"./mordred_processed/om_mordred_processed.npy")
mordred_metadata    = Path(r"../mordred_descriptors/mordred_output/coconut_mordred_descriptors_metadata.csv")
unimol_features     = Path(r"./unimol2_odor_feature/coconut_calculation_molecular_features.npy")
unimol_success_idx  = Path(r"./unimol2_odor_feature/coconut_calculation_success_indices.npy")
unimol_filtered_csv = Path(r"./unimol2_odor_feature/coconut_calculation_filtered_data.csv")
output_dir          = Path(r"./aligned_features")


def main() -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate prerequisites
    for p in (mordred_processed, mordred_metadata,
              unimol_features, unimol_success_idx, unimol_filtered_csv):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing required file: {p}\n"
                "Run preprocess_mordred_odor.py first if om_mordred_processed.npy is missing."
            )

    # Load Mordred processed features (memmap)
    print("Loading Mordred processed features (memmap)...")
    mordred_full = np.lib.format.open_memmap(str(mordred_processed), mode="r")
    print(f"  Mordred full shape : {mordred_full.shape}")

    # Load Mordred metadata and build SMILES lookup
    print("Loading Mordred metadata and building SMILES index...")
    mordred_meta = pd.read_csv(mordred_metadata, usecols=["id", "SMILES"])
    assert len(mordred_meta) == mordred_full.shape[0], (
        f"Mordred metadata rows ({len(mordred_meta)}) != feature rows ({mordred_full.shape[0]})"
    )

    # Map SMILES → Mordred row index (first occurrence wins; log duplicates)
    smiles_to_mordred_row: dict[str, int] = {}
    mordred_dup_smiles: list[str] = []
    for row_idx, smi in enumerate(mordred_meta["SMILES"]):
        if smi not in smiles_to_mordred_row:
            smiles_to_mordred_row[smi] = row_idx
        else:
            mordred_dup_smiles.append(smi)
    print(f"  Unique SMILES in Mordred : {len(smiles_to_mordred_row):,}")
    if mordred_dup_smiles:
        print(f"  Duplicate SMILES in Mordred (first kept) : {len(mordred_dup_smiles):,}")

    # Load UniMol features and derive per-row SMILES
    print("Loading UniMol features and success indices...")
    unimol_feat = np.load(str(unimol_features))
    success_idx = np.load(str(unimol_success_idx))
    print(f"  UniMol features shape  : {unimol_feat.shape}")
    assert unimol_feat.shape[0] == success_idx.shape[0], (
        "Mismatch between UniMol feature rows and success_indices length"
    )

    print("Loading UniMol filtered_data (SMILES)...")
    # filtered_data.csv = df.iloc[success_indices] already — row j aligns with unimol_feat row j
    unimol_success_smiles = pd.read_csv(unimol_filtered_csv, usecols=["SMILES"])["SMILES"].values
    print(f"  UniMol successful SMILES : {len(unimol_success_smiles):,}")
    assert len(unimol_success_smiles) == unimol_feat.shape[0], (
        f"filtered_data rows ({len(unimol_success_smiles)}) != UniMol feature rows ({unimol_feat.shape[0]})"
    )

    # SMILES-based matching
    print("\nMatching UniMol SMILES → Mordred rows...")
    unimol_rows:  list[int] = []   # index into unimol_feat
    mordred_rows: list[int] = []   # index into mordred_full
    matched_smiles: list[str] = []
    unmatched: list[str] = []

    for j, smi in enumerate(unimol_success_smiles):
        mordred_row = smiles_to_mordred_row.get(smi)
        if mordred_row is not None:
            unimol_rows.append(j)
            mordred_rows.append(mordred_row)
            matched_smiles.append(smi)
        else:
            unmatched.append(smi)

    print(f"  Matched   : {len(unimol_rows):,}")
    print(f"  Unmatched : {len(unmatched):,}")
    if unmatched:
        print("  Unmatched SMILES (check om_unmatched.txt for full list):")
        for s in unmatched[:5]:
            print(f"    {s}")

    if len(unimol_rows) == 0:
        raise RuntimeError("No SMILES matched. Check that both datasets derive from the same source.")

    # Build aligned arrays
    print("\nBuilding aligned feature arrays...")
    unimol_rows_arr  = np.array(unimol_rows,  dtype=np.intp)
    mordred_rows_arr = np.array(mordred_rows, dtype=np.intp)

    unimol_aligned  = unimol_feat[unimol_rows_arr].astype(np.float32)    # (N, 768)
    mordred_aligned = mordred_full[mordred_rows_arr].astype(np.float32)  # (N, 300)
    print(f"  Aligned UniMol  shape : {unimol_aligned.shape}")
    print(f"  Aligned Mordred shape : {mordred_aligned.shape}")

    # Save outputs
    mordred_out  = output_dir / "om_aligned_mordred.npy"
    unimol_out   = output_dir / "om_aligned_unimol.npy"
    smiles_out   = output_dir / "om_aligned_smiles.txt"
    meta_out     = output_dir / "om_aligned_metadata.csv"
    unmatch_out  = output_dir / "om_unmatched.txt"

    print(f"\nSaving → {mordred_out}")
    np.save(str(mordred_out), mordred_aligned)

    print(f"Saving → {unimol_out}")
    np.save(str(unimol_out), unimol_aligned)

    print(f"Saving → {smiles_out}")
    with open(smiles_out, "w", encoding="utf-8") as f:
        f.write("\n".join(matched_smiles))

    print(f"Saving → {meta_out}")
    aligned_meta = mordred_meta.iloc[mordred_rows_arr].reset_index(drop=True)
    aligned_meta.to_csv(meta_out, index=False)

    if unmatched:
        print(f"Saving → {unmatch_out}")
        with open(unmatch_out, "w", encoding="utf-8") as f:
            f.write("\n".join(unmatched))

    # Final summary
    print("Alignment complete!")
    print(f"  UniMol successes  : {len(unimol_success_smiles):,}")
    print(f"  Matched & aligned : {len(unimol_rows):,}")
    print(f"  Unmatched         : {len(unmatched):,}")
    print(f"  Mordred features  : {mordred_aligned.shape}  → {mordred_out.name}")
    print(f"  UniMol  features  : {unimol_aligned.shape}  → {unimol_out.name}")
    print(f"  SMILES list       : {smiles_out.name}")
    print(f"  Metadata CSV      : {meta_out.name}")
    print(f"  Output dir        : {output_dir}")


if __name__ == "__main__":
    main()
