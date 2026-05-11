#!/usr/bin/env python3
"""
High-Throughput Screening (HTS) Prediction — Odor DLMTF-Net Ensemble Inference
No labels required. Outputs per-molecule probabilities and binary predictions.

Inputs:
  pom_aligned_unimol.npy     (669052, 768)
  pom_aligned_mordred.npy    (669052, 300)
  pom_aligned_metadata.csv   id / SMILES

Outputs (in OUTPUT_DIR):
  pom_odor_hts_probabilities.npy    (N, C) float32  — raw ensemble probabilities
  pom_odor_hts_predictions.npy      (N, C) int8     — binary predictions (prob >= threshold)
  pom_odor_hts_results.csv          id, smiles, prob_*, pred_*
  pom_odor_hts_fusion_features.npy  (N, 384) float32 — fusion h for downstream POM
  pom_odor_hts_summary.json         thresholds + positive counts per label

"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import odor model from current project
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from train_odor_model import EnhancedMultiLabelModel, ensemble_probs


# ── Default paths ─────────────────────────────────────────────────────────────
UNIMOL_FEAT   = Path(r"./aligned_features/om_aligned_unimol.npy")
MORDRED_FEAT  = Path(r"./aligned_features/om_aligned_mordred.npy")
METADATA_CSV  = Path(r"./aligned_features/om_aligned_metadata.csv")
MODEL_DIR     = Path(r"../../Odor/model/odor_train_finetune")
OUTPUT_DIR    = Path(r"../odor_hts_results")

BATCH_SIZE    = 1024

# ─────────────────────────────────────────────────────────────────────────────


class HTSDataset(Dataset):
    """Label-free dataset for inference."""
    def __init__(self, X_embed: np.ndarray, X_desc: np.ndarray):
        self.X_embed = torch.FloatTensor(X_embed)
        self.X_desc  = torch.FloatTensor(X_desc)

    def __len__(self):
        return len(self.X_embed)

    def __getitem__(self, idx):
        return self.X_embed[idx], self.X_desc[idx]


@torch.no_grad()
def predict_fold(model: nn.Module, loader: DataLoader,
                 device: torch.device, infer_weights):
    """Returns (probs, fusion_h) for one fold."""
    model.eval()
    all_probs = []
    all_h     = []
    for x_embed, x_desc in loader:
        x_embed, x_desc = x_embed.to(device), x_desc.to(device)
        out = model(x_embed, x_desc)
        all_probs.append(ensemble_probs(out, infer_weights))
        all_h.append(out['h'].cpu().numpy())
    return np.vstack(all_probs), np.vstack(all_h)


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    # ── Load model config ─────────────────────────────────────────────────────
    with open(model_dir / "config.json")    as f: config    = json.load(f)
    with open(model_dir / "label_cols.json") as f: label_cols = json.load(f)
    print(f"Model dir  : {model_dir}")
    print(f"Labels     : {len(label_cols)} odor labels")

    # ── Load features ─────────────────────────────────────────────────────────
    print("\nLoading features...")
    X_embed = np.load(args.unimol_feat).astype(np.float32)
    X_desc  = np.load(args.mordred_feat).astype(np.float32)
    meta_df = pd.read_csv(args.metadata)
    meta_df.rename(columns={"SMILES": "smiles"}, inplace=True)  # normalise column name

    assert X_embed.shape[0] == X_desc.shape[0] == len(meta_df), \
        f"Row count mismatch: UniMol={X_embed.shape[0]}, Mordred={X_desc.shape[0]}, meta={len(meta_df)}"

    N = X_embed.shape[0]
    print(f"  X_embed : {X_embed.shape}")
    print(f"  X_desc  : {X_desc.shape}")
    print(f"  Metadata: {N:,} molecules")

    config["embed_dim"]   = X_embed.shape[1]
    config["desc_dim"]    = X_desc.shape[1]
    config["num_classes"] = len(label_cols)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device  : {device}")

    # ── Build DataLoader ──────────────────────────────────────────────────────
    ds     = HTSDataset(X_embed, X_desc)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    # ── Ensemble over all folds ───────────────────────────────────────────────
    fold_dirs = sorted([d for d in os.listdir(model_dir) if d.startswith("fold_")])
    print(f"\nFound {len(fold_dirs)} fold models")

    all_fold_probs      = []
    all_fold_h          = []
    all_fold_thresholds = []

    for fd in fold_dirs:
        ckpt_path = model_dir / fd / "model.pt"
        ckpt      = torch.load(str(ckpt_path), map_location=device, weights_only=False)

        model = EnhancedMultiLabelModel(
            embed_dim  = config["embed_dim"],
            desc_dim   = config["desc_dim"],
            hidden_dim = config["hidden_dim"],
            query_dim  = config["query_dim"],
            num_classes= config["num_classes"],
            dropout    = config["dropout"],
            use_aux_loss = config["use_aux_loss"],
        ).to(device)

        model.load_state_dict(ckpt["model_state_dict"])
        probs, h_feat = predict_fold(model, loader, device, config["infer_weights"])
        all_fold_probs.append(probs)
        all_fold_h.append(h_feat)
        all_fold_thresholds.append(ckpt["thresholds"])
        print(f"  {fd}: probs={probs.shape}  h={h_feat.shape}  prob range [{probs.min():.3f}, {probs.max():.3f}]")

        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    avg_probs      = np.mean(all_fold_probs,      axis=0).astype(np.float32)  # (N, C)
    avg_h          = np.mean(all_fold_h,           axis=0).astype(np.float32)  # (N, H)
    avg_thresholds = np.mean(all_fold_thresholds, axis=0)                     # (C,)
    predictions    = (avg_probs >= avg_thresholds).astype(np.int8)            # (N, C)
    print(f"\nFusion features h: {avg_h.shape}  (averaged over {len(fold_dirs)} folds)")
    print(f"Thresholds: {dict(zip(label_cols, avg_thresholds.round(3)))}")

    # ── Build results CSV ─────────────────────────────────────────────────────
    print("\nBuilding results CSV...")
    prob_cols = {f"prob_{col}": avg_probs[:, i] for i, col in enumerate(label_cols)}
    pred_cols = {f"pred_{col}": predictions[:, i] for i, col in enumerate(label_cols)}
    result_df = pd.concat(
        [meta_df[["id", "smiles"]].reset_index(drop=True),
         pd.DataFrame(prob_cols),
         pd.DataFrame(pred_cols)],
        axis=1,
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    probs_path  = output_dir / f"pom_odor_hts_probabilities_{ts}.npy"
    preds_path  = output_dir / f"pom_odor_hts_predictions_{ts}.npy"
    csv_path    = output_dir / f"pom_odor_hts_results_{ts}.csv"
    summary_path= output_dir / f"pom_odor_hts_summary_{ts}.json"

    fusion_path = output_dir / f"pom_odor_hts_fusion_features_{ts}.npy"

    np.save(str(probs_path), avg_probs)
    np.save(str(preds_path), predictions)
    np.save(str(fusion_path), avg_h)
    result_df.to_csv(csv_path, index=False)

    # ── Summary stats ─────────────────────────────────────────────────────────
    summary = {
        "timestamp":   ts,
        "n_molecules": int(N),
        "n_folds":     len(fold_dirs),
        "model_dir":   str(model_dir),
        "thresholds":  {col: float(t) for col, t in zip(label_cols, avg_thresholds)},
        "positives":   {},
    }
    print("\nPositive predictions per label:")
    for i, col in enumerate(label_cols):
        name  = col
        n_pos = int(predictions[:, i].sum())
        pct   = 100 * n_pos / N
        summary["positives"][name] = {"n": n_pos, "pct": round(pct, 3)}
        print(f"  {name:<8}: {n_pos:>8,} ({pct:.2f}%)  avg_prob={avg_probs[:, i].mean():.4f}")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Done! Results saved to: {output_dir}")
    print(f"  Probabilities : {probs_path.name}")
    print(f"  Predictions   : {preds_path.name}")
    print(f"  Results CSV   : {csv_path.name}")
    print(f"  Fusion feats  : {fusion_path.name}")
    print(f"  Summary       : {summary_path.name}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Odor HTS prediction for POM aligned features using DLMTF-Net"
    )
    parser.add_argument("--model_dir",   default=str(MODEL_DIR))
    parser.add_argument("--unimol_feat", default=str(UNIMOL_FEAT))
    parser.add_argument("--mordred_feat",default=str(MORDRED_FEAT))
    parser.add_argument("--metadata",    default=str(METADATA_CSV))
    parser.add_argument("--output_dir",  default=str(OUTPUT_DIR))
    parser.add_argument("--batch_size",  type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    main(args)
