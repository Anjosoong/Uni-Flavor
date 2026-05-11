"""
Ablation Study Test Script — Ensemble inference across all folds for single-feature models.

Usage:
    python test_ablation.py --mode unimol_only
    python test_ablation.py --mode mordred_only
    python test_ablation.py --mode unimol_only --model_dir <path>
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    average_precision_score, roc_auc_score
)

from train_ablation import SingleBranchModel, SingleFeatureDataset, ensemble_probs


def safe_roc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


def safe_ap(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return average_precision_score(y_true, y_score)
    except Exception:
        return np.nan


def compute_per_label_metrics(y_true, y_pred, y_prob, label_cols):
    results = []
    for i, col in enumerate(label_cols):
        results.append({
            'Label': col,
            'Accuracy': accuracy_score(y_true[:, i], y_pred[:, i]),
            'AUROC':    safe_roc(y_true[:, i], y_prob[:, i]),
            'AUPRC':    safe_ap(y_true[:, i], y_prob[:, i]),
            'F1':       f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'Precision': precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'Recall':   recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'N_pos':    int(y_true[:, i].sum()),
        })
    return results


def compute_overall_metrics(y_true, y_pred, y_prob, label_cols):
    C = y_true.shape[1]
    per_label = compute_per_label_metrics(y_true, y_pred, y_prob, label_cols)

    auroc_macro = float(np.nanmean([r['AUROC'] for r in per_label]))
    auprc_macro = float(np.nanmean([r['AUPRC'] for r in per_label]))

    try:
        auroc_micro = roc_auc_score(y_true, y_prob, average='micro')
    except Exception:
        auroc_micro = np.nan
    try:
        auprc_micro = average_precision_score(y_true, y_prob, average='micro')
    except Exception:
        auprc_micro = np.nan

    return {
        'Accuracy_Macro': float(np.mean([accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(C)])),
        'Accuracy_Micro': float(accuracy_score(y_true.flatten(), y_pred.flatten())),
        'F1_Macro':       f1_score(y_true, y_pred, average='macro',  zero_division=0),
        'F1_Micro':       f1_score(y_true, y_pred, average='micro',  zero_division=0),
        'Precision_Macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Precision_Micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'Recall_Macro':   recall_score(y_true, y_pred, average='macro', zero_division=0),
        'Recall_Micro':   recall_score(y_true, y_pred, average='micro', zero_division=0),
        'AUROC_Macro':    auroc_macro,
        'AUROC_Micro':    auroc_micro,
        'AUPRC_Macro':    auprc_macro,
        'AUPRC_Micro':    auprc_micro,
    }


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device,
            infer_weights) -> np.ndarray:
    model.eval()
    all_probs = []
    for x, _ in loader:
        x = x.to(device)
        out = model(x)
        all_probs.append(ensemble_probs(out, infer_weights))
    return np.vstack(all_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                        choices=['unimol_only', 'mordred_only'])
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--unimol_test', type=str,
                        default=r'./unimol_feature/fintune_feature/test_output/test_data_molecular_features.npy')
    parser.add_argument('--mordred_test', type=str,
                        default=r'./mordred/taste_descriptors/test_processed.csv')
    parser.add_argument('--test_labels', type=str,
                        default=r'./Data/processed_data/test_data.csv')
    args = parser.parse_args()

    # Auto-detect model directory
    base_dir = r'./ablation_output'
    if args.model_dir:
        model_dir = args.model_dir
    else:
        latest_txt = os.path.join(base_dir, f'latest_{args.mode}.txt')
        if os.path.exists(latest_txt):
            with open(latest_txt) as f:
                model_dir = f.read().strip()
        else:
            prefix = f'ablation_{args.mode}_'
            dirs = sorted([d for d in os.listdir(base_dir)
                           if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(prefix)])
            if not dirs:
                raise FileNotFoundError(f"No {args.mode} model directories in {base_dir}")
            model_dir = os.path.join(base_dir, dirs[-1])

    print(f"Mode: {args.mode}")
    print(f"Model directory: {model_dir}")

    with open(os.path.join(model_dir, 'config.json')) as f:
        config = json.load(f)

    with open(os.path.join(model_dir, 'label_cols.json')) as f:
        label_cols = json.load(f)

    # Load test data based on mode
    print("\nLoading test data...")
    df_labels = pd.read_csv(args.test_labels)
    Y = df_labels[label_cols].values.astype(np.float32)

    if args.mode == 'unimol_only':
        X = np.load(args.unimol_test).astype(np.float32)
        print(f"  UniMol features: {X.shape}")
    else:
        df_mordred = pd.read_csv(args.mordred_test)
        mordred_cols = [c for c in df_mordred.columns if c not in ['SMILES'] + label_cols]
        X = df_mordred[mordred_cols].values.astype(np.float32)
        print(f"  Mordred features: {X.shape}")

    print(f"  Y={Y.shape}")

    # Override config dims with actual test data dims
    config['input_dim'] = X.shape[1]
    config['num_classes'] = Y.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    test_ds = SingleFeatureDataset(X, Y)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'])

    fold_dirs = sorted([d for d in os.listdir(model_dir) if d.startswith('fold_')])
    print(f"\nFound {len(fold_dirs)} fold models")

    all_fold_probs = []
    all_fold_thresholds = []

    for fd in fold_dirs:
        ckpt = torch.load(os.path.join(model_dir, fd, 'model.pt'),
                          map_location=device, weights_only=False)

        model = SingleBranchModel(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            query_dim=config['query_dim'],
            num_classes=config['num_classes'],
            dropout=config['dropout'],
            use_aux_loss=config['use_aux_loss'],
            branch_type=config['branch_type'],
        ).to(device)

        model.load_state_dict(ckpt['model_state_dict'])
        probs = predict(model, test_loader, device, config['infer_weights'])
        all_fold_probs.append(probs)
        all_fold_thresholds.append(ckpt['thresholds'])
        print(f"  {fd}: loaded")

    avg_probs = np.mean(all_fold_probs, axis=0)
    avg_thresholds = np.mean(all_fold_thresholds, axis=0)
    ensemble_pred = (avg_probs >= avg_thresholds).astype(int)

    print(f"\nThresholds: min={avg_thresholds.min():.2f}  "
          f"mean={avg_thresholds.mean():.2f}  max={avg_thresholds.max():.2f}")

    print("\nComputing metrics...")
    mode_label = 'UniMol-Only' if args.mode == 'unimol_only' else 'Mordred-Only'
    overall = compute_overall_metrics(Y, ensemble_pred, avg_probs, label_cols)
    overall['Model'] = mode_label
    overall['N_Folds'] = len(fold_dirs)

    per_label = compute_per_label_metrics(Y, ensemble_pred, avg_probs, label_cols)

    out_dir = os.path.join(model_dir, 'test_results')
    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame([overall]).to_csv(os.path.join(out_dir, 'overall_results.csv'), index=False)
    pd.DataFrame(per_label).to_csv(os.path.join(out_dir, 'per_label_results.csv'), index=False)
    np.save(os.path.join(out_dir, 'probabilities.npy'), avg_probs)
    np.save(os.path.join(out_dir, 'predictions.npy'), ensemble_pred)
    np.save(os.path.join(out_dir, 'thresholds.npy'), avg_thresholds)

    print("\n" + "=" * 60)
    print(f"TEST RESULTS — {mode_label}")
    print("=" * 60)
    for k, v in overall.items():
        if k not in ('Model', 'N_Folds'):
            print(f"  {k:20s}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nPer-label results (sample):")
    per_df = pd.DataFrame(per_label)
    print(per_df.sort_values('F1', ascending=False)
          .head(10).to_string(index=False, float_format='%.4f'))

    print(f"\nResults saved to: {out_dir}")


if __name__ == '__main__':
    main()
