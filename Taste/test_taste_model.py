"""
DLMTF-Net Test Script — Ensemble inference across all folds with per-label tuned thresholds.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    average_precision_score, roc_auc_score
)

from train_taste_model import (
    MolecularDataset,
    resolve_model_dir,
    load_run_config,
    load_fold_models,
    predict_batch,
)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--unimol_test', type=str,
                        default=r'./unimol_feature/fintune_feature/test_output/test_data_molecular_features.npy')
    parser.add_argument('--mordred_test', type=str,
                        default=r'./mordred/taste_descriptors/test_processed.csv')
    parser.add_argument('--test_labels', type=str,
                        default=r'./Data/processed_data/test_data.csv')
    args = parser.parse_args()

    # Auto-detect model directory (reads ./model/latest.txt written by train_taste_model.py)
    model_dir = resolve_model_dir(args.model_dir or r'./model')

    print(f"Model directory: {model_dir}")

    config, label_cols = load_run_config(model_dir)

    # Load data
    print("\nLoading test data...")
    df_mordred = pd.read_csv(args.mordred_test)
    mordred_cols = [c for c in df_mordred.columns if c not in ['SMILES'] + label_cols]
    X_desc  = df_mordred[mordred_cols].values.astype(np.float32)
    X_embed = np.load(args.unimol_test).astype(np.float32)
    df_labels = pd.read_csv(args.test_labels)
    Y = df_labels[label_cols].values.astype(np.float32)

    assert X_embed.shape[0] == len(df_mordred), \
        f"Row count mismatch: UniMol={X_embed.shape[0]}, Mordred={len(df_mordred)}"
    print(f"  X_embed={X_embed.shape}, X_desc={X_desc.shape}, Y={Y.shape}")

    # Override config dims with actual test data dims (handles 100 vs 300 feature mismatch)
    config['embed_dim'] = X_embed.shape[1]
    config['desc_dim'] = X_desc.shape[1]
    config['num_classes'] = Y.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    test_ds = MolecularDataset(X_embed, X_desc, Y)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'])

    print("\nLoading fold models...")
    models, avg_thresholds, config, label_cols = load_fold_models(model_dir, device)
    print(f"Found {len(models)} fold models")

    all_fold_probs = []
    for i, model in enumerate(models):
        probs = predict_batch(model, test_loader, device, config['infer_weights'])
        all_fold_probs.append(probs)
        print(f"  fold_{i + 1}: loaded")

    avg_probs = np.mean(all_fold_probs, axis=0)
    ensemble_pred = (avg_probs >= avg_thresholds).astype(int)

    print(f"\nThresholds: min={avg_thresholds.min():.2f}  "
          f"mean={avg_thresholds.mean():.2f}  max={avg_thresholds.max():.2f}")

    print("\nComputing metrics...")
    overall = compute_overall_metrics(Y, ensemble_pred, avg_probs, label_cols)
    overall['Model'] = 'DLMTF-Net'
    overall['N_Folds'] = len(models)

    per_label = compute_per_label_metrics(Y, ensemble_pred, avg_probs, label_cols)

    out_dir = os.path.join(model_dir, 'test_results')
    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame([overall]).to_csv(os.path.join(out_dir, 'overall_results.csv'), index=False)
    pd.DataFrame(per_label).to_csv(os.path.join(out_dir, 'per_label_results.csv'), index=False)
    np.save(os.path.join(out_dir, 'probabilities.npy'), avg_probs)
    np.save(os.path.join(out_dir, 'predictions.npy'), ensemble_pred)
    np.save(os.path.join(out_dir, 'thresholds.npy'), avg_thresholds)

    print("\n" + "=" * 60)
    print("TEST RESULTS — DLMTF-Net")
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
