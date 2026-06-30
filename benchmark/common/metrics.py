"""
Shared evaluation metrics for multilabel odor/taste benchmarks.

Aligned with Odor/train_odor_model.py and Taste/train_taste_model.py.
"""

import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    average_precision_score, roc_auc_score,
)


def safe_roc(y_true, y_score):
    """AUROC for a single label; return NaN when only one class is present."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


def safe_ap(y_true, y_score):
    """AUPRC for a single label; return NaN when only one class is present."""
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
            'AUROC': safe_roc(y_true[:, i], y_prob[:, i]),
            'AUPRC': safe_ap(y_true[:, i], y_prob[:, i]),
            'F1': f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'Precision': precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'Recall': recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'N_pos': int(y_true[:, i].sum()),
        })
    return results


def compute_overall_metrics(y_true, y_pred, y_prob, label_cols=None):
    """Macro/micro aggregates used in overall_results.csv."""
    num_classes = y_true.shape[1]
    per_label = compute_per_label_metrics(
        y_true, y_pred, y_prob,
        label_cols or [f'TARGET_{i+1:02d}' for i in range(num_classes)],
    )

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
        'Accuracy_Macro': float(np.mean([
            accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(num_classes)
        ])),
        'Accuracy_Micro': float(accuracy_score(y_true.flatten(), y_pred.flatten())),
        'F1_Macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'F1_Micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'Precision_Macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Precision_Micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'Recall_Macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'Recall_Micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'AUROC_Macro': auroc_macro,
        'AUROC_Micro': auroc_micro,
        'AUPRC_Macro': auprc_macro,
        'AUPRC_Micro': auprc_micro,
    }


def compute_cv_metrics(y_true, y_prob, thresholds):
    """Per-fold validation metrics written to cv_results.csv."""
    y_pred = (y_prob >= thresholds).astype(int)
    metrics = {
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'micro_f1': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'macro_auprc': 0.0,
        'micro_auprc': 0.0,
    }
    try:
        metrics['macro_auprc'] = average_precision_score(y_true, y_prob, average='macro')
    except Exception:
        pass
    try:
        metrics['micro_auprc'] = average_precision_score(y_true, y_prob, average='micro')
    except Exception:
        pass

    auroc_list = []
    for c in range(y_true.shape[1]):
        auroc_list.append(
            roc_auc_score(y_true[:, c], y_prob[:, c])
            if len(np.unique(y_true[:, c])) > 1 else np.nan
        )
    metrics['macro_auroc'] = float(np.nanmean(auroc_list))
    metrics['monitor'] = (metrics['macro_auroc'] + metrics['macro_auprc']) / 2
    return metrics


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray, step: float = 0.02) -> np.ndarray:
    """Grid-search per-label thresholds on validation predictions (maximize F1)."""
    num_classes = y_true.shape[1]
    thr = np.full(num_classes, 0.5)
    grid = np.arange(0.05, 0.95 + step, step)
    for c in range(num_classes):
        if y_true[:, c].sum() == 0:
            continue
        best_f1, best_t = 0.0, 0.5
        for t in grid:
            f1 = f1_score(y_true[:, c], (y_prob[:, c] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thr[c] = best_t
    return thr


def resolve_thresholds(config, y_val, val_prob):
    if config.get('tune_thresholds', True):
        step = config.get('threshold_grid_step', 0.02)
        return tune_thresholds(y_val, val_prob, step)
    return np.full(y_val.shape[1], 0.5)
