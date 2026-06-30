"""
5-fold CV training loop and held-out test evaluation.

Protocol matches DLMOF-Net / DLMTF-Net:
  - StratifiedKFold on training set
  - Per-label dynamic thresholds on each validation fold
  - Test: average fold probabilities + average thresholds
"""

import os
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import save_test_results
from .metrics import compute_cv_metrics, resolve_thresholds


def ensemble_fold_probabilities(prob_folds: List[np.ndarray]) -> np.ndarray:
    return np.mean(prob_folds, axis=0)


def ensemble_fold_thresholds(threshold_folds: List[np.ndarray]) -> np.ndarray:
    return np.mean(threshold_folds, axis=0)


def run_cv_training(
    config: dict,
    y_train: np.ndarray,
    label_cols: list,
    output_dir: str,
    train_fold_fn: Callable,
    save_fold_fn: Optional[Callable] = None,
) -> tuple:
    """
    Run stratified K-fold training on the training split.

    ``train_fold_fn(fold, train_idx, val_idx)`` must return
    ``(val_prob, fold_artifact)`` or ``(val_prob, fold_artifact, y_val)``.
    """
    from .data import get_cv_splits

    splits = get_cv_splits(len(y_train), y_train, config['n_folds'], config['random_state'])
    fold_results = []
    threshold_folds = []

    fold_iter = tqdm(
        enumerate(splits),
        total=len(splits),
        desc='CV folds',
        unit='fold',
    )
    for fold, (train_idx, val_idx) in fold_iter:
        fold_iter.set_postfix_str(f'fold {fold + 1}/{config["n_folds"]}')
        fold_out = train_fold_fn(fold, train_idx, val_idx)
        if len(fold_out) == 3:
            val_prob, fold_artifact, y_val = fold_out
        else:
            val_prob, fold_artifact = fold_out
            y_val = y_train[val_idx]

        thresholds = resolve_thresholds(config, y_val, val_prob)
        threshold_folds.append(thresholds)
        metrics = compute_cv_metrics(y_val, val_prob, thresholds)
        fold_results.append({'fold': fold + 1, **metrics})

        fold_dir = os.path.join(output_dir, f'fold_{fold + 1}')
        os.makedirs(fold_dir, exist_ok=True)
        np.save(os.path.join(fold_dir, 'thresholds.npy'), thresholds)

        if save_fold_fn is not None:
            save_fold_fn(fold_dir, fold_artifact, thresholds)

        print(
            f'  macro_f1={metrics["macro_f1"]:.4f}  '
            f'macro_auroc={metrics["macro_auroc"]:.4f}  '
            f'monitor={metrics["monitor"]:.4f}'
        )

    pd.DataFrame(fold_results).to_csv(os.path.join(output_dir, 'cv_results.csv'), index=False)

    print('\nCROSS-VALIDATION RESULTS')
    for key in ['macro_f1', 'micro_f1', 'macro_auprc', 'micro_auprc', 'macro_auroc']:
        vals = [r[key] for r in fold_results]
        print(f'  {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')

    avg_thresholds = ensemble_fold_thresholds(threshold_folds)
    return threshold_folds, avg_thresholds, fold_results


def run_test_evaluation(
    config: dict,
    output_dir: str,
    model_name: str,
    y_test: np.ndarray,
    label_cols: list,
    predict_test_fn: Callable,
    threshold_folds: List[np.ndarray],
    avg_thresholds: np.ndarray,
) -> str:
    """Ensemble test predictions across saved fold models."""
    print('\nEvaluating on held-out test set...')
    prob_folds = []
    for fold in tqdm(range(config['n_folds']), desc='Test ensemble', unit='fold'):
        fold_dir = os.path.join(output_dir, f'fold_{fold + 1}')
        prob_folds.append(predict_test_fn(fold_dir, fold))

    test_prob = ensemble_fold_probabilities(prob_folds)
    test_dir = save_test_results(
        os.path.join(output_dir, 'test_results'),
        model_name=model_name,
        y_true=y_test,
        y_prob=test_prob,
        thresholds=avg_thresholds,
        label_cols=label_cols,
        n_folds=config['n_folds'],
    )
    print(f'\nTest results saved to: {test_dir}')
    return test_dir
