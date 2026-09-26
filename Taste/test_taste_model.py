"""Evaluate repeated taste models on the independent test set.

The five folds from each training repeat are averaged into one seed-level
ensemble. Seed-level t intervals quantify training reproducibility. Separately,
test molecules can be bootstrap-resampled while the trained models, OOF-derived
thresholds, probabilities, and predictions remain fixed. Bootstrap therefore
does not rerun inference or retrain any model.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t
from sklearn.metrics import average_precision_score, roc_auc_score

from train_taste_model import (
    GLOBAL_SEED,
    load_repeated_fold_models,
    load_run_config,
    resolve_model_dir,
    set_global_seed,
)


SIGNIFICANCE_METRICS = (
    'F1_Macro', 'F1_Micro', 'AUROC_Macro', 'AUPRC_Macro'
)


def safe_divide(numerator, denominator):
    """Divide arrays elementwise, returning zero where the denominator is zero."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    result = np.zeros_like(numerator, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def safe_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute binary AUROC, or NaN when the label has one observed class."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return np.nan


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute binary average precision, or NaN for a one-class label."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return np.nan


def labelwise_scores(y_true: np.ndarray, y_score: np.ndarray,
                     score_function) -> np.ndarray:
    """Apply a scikit-learn score per evaluable label and preserve NaNs."""
    valid = np.array([
        len(np.unique(y_true[:, index])) > 1
        for index in range(y_true.shape[1])
    ])
    scores = np.full(y_true.shape[1], np.nan, dtype=float)
    if np.any(valid):
        scores[valid] = np.atleast_1d(score_function(
            y_true[:, valid], y_score[:, valid], average=None))
    return scores


def support_level(n_positive: int) -> str:
    """Map positive-label count to the reporting support category."""
    if n_positive < 5:
        return 'insufficient'
    if n_positive < 10:
        return 'low'
    return 'adequate'


def classification_counts(y_true: np.ndarray,
                          y_pred: np.ndarray) -> tuple:
    """Return per-label true-positive, false-positive, false-negative, and true-negative counts."""
    true = y_true.astype(bool)
    pred = y_pred.astype(bool)
    tp = np.sum(true & pred, axis=0)
    fp = np.sum(~true & pred, axis=0)
    fn = np.sum(true & ~pred, axis=0)
    tn = np.sum(~true & ~pred, axis=0)
    return tp, fp, fn, tn


def compute_per_label_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              y_prob: np.ndarray, label_cols: list) -> list:
    """Build one detailed metric record per label."""
    tp, fp, fn, tn = classification_counts(y_true, y_pred)
    accuracy = safe_divide(tp + tn, tp + fp + fn + tn)
    f1 = safe_divide(2 * tp, 2 * tp + fp + fn)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    auroc = labelwise_scores(y_true, y_prob, roc_auc_score)
    auprc = labelwise_scores(y_true, y_prob, average_precision_score)
    rows = []
    for index, label in enumerate(label_cols):
        n_positive = int(y_true[:, index].sum())
        rows.append({
            'Label': label,
            'Accuracy': float(accuracy[index]),
            'AUROC': float(auroc[index]),
            'AUPRC': float(auprc[index]),
            'F1': float(f1[index]),
            'Precision': float(precision[index]),
            'Recall': float(recall[index]),
            'N_Positive': n_positive,
            'Support_Level': support_level(n_positive),
        })
    return rows


def compute_overall_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            y_prob: np.ndarray, label_cols: list) -> dict:
    """Compute macro and micro metrics for one repeated-model ensemble."""
    del label_cols
    tp, fp, fn, tn = classification_counts(y_true, y_pred)
    accuracy = safe_divide(tp + tn, tp + fp + fn + tn)
    f1 = safe_divide(2 * tp, 2 * tp + fp + fn)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    tp_all, fp_all = tp.sum(), fp.sum()
    fn_all, tn_all = fn.sum(), tn.sum()
    auroc = labelwise_scores(y_true, y_prob, roc_auc_score)
    auprc = labelwise_scores(y_true, y_prob, average_precision_score)
    return {
        'Accuracy_Macro': float(np.nanmean(accuracy)),
        'Accuracy_Micro': float(safe_divide(
            tp_all + tn_all, tp_all + fp_all + fn_all + tn_all)),
        'F1_Macro': float(np.nanmean(f1)),
        'F1_Micro': float(safe_divide(
            2 * tp_all, 2 * tp_all + fp_all + fn_all)),
        'Precision_Macro': float(np.nanmean(precision)),
        'Precision_Micro': float(safe_divide(tp_all, tp_all + fp_all)),
        'Recall_Macro': float(np.nanmean(recall)),
        'Recall_Micro': float(safe_divide(tp_all, tp_all + fn_all)),
        'AUROC_Macro': float(np.nanmean(auroc)),
        'AUROC_Micro': safe_roc(y_true.ravel(), y_prob.ravel()),
        'AUPRC_Macro': float(np.nanmean(auprc)),
        'AUPRC_Micro': safe_ap(y_true.ravel(), y_prob.ravel()),
    }


def summarize_repeated_metrics(repeat_df: pd.DataFrame,
                               confidence: float) -> pd.DataFrame:
    """Summarize repeat-level metrics with Student t confidence intervals."""
    metadata = {'Repeat', 'Seed', 'N_Folds'}
    rows = []
    for metric in (column for column in repeat_df.columns
                   if column not in metadata):
        values = repeat_df[metric].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        count = len(values)
        mean = float(np.mean(values)) if count else np.nan
        std = float(np.std(values, ddof=1)) if count > 1 else np.nan
        if count > 1:
            critical = student_t.ppf((1 + confidence) / 2, df=count - 1)
            margin = critical * std / np.sqrt(count)
            lower, upper = mean - margin, mean + margin
        else:
            lower = upper = np.nan
        rows.append({
            'Metric': metric,
            'Mean': mean,
            'Median': float(np.median(values)) if count else np.nan,
            'Std': std,
            'CI_Lower': lower,
            'CI_Upper': upper,
            'Confidence_Level': confidence,
            'N_Repeats': count,
        })
    return pd.DataFrame(rows)


def weighted_bootstrap_ranking_metrics(
        y_true: np.ndarray, y_score: np.ndarray,
        bootstrap_counts: np.ndarray) -> tuple:
    """Vectorized weighted AUROC and average precision for fixed scores."""
    if len(np.unique(y_true)) < 2:
        missing = np.full(len(bootstrap_counts), np.nan, dtype=float)
        return missing, missing.copy()

    order = np.argsort(-y_score, kind='mergesort')
    sorted_score = y_score[order]
    sorted_true = y_true[order].astype(float)
    group_starts = np.r_[
        0, np.flatnonzero(sorted_score[1:] != sorted_score[:-1]) + 1
    ]
    sorted_weights = bootstrap_counts[:, order]
    positive_groups = np.add.reduceat(
        sorted_weights * sorted_true, group_starts, axis=1)
    total_groups = np.add.reduceat(
        sorted_weights, group_starts, axis=1)
    negative_groups = total_groups - positive_groups

    total_positive = positive_groups.sum(axis=1)
    total_negative = negative_groups.sum(axis=1)
    cumulative_negative = np.cumsum(negative_groups, axis=1)
    negative_below = total_negative[:, None] - cumulative_negative
    auc_numerator = np.sum(
        positive_groups * (negative_below + 0.5 * negative_groups), axis=1)
    auroc = safe_divide(
        auc_numerator, total_positive * total_negative)
    auroc[(total_positive == 0) | (total_negative == 0)] = np.nan

    cumulative_positive = np.cumsum(positive_groups, axis=1)
    cumulative_total = np.cumsum(total_groups, axis=1)
    precision = safe_divide(cumulative_positive, cumulative_total)
    average_precision = safe_divide(
        np.sum(positive_groups * precision, axis=1), total_positive)
    average_precision[total_positive == 0] = np.nan
    return auroc, average_precision


def bootstrap_test_metrics(
        y_true: np.ndarray, probability_runs: np.ndarray,
        prediction_runs: np.ndarray, label_cols: list,
        repeat_df: pd.DataFrame, n_bootstrap: int, confidence: float,
        random_seed: int) -> tuple:
    """Percentile CI from paired test-molecule resampling.

    Each bootstrap replicate uses the same sampled molecule indices for every
    fixed seed-level ensemble. Metrics are computed per seed and then averaged,
    matching the point estimates in ``repeated_test_summary.csv``.
    """
    del label_cols
    metric_names = [
        'Accuracy_Macro', 'F1_Macro', 'Precision_Macro', 'Recall_Macro',
        'AUROC_Macro', 'AUPRC_Macro',
    ]
    if probability_runs.shape != prediction_runs.shape:
        raise ValueError(
            'Bootstrap probabilities and predictions must have equal shapes.')
    if probability_runs.ndim != 3:
        raise ValueError(
            'Bootstrap arrays must have shape [repeats, samples, labels].')
    if probability_runs.shape[1:] != y_true.shape:
        raise ValueError(
            f'Bootstrap array shape {probability_runs.shape[1:]} does not '
            f'match test labels {y_true.shape}.')

    rng = np.random.default_rng(random_seed)
    bootstrap_counts = rng.multinomial(
        len(y_true), np.full(len(y_true), 1.0 / len(y_true)),
        size=n_bootstrap).astype(float)
    true_binary = y_true.astype(bool)
    positive_counts = bootstrap_counts @ true_binary.astype(float)

    values_by_metric = {
        metric: np.empty((n_bootstrap, probability_runs.shape[0]),
                         dtype=float)
        for metric in metric_names
    }
    for repeat_index, (probabilities, predictions) in enumerate(zip(
            probability_runs, prediction_runs)):
        pred_binary = predictions.astype(bool)
        predicted_counts = bootstrap_counts @ pred_binary.astype(float)
        tp = bootstrap_counts @ (true_binary & pred_binary).astype(float)
        fp = predicted_counts - tp
        fn = positive_counts - tp
        tn = len(y_true) - tp - fp - fn

        values_by_metric['Accuracy_Macro'][:, repeat_index] = np.mean(
            safe_divide(tp + tn, tp + fp + fn + tn), axis=1)
        values_by_metric['F1_Macro'][:, repeat_index] = np.mean(
            safe_divide(2 * tp, 2 * tp + fp + fn), axis=1)
        values_by_metric['Precision_Macro'][:, repeat_index] = np.mean(
            safe_divide(tp, tp + fp), axis=1)
        values_by_metric['Recall_Macro'][:, repeat_index] = np.mean(
            safe_divide(tp, tp + fn), axis=1)

        label_auroc = np.full(
            (n_bootstrap, y_true.shape[1]), np.nan, dtype=float)
        label_auprc = np.full_like(label_auroc, np.nan)
        for label_index in range(y_true.shape[1]):
            label_auroc[:, label_index], label_auprc[:, label_index] = (
                weighted_bootstrap_ranking_metrics(
                    y_true[:, label_index], probabilities[:, label_index],
                    bootstrap_counts))
        values_by_metric['AUROC_Macro'][:, repeat_index] = np.nanmean(
            label_auroc, axis=1)
        values_by_metric['AUPRC_Macro'][:, repeat_index] = np.nanmean(
            label_auprc, axis=1)
        print(f'  Bootstrap metrics: repeat {repeat_index + 1}/'
              f'{probability_runs.shape[0]} completed')

    samples = pd.DataFrame({
        metric: np.nanmean(values_by_metric[metric], axis=1)
        for metric in metric_names
    })
    samples.insert(0, 'Bootstrap', np.arange(1, n_bootstrap + 1))
    alpha = (1.0 - confidence) / 2.0
    summary_rows = []
    for metric in metric_names:
        values = samples[metric].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite):
            lower, upper = np.quantile(finite, [alpha, 1.0 - alpha])
            bootstrap_mean = float(np.mean(finite))
            bootstrap_std = (
                float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan)
        else:
            lower = upper = bootstrap_mean = bootstrap_std = np.nan
        summary_rows.append({
            'Metric': metric,
            'Point_Estimate': float(repeat_df[metric].mean()),
            'Bootstrap_Mean': bootstrap_mean,
            'Bootstrap_Std': bootstrap_std,
            'CI_Lower': float(lower),
            'CI_Upper': float(upper),
            'Confidence_Level': confidence,
            'N_Bootstrap': n_bootstrap,
            'N_Test_Samples': len(y_true),
            'N_Repeats': probability_runs.shape[0],
            'Bootstrap_Seed': random_seed,
            'Resampling_Unit': 'test_molecule',
            'Aggregation': 'mean_metric_across_fixed_seed_ensembles',
        })
    return pd.DataFrame(summary_rows), samples


def summarize_per_label_metrics(per_label_df: pd.DataFrame,
                                confidence: float) -> pd.DataFrame:
    """Summarize each label across repeat seeds with t intervals."""
    metadata = {
        'Repeat', 'Seed', 'Label', 'N_Positive', 'Support_Level'
    }
    metric_cols = [
        column for column in per_label_df.columns if column not in metadata
    ]
    rows = []
    for label, group in per_label_df.groupby('Label', sort=False):
        for metric in metric_cols:
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            count = len(values)
            mean = float(np.mean(values)) if count else np.nan
            std = float(np.std(values, ddof=1)) if count > 1 else np.nan
            if count > 1:
                critical = student_t.ppf((1 + confidence) / 2, df=count - 1)
                margin = critical * std / np.sqrt(count)
                lower, upper = mean - margin, mean + margin
            else:
                lower = upper = np.nan
            rows.append({
                'Label': label,
                'Metric': metric,
                'Mean': mean,
                'Median': float(np.median(values)) if count else np.nan,
                'Std': std,
                'CI_Lower': lower,
                'CI_Upper': upper,
                'Confidence_Level': confidence,
                'N_Repeats': count,
                'N_Positive': int(group['N_Positive'].iloc[0]),
                'Support_Level': group['Support_Level'].iloc[0],
            })
    return pd.DataFrame(rows)


def paired_seed_significance_test(
        y_true: np.ndarray, model_prob: np.ndarray, model_pred: np.ndarray,
        baseline_prob: np.ndarray, baseline_pred: np.ndarray,
        label_cols: list, confidence: float) -> pd.DataFrame:
    """Compare model and baseline using paired differences across seeds."""
    model_metrics = [
        compute_overall_metrics(y_true, pred, prob, label_cols)
        for prob, pred in zip(model_prob, model_pred)
    ]
    baseline_metrics = [
        compute_overall_metrics(y_true, pred, prob, label_cols)
        for prob, pred in zip(baseline_prob, baseline_pred)
    ]
    rows = []
    for metric in SIGNIFICANCE_METRICS:
        model_values = np.asarray([row[metric] for row in model_metrics])
        baseline_values = np.asarray([row[metric] for row in baseline_metrics])
        differences = model_values - baseline_values
        differences = differences[np.isfinite(differences)]
        count = len(differences)
        mean_difference = float(np.mean(differences)) if count else np.nan
        std = float(np.std(differences, ddof=1)) if count > 1 else np.nan
        if count > 1 and std > 0:
            critical = student_t.ppf((1 + confidence) / 2, count - 1)
            margin = critical * std / np.sqrt(count)
            t_stat = mean_difference / (std / np.sqrt(count))
            p_value = float(2 * student_t.sf(abs(t_stat), count - 1))
            lower, upper = mean_difference - margin, mean_difference + margin
        elif count > 1:
            lower = upper = mean_difference
            p_value = 0.0 if mean_difference else 1.0
        else:
            lower = upper = np.nan
            p_value = np.nan
        rows.append({
            'Metric': metric,
            'Model_Mean': float(np.mean(model_values)),
            'Baseline_Mean': float(np.mean(baseline_values)),
            'Mean_Difference': mean_difference,
            'Difference_Std': std,
            'Difference_CI_Lower': lower,
            'Difference_CI_Upper': upper,
            'Paired_T_P_Value': p_value,
            'Significant_At_0.05': bool(
                np.isfinite(p_value) and p_value < 0.05),
            'N_Repeats': count,
        })
    return pd.DataFrame(rows)


def load_baseline_results(baseline_dir: str, expected_shape: tuple,
                          expected_seeds: np.ndarray,
                          expected_labels: list) -> tuple:
    """Load and validate saved baseline probabilities and metadata."""
    baseline_prob = np.load(os.path.join(
        baseline_dir, 'repeat_probabilities.npy'))
    baseline_thresholds = np.load(os.path.join(
        baseline_dir, 'repeat_thresholds.npy'))
    baseline_seeds = np.load(os.path.join(
        baseline_dir, 'repeat_seeds.npy'))
    with open(os.path.join(baseline_dir, 'label_cols.json'),
              encoding='utf-8') as handle:
        baseline_labels = json.load(handle)
    if baseline_prob.shape != expected_shape:
        raise ValueError(
            f'Baseline probability shape {baseline_prob.shape} does not match '
            f'{expected_shape}.')
    if baseline_thresholds.shape != (expected_shape[0], expected_shape[2]):
        raise ValueError('Baseline threshold shape does not match repeats.')
    if not np.array_equal(baseline_seeds, expected_seeds):
        raise ValueError('Baseline and model repeat seeds must match.')
    if baseline_labels != expected_labels:
        raise ValueError('Baseline and model label order must match.')
    baseline_pred = (
        baseline_prob >= baseline_thresholds[:, None, :]).astype(int)
    return baseline_prob, baseline_pred


@torch.inference_mode()
def predict_model(model: torch.nn.Module, x_embed: torch.Tensor,
                  x_desc: torch.Tensor, batch_size: int,
                  infer_weights) -> np.ndarray:
    """Run inference on feature tensors already resident on the device."""
    model.eval()
    weights = torch.as_tensor(
        infer_weights, dtype=x_embed.dtype, device=x_embed.device)
    all_probabilities = []
    for start in range(0, len(x_embed), batch_size):
        stop = min(start + batch_size, len(x_embed))
        outputs = model(x_embed[start:stop], x_desc[start:stop])
        probabilities = weights[0] * torch.sigmoid(outputs['logits'])
        if outputs.get('logits_embed') is not None:
            if len(weights) != 3:
                raise ValueError(
                    'infer_weights must contain three auxiliary-head weights.')
            probabilities = (
                probabilities
                + weights[1] * torch.sigmoid(outputs['logits_embed'])
                + weights[2] * torch.sigmoid(outputs['logits_desc'])
            )
        all_probabilities.append(probabilities.cpu())
    return torch.cat(all_probabilities).numpy()


def main() -> None:
    """Evaluate repeated taste ensembles and save metrics and uncertainty."""
    parser = argparse.ArgumentParser(
        description='Evaluate repeated five-fold taste models.')
    parser.add_argument(
        '--model_dir', type=str, default=None,
        help='Run directory, model root containing latest.txt, or newest run root')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Defaults to MODEL_DIR/evaluation_results.')
    parser.add_argument('--unimol_test', type=str,
                        default=(
                            r'./unimol_feature/finetune_feature/test_output/'
                            r'test_data_molecular_features.npy'),
                        help='NumPy matrix of test-set Uni-Mol features')
    parser.add_argument('--mordred_test', type=str,
                        default=r'./mordred/taste_descriptors/test_processed.csv',
                        help='Processed test-set Mordred descriptor CSV')
    parser.add_argument('--test_labels', type=str,
                        default=r'./Data/processed_data/test_data.csv',
                        help='Test CSV containing TARGET_* label columns')
    parser.add_argument('--seed', type=int, default=GLOBAL_SEED,
                        help='Global inference seed')
    parser.add_argument('--confidence', type=float, default=0.95,
                        help='Confidence level for interval estimates')
    parser.add_argument('--inference_batch_size', type=int, default=2048,
                        help='Inference batch size (default: 2048).')
    parser.add_argument('--n_bootstrap', '--n-bootstrap', type=int,
                        default=1000,
                        help='Test-molecule bootstrap replicates; 0 disables '
                             '(default: 1000).')
    parser.add_argument('--bootstrap_seed', '--bootstrap-seed', type=int,
                        default=GLOBAL_SEED,
                        help='Random seed for test bootstrap sampling.')
    parser.add_argument('--baseline_dir', type=str, default=None,
                        help='Evaluation directory containing repeated baseline predictions.')
    parser.add_argument('--baseline_name', type=str, default='Baseline',
                        help='Display name used in paired significance output')
    args = parser.parse_args()
    if args.inference_batch_size < 1:
        parser.error('--inference_batch_size must be positive.')
    if not 0 < args.confidence < 1:
        parser.error('--confidence must be between 0 and 1.')
    if args.n_bootstrap < 0:
        parser.error('--n_bootstrap must be non-negative.')
    set_global_seed(args.seed)

    model_dir = resolve_model_dir(args.model_dir or r'./model')
    print(f'Model directory: {model_dir}')
    config, label_cols = load_run_config(model_dir)

    print('\nLoading test data...')
    df_mordred = pd.read_csv(args.mordred_test)
    df_labels = pd.read_csv(args.test_labels)
    if 'SMILES' in df_mordred and 'SMILES' in df_labels:
        if not np.array_equal(
                df_mordred['SMILES'].astype(str).to_numpy(),
                df_labels['SMILES'].astype(str).to_numpy()):
            raise ValueError('Mordred and label SMILES rows are not aligned.')
    missing_labels = [label for label in label_cols
                      if label not in df_labels.columns]
    if missing_labels:
        raise ValueError(f'Missing test labels: {missing_labels}')
    mordred_cols = [
        column for column in df_mordred.columns
        if column not in ['SMILES'] + label_cols
    ]
    x_desc = df_mordred[mordred_cols].values.astype(np.float32)
    x_embed = np.load(args.unimol_test).astype(np.float32)
    y_true = df_labels[label_cols].values.astype(np.float32)
    if not (len(x_embed) == len(x_desc) == len(y_true)):
        raise ValueError(
            f'Row count mismatch: UniMol={len(x_embed)}, '
            f'Mordred={len(x_desc)}, labels={len(y_true)}.')
    if (x_embed.shape[1] != config['embed_dim'] or
            x_desc.shape[1] != config['desc_dim'] or
            y_true.shape[1] != config['num_classes']):
        raise ValueError(
            f'Dimension mismatch: test=({x_embed.shape[1]}, '
            f'{x_desc.shape[1]}, {y_true.shape[1]}), '
            f"model=({config['embed_dim']}, {config['desc_dim']}, "
            f"{config['num_classes']}).")
    print(f'  X_embed={x_embed.shape}, X_desc={x_desc.shape}, Y={y_true.shape}')

    support_counts = pd.Series([
        support_level(int(y_true[:, index].sum()))
        for index in range(y_true.shape[1])
    ]).value_counts()
    print('  Label support: ' + ', '.join(
        f'{level}={int(support_counts.get(level, 0))}'
        for level in ('insufficient', 'low', 'adequate')))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  Device: {device}')
    print(f'  Inference batch size: {args.inference_batch_size}')
    x_embed_device = torch.from_numpy(x_embed).to(device)
    x_desc_device = torch.from_numpy(x_desc).to(device)

    print('\nLoading repeated fold models...')
    inference_started = time.perf_counter()
    repeated_runs, config, label_cols = load_repeated_fold_models(
        model_dir, device)
    expected_repeats = int(config.get('n_repeats', 1))
    if len(repeated_runs) != expected_repeats:
        raise ValueError(
            f'Expected {expected_repeats} repeats, found {len(repeated_runs)}.')

    probability_runs = []
    threshold_runs = []
    repeat_rows = []
    per_label_rows = []
    repeat_seeds = []
    for run in repeated_runs:
        expected_folds = int(config.get('n_folds', len(run['models'])))
        if len(run['models']) != expected_folds:
            raise ValueError(
                f"{run['name']} contains {len(run['models'])} folds; "
                f'expected {expected_folds}.')
        fold_probabilities = [
            predict_model(
                model, x_embed_device, x_desc_device,
                args.inference_batch_size, config['infer_weights'])
            for model in run['models']
        ]
        probabilities = np.mean(fold_probabilities, axis=0)
        thresholds = np.asarray(run['thresholds'], dtype=float)
        if thresholds.shape != (y_true.shape[1],):
            raise ValueError(
                f"{run['name']} threshold shape is {thresholds.shape}; "
                f'expected {(y_true.shape[1],)}.')
        predictions = (probabilities >= thresholds).astype(int)
        metrics = compute_overall_metrics(
            y_true, predictions, probabilities, label_cols)
        probability_runs.append(probabilities)
        threshold_runs.append(thresholds)
        repeat_seeds.append(run['seed'])
        repeat_rows.append({
            'Repeat': run['repeat'],
            'Seed': run['seed'],
            'N_Folds': len(run['models']),
            **metrics,
        })
        for row in compute_per_label_metrics(
                y_true, predictions, probabilities, label_cols):
            per_label_rows.append({
                'Repeat': run['repeat'],
                'Seed': run['seed'],
                **row,
            })
        print(f"  {run['name']}: {len(run['models'])} folds evaluated")
    print('  Model loading and inference completed in '
          f'{time.perf_counter() - inference_started:.1f} seconds.')

    probability_runs = np.stack(probability_runs)
    threshold_runs = np.stack(threshold_runs)
    prediction_runs = (
        probability_runs >= threshold_runs[:, None, :]).astype(int)
    repeat_seeds = np.asarray(repeat_seeds, dtype=np.int64)
    repeat_df = pd.DataFrame(repeat_rows)
    repeated_summary = summarize_repeated_metrics(
        repeat_df, args.confidence)
    per_label_df = pd.DataFrame(per_label_rows)
    per_label_summary = summarize_per_label_metrics(
        per_label_df, args.confidence)

    out_dir = args.output_dir or os.path.join(model_dir, 'evaluation_results')
    os.makedirs(out_dir, exist_ok=True)
    legacy_bootstrap_file = os.path.join(
        out_dir, 'metric_confidence_intervals.csv')
    if os.path.exists(legacy_bootstrap_file):
        os.remove(legacy_bootstrap_file)
    repeat_df.to_csv(os.path.join(
        out_dir, 'repeated_test_results.csv'), index=False)
    repeated_summary.to_csv(os.path.join(
        out_dir, 'repeated_test_summary.csv'), index=False)
    per_label_df.to_csv(os.path.join(
        out_dir, 'per_label_repeated_results.csv'), index=False)
    per_label_summary.to_csv(os.path.join(
        out_dir, 'per_label_summary.csv'), index=False)
    np.save(os.path.join(out_dir, 'repeat_probabilities.npy'), probability_runs)
    np.save(os.path.join(out_dir, 'repeat_predictions.npy'), prediction_runs)
    np.save(os.path.join(out_dir, 'repeat_thresholds.npy'), threshold_runs)
    np.save(os.path.join(out_dir, 'repeat_seeds.npy'), repeat_seeds)
    with open(os.path.join(out_dir, 'label_cols.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(label_cols, handle, indent=2)

    bootstrap_summary_path = os.path.join(
        out_dir, 'test_bootstrap_summary.csv')
    bootstrap_samples_path = os.path.join(
        out_dir, 'test_bootstrap_samples.csv')
    bootstrap_summary = None
    if args.n_bootstrap:
        print('\nTest-set Bootstrap: fixed models, probabilities, and OOF '
              'thresholds')
        print(f'  Replicates: {args.n_bootstrap}; seed: '
              f'{args.bootstrap_seed}')
        bootstrap_started = time.perf_counter()
        bootstrap_summary, bootstrap_samples = bootstrap_test_metrics(
            y_true, probability_runs, prediction_runs, label_cols,
            repeat_df, args.n_bootstrap, args.confidence,
            args.bootstrap_seed)
        bootstrap_summary.to_csv(bootstrap_summary_path, index=False)
        bootstrap_samples.to_csv(bootstrap_samples_path, index=False)
        print('  Bootstrap metrics completed in '
              f'{time.perf_counter() - bootstrap_started:.1f} seconds.')
    else:
        for stale_path in (bootstrap_summary_path, bootstrap_samples_path):
            if os.path.exists(stale_path):
                os.remove(stale_path)
        print('\nTest-set Bootstrap skipped: --n_bootstrap=0.')

    if args.baseline_dir:
        baseline_prob, baseline_pred = load_baseline_results(
            args.baseline_dir, probability_runs.shape,
            repeat_seeds, label_cols)
        significance = paired_seed_significance_test(
            y_true, probability_runs, prediction_runs,
            baseline_prob, baseline_pred, label_cols, args.confidence)
        significance.insert(1, 'Baseline', args.baseline_name)
        significance.to_csv(os.path.join(
            out_dir, 'paired_significance_tests.csv'), index=False)
    else:
        print('\nSignificance test skipped: provide --baseline_dir to compare models.')

    summary_by_metric = repeated_summary.set_index('Metric')
    print('\n' + '=' * 88)
    print(f'TEST RESULTS - repeated seed mean +/- SD and '
          f'{args.confidence:.0%} seed-level t CI')
    print('=' * 88)
    for metric in repeated_summary['Metric']:
        summary = summary_by_metric.loc[metric]
        print(f"  {metric:20s}: {summary['Mean']:.4f} +/- "
              f"{summary['Std']:.4f} [{summary['CI_Lower']:.4f}, "
              f"{summary['CI_Upper']:.4f}]")
    if bootstrap_summary is not None:
        bootstrap_by_metric = bootstrap_summary.set_index('Metric')
        print('\n' + '=' * 88)
        print(f'TEST-MOLECULE BOOTSTRAP - {args.n_bootstrap} resamples, '
              f'{args.confidence:.0%} percentile CI')
        print('=' * 88)
        for metric in bootstrap_summary['Metric']:
            summary = bootstrap_by_metric.loc[metric]
            print(f"  {metric:20s}: {summary['Point_Estimate']:.4f} "
                  f"[{summary['CI_Lower']:.4f}, "
                  f"{summary['CI_Upper']:.4f}]")
    print(f'\nResults saved to: {out_dir}')


if __name__ == '__main__':
    main()
