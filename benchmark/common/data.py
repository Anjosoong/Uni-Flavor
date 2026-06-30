"""Data loading, CV splits, and test-result export for benchmark models."""

import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


benchmark_root = Path(__file__).resolve().parent.parent


def resolve_config_path(config_path: str) -> Path:
    """Resolve a config path from cwd, benchmark root, or configs/ by filename."""
    p = Path(config_path)
    if p.is_absolute():
        return p.resolve()

    candidates = [
        Path.cwd() / p,
        benchmark_root / p,
    ]
    if 'configs' in p.parts:
        idx = p.parts.index('configs')
        candidates.append(benchmark_root / Path(*p.parts[idx:]))

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved

    if p.suffix == '.json':
        return (benchmark_root / 'configs' / p.name).resolve()
    return (benchmark_root / p).resolve()


def load_task_config(config_path: str) -> dict:
    """Load task JSON and resolve label CSV paths against ``base_dir``."""
    with open(config_path, encoding='utf-8') as f:
        cfg = json.load(f)

    base = Path(cfg['base_dir'])
    if not base.is_absolute():
        base = (benchmark_root / base).resolve()

    cfg['base_dir_resolved'] = str(base)

    def resolve(key: str) -> str:
        if key not in cfg:
            return ''
        p = Path(cfg[key])
        if p.is_absolute():
            return str(p)
        return str((base / p).resolve())

    for key in (
        'labels_train_path', 'labels_test_path',
        'train_script', 'test_script',
    ):
        if key in cfg:
            cfg[key] = resolve(key)

    return cfg


def load_task_config_raw(config_path: str) -> dict:
    """Load task JSON without resolving paths (portable relative paths)."""
    with open(config_path, encoding='utf-8') as f:
        return json.load(f)


def save_run_config(config_path: str, output_path: str) -> None:
    """Write a copy of the source task config into a run output directory."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(load_task_config_raw(config_path), f, indent=2)
        f.write('\n')


def label_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith('TARGET_')]


def load_smiles_labels(labels_path: str) -> Tuple[List[str], List[str], np.ndarray]:
    """Load SMILES strings and multilabel matrix ``y`` with shape (N, C)."""
    df = pd.read_csv(labels_path)
    cols = label_cols(df)
    smiles = df['SMILES'].astype(str).tolist()
    y = df[cols].values.astype(np.float32)
    return smiles, cols, y


def get_cv_splits(
    n_samples: int,
    y: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Stratified K-fold splits; stratify on the first label column."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return list(skf.split(np.arange(n_samples), y[:, 0]))


def save_test_results(
    output_dir: str,
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    label_cols: List[str],
    n_folds: int = 5,
) -> str:
    """Write overall/per-label CSV metrics and numpy probability arrays."""
    from .metrics import compute_overall_metrics, compute_per_label_metrics

    os.makedirs(output_dir, exist_ok=True)
    y_pred = (y_prob >= thresholds).astype(int)

    overall = compute_overall_metrics(y_true, y_pred, y_prob, label_cols)
    overall['Model'] = model_name
    overall['N_Folds'] = n_folds
    per_label = compute_per_label_metrics(y_true, y_pred, y_prob, label_cols)

    pd.DataFrame([overall]).to_csv(os.path.join(output_dir, 'overall_results.csv'), index=False)
    pd.DataFrame(per_label).to_csv(os.path.join(output_dir, 'per_label_results.csv'), index=False)
    np.save(os.path.join(output_dir, 'probabilities.npy'), y_prob)
    np.save(os.path.join(output_dir, 'predictions.npy'), y_pred)
    np.save(os.path.join(output_dir, 'thresholds.npy'), thresholds)
    return output_dir
