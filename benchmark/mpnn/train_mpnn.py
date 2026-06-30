"""
MPNN baseline (ChemProp 2.2.3): SMILES-only multilabel graph neural network.

Evaluation protocol matches DLMOF/DLMTF-Net:
  5-fold CV -> per-label dynamic thresholds -> 5-fold test ensemble.

Prerequisites:
    python setup_deps.py

Usage:
    python mpnn/train_mpnn.py --config configs/odor.json
    python mpnn/train_mpnn.py --config configs/taste.json --device cuda
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

benchmark_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(benchmark_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.data import load_task_config, load_smiles_labels, resolve_config_path, save_run_config
from common.deps import chemprop_version
from common.device import resolve_device
from common.evaluation import run_cv_training, run_test_evaluation
from common.metrics import compute_cv_metrics
from chemprop_backend import (
    build_dataloader,
    build_model,
    load_model,
    predict_probs,
    save_model,
    train_step,
)


def train_one_fold(train_idx, val_idx, smiles, y, config, device):
    """Train one CV fold with early stopping on validation monitor."""
    tr_smiles = [smiles[i] for i in train_idx]
    val_smiles = [smiles[i] for i in val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    train_loader, train_valid_idx = build_dataloader(
        tr_smiles, y_tr, config['mpnn_batch_size'], shuffle=True,
    )
    val_loader, val_valid_idx = build_dataloader(
        val_smiles, y_val, config['mpnn_batch_size'], shuffle=False,
    )
    y_val_aligned = y_val[val_valid_idx]

    model = build_model(y.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['mpnn_lr'])

    best_monitor = -1.0
    best_state = None
    patience_counter = 0

    epoch_bar = tqdm(
        range(config['mpnn_num_epochs']),
        desc='  Epochs',
        leave=False,
    )
    for epoch in epoch_bar:
        model.train()
        for batch in tqdm(
            train_loader, desc=f'  Ep {epoch + 1} train', leave=False,
        ):
            optimizer.zero_grad()
            loss = train_step(model, batch, device)
            loss.backward()
            optimizer.step()

        val_prob = predict_probs(model, val_loader, device, desc='  Val')
        metrics = compute_cv_metrics(
            y_val_aligned, val_prob,
            np.full(y.shape[1], 0.5),
        )
        monitor = metrics['monitor']
        if np.isnan(monitor):
            monitor = metrics['macro_f1']

        if monitor > best_monitor:
            best_monitor = monitor
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['mpnn_patience']:
                break

        epoch_bar.set_postfix(
            monitor=f'{monitor:.4f}',
            best=f'{best_monitor:.4f}',
            patience=f'{patience_counter}/{config["mpnn_patience"]}',
        )

    model.load_state_dict(best_state)
    model.to(device)
    val_prob = predict_probs(model, val_loader, device)
    return val_prob, model, y_val_aligned


def main():
    parser = argparse.ArgumentParser(description='MPNN benchmark (ChemProp 2.2.3)')
    parser.add_argument('--config', type=str, default='configs/odor.json')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    args = parser.parse_args()

    config_path = str(resolve_config_path(args.config))
    config = load_task_config(config_path)

    ts = datetime.now().strftime('%Y%m%d_%H%M')
    output_dir = os.path.join(config['output_base'], f'mpnn_{ts}')
    os.makedirs(output_dir, exist_ok=True)
    save_run_config(config_path, os.path.join(output_dir, 'config.json'))

    device = resolve_device(args.device)
    print(f'Device: {device}')
    print(f'ChemProp: {chemprop_version()}')

    smiles, label_cols, y_train = load_smiles_labels(config['labels_train_path'])
    print(f'Train: {len(smiles)} molecules, {y_train.shape[1]} labels')

    def train_fold_fn(fold, train_idx, val_idx):
        return train_one_fold(train_idx, val_idx, smiles, y_train, config, device)

    def save_fold_fn(fold_dir, model, thresholds):
        save_model(model, os.path.join(fold_dir, 'model.pt'))

    threshold_folds, avg_thresholds, _ = run_cv_training(
        config, y_train, label_cols, output_dir, train_fold_fn, save_fold_fn,
    )

    test_smiles, _, y_test = load_smiles_labels(config['labels_test_path'])
    test_loader, test_valid_idx = build_dataloader(
        test_smiles, y_test, config['mpnn_batch_size'], shuffle=False,
    )
    y_test_valid = y_test[test_valid_idx]

    def predict_test_fn(fold_dir, fold):
        model = load_model(os.path.join(fold_dir, 'model.pt'), device)
        return predict_probs(model, test_loader, device)

    run_test_evaluation(
        config, output_dir, 'MPNN (ChemProp)', y_test_valid, label_cols,
        predict_test_fn, threshold_folds, avg_thresholds,
    )

    with open(os.path.join(config['output_base'], 'mpnn_latest.txt'), 'w') as f:
        f.write(output_dir)


if __name__ == '__main__':
    main()
