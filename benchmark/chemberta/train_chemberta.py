"""
ChemBERTa baseline: fine-tune pretrained SMILES LM for multilabel prediction.

Evaluation protocol matches DLMOF/DLMTF-Net:
  5-fold CV -> per-label dynamic thresholds -> 5-fold test ensemble.

Prerequisites:
    python setup_deps.py

Usage:
    python chemberta/train_chemberta.py --config configs/odor.json
    python chemberta/train_chemberta.py --config configs/taste.json --device cuda
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

benchmark_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(benchmark_root))

from common.asymmetric_loss import AsymmetricLoss, build_label_weights
from common.data import load_task_config, load_smiles_labels, resolve_config_path, save_run_config
from common.device import resolve_device, dataloader_kwargs
from common.evaluation import run_cv_training, run_test_evaluation
from common.hf_load import ensure_pretrained, load_tokenizer, load_encoder, resolve_model_dir
from common.metrics import compute_cv_metrics, resolve_thresholds

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError as e:
    raise ImportError(
        'ChemBERTa requires transformers. Run: python setup_deps.py'
    ) from e


class SmilesDataset(Dataset):
    """Tokenized SMILES + multilabel target vector."""

    def __init__(self, smiles, labels, tokenizer, max_length):
        self.smiles = smiles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.smiles[i],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(self.labels[i], dtype=torch.float32)
        return item


class ChemBERTaClassifier(nn.Module):
    """Mean-pooled ChemBERTa encoder + MLP head -> C logits (multilabel)."""

    def __init__(self, model_name: str, num_classes: int, encoder=None, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder or load_encoder(
            AutoModel, model_name, trust_remote_code=True,
        )
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.classifier(self.dropout(pooled))


@torch.no_grad()
def predict(model, loader, device, desc='Predict'):
    model.eval()
    probs = []
    for batch in tqdm(loader, desc=desc, leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        logits = model(input_ids, attention_mask)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(probs)


def build_loss_fn(y_train_fold: np.ndarray, config: dict, device: torch.device) -> nn.Module:
    label_weights = None
    if config.get('chemberta_use_label_weights', True):
        label_weights = build_label_weights(
            y_train_fold,
            power=config.get('chemberta_label_weight_power', 0.5),
            clip=config.get('chemberta_label_weight_clip', 20.0),
            device=device,
        )
    return AsymmetricLoss(
        gamma_neg=config.get('chemberta_asl_gamma_neg', 4),
        gamma_pos=config.get('chemberta_asl_gamma_pos', 0),
        clip=config.get('chemberta_asl_clip', 0.05),
        label_weights=label_weights,
    )


def build_optimizer(model: ChemBERTaClassifier, config: dict):
    encoder_lr = config['chemberta_lr'] * config.get('chemberta_encoder_lr_scale', 0.1)
    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())
    encoder_params = list(model.encoder.parameters())
    return torch.optim.AdamW(
        [
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': head_params, 'lr': config['chemberta_lr']},
        ],
        weight_decay=config.get('chemberta_weight_decay', 0.01),
    )


def train_one_fold(train_idx, val_idx, smiles, y, config, device, tokenizer, loader_kw):
    """Train one CV fold with ASL loss and early stopping."""
    tr_smiles = [smiles[i] for i in train_idx]
    val_smiles = [smiles[i] for i in val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    tr_ds = SmilesDataset(
        tr_smiles, y_tr, tokenizer, config['chemberta_max_length'],
    )
    val_ds = SmilesDataset(
        val_smiles, y_val, tokenizer, config['chemberta_max_length'],
    )
    train_loader = DataLoader(
        tr_ds, batch_size=config['chemberta_batch_size'], shuffle=True, **loader_kw,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config['chemberta_batch_size'], **loader_kw,
    )

    model_dir = resolve_model_dir(config.get('chemberta_model_dir'))
    encoder = load_encoder(AutoModel, model_dir, trust_remote_code=True)
    model = ChemBERTaClassifier(model_dir, y.shape[1], encoder=encoder).to(device)
    optimizer = build_optimizer(model, config)
    loss_fn = build_loss_fn(y_tr, config, device)

    best_monitor = -1.0
    best_state = None
    patience_counter = 0
    max_grad_norm = config.get('chemberta_max_grad_norm', 1.0)

    epoch_bar = tqdm(
        range(config['chemberta_num_epochs']),
        desc='  Epochs',
        leave=False,
    )
    for epoch in epoch_bar:
        model.train()
        for batch in tqdm(
            train_loader, desc=f'  Ep {epoch + 1} train', leave=False,
        ):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        val_prob = predict(model, val_loader, device, desc='  Val')
        val_thresholds = resolve_thresholds(config, y_val, val_prob)
        metrics = compute_cv_metrics(y_val, val_prob, val_thresholds)
        monitor = metrics['monitor']
        if np.isnan(monitor):
            monitor = metrics['macro_f1']

        if monitor > best_monitor:
            best_monitor = monitor
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['chemberta_patience']:
                break

        epoch_bar.set_postfix(
            monitor=f'{monitor:.4f}',
            macro_f1=f'{metrics["macro_f1"]:.4f}',
            best=f'{best_monitor:.4f}',
            patience=f'{patience_counter}/{config["chemberta_patience"]}',
        )

    model.load_state_dict(best_state)
    model.to(device)
    val_prob = predict(model, val_loader, device)
    return val_prob, model


def main():
    parser = argparse.ArgumentParser(description='ChemBERTa multilabel benchmark')
    parser.add_argument('--config', type=str, default='configs/odor.json')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    args = parser.parse_args()

    config_path = str(resolve_config_path(args.config))
    config = load_task_config(config_path)
    model_dir = resolve_model_dir(config.get('chemberta_model_dir'))
    ensure_pretrained(config['chemberta_model_name'], model_dir)

    ts = datetime.now().strftime('%Y%m%d_%H%M')
    output_dir = os.path.join(config['output_base'], f'chemberta_{ts}')
    os.makedirs(output_dir, exist_ok=True)
    save_run_config(config_path, os.path.join(output_dir, 'config.json'))

    device = resolve_device(args.device)
    loader_kw = dataloader_kwargs(device)

    tokenizer = load_tokenizer(AutoTokenizer, model_dir, trust_remote_code=True)
    smiles, label_cols, y_train = load_smiles_labels(config['labels_train_path'])
    print(f'Train: {len(smiles)} molecules, {y_train.shape[1]} labels')

    def train_fold_fn(fold, train_idx, val_idx):
        val_prob, model = train_one_fold(
            train_idx, val_idx, smiles, y_train, config, device, tokenizer, loader_kw,
        )
        return val_prob, model

    def save_fold_fn(fold_dir, model, thresholds):
        torch.save(
            {'model_state_dict': model.state_dict(), 'thresholds': thresholds},
            os.path.join(fold_dir, 'model.pt'),
        )

    threshold_folds, avg_thresholds, _ = run_cv_training(
        config, y_train, label_cols, output_dir, train_fold_fn, save_fold_fn,
    )

    test_smiles, _, y_test = load_smiles_labels(config['labels_test_path'])
    test_ds = SmilesDataset(
        test_smiles, y_test, tokenizer, config['chemberta_max_length'],
    )
    test_loader = DataLoader(
        test_ds, batch_size=config['chemberta_batch_size'], **loader_kw,
    )

    def predict_test_fn(fold_dir, fold):
        ckpt = torch.load(
            os.path.join(fold_dir, 'model.pt'),
            map_location=device, weights_only=False,
        )
        model_dir = resolve_model_dir(config.get('chemberta_model_dir'))
        encoder = load_encoder(AutoModel, model_dir, trust_remote_code=True)
        model = ChemBERTaClassifier(model_dir, y_train.shape[1], encoder=encoder).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        return predict(model, test_loader, device)

    run_test_evaluation(
        config, output_dir, 'ChemBERTa', y_test, label_cols,
        predict_test_fn, threshold_folds, avg_thresholds,
    )

    with open(os.path.join(config['output_base'], 'chemberta_latest.txt'), 'w') as f:
        f.write(output_dir)


if __name__ == '__main__':
    main()
