"""
Ablation Study — Single-feature models for comparison with DLMTF-Net fusion model.

Modes:
  unimol_only  — uses only UniMol 768-dim embeddings
  mordred_only — uses only Mordred 300-dim descriptors

Everything else (ASL loss, augmentation, training schedule, EMA, evaluation)
is kept identical to the fusion model in train_taste_model.py.

Usage:
    python train_ablation.py --mode unimol_only
    python train_ablation.py --mode mordred_only
"""

import os
import json
import math
import logging
import argparse
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


DEFAULT_CONFIG = {
    # Data paths (same as fusion model)
    'unimol_train_path':  r'./unimol_feature/base_feature/train_output/train_data_molecular_features.npy',
    'mordred_train_path': r'./mordred/taste_descriptors/train_processed.csv',
    'labels_train_path':  r'./Data/processed_data/train_data.csv',
    'output_base': r'./ablation_output',
    'run_name': 'ablation',

    # Feature dims (auto-updated from data)
    'num_classes': 6,
    'input_dim': 768,   # will be set based on mode

    # Model (same hidden_dim / query_dim as fusion)
    'hidden_dim': 384,
    'query_dim': 64,
    'dropout': 0.25,

    # Auxiliary loss
    'use_aux_loss': True,
    'aux_weight': 0.3,
    'infer_weights': [0.7, 0.3],   # main / aux

    # ASL (same as fusion)
    'asl_gamma_neg': 1,
    'asl_gamma_pos': 0,
    'asl_clip': 0.05,

    # Training (same as fusion)
    'batch_size': 64,
    'num_epochs': 200,
    'lr': 1e-4,
    'weight_decay': 1e-3,
    'max_grad_norm': 5.0,
    'num_warmup_epochs': 5,

    # Augmentation (same as fusion)
    'noise_std': 0.03,
    'feat_mask_prob': 0.15,
    'use_mixup': True,
    'mixup_alpha': 0.2,
    'label_smoothing': 0.05,

    'use_smote': True,
    'smote_label_indices': [4, 5],
    'smote_k': 5,
    'smote_aug_factor': 4.0,
    'smote_aug_factor_per_label': {4: 4.0, 5: 0.1},

    # EMA (same as fusion)
    'use_ema': True,
    'ema_decay': 0.999,

    # Weighted sampling
    'use_weighted_sampler': False,

    # Threshold tuning (same as fusion)
    'tune_thresholds': True,
    'threshold_grid_step': 0.02,

    # Label weights (same as fusion)
    'use_label_weights': True,
    'label_weight_power': 0.5,
    'label_weight_clip': 20.0,

    # Early stopping
    'es_monitor': 'macro_f1',

    # CV (same as fusion)
    'n_folds': 5,
    'patience': 15,
    'random_state': 42,
    'use_amp': False,
}


# ── Loss (identical to fusion model) ─────────────────────────────────────────

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg: float = 4, gamma_pos: float = 0,
                 clip: float = 0.05, eps: float = 1e-8,
                 label_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        if label_weights is not None:
            self.register_buffer('label_weights', label_weights)
        else:
            self.label_weights = None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xs_pos = torch.sigmoid(x)
        xs_neg = 1 - xs_pos
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        loss = y * torch.log(xs_pos.clamp(min=self.eps)) + \
               (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = xs_pos * y + xs_neg * (1 - y)
            loss = loss * torch.pow(1 - pt, self.gamma_pos * y + self.gamma_neg * (1 - y))
        loss = -loss
        if self.label_weights is not None:
            loss = loss * self.label_weights
        return loss.mean()


# ── SMOTE (single-feature version) ───────────────────────────────────────────

def smote_minority(X: np.ndarray, Y: np.ndarray,
                   label_indices, k: int = 3, aug_factor: float = 2.0,
                   aug_per_label: dict = None) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.neighbors import NearestNeighbors
    X_parts, Y_parts = [X], [Y]

    for label_idx in label_indices:
        factor = (aug_per_label or {}).get(label_idx, aug_factor)
        pos_mask = Y[:, label_idx] == 1
        X_pos, y_pos = X[pos_mask], Y[pos_mask]
        n = len(X_pos)
        if n < 2:
            continue
        k_eff = min(k, n - 1)
        nbrs = NearestNeighbors(n_neighbors=k_eff + 1).fit(X_pos)
        neighbors = nbrs.kneighbors(X_pos, return_distance=False)[:, 1:]
        n_syn = int(n * factor)
        syn_X, syn_y = [], []
        for _ in range(n_syn):
            i = np.random.randint(n)
            j = neighbors[i, np.random.randint(k_eff)]
            lam = np.random.random()
            syn_X.append(lam * X_pos[i] + (1 - lam) * X_pos[j])
            syn_y.append(y_pos[i])
        X_parts.append(np.array(syn_X, dtype=np.float32))
        Y_parts.append(np.array(syn_y, dtype=np.float32))

    return np.vstack(X_parts), np.vstack(Y_parts)


# ── Mixup (single-feature version) ───────────────────────────────────────────

def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    lam = max(lam, 1 - lam)
    B = x.size(0)
    idx = torch.randperm(B, device=x.device)
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]


# ── EMA (identical to fusion model) ──────────────────────────────────────────

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ep, p in zip(self.ema_model.parameters(), model.parameters()):
            ep.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def get_model(self) -> nn.Module:
        return self.ema_model


def compute_sample_weights(Y: np.ndarray) -> np.ndarray:
    N = len(Y)
    freq = Y.sum(axis=0).clip(min=1) / N
    rare_w = 1.0 / (freq + 1e-6)
    rare_w /= rare_w.mean()
    sw = (Y * rare_w).sum(axis=1).clip(min=0.1)
    return (sw / sw.mean()).astype(np.float32)


# ── Dataset (single-feature) ─────────────────────────────────────────────────

class SingleFeatureDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray,
                 noise_std: float = 0.0, feat_mask_prob: float = 0.0,
                 training: bool = True):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
        self.noise_std = noise_std
        self.feat_mask_prob = feat_mask_prob
        self.training = training

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.training:
            if self.noise_std > 0:
                x += torch.randn_like(x) * self.noise_std
            if self.feat_mask_prob > 0:
                x *= torch.bernoulli(torch.ones_like(x) * (1 - self.feat_mask_prob))
        return x, self.Y[idx]


# ── Model building block (identical to fusion model) ─────────────────────────

def _mlp_block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(dropout)
    )


# ── Single-branch ablation model ─────────────────────────────────────────────

class SingleBranchModel(nn.Module):
    """
    Single-branch ablation model.
    Branch architecture is identical to the corresponding branch in DLMTF-Net:
      embed branch: input_dim → 512 → hidden_dim   (for unimol_only)
      desc  branch: input_dim → 256 → hidden_dim   (for mordred_only)
    Label query head and aux head are identical to the fusion model.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 384,
                 query_dim: int = 64, num_classes: int = 6,
                 dropout: float = 0.25, use_aux_loss: bool = True,
                 branch_type: str = 'embed'):
        super().__init__()
        self.use_aux_loss = use_aux_loss
        self.num_classes = num_classes

        if branch_type == 'embed':
            # Same as fusion model embed_branch: input_dim → 512 → H
            self.branch = nn.Sequential(
                _mlp_block(input_dim, 512, dropout * 0.5),
                _mlp_block(512, hidden_dim, dropout * 0.5),
            )
        else:
            # Same as fusion model desc_branch: input_dim → 256 → H
            self.branch = nn.Sequential(
                _mlp_block(input_dim, 256, dropout * 0.5),
                _mlp_block(256, hidden_dim, dropout * 0.5),
            )

        # Label query head (identical to fusion model)
        self.label_emb = nn.Embedding(num_classes, query_dim)
        nn.init.normal_(self.label_emb.weight, 0.0, 0.01)
        self.h_to_query = nn.Linear(hidden_dim, query_dim, bias=False)
        self.label_bias = nn.Parameter(torch.zeros(num_classes))

        if use_aux_loss:
            self.aux_head = nn.Linear(hidden_dim, num_classes)

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 64)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict:
        h = self.branch(x)
        logits = self.h_to_query(h) @ self.label_emb.weight.T + self.label_bias
        return {
            'logits': logits,
            'logits_aux': self.aux_head(h) if self.use_aux_loss else None,
            'z': F.normalize(self.projection(h), p=2, dim=1),
            'h': h,
        }


# ── Contrastive loss (identical to fusion model) ─────────────────────────────

def multilabel_supcon_loss(z: torch.Tensor, y: torch.Tensor,
                           temperature: float = 0.07) -> torch.Tensor:
    B, device = z.size(0), z.device
    sim = torch.matmul(z, z.T) / temperature
    pos_mask = (torch.matmul(y, y.T) > 0).float()
    pos_mask.fill_diagonal_(0)
    mask = ~torch.eye(B, device=device, dtype=torch.bool)
    log_prob = sim - torch.log((torch.exp(sim) * mask.float()).sum(1, keepdim=True) + 1e-12)
    valid = pos_mask.sum(1) > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)
    return -(pos_mask * log_prob).sum(1)[valid].div(pos_mask.sum(1)[valid]).mean()


# ── Scheduler (identical to fusion model) ─────────────────────────────────────

def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Metrics (identical to fusion model) ───────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    thresholds: Optional[np.ndarray] = None) -> Dict:
    if thresholds is None:
        thresholds = np.full(y_true.shape[1], 0.5)
    y_pred = (y_prob >= thresholds).astype(int)
    metrics = {
        'macro_f1':   f1_score(y_true, y_pred, average='macro',  zero_division=0),
        'micro_f1':   f1_score(y_true, y_pred, average='micro',  zero_division=0),
        'macro_auprc': 0.0, 'micro_auprc': 0.0,
    }
    try:
        metrics['macro_auprc'] = average_precision_score(y_true, y_prob, average='macro')
    except Exception:
        pass
    try:
        metrics['micro_auprc'] = average_precision_score(y_true, y_prob, average='micro')
    except Exception:
        pass
    return metrics


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray,
                    step: float = 0.02) -> np.ndarray:
    C = y_true.shape[1]
    thr = np.full(C, 0.5)
    grid = np.arange(0.05, 0.95 + step, step)
    for c in range(C):
        if y_true[:, c].sum() == 0:
            continue
        best_f1, best_t = 0.0, 0.5
        for t in grid:
            f1 = f1_score(y_true[:, c], (y_prob[:, c] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thr[c] = best_t
    return thr


def ensemble_probs(outputs: Dict, weights) -> np.ndarray:
    """Weighted average of main + auxiliary branch sigmoid probabilities."""
    p = torch.sigmoid(outputs['logits']).cpu().numpy()
    if outputs.get('logits_aux') is not None:
        p_aux = torch.sigmoid(outputs['logits_aux']).cpu().numpy()
        return weights[0] * p + weights[1] * p_aux
    return p


# ── Training / validation loops ───────────────────────────────────────────────

def train_epoch(model: nn.Module, loader: DataLoader, asl_fn: AsymmetricLoss,
                optimizer, scheduler, ema: Optional[ModelEMA],
                device: torch.device, config: Dict, epoch: int,
                scaler=None) -> Dict:
    model.train()
    total = total_main = total_aux = 0.0
    N = 0
    use_supcon = config.get('use_supcon', False) and epoch >= config.get('supcon_start_epoch', 999)
    smooth = config.get('label_smoothing', 0.0)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if config['use_mixup']:
            x, y = mixup_batch(x, y, config['mixup_alpha'])

        y_s = y * (1 - smooth) + smooth * 0.5 if smooth > 0 else y

        optimizer.zero_grad()

        def fwd():
            out = model(x)
            l_main = asl_fn(out['logits'], y_s)
            l_aux = torch.tensor(0.0, device=device)
            if config['use_aux_loss'] and out['logits_aux'] is not None:
                l_aux = config['aux_weight'] * asl_fn(out['logits_aux'], y_s)
            l_sc = torch.tensor(0.0, device=device)
            if use_supcon:
                l_sc = config['lambda_supcon'] * multilabel_supcon_loss(
                    out['z'], y.clamp(0, 1).round(), config['temperature'])
            return l_main + l_aux + l_sc, l_main, l_aux

        if config['use_amp'] and scaler:
            with torch.cuda.amp.autocast():
                loss, l_main, l_aux = fwd()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, l_main, l_aux = fwd()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()

        scheduler.step()
        if ema:
            ema.update(model)

        B = x.size(0)
        total += loss.item() * B
        total_main += l_main.item() * B
        total_aux += l_aux.item() * B
        N += B

    return {'loss': total / N, 'loss_main': total_main / N, 'loss_aux': total_aux / N}


@torch.no_grad()
def val_epoch(model: nn.Module, loader: DataLoader, asl_fn: AsymmetricLoss,
              device: torch.device, config: Dict) -> Tuple[Dict, np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    total_loss = N = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += asl_fn(out['logits'], y).item() * x.size(0)
        all_probs.append(ensemble_probs(out, config['infer_weights']))
        all_labels.append(y.cpu().numpy())
        N += x.size(0)

    y_true = np.vstack(all_labels)
    y_prob = np.vstack(all_probs)
    metrics = compute_metrics(y_true, y_prob)
    metrics['loss'] = total_loss / N

    auroc_list = []
    for c in range(y_true.shape[1]):
        try:
            auroc_list.append(roc_auc_score(y_true[:, c], y_prob[:, c])
                              if len(np.unique(y_true[:, c])) > 1 else np.nan)
        except Exception:
            auroc_list.append(np.nan)
    metrics['macro_auroc'] = float(np.nanmean(auroc_list))
    metrics['monitor'] = (metrics['macro_auroc'] + metrics['macro_auprc']) / 2

    return metrics, y_true, y_prob


# ── Fold training ─────────────────────────────────────────────────────────────

def train_fold(fold: int, train_idx: np.ndarray, val_idx: np.ndarray,
               X: np.ndarray, Y: np.ndarray,
               config: Dict, device: torch.device, logger: logging.Logger) -> Dict:
    logger.info(f"\nFold {fold+1}/{config['n_folds']}")

    X_tr, X_val = X[train_idx], X[val_idx]
    Y_tr, Y_val = Y[train_idx], Y[val_idx]

    # SMOTE
    if config.get('use_smote') and config.get('smote_label_indices'):
        n_before = len(Y_tr)
        X_tr, Y_tr = smote_minority(
            X_tr, Y_tr,
            label_indices=config['smote_label_indices'],
            k=config['smote_k'],
            aug_factor=config['smote_aug_factor'],
            aug_per_label=config.get('smote_aug_factor_per_label'),
        )
        logger.info(f"  SMOTE: {n_before} → {len(Y_tr)} samples")

    train_ds = SingleFeatureDataset(X_tr, Y_tr,
                                     noise_std=config.get('noise_std', 0.0),
                                     feat_mask_prob=config.get('feat_mask_prob', 0.0),
                                     training=True)
    val_ds = SingleFeatureDataset(X_val, Y_val, training=False)

    if config['use_weighted_sampler']:
        sw = compute_sample_weights(Y_tr)
        sampler = WeightedRandomSampler(torch.FloatTensor(sw), len(sw), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                                  sampler=sampler, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                                  shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'])

    model = SingleBranchModel(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        query_dim=config['query_dim'],
        num_classes=config['num_classes'],
        dropout=config['dropout'],
        use_aux_loss=config['use_aux_loss'],
        branch_type=config['branch_type'],
    ).to(device)

    # Per-label inverse-frequency weights (identical to fusion)
    label_weights = None
    if config.get('use_label_weights', False):
        pos_freq = Y_tr.sum(0).clip(min=1) / len(Y_tr)
        pw = config.get('label_weight_power', 0.5)
        lw = np.power(1.0 / pos_freq, pw)
        lw = lw / lw.mean()
        lw = lw.clip(max=config.get('label_weight_clip', 20.0))
        label_weights = torch.FloatTensor(lw).to(device)
        logger.info(f"  Label weights: min={label_weights.min():.2f}  mean=1.00  max={label_weights.max():.2f}")

    asl_fn = AsymmetricLoss(config['asl_gamma_neg'], config['asl_gamma_pos'], config['asl_clip'],
                            label_weights=label_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'],
                                  weight_decay=config['weight_decay'])
    steps_per_epoch = len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=config['num_warmup_epochs'] * steps_per_epoch,
        total_steps=config['num_epochs'] * steps_per_epoch,
    )
    ema = ModelEMA(model, config['ema_decay']) if config['use_ema'] else None
    scaler = torch.cuda.amp.GradScaler() if config['use_amp'] else None

    best_monitor = 0.0
    patience = 0
    best_state = None
    history = []

    for epoch in range(config['num_epochs']):
        train_m = train_epoch(model, train_loader, asl_fn, optimizer, scheduler,
                              ema, device, config, epoch, scaler)
        val_model = ema.get_model() if ema else model
        val_m, _, _ = val_epoch(val_model, val_loader, asl_fn, device, config)

        history.append({'epoch': epoch,
                        **{f'train_{k}': v for k, v in train_m.items()},
                        **{f'val_{k}': v for k, v in val_m.items()}})

        if (epoch + 1) % 5 == 0 or epoch < 3:
            logger.info(
                f"Ep {epoch+1:3d}  "
                f"Loss={train_m['loss']:.4f} (main={train_m['loss_main']:.4f} aux={train_m['loss_aux']:.4f})  "
                f"F1={val_m['macro_f1']:.4f}  AUPRC={val_m['macro_auprc']:.4f}  "
                f"Monitor={(val_m['macro_auroc']+val_m['macro_auprc'])/2:.4f}"
            )

        monitor_val = (val_m['macro_auroc'] + val_m['macro_auprc']) / 2
        if monitor_val > best_monitor:
            best_monitor = monitor_val
            patience = 0
            best_state = deepcopy(val_model.state_dict())
        else:
            patience += 1
            if patience >= config['patience']:
                logger.info(f"  Early stop ep {epoch+1}  best_monitor={(best_monitor):.4f}")
                break

    eval_model = ema.get_model() if ema else model
    eval_model.load_state_dict(best_state)

    _, y_val_true, y_val_prob = val_epoch(eval_model, val_loader, asl_fn, device, config)
    thresholds = (tune_thresholds(y_val_true, y_val_prob, config['threshold_grid_step'])
                  if config['tune_thresholds'] else np.full(config['num_classes'], 0.5))
    val_metrics = compute_metrics(y_val_true, y_val_prob, thresholds)

    logger.info(f"  Fold {fold+1} result: F1={val_metrics['macro_f1']:.4f}  "
                f"AUPRC={val_metrics['macro_auprc']:.4f}")

    fold_dir = Path(config['output_dir']) / f'fold_{fold+1}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': eval_model.state_dict(),
        'config': config,
        'thresholds': thresholds,
        'history': history,
        'val_metrics': val_metrics,
    }, fold_dir / 'model.pt')
    pd.DataFrame(history).to_csv(fold_dir / 'history.csv', index=False)

    return val_metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                        choices=['unimol_only', 'mordred_only'],
                        help='Ablation mode: unimol_only or mordred_only')
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config.update(json.load(f))

    # Set mode-specific parameters
    config['mode'] = args.mode
    if args.mode == 'unimol_only':
        config['branch_type'] = 'embed'
        config['run_name'] = 'ablation_unimol_only'
    else:
        config['branch_type'] = 'desc'
        config['run_name'] = 'ablation_mordred_only'

    ts = datetime.now().strftime('%Y%m%d_%H%M')
    config['output_dir'] = os.path.join(config['output_base'], f"{config['run_name']}_{ts}")
    os.makedirs(config['output_dir'], exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config['output_dir'], 'train.log')),
            logging.StreamHandler(),
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Ablation Study: {args.mode}")
    logger.info(f"Branch type: {config['branch_type']}")

    # Load labels
    df_labels  = pd.read_csv(config['labels_train_path'])
    label_cols = [c for c in df_labels.columns if c != 'SMILES']

    # Load features based on mode
    if args.mode == 'unimol_only':
        X = np.load(config['unimol_train_path']).astype(np.float32)
        logger.info(f"UniMol features: {X.shape}")
        # Need labels aligned — use mordred CSV for label columns
        df_mordred = pd.read_csv(config['mordred_train_path'])
        Y = df_mordred[label_cols].values.astype(np.float32)
        assert X.shape[0] == len(df_mordred), \
            f"Row count mismatch: UniMol={X.shape[0]}, Mordred={len(df_mordred)}"
    else:
        df_mordred = pd.read_csv(config['mordred_train_path'])
        mordred_cols = [c for c in df_mordred.columns if c not in ['SMILES'] + label_cols]
        X = df_mordred[mordred_cols].values.astype(np.float32)
        Y = df_mordred[label_cols].values.astype(np.float32)
        logger.info(f"Mordred features: {X.shape}")

    config['input_dim']   = X.shape[1]
    config['num_classes'] = Y.shape[1]

    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    logger.info(f"X={X.shape}, Y={Y.shape}  "
                f"pos/label: min={Y.sum(0).min():.0f} mean={Y.sum(0).mean():.1f} max={Y.sum(0).max():.0f}")
    logger.info(f"ASL(γ-={config['asl_gamma_neg']}, γ+={config['asl_gamma_pos']}, clip={config['asl_clip']})  "
                f"noise={config['noise_std']}  mask={config['feat_mask_prob']}  "
                f"smooth={config['label_smoothing']}  smote={config['use_smote']}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    with open(os.path.join(config['output_dir'], 'label_cols.json'), 'w') as f:
        json.dump(label_cols, f)

    skf = StratifiedKFold(n_splits=config['n_folds'], shuffle=True,
                          random_state=config['random_state'])
    fold_results = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, Y[:, 0])):
        r = train_fold(fold, tr_idx, val_idx, X, Y, config, device, logger)
        fold_results.append(r)

    logger.info("\nCROSS-VALIDATION RESULTS")
    for key in ['macro_f1', 'micro_f1', 'macro_auprc', 'micro_auprc']:
        vals = [r[key] for r in fold_results]
        logger.info(f"  {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    pd.DataFrame(fold_results).to_csv(os.path.join(config['output_dir'], 'cv_results.csv'), index=False)
    with open(os.path.join(config['output_base'], f'latest_{args.mode}.txt'), 'w') as f:
        f.write(config['output_dir'])
    logger.info(f"Saved to: {config['output_dir']}")


if __name__ == '__main__':
    main()
