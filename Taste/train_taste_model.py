"""
Dual-branch Label-aware Molecular Taste Fusion Network (DLMTF-Net)

Architecture:
  UniMol branch  : 768 -> 512 -> H  (LayerNorm + GELU + Dropout)
  Mordred branch : D   -> 256 -> H  (LayerNorm + GELU + Dropout)
  Fusion         : cat(h_e, h_d, h_e * h_d) -> 3H -> 2H -> H
  Label head     : logits = Linear(H->Q) @ E_label^T + b
  Aux heads      : Linear(H->C) per branch (deep supervision; blended at inference)

Loss: Asymmetric Loss (Ridnik et al., ICCV 2021) with per-label inverse-frequency weighting.
Augmentation: Gaussian noise, feature masking, asymmetric mixup.
Training: EMA, cosine LR with linear warmup, 5-fold cross-validation.
"""

import os
import json
import math
import logging
import argparse
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


DEFAULT_CONFIG = {
    # Data paths
    'unimol_train_path':  r'./unimol_feature/fintune_feature/train_output/train_data_molecular_features.npy',
    'mordred_train_path': r'./mordred/taste_descriptors/train_processed.csv',
    'labels_train_path':  r'./Data/processed_data/train_data.csv',
    'output_base': r'./model',
    'run_name': 'taste_train',

    # Feature dims (auto-updated from data)
    'num_classes': 6,
    'embed_dim': 768,
    'desc_dim': 300,

    # Model
    'hidden_dim': 384,
    'query_dim': 64,
    'dropout': 0.25,    # reduced from 0.3 — EMA + noise already provide regularization

    # Auxiliary loss weights and inference blend
    'use_aux_loss': True,
    'aux_weight_embed': 0.3,
    'aux_weight_desc': 0.2,
    'infer_weights': [0.6, 0.25, 0.15],  # main / embed_aux / desc_aux

    # ASL
    'asl_gamma_neg': 1,
    'asl_gamma_pos': 0,
    'asl_clip': 0.05,

    # Training
    'batch_size': 64,
    'num_epochs': 200,
    'lr': 1e-4,
    'weight_decay': 1e-3,
    'max_grad_norm': 5.0,
    'num_warmup_epochs': 5,

    # Augmentation
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

    # EMA
    'use_ema': True,
    'ema_decay': 0.999,

    # Weighted sampling (disabled: ASL label weights already handle imbalance)
    'use_weighted_sampler': False,

    # Threshold tuning
    'tune_thresholds': True,
    'threshold_grid_step': 0.02,

    # Label-frequency weighted ASL: upweight rare labels, w_c = (N/pos_c)^power, clipped
    'use_label_weights': True,
    'label_weight_power': 0.5,    # sqrt of inverse frequency (softer than linear)
    'label_weight_clip': 20.0,    # cap to avoid extreme weights on very rare labels

    # Early stopping monitor: (macro_auroc + macro_auprc) / 2
    'es_monitor': 'auroc_auprc',

    # CV
    'n_folds': 5,
    'patience': 15,
    'random_state': 42,
    'use_amp': False,
}


class AsymmetricLoss(nn.Module):
    """
    gamma_neg >> gamma_pos: easy negatives (abundant) get down-weighted aggressively.
    probability_shift (clip): completely ignores very confident correct negatives.
    label_weights [C]: per-label inverse-frequency weights, upweights rare labels.
    """
    def __init__(self, gamma_neg: float = 4, gamma_pos: float = 0,
                 clip: float = 0.05, eps: float = 1e-8,
                 label_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        if label_weights is not None:
            self.register_buffer('label_weights', label_weights)  # [C]
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
        loss = -loss  # [B, C]
        if self.label_weights is not None:
            loss = loss * self.label_weights  # broadcast [1, C]
        return loss.mean()


def smote_minority(X_embed: np.ndarray, X_desc: np.ndarray, Y: np.ndarray,
                   label_indices, k: int = 3, aug_factor: float = 2.0,
                   aug_per_label: dict = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    k-NN interpolation in the concatenated [embed|desc] feature space.
    Synthetic samples preserve cross-modal correlations; split back after synthesis.
    """
    from sklearn.neighbors import NearestNeighbors
    embed_dim = X_embed.shape[1]
    X = np.concatenate([X_embed, X_desc], axis=1).astype(np.float32)
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

    X_aug = np.vstack(X_parts)
    Y_aug = np.vstack(Y_parts)
    return X_aug[:, :embed_dim], X_aug[:, embed_dim:], Y_aug


def mixup_batch(x_embed: torch.Tensor, x_desc: torch.Tensor,
                y: torch.Tensor, alpha: float = 0.2):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    lam = max(lam, 1 - lam)  # asymmetric: always ≥ 0.5, dominant sample preserved
    B = x_embed.size(0)
    idx = torch.randperm(B, device=x_embed.device)
    return (lam * x_embed + (1 - lam) * x_embed[idx],
            lam * x_desc + (1 - lam) * x_desc[idx],
            lam * y + (1 - lam) * y[idx])


class ModelEMA:
    """Maintains exponential moving average of model weights for smoother inference."""
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


class MolecularDataset(Dataset):
    """
    noise_std > 0: adds Gaussian noise to both branches during training.
    feat_mask_prob > 0: randomly zeros out features (independently per branch).
    Both disabled at validation/test time.
    """
    def __init__(self, X_embed: np.ndarray, X_desc: np.ndarray, Y: np.ndarray,
                 noise_std: float = 0.0, feat_mask_prob: float = 0.0,
                 training: bool = True):
        self.X_embed = torch.FloatTensor(X_embed)
        self.X_desc = torch.FloatTensor(X_desc)
        self.Y = torch.FloatTensor(Y)
        self.noise_std = noise_std
        self.feat_mask_prob = feat_mask_prob
        self.training = training

    def __len__(self):
        return len(self.X_embed)

    def __getitem__(self, idx):
        xe = self.X_embed[idx].clone()
        xd = self.X_desc[idx].clone()
        if self.training:
            if self.noise_std > 0:
                xe += torch.randn_like(xe) * self.noise_std
                xd += torch.randn_like(xd) * self.noise_std
            if self.feat_mask_prob > 0:
                xe *= torch.bernoulli(torch.ones_like(xe) * (1 - self.feat_mask_prob))
                xd *= torch.bernoulli(torch.ones_like(xd) * (1 - self.feat_mask_prob))
        return xe, xd, self.Y[idx]


def _mlp_block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(dropout)
    )


class EnhancedMultiLabelModel(nn.Module):
    def __init__(self, embed_dim: int = 768, desc_dim: int = 300,
                 hidden_dim: int = 256, query_dim: int = 64,
                 num_classes: int = 138, dropout: float = 0.3,
                 use_aux_loss: bool = True):
        super().__init__()
        self.use_aux_loss = use_aux_loss
        self.num_classes = num_classes

        self.embed_branch = nn.Sequential(
            _mlp_block(embed_dim, 512, dropout * 0.5),
            _mlp_block(512, hidden_dim, dropout * 0.5),
        )
        self.desc_branch = nn.Sequential(
            _mlp_block(desc_dim, 256, dropout * 0.5),
            _mlp_block(256, hidden_dim, dropout * 0.5),
        )

        # fusion: cat(h_e, h_d, h_e*h_d) → 3H → 2H → H
        self.fusion = nn.Sequential(
            _mlp_block(hidden_dim * 3, hidden_dim * 2, dropout),
            _mlp_block(hidden_dim * 2, hidden_dim, dropout * 0.5),
        )

        # label query head: each label has its own 64-dim vector
        self.label_emb = nn.Embedding(num_classes, query_dim)
        nn.init.normal_(self.label_emb.weight, 0.0, 0.01)
        self.h_to_query = nn.Linear(hidden_dim, query_dim, bias=False)
        self.label_bias = nn.Parameter(torch.zeros(num_classes))

        if use_aux_loss:
            self.aux_head_embed = nn.Linear(hidden_dim, num_classes)
            self.aux_head_desc = nn.Linear(hidden_dim, num_classes)

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

    def forward(self, x_embed: torch.Tensor, x_desc: torch.Tensor) -> Dict:
        h_e = self.embed_branch(x_embed)
        h_d = self.desc_branch(x_desc)
        h = self.fusion(torch.cat([h_e, h_d, h_e * h_d], dim=1))

        # label query: each label attends to query projection of h
        logits = self.h_to_query(h) @ self.label_emb.weight.T + self.label_bias

        return {
            'logits': logits,
            'logits_embed': self.aux_head_embed(h_e) if self.use_aux_loss else None,
            'logits_desc':  self.aux_head_desc(h_d)  if self.use_aux_loss else None,
            'z': F.normalize(self.projection(h), p=2, dim=1),
            'h': h,
        }


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


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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
    """Per-label F1-optimal threshold search on validation set."""
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
    if outputs.get('logits_embed') is not None:
        pe = torch.sigmoid(outputs['logits_embed']).cpu().numpy()
        pd_ = torch.sigmoid(outputs['logits_desc']).cpu().numpy()
        return weights[0] * p + weights[1] * pe + weights[2] * pd_
    return p


def resolve_model_dir(model_dir: str) -> str:
    """Resolve to a run directory containing config.json."""
    if os.path.exists(os.path.join(model_dir, 'config.json')):
        return model_dir
    latest_txt = os.path.join(model_dir, 'latest.txt')
    if os.path.exists(latest_txt):
        with open(latest_txt) as f:
            return f.read().strip()
    subdirs = sorted(d for d in os.listdir(model_dir)
                     if os.path.isdir(os.path.join(model_dir, d)))
    if not subdirs:
        raise FileNotFoundError(f"No model found in {model_dir}")
    return os.path.join(model_dir, subdirs[-1])


def load_run_config(model_dir: str) -> Tuple[dict, list]:
    with open(os.path.join(model_dir, 'config.json')) as f:
        config = json.load(f)
    with open(os.path.join(model_dir, 'label_cols.json')) as f:
        label_cols = json.load(f)
    return config, label_cols


def build_model(config: dict, device: torch.device) -> EnhancedMultiLabelModel:
    return EnhancedMultiLabelModel(
        embed_dim=config['embed_dim'],
        desc_dim=config['desc_dim'],
        hidden_dim=config['hidden_dim'],
        query_dim=config['query_dim'],
        num_classes=config['num_classes'],
        dropout=config['dropout'],
        use_aux_loss=config['use_aux_loss'],
    ).to(device)


def load_fold_models(model_dir: str, device: torch.device,
                     verbose: bool = True) -> Tuple[List[EnhancedMultiLabelModel],
                                                    np.ndarray, dict, list]:
    """Load all fold checkpoints. Returns models, avg_thresholds, config, label_cols."""
    config, label_cols = load_run_config(model_dir)
    fold_dirs = sorted(d for d in os.listdir(model_dir) if d.startswith('fold_'))
    models, thresholds_list = [], []

    for fd in fold_dirs:
        ckpt = torch.load(os.path.join(model_dir, fd, 'model.pt'),
                          map_location=device, weights_only=False)
        model = build_model(config, device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        models.append(model)
        thresholds_list.append(ckpt['thresholds'])
        if verbose:
            print(f"  Loaded {fd}")

    return models, np.mean(thresholds_list, axis=0), config, label_cols


@torch.no_grad()
def predict_batch(model: nn.Module, loader: DataLoader, device: torch.device,
                  infer_weights) -> np.ndarray:
    model.eval()
    all_probs = []
    for x_embed, x_desc, _ in loader:
        x_embed, x_desc = x_embed.to(device), x_desc.to(device)
        out = model(x_embed, x_desc)
        all_probs.append(ensemble_probs(out, infer_weights))
    return np.vstack(all_probs)


@torch.no_grad()
def predict_single_ensemble(models: List[EnhancedMultiLabelModel],
                            x_embed: np.ndarray, x_desc: np.ndarray,
                            infer_weights, device: torch.device) -> np.ndarray:
    """Ensemble inference for one sample. Returns average probabilities (C,)."""
    x_embed_t = torch.tensor(x_embed[np.newaxis], dtype=torch.float32, device=device)
    x_desc_t = torch.tensor(x_desc[np.newaxis], dtype=torch.float32, device=device)
    all_probs = []
    for model in models:
        out = model(x_embed_t, x_desc_t)
        all_probs.append(ensemble_probs(out, infer_weights)[0])
    return np.mean(all_probs, axis=0)


def train_epoch(model: nn.Module, loader: DataLoader, asl_fn: AsymmetricLoss,
                optimizer, scheduler, ema: Optional[ModelEMA],
                device: torch.device, config: Dict, epoch: int,
                scaler=None) -> Dict:
    model.train()
    total = total_main = total_aux = 0.0
    N = 0
    use_supcon = config.get('use_supcon', False) and epoch >= config.get('supcon_start_epoch', 999)
    smooth = config.get('label_smoothing', 0.0)

    for xe, xd, y in loader:
        xe, xd, y = xe.to(device), xd.to(device), y.to(device)
        if config['use_mixup']:
            xe, xd, y = mixup_batch(xe, xd, y, config['mixup_alpha'])

        # Label smoothing: push hard 0/1 targets towards ε/2 and 1-ε/2
        y_s = y * (1 - smooth) + smooth * 0.5 if smooth > 0 else y

        optimizer.zero_grad()

        def fwd():
            out = model(xe, xd)
            l_main = asl_fn(out['logits'], y_s)
            l_aux = torch.tensor(0.0, device=device)
            if config['use_aux_loss'] and out['logits_embed'] is not None:
                l_aux = (config['aux_weight_embed'] * asl_fn(out['logits_embed'], y_s) +
                         config['aux_weight_desc']  * asl_fn(out['logits_desc'],  y_s))
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

        B = xe.size(0)
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

    for xe, xd, y in loader:
        xe, xd, y = xe.to(device), xd.to(device), y.to(device)
        out = model(xe, xd)
        total_loss += asl_fn(out['logits'], y).item() * xe.size(0)
        all_probs.append(ensemble_probs(out, config['infer_weights']))
        all_labels.append(y.cpu().numpy())
        N += xe.size(0)

    y_true = np.vstack(all_labels)
    y_prob = np.vstack(all_probs)
    metrics = compute_metrics(y_true, y_prob)
    metrics['loss'] = total_loss / N

    # Per-label AUROC for combined monitor (same as MLP baseline)
    auroc_list = []
    for c in range(y_true.shape[1]):
        try:
            auroc_list.append(roc_auc_score(y_true[:, c], y_prob[:, c])
                              if len(np.unique(y_true[:, c])) > 1 else np.nan)
        except Exception:
            auroc_list.append(np.nan)
    metrics['macro_auroc'] = float(np.nanmean(auroc_list))
    metrics['monitor'] = (metrics['macro_auroc'] + metrics['macro_auprc']) / 2  # combined monitor

    return metrics, y_true, y_prob


def train_fold(fold: int, train_idx: np.ndarray, val_idx: np.ndarray,
               X_embed: np.ndarray, X_desc: np.ndarray, Y: np.ndarray,
               config: Dict, device: torch.device, logger: logging.Logger) -> Dict:
    logger.info(f"\nFold {fold+1}/{config['n_folds']}")

    X_e_tr, X_e_val = X_embed[train_idx], X_embed[val_idx]
    X_d_tr, X_d_val = X_desc[train_idx], X_desc[val_idx]
    Y_tr, Y_val = Y[train_idx], Y[val_idx]

    # SMOTE: augment rare-label positives in concatenated feature space
    if config.get('use_smote') and config.get('smote_label_indices'):
        n_before = len(Y_tr)
        X_e_tr, X_d_tr, Y_tr = smote_minority(
            X_e_tr, X_d_tr, Y_tr,
            label_indices=config['smote_label_indices'],
            k=config['smote_k'],
            aug_factor=config['smote_aug_factor'],
            aug_per_label=config.get('smote_aug_factor_per_label'),
        )
        logger.info(f"  SMOTE: {n_before} → {len(Y_tr)} samples")

    # Datasets with online noise + masking
    train_ds = MolecularDataset(X_e_tr, X_d_tr, Y_tr,
                                noise_std=config.get('noise_std', 0.0),
                                feat_mask_prob=config.get('feat_mask_prob', 0.0),
                                training=True)
    val_ds = MolecularDataset(X_e_val, X_d_val, Y_val, training=False)

    # Weighted sampler (computed on original + SMOTE-augmented train labels)
    if config['use_weighted_sampler']:
        sw = compute_sample_weights(Y_tr)
        sampler = WeightedRandomSampler(torch.FloatTensor(sw), len(sw), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                                  sampler=sampler, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                                  shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'])

    model = EnhancedMultiLabelModel(
        embed_dim=config['embed_dim'], desc_dim=config['desc_dim'],
        hidden_dim=config['hidden_dim'], query_dim=config['query_dim'],
        num_classes=config['num_classes'], dropout=config['dropout'],
        use_aux_loss=config['use_aux_loss'],
    ).to(device)

    # Per-label inverse-frequency weights: rare labels get higher loss weight
    label_weights = None
    if config.get('use_label_weights', False):
        pos_freq = Y_tr.sum(0).clip(min=1) / len(Y_tr)  # [C] positive frequency per label
        pw = config.get('label_weight_power', 0.5)
        lw = np.power(1.0 / pos_freq, pw)               # inverse-freq raised to power
        lw = lw / lw.mean()                              # normalize so mean weight = 1
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

        monitor_val = (val_m['macro_auroc'] + val_m['macro_auprc']) / 2  # es_monitor: auroc_auprc
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config.update(json.load(f))

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
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    logger.info("DLMTF-Net: Dual-branch Label-aware Molecular Taste Fusion Network")
    logger.info(f"ASL(γ-={config['asl_gamma_neg']}, γ+={config['asl_gamma_pos']}, clip={config['asl_clip']})  "
                f"noise={config['noise_std']}  mask={config['feat_mask_prob']}  "
                f"smooth={config['label_smoothing']}  smote={config['use_smote']}")

    df_labels  = pd.read_csv(config['labels_train_path'])
    label_cols = [c for c in df_labels.columns if c != 'SMILES']

    df_mordred   = pd.read_csv(config['mordred_train_path'])
    mordred_cols = [c for c in df_mordred.columns if c not in ['SMILES'] + label_cols]

    X_embed = np.load(config['unimol_train_path']).astype(np.float32)
    X_desc  = df_mordred[mordred_cols].values.astype(np.float32)
    Y       = df_mordred[label_cols].values.astype(np.float32)
    assert X_embed.shape[0] == len(df_mordred), \
        f"Row count mismatch: UniMol={X_embed.shape[0]}, Mordred={len(df_mordred)}"

    config['embed_dim']   = X_embed.shape[1]
    config['desc_dim']    = X_desc.shape[1]
    config['num_classes'] = Y.shape[1]

    logger.info(f"X_embed={X_embed.shape}, X_desc={X_desc.shape}, Y={Y.shape}  "
                f"pos/label: min={Y.sum(0).min():.0f} mean={Y.sum(0).mean():.1f} max={Y.sum(0).max():.0f}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    with open(os.path.join(config['output_dir'], 'label_cols.json'), 'w') as f:
        json.dump(label_cols, f)

    skf = StratifiedKFold(n_splits=config['n_folds'], shuffle=True,
                          random_state=config['random_state'])
    fold_results = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_embed, Y[:, 0])):
        r = train_fold(fold, tr_idx, val_idx, X_embed, X_desc, Y, config, device, logger)
        fold_results.append(r)

    logger.info("\nCROSS-VALIDATION RESULTS")
    for key in ['macro_f1', 'micro_f1', 'macro_auprc', 'micro_auprc']:
        vals = [r[key] for r in fold_results]
        logger.info(f"  {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    pd.DataFrame(fold_results).to_csv(os.path.join(config['output_dir'], 'cv_results.csv'), index=False)
    with open(os.path.join(config['output_base'], 'latest.txt'), 'w') as f:
        f.write(config['output_dir'])
    logger.info(f"Saved to: {config['output_dir']}")


if __name__ == '__main__':
    main()
