"""Asymmetric loss (ASL) for imbalanced multilabel classification."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """ASL with optional per-label reweighting for rare positives."""

    def __init__(
        self,
        gamma_neg: float = 4,
        gamma_pos: float = 0,
        clip: float = 0.05,
        eps: float = 1e-8,
        label_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        if label_weights is not None:
            self.register_buffer('label_weights', label_weights)
        else:
            self.label_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1 - xs_pos
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        loss = targets * torch.log(xs_pos.clamp(min=self.eps)) + \
            (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = xs_pos * targets + xs_neg * (1 - targets)
            loss = loss * torch.pow(
                1 - pt,
                self.gamma_pos * targets + self.gamma_neg * (1 - targets),
            )
        loss = -loss
        if self.label_weights is not None:
            loss = loss * self.label_weights
        return loss.mean()


def build_label_weights(
    y: np.ndarray,
    power: float = 0.5,
    clip: float = 20.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Inverse-frequency style weights per label column."""
    pos = y.sum(axis=0)
    n = len(y)
    weights = np.power((n - pos) / np.maximum(pos, 1.0), power)
    weights = weights / weights.mean()
    weights = np.clip(weights, max=clip)
    tensor = torch.as_tensor(weights, dtype=torch.float32)
    return tensor.to(device) if device is not None else tensor
