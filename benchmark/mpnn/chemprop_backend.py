"""ChemProp MPNN backend: SMILES -> multilabel probabilities (one shared graph network)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.deps import check_chemprop_import

check_chemprop_import()

from chemprop import data, models, nn  # noqa: E402


def build_datapoints(smiles: List[str], y: np.ndarray) -> Tuple[list, List[int]]:
    """Convert SMILES strings to ChemProp datapoints; skip invalid structures."""
    datapoints = []
    valid_idx = []
    for i, smi in enumerate(smiles):
        if Chem.MolFromSmiles(smi) is None:
            continue
        datapoints.append(data.MoleculeDatapoint.from_smi(smi, y[i]))
        valid_idx.append(i)
    return datapoints, valid_idx


def build_dataloader(
    smiles: List[str],
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False,
) -> Tuple[DataLoader, List[int]]:
    """Build ChemProp dataloader; ``drop_last=False`` avoids batch-size-1 edge cases."""
    datapoints, valid_idx = build_datapoints(smiles, y)
    dataset = data.MoleculeDataset(datapoints)
    loader = data.build_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=drop_last,
    )
    return loader, valid_idx


def build_model(num_classes: int, config: dict) -> models.MPNN:
    """Bond message passing + C-task binary classification head (multilabel)."""
    hidden_dim = config['mpnn_hidden_dim']
    depth = config['mpnn_num_layers']
    lr = config['mpnn_lr']

    message_passing = nn.BondMessagePassing(d_h=hidden_dim, depth=depth)
    agg = nn.MeanAggregation()
    ffn = nn.BinaryClassificationFFN(
        n_tasks=num_classes,
        input_dim=message_passing.output_dim,
        hidden_dim=hidden_dim,
    )
    return models.MPNN(
        message_passing,
        agg,
        ffn,
        init_lr=lr,
        max_lr=lr,
        final_lr=lr,
    )


def _batch_to_device(batch, device):
    """Move ChemProp batch tensors to device (BatchMolGraph.to is in-place)."""
    bmg, v_d, x_d, targets, weights, lt_mask, gt_mask = batch
    bmg.to(device)
    if v_d is not None:
        v_d = v_d.to(device)
    if x_d is not None:
        x_d = x_d.to(device)
    targets = targets.to(device)
    weights = weights.to(device)
    if lt_mask is not None:
        lt_mask = lt_mask.to(device)
    if gt_mask is not None:
        gt_mask = gt_mask.to(device)
    return bmg, v_d, x_d, targets, weights, lt_mask, gt_mask


def train_step(model: models.MPNN, batch, device) -> torch.Tensor:
    """Single optimization step; returns scalar BCE loss."""
    bmg, v_d, x_d, targets, weights, lt_mask, gt_mask = _batch_to_device(batch, device)
    mask = targets.isfinite()
    targets = targets.nan_to_num(nan=0.0)
    z = model.fingerprint(bmg, v_d, x_d)
    preds = model.predictor.train_step(z)
    return model.criterion(preds, targets, mask, weights, lt_mask, gt_mask)


@torch.no_grad()
def predict_probs(model: models.MPNN, loader: DataLoader, device, desc: str = 'Predict') -> np.ndarray:
    """Return sigmoid probabilities with shape (N, num_labels)."""
    model.eval()
    probs = []
    for batch in tqdm(loader, desc=desc, leave=False):
        bmg, v_d, x_d, *_ = _batch_to_device(batch, device)
        preds = model(bmg, v_d, x_d)
        probs.append(preds.cpu().numpy())
    return np.vstack(probs)


def save_model(model: models.MPNN, path: str) -> None:
    torch.save(
        {
            'hyper_parameters': dict(model.hparams),
            'state_dict': model.state_dict(),
        },
        path,
    )


def load_model(path: str, device: torch.device) -> models.MPNN:
    model = models.MPNN.load_from_file(path, map_location=device)
    model.to(device)
    return model
