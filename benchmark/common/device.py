"""PyTorch device helpers shared by MPNN and ChemBERTa training scripts."""

from __future__ import annotations

import torch


def resolve_device(name: str = 'auto') -> torch.device:
    """Resolve training device from CLI flag ``auto`` / ``cuda`` / ``cpu``."""
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def dataloader_kwargs(device: torch.device) -> dict:
    """Default DataLoader kwargs; pin memory only when using CUDA."""
    return {
        'num_workers': 0,
        'pin_memory': device.type == 'cuda',
    }
