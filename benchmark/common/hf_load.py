"""Download and load Hugging Face ChemBERTa weights from a local directory."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch

benchmark_root = Path(__file__).resolve().parent.parent

REQUIRED_MODEL_FILES = ('config.json',)
WEIGHT_FILES = ('model.safetensors', 'pytorch_model.bin')
TOKENIZER_FILES = ('tokenizer.json', 'vocab.json')

SETUP_HINT = 'Run: python setup_deps.py --chemberta-only'


def resolve_model_dir(model_dir: str | None, default: str = 'chemberta/pretrained/DeepChem-ChemBERTa-77M-MTR') -> str:
    """Resolve pretrained directory relative to benchmark root or cwd."""
    raw = model_dir or default
    path = Path(raw)
    if path.is_absolute():
        return str(path.resolve())
    for candidate in (Path.cwd() / path, benchmark_root / path):
        if candidate.parent.exists() or candidate.exists():
            return str(candidate.resolve())
    return str((benchmark_root / path).resolve())


def _has_file(directory: str, names: tuple[str, ...]) -> bool:
    return any(os.path.isfile(os.path.join(directory, name)) for name in names)


def is_model_ready(local_dir: str) -> bool:
    if not os.path.isdir(local_dir):
        return False
    if not all(os.path.isfile(os.path.join(local_dir, name)) for name in REQUIRED_MODEL_FILES):
        return False
    return _has_file(local_dir, WEIGHT_FILES)


def is_tokenizer_ready(local_dir: str) -> bool:
    if not os.path.isdir(local_dir):
        return False
    return _has_file(local_dir, TOKENIZER_FILES)


def _convert_bin_to_safetensors(local_dir: str) -> bool:
    """Convert pytorch_model.bin to model.safetensors for transformers >= 4.5 / torch 2.5."""
    dst = os.path.join(local_dir, 'model.safetensors')
    if os.path.isfile(dst):
        return True

    src = os.path.join(local_dir, 'pytorch_model.bin')
    if not os.path.isfile(src):
        return False

    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError('Install safetensors: pip install safetensors') from exc

    state_dict = torch.load(src, map_location='cpu', weights_only=True)
    # ChemBERTa tied embeddings: clone tensors to satisfy safetensors writer.
    state_dict = {key: tensor.clone() for key, tensor in state_dict.items()}
    save_file(state_dict, dst)
    print(f'Converted pytorch_model.bin -> model.safetensors in {local_dir}')
    return True


def _ensure_safetensors_in_dir(model_name: str, local_dir: str, *, allow_download: bool = False) -> None:
    dst = os.path.join(local_dir, 'model.safetensors')
    if os.path.isfile(dst):
        return

    if _convert_bin_to_safetensors(local_dir):
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return

    for local_only in (True, False):
        if local_only and not allow_download:
            continue
        try:
            src = hf_hub_download(
                model_name, 'model.safetensors', local_files_only=local_only,
            )
        except OSError:
            continue
        if os.path.isfile(src) and os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
            print(f'Added model.safetensors in {local_dir}')
            return


def _copy_from_hf_cache(model_name: str, local_dir: str) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False

    os.makedirs(local_dir, exist_ok=True)
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_files_only=True,
        )
    except OSError:
        return False

    _ensure_safetensors_in_dir(model_name, local_dir)
    return is_model_ready(local_dir) and is_tokenizer_ready(local_dir)


def ensure_pretrained(model_name: str, local_dir: str, *, allow_download: bool = True) -> str:
    """Ensure ChemBERTa weights exist under ``local_dir``; download if allowed."""
    local_dir = os.path.abspath(local_dir)
    if is_model_ready(local_dir) and is_tokenizer_ready(local_dir):
        _ensure_safetensors_in_dir(model_name, local_dir, allow_download=allow_download)
        print(f'Using local pretrained model: {local_dir}')
        return local_dir

    if _copy_from_hf_cache(model_name, local_dir):
        print(f'Copied pretrained model from HF cache: {local_dir}')
        return local_dir

    if not allow_download:
        raise RuntimeError(
            f'Pretrained model not found in {local_dir}. {SETUP_HINT}'
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError('Install huggingface_hub: pip install huggingface_hub') from exc

    os.makedirs(local_dir, exist_ok=True)
    print(f'Downloading {model_name} -> {local_dir}')
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
    )
    _ensure_safetensors_in_dir(model_name, local_dir, allow_download=True)
    print(f'Download complete: {local_dir}')
    return local_dir


def _load_from_local(cls, local_dir: str, **kwargs):
    if not os.path.isfile(os.path.join(local_dir, 'model.safetensors')):
        _convert_bin_to_safetensors(local_dir)

    safetensors_path = os.path.join(local_dir, 'model.safetensors')
    if not os.path.isfile(safetensors_path):
        raise RuntimeError(f'No model.safetensors in {local_dir}. {SETUP_HINT}')

    load_kwargs = {
        **kwargs,
        'local_files_only': True,
        'use_safetensors': True,
    }
    obj = cls.from_pretrained(local_dir, **load_kwargs)
    print(f'Loaded from {local_dir}')
    return obj


def load_tokenizer(tokenizer_cls, model_source: str, **kwargs):
    return _load_from_local(tokenizer_cls, model_source, **kwargs)


def load_encoder(model_cls, model_source: str, **kwargs):
    return _load_from_local(model_cls, model_source, **kwargs)
