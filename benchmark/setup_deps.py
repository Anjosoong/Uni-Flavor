"""
Install benchmark dependencies and download ChemBERTa pretrained weights.

First-time setup (requires network):
    cd benchmark
    python setup_deps.py

Options:
    python setup_deps.py --check          # verify only, no install
    python setup_deps.py --chemberta-only # skip ChemProp pip install
    python setup_deps.py --chemprop-only  # skip ChemBERTa download
    python setup_deps.py --force-chemprop # reinstall ChemProp 2.2.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

benchmark_root = Path(__file__).resolve().parent
sys.path.insert(0, str(benchmark_root))

from common.data import load_task_config, resolve_config_path  # noqa: E402
from common.deps import (  # noqa: E402
    CHEMBERTA_MODEL_DIR,
    CHEMBERTA_MODEL_NAME,
    CHEMPROP_VERSION,
    check_chemprop,
    check_chemprop_import,
    chemprop_version,
    pip_install_chemprop,
)
from common.hf_load import ensure_pretrained, is_model_ready, is_tokenizer_ready, resolve_model_dir  # noqa: E402


def _resolve_chemberta_targets(config_path: str | None) -> tuple[str, str]:
    if config_path:
        config = load_task_config(str(resolve_config_path(config_path)))
        model_name = config.get('chemberta_model_name', CHEMBERTA_MODEL_NAME)
        model_dir = resolve_model_dir(config.get('chemberta_model_dir', CHEMBERTA_MODEL_DIR))
    else:
        model_name = CHEMBERTA_MODEL_NAME
        model_dir = resolve_model_dir(CHEMBERTA_MODEL_DIR)
    return model_name, model_dir


def setup_chemprop(*, force: bool = False) -> None:
    pip_install_chemprop(CHEMPROP_VERSION, force=force)


def setup_chemberta(config_path: str | None = None) -> str:
    model_name, model_dir = _resolve_chemberta_targets(config_path)
    ensure_pretrained(model_name, model_dir, allow_download=True)
    return model_dir


def verify_all(config_path: str | None = None) -> bool:
    ok = True

    try:
        check_chemprop_import()
        print(f'[ok] ChemProp {chemprop_version()}')
    except RuntimeError as exc:
        ok = False
        print(f'[missing] {exc}')

    model_name, model_dir = _resolve_chemberta_targets(config_path)
    if is_model_ready(model_dir) and is_tokenizer_ready(model_dir):
        print(f'[ok] ChemBERTa weights: {model_dir}')
    else:
        ok = False
        print(
            f'[missing] ChemBERTa weights not found at {model_dir}. '
            f'Run: python setup_deps.py --chemberta-only'
        )
        print(f'          Expected Hugging Face model: {model_name}')

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Install ChemProp 2.2.3 and download ChemBERTa MTR weights',
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Optional task config (e.g. configs/odor.json) for ChemBERTa paths',
    )
    parser.add_argument('--check', action='store_true', help='Verify dependencies only')
    parser.add_argument('--chemberta-only', action='store_true', help='Download ChemBERTa only')
    parser.add_argument('--chemprop-only', action='store_true', help='Install ChemProp only')
    parser.add_argument('--force-chemprop', action='store_true', help='Reinstall ChemProp 2.2.3')
    args = parser.parse_args()

    if args.check:
        sys.exit(0 if verify_all(args.config) else 1)

    if args.chemberta_only:
        model_dir = setup_chemberta(args.config)
        print(f'ChemBERTa ready: {model_dir}')
        return

    if args.chemprop_only:
        setup_chemprop(force=args.force_chemprop)
        return

    setup_chemprop(force=args.force_chemprop)
    model_dir = setup_chemberta(args.config)
    print('\nSetup complete.')
    print(f'  ChemProp: {CHEMPROP_VERSION}')
    print(f'  ChemBERTa: {model_dir}')
    if not verify_all(args.config):
        sys.exit(1)


if __name__ == '__main__':
    main()
