"""Pinned third-party dependency versions and availability checks."""

from __future__ import annotations

import importlib.metadata
import sys

# Pinned versions used in this benchmark (keep in sync with requirements-benchmark.txt).
CHEMPROP_VERSION = '2.2.3'
CHEMBERTA_MODEL_NAME = 'DeepChem/ChemBERTa-77M-MTR'
CHEMBERTA_MODEL_DIR = 'chemberta/pretrained/DeepChem-ChemBERTa-77M-MTR'


def _import_or_none(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


def chemprop_version() -> str | None:
    """Return installed ChemProp version, or None if not installed."""
    try:
        return importlib.metadata.version('chemprop')
    except importlib.metadata.PackageNotFoundError:
        return None


def check_chemprop(required_version: str = CHEMPROP_VERSION) -> None:
    """Raise RuntimeError when ChemProp is missing or the version does not match."""
    version = chemprop_version()
    if version is None:
        raise RuntimeError(
            f'ChemProp {required_version} is not installed. '
            f'Run: python setup_deps.py'
        )
    if version != required_version:
        raise RuntimeError(
            f'ChemProp {required_version} is required, but {version} is installed. '
            f'Run: python setup_deps.py --force-chemprop'
        )


def check_chemprop_import() -> None:
    """Verify ChemProp Python API is importable after pip install."""
    check_chemprop()
    if _import_or_none('chemprop') is None:
        raise RuntimeError(
            'ChemProp is listed in pip but cannot be imported. '
            'Re-run: python setup_deps.py --force-chemprop'
        )


def pip_install_chemprop(version: str = CHEMPROP_VERSION, *, force: bool = False) -> None:
    """Install pinned ChemProp via pip."""
    import subprocess

    current = chemprop_version()
    if current == version and not force:
        print(f'ChemProp {version} already installed.')
        return

    print(f'Installing chemprop=={version} ...')
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', f'chemprop=={version}'],
    )
    check_chemprop_import()
    print(f'ChemProp {version} ready.')
