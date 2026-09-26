"""Train controlled ablations of the odor prediction model.

This task-specific wrapper supplies odor paths and model defaults to the shared
implementation in ``ablation_common.py``. The available modes are
``unimol_only``, ``mordred_only``, and ``no_augmentation``; all other training
and repeated-cross-validation behavior is inherited from the main odor model.
"""

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
REPO_ROOT = TASK_DIR.parent
for path in (TASK_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_odor_model as main_train  # noqa: E402
from ablation_common import (  # noqa: E402,F401
    ABLATION_MODES,
    AblationSpec,
    SingleBranchModel,
    SingleFeatureDataset,
    apply_ablation_constraints,
    build_model as _build_model,
    load_repeated_fold_models as _load_repeated_fold_models,
    load_run_config,
    make_default_config,
    train_main,
)


SPEC = AblationSpec(
    task_name='odor',
    task_label='Odor',
    task_dir=TASK_DIR,
    ablation_dir=SCRIPT_DIR,
    unimol_train=(
        'unimol_feature/random_init/train_output/'
        'train_data_molecular_features.npy'),
    unimol_test=(
        'unimol_feature/random_init/test_output/'
        'test_data_molecular_features.npy'),
    mordred_train='mordred/odor_descriptors/train_processed.csv',
    mordred_test='mordred/odor_descriptors/test_processed.csv',
    labels_train='Data/processed_data/train_data.csv',
    labels_test='Data/processed_data/test_data.csv',
)
DEFAULT_CONFIG = make_default_config(SPEC, main_train)


def build_model(config, device):
    """Compatibility wrapper used by existing analysis code."""
    return _build_model(config, main_train, device)


def load_repeated_fold_models(model_dir, device, verbose=True):
    """Compatibility wrapper used by existing analysis code."""
    return _load_repeated_fold_models(
        model_dir, device, main_train, verbose=verbose)


if __name__ == '__main__':
    train_main(SPEC, main_train, DEFAULT_CONFIG)
