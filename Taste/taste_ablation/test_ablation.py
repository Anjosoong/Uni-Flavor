"""Evaluate a trained taste ablation on the independent test partition.

Model loading, repeated-fold ensembling, threshold application, uncertainty
summaries, and optional baseline comparison are implemented in
``ablation_common.py`` and the main taste evaluation module.
"""

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
REPO_ROOT = TASK_DIR.parent
for path in (TASK_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import test_taste_model as main_test  # noqa: E402
import train_taste_model as main_train  # noqa: E402
from ablation_common import test_main  # noqa: E402
from train_ablation import SPEC  # noqa: E402


if __name__ == '__main__':
    test_main(SPEC, main_train, main_test)
