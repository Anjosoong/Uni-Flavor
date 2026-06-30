# External Benchmark

SMILES-based baselines for **odor** (138 labels) and **taste** (6 labels) multilabel prediction.  
This folder is part of the Uni-Flavor project (not a standalone repo). Main models (DLMOF-Net / DLMTF-Net) live under `Odor/` and `Taste/`.

Both baselines follow the **same protocol as DLMOF-Net / DLMTF-Net**:

- 5-fold stratified CV on the training set (`random_state=42`)
- Per-label dynamic thresholds on each validation fold (grid 0.05–0.95, step 0.02)
- Test evaluation: average fold probabilities + averaged thresholds

| Model | Input | Dependency |
|-------|-------|------------|
| **MPNN** | SMILES | ChemProp **2.2.3** (pip) |
| **ChemBERTa** | SMILES | `DeepChem/ChemBERTa-77M-MTR` (Hugging Face) |

## Prerequisites

1. Python environment with PyTorch (see repo `environment.yml`).
2. Task data under `../Odor` or `../Taste` (paths in `configs/*.json`).
3. One-time dependency setup:

```bash
cd benchmark
pip install -r requirements-benchmark.txt   # optional: install all Python deps
python setup_deps.py                        # ChemProp 2.2.3 + ChemBERTa weights
python setup_deps.py --check                # verify installation only
```

`setup_deps.py` will:

- `pip install chemprop==2.2.3`
- Download ChemBERTa MTR weights to `chemberta/pretrained/DeepChem-ChemBERTa-77M-MTR/`

### ChemBERTa pretrained weights

Weights are **not** included in git (see `chemberta/pretrained/.gitkeep` — folder is created locally after setup).

- Default model: `DeepChem/ChemBERTa-77M-MTR`
- Local path: `chemberta/pretrained/DeepChem-ChemBERTa-77M-MTR/`
- Loading: offline via `common/hf_load.py` during training
- If only `pytorch_model.bin` is present, `setup_deps.py` converts it to `model.safetensors` automatically

```bash
cd benchmark
python setup_deps.py              # ChemProp + ChemBERTa
python setup_deps.py --chemberta-only
```

## Quick start

```powershell
cd benchmark

# Setup (first time)
python setup_deps.py

# Run all baselines
.\run_benchmarks.ps1 -Task odor

# Single model
python mpnn/train_mpnn.py --config configs/odor.json
python chemberta/train_chemberta.py --config configs/taste.json

# Aggregate + optional main-model CSV
python aggregate_results.py --task odor `
  --extra-csv "DLMOF-Net=../Odor/model/odor_train_xxx/test_results/overall_results.csv"
```

Replace `odor` with `taste` for taste benchmarks.

## Directory layout

```
benchmark/
├── README.md
├── setup_deps.py              # install ChemProp + download ChemBERTa
├── requirements-benchmark.txt
├── run_benchmarks.ps1
├── aggregate_results.py
├── configs/
├── common/
├── mpnn/
└── chemberta/
    └── pretrained/            # gitignored weights; .gitkeep tracks empty dir
```

## Config keys

| Key | Description |
|-----|-------------|
| `base_dir` | Path to `Odor/` or `Taste/` data root |
| `labels_train_path` / `labels_test_path` | SMILES + multilabel CSV |
| `mpnn_*` | ChemProp MPNN hyperparameters |
| `chemberta_*` | ChemBERTa fine-tuning hyperparameters |

Training outputs are written to `output/{task}/` (gitignored).
