# Uni-Flavor

Uni-Flavor is a research codebase for molecular odor and taste prediction. It
combines Uni-Mol2 molecular representations with Mordred descriptors in two
multilabel fusion models:

- **DLMOF-Net** predicts 138 odor labels.
- **DLMTF-Net** predicts 6 taste labels.

The repository includes data-processing utilities, Uni-Mol2 LoRA fine-tuning,
repeated cross-validation, test-set uncertainty estimation, ablation studies,
external baselines, a SIDER cross-task experiment, virtual screening, and
structure-oriented downstream analyses.

![Uni-Flavor model architecture](picture.jpg)

## Contents

```text
Uni-Flavor/
|-- finetune/                         # Uni-Mol2 LMDB preparation and LoRA fine-tuning
|-- Odor/                             # DLMOF-Net: 138-label odor prediction
|   |-- Data/                         # raw and processed odor data
|   |-- mordred/                      # descriptor processing and feature selection
|   |-- odor_ablation/                # odor ablation experiments and summaries
|   |-- train_odor_model.py
|   `-- test_odor_model.py
|-- Taste/                            # DLMTF-Net: 6-label taste prediction
|   |-- Data/                         # raw and processed taste data
|   |-- mordred/                      # descriptor processing and feature selection
|   |-- taste_ablation/               # taste ablation experiments and summaries
|   |-- train_taste_model.py
|   `-- test_taste_model.py
|-- benchmark/                        # ChemProp MPNN and ChemBERTa baselines
|-- SIDER_Cross_Task_Applicability/   # SIDER cross-task experiment
|-- molecule_HTS/                     # screening and odor/taste map generation
|-- molecule_design/                  # scaffold and functional-group analyses
|-- Ligand_preparation/               # SMILES-to-MOL2 conversion for docking
|-- environment.yml
`-- LICENSE
```

See [`benchmark/README.md`](benchmark/README.md) for the benchmark-specific
setup and commands.

## Installation

The provided Conda environment uses Python 3.11. A CUDA-capable GPU is strongly
recommended for Uni-Mol2 feature extraction and model training.

```bash
git clone https://github.com/Anjosoong/Uni-Flavor.git
cd Uni-Flavor
conda env create -f environment.yml
```

`environment.yml` records the Windows environment used for this repository and
includes Windows-specific runtime packages. On Linux or macOS, create a Python
3.11 environment and install the corresponding packages from the file without
the Windows-only entries. The recorded environment includes PyTorch, RDKit,
`mordredcommunity`, NumPy, scikit-learn, and `unimol-tools`. The external
baselines require the additional packages in
`benchmark/requirements-benchmark.txt`.

Uni-Mol2 LoRA fine-tuning also uses
[`Uni-Core`](https://github.com/dptech-corp/Uni-Core) and
[`Uni-Mol`](https://github.com/deepmodeling/Uni-Mol). Clone both projects next
to this repository, or pass their locations explicitly to the fine-tuning
runner:

```text
parent-directory/
|-- Uni-Flavor/
|-- Uni-Core/
`-- Uni-Mol/
```

## Data and feature contract

Odor and taste training expect the following task-local files:

```text
<TASK>/Data/processed_data/train_data.csv
<TASK>/Data/processed_data/test_data.csv
<TASK>/unimol_feature/<variant>/train_output/train_data_molecular_features.npy
<TASK>/unimol_feature/<variant>/test_output/test_data_molecular_features.npy
<TASK>/mordred/<task>_descriptors/train_processed.csv
<TASK>/mordred/<task>_descriptors/test_processed.csv
```

`<TASK>` is `Odor` or `Taste`. The label CSV files must contain a canonical
`SMILES` column and binary `TARGET_*` columns. The label rows, Uni-Mol2 feature
rows, and Mordred descriptor rows must have identical order and length. The
training and evaluation scripts validate row counts and check SMILES alignment
when the relevant columns are available.

Dataset splitting utilities are provided for both tasks:

```bash
python Odor/Data/raw_data/split_dataset_odor.py path/to/odor_data_all.csv
python Taste/Data/raw_data/split_dataset_taste.py path/to/taste_data_all.csv
```

Both utilities use seed `42`. Keep the resulting held-out test set fixed when
comparing models or ablations.

## Feature preparation

Run task commands from the corresponding `Odor/` or `Taste/` directory so that
the relative paths in model configurations resolve consistently.

### Fine-tuned Uni-Mol2 features

The feature extractors accept explicit input, output, and checkpoint paths. For
example:

```bash
cd Odor
python unimol_feature_extractor_odor.py --input Data/processed_data/train_data.csv --output unimol_feature/finetune_feature/train_output --checkpoint checkpoint/checkpoint_odor_finetune.pt --use-gpu
python unimol_feature_extractor_odor.py --input Data/processed_data/test_data.csv --output unimol_feature/finetune_feature/test_output --checkpoint checkpoint/checkpoint_odor_finetune.pt --use-gpu
```

For taste, run the analogous commands with
`Taste/unimol_feature_extractor.py` and the taste checkpoint. Use the same row
source as the label table and inspect the generated `*_success_indices.npy` and
`*_filtered_data.csv` files before training.

### Mordred descriptors

Descriptor generation and feature-selection scripts are located in each task
directory:

```bash
cd Odor
python mordred_descriptors.py
python mordred/feature_selection.py
```

Use the corresponding scripts under `Taste/` for taste. The descriptor scripts
currently define their input and output paths in a configuration block near the
top of each file; set those paths before execution.

## Model training

The released main-model configurations are:

- `Odor/model/UM-FT+Mordred_w_Aug/config.json`
- `Taste/model/UM-FT+Mordred_w_Aug/config.json`

From the relevant task directory, run:

```bash
cd Odor
python train_odor_model.py --config model/UM-FT+Mordred_w_Aug/config.json

cd ../Taste
python train_taste_model.py --config model/UM-FT+Mordred_w_Aug/config.json
```

Each main configuration uses 5 multilabel-stratified folds and 5 independent
training seeds (`42` through `46`), producing 25 fold models. For each seed, the
pipeline saves out-of-fold probabilities and derives per-label decision
thresholds only from training-set out-of-fold predictions. The test labels are
not used to select thresholds.

Training creates a timestamped directory under the configured `output_base`.
The `output_dir` value stored in an existing result configuration is replaced
when a new run starts.

## Evaluation

Evaluate a completed training directory from the matching task directory:

```bash
cd Odor
python test_odor_model.py --model_dir model/<run_directory> --unimol_test unimol_feature/finetune_feature/test_output/test_data_molecular_features.npy --mordred_test mordred/odor_descriptors/test_processed.csv --test_labels Data/processed_data/test_data.csv --n_bootstrap 1000 --bootstrap_seed 42

cd ../Taste
python test_taste_model.py --model_dir model/<run_directory> --unimol_test unimol_feature/finetune_feature/test_output/test_data_molecular_features.npy --mordred_test mordred/taste_descriptors/test_processed.csv --test_labels Data/processed_data/test_data.csv --n_bootstrap 1000 --bootstrap_seed 42
```

The five fold models form one ensemble per training seed. Evaluation reports
the mean, standard deviation, and Student t confidence interval across the five
seed-level ensembles.

By default, the evaluation scripts also perform 1,000 molecule-level bootstrap
resamples of the fixed test set. Models, probabilities, and out-of-fold-derived
thresholds remain fixed during this analysis. Set `--n_bootstrap 0` to disable
bootstrap estimation. Use `--baseline_dir <evaluation_results>` to compute
paired seed-level significance tests against another compatible evaluation
directory.

## Random-initialization control

The random-initialization extractors construct Uni-Mol2 without loading a
pretrained checkpoint. Use the same encoder seed for train and test extraction:

```bash
cd Odor
python unimol_feature_extractor_random_init.py --input Data/processed_data/train_data.csv --output unimol_feature/random_init/train_output --seed 42 --use-gpu
python unimol_feature_extractor_random_init.py --input Data/processed_data/test_data.csv --output unimol_feature/random_init/test_output --seed 42 --use-gpu
```

Run the analogous commands under `Taste/` for the taste task. The included
control fixes the encoder initialization at seed `42` and varies the downstream
classifier-training seeds from `42` to `46`. These are separate sources of
randomness and should not be interpreted as five independently initialized
encoders.

## Ablation studies

Odor and taste ablation configurations and result summaries are stored in:

```text
Odor/odor_ablation/ablation_output/
Taste/taste_ablation/ablation_output/
```

The included variants cover fine-tuned Uni-Mol2 features, base Uni-Mol2
features, random-initialized Uni-Mol2 features, Mordred-only input, removal of
Mordred descriptors, and removal of feature augmentation. Paths in the ten
released ablation `config.json` files are relative to their task directory.

Run ablations from `Odor/` or `Taste/` using the task-specific ablation scripts
and a selected configuration. Preserve the same split, folds, repeat seeds,
thresholding procedure, and evaluation data when comparing variants.

## Output layout

A complete repeated training run has the following structure:

```text
<run_directory>/
|-- config.json
|-- label_cols.json
|-- train.log
|-- cv_results.csv
|-- repeated_cv_results.csv
|-- repeat_1_seed_42/
|   |-- fold_1/model.pt
|   |-- ...
|   |-- fold_5/model.pt
|   |-- oof_probabilities.npy
|   `-- repeat_thresholds.npy
|-- ...
`-- repeat_5_seed_46/
```

Evaluation writes to `<run_directory>/evaluation_results/` by default:

- `repeated_test_results.csv`: one metric row per training seed.
- `repeated_test_summary.csv`: seed-level mean, standard deviation, and
  confidence interval.
- `per_label_repeated_results.csv` and `per_label_summary.csv`: per-label
  metrics.
- `test_bootstrap_samples.csv` and `test_bootstrap_summary.csv`: bootstrap
  uncertainty results.
- `repeat_probabilities.npy`, `repeat_predictions.npy`, and
  `repeat_thresholds.npy`: reusable seed-level arrays.
- `paired_significance_tests.csv`: created when `--baseline_dir` is supplied.

Compact summaries are included under `Odor/model/`, `Taste/model/`, and the
task-specific ablation directories. Some summary directories omit fold
checkpoints; checkpoint-based inference requires a complete training run or
separately downloaded checkpoints.

## Uni-Mol2 LoRA fine-tuning

The `finetune/` directory contains:

- `unimol_lmdb_preparation.py` for conformer generation and LMDB creation.
- `unimol2_lora_finetune_runner.py` for parameter-efficient fine-tuning.
- `merge.py` for merging trained LoRA weights into a standard checkpoint.

Set the dataset paths in `unimol_lmdb_preparation.py`, then run the fine-tuning
runner with explicit portable paths. The example below assumes `Uni-Core` and
`Uni-Mol` are siblings of this repository:

```bash
python finetune/unimol2_lora_finetune_runner.py --data_path finetune/lmdb_preparation_odor --weight_path path/to/84M/checkpoint.pt --save_dir finetune/result/lora_finetune_odor --user_dir ../Uni-Mol/unimol2/unimol2 --unicore_dir ../Uni-Core
python finetune/merge.py --lora-checkpoint finetune/result/lora_finetune_odor/checkpoint_best.pt --output Odor/checkpoint/checkpoint_odor_finetune.pt --lora-rank 16 --lora-alpha 16
```

The SIDER task has its own fine-tuning runner and merge utility under
`SIDER_Cross_Task_Applicability/SIDER_finetune/`.

## Additional workflows

- **External baselines:** `benchmark/` implements ChemProp MPNN and ChemBERTa
  baselines for odor and taste.
- **Cross-task applicability:** `SIDER_Cross_Task_Applicability/` contains the
  SIDER preprocessing, feature extraction, training, and evaluation workflow.
- **High-throughput screening:** `molecule_HTS/` contains COCONUT filtering,
  descriptor and representation generation, toxicity/ADMET filtering, batch
  inference, and odor/taste map generation.
- **Molecular analysis:** `molecule_design/` contains scaffold and functional-
  group analyses.
- **Docking preparation:** `Ligand_preparation/` converts SMILES to MOL2 files.

These directories contain research workflows rather than a single unified CLI.
Inspect the configuration block or `--help` output of the relevant entry point
before starting a new dataset or task.

## Data and checkpoints

Large screening artifacts and fine-tuned checkpoints are available from
[Zenodo](https://doi.org/10.5281/zenodo.20071913). The deposit includes:

- `mordred_output.tar.gz`
- `taste_hts_feature.tar.gz`
- `odor_hts_feature.tar.gz`
- `odor_hts_results.tar.gz`
- `taste_hts_results.tar.gz`
- `checkpoint_odor_finetune.pt`
- `checkpoint_taste_finetune.pt`

Place downloaded checkpoints at the locations supplied to the feature
extractors. Raw datasets and pretrained models may be subject to licenses or
terms from their original providers. The repository's MIT License applies to
the source code and does not automatically relicense third-party data, model
weights, or software.

## Reproducibility

- Main and ablation comparisons use the same fixed train/test split.
- Complete configurations use 5 repeats by 5 folds.
- Repeat seeds are `random_state + i` for `i = 0, ..., n_repeats - 1`, producing
  seeds `42` through `46` in the released configurations.
- Per-label thresholds are selected independently from each repeat's complete
  training-set out-of-fold predictions.
- Test bootstrap resampling changes only molecule multiplicities; it does not
  retrain models or retune thresholds.
- Deterministic PyTorch operations are requested where available, but exact
  bitwise results can still depend on hardware, CUDA, drivers, and library
  versions.
- Run task-level commands from `Odor/` or `Taste/` unless a command explicitly
  states otherwise.

## Contributing

Bug reports and focused pull requests are welcome. Include the affected task,
configuration, random seeds, environment details, and a minimal reproduction.
Do not commit credentials, private data, downloaded model weights, or large
generated artifacts.

## License and intended use

The source code is released under the [MIT License](LICENSE).

This software is intended for research use. Molecular predictions, toxicity
screens, and docking preparations are computational estimates and must not be
treated as clinical, safety, or regulatory conclusions without independent
experimental validation.
