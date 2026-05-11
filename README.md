# Uni-Flavor

Flavor (odor and taste) molecule prediction, virtual screening, and reverse design pipeline based on [Uni-Mol2](https://github.com/deepmodeling/Uni-Mol).

![Model architecture](picture.tif)

> Dual-branch fusion architecture shared by DLMOF-Net (138-class odor) and DLMTF-Net (6-class taste).

## What is included

- LoRA fine-tuning of Uni-Mol2 on flavor data.
- Dual-branch multi-label classifiers: 138 odor labels (DLMOF-Net) and 6 taste labels (DLMTF-Net).
- High-throughput screening on the [COCONUT](https://coconut.naturalproducts.net/) natural-product database, with cleaning and toxicity / ADMET filters.
- Principal Odor Map (OM) and Taste Map (TM) visualization.
- Scaffold-based reverse search and functional-group analysis.
- SMILES → MOL2 conversion for AutoDock Vina / GOLD / Glide.

## Modules

- **`finetune/`** — domain-adapt Uni-Mol2 to flavor data via LoRA: build LMDB datasets, run parameter-efficient fine-tuning, then merge the LoRA weights back into the base checkpoint.
- **`Odor/`** — train and evaluate **DLMOF-Net** for 138-class odor multi-label prediction. Covers Uni-Mol2 / Mordred feature extraction, 5-fold CV training, held-out testing, and SHAP-style interpretability + treemap visualization of odor tags.
- **`Taste/`** — train and evaluate **DLMTF-Net** for 6-class taste multi-label prediction (sweet, bitter, sour, salty, umami, secondary taste). Same feature pipeline and training recipe as `Odor/`.
- **`molecule_HTS/`** — large-scale virtual screening of the COCONUT natural-product database (~700k molecules). Includes 7-stage molecule cleaning, structural-alert / ADMET toxicity screening, batch feature extraction, batch prediction, and Principal Odor Map (`OM/`) / Taste Map (`TM/`) visualization.
- **`molecule_design/`** — reverse molecule design from a property profile: scaffold mining and reverse search (`scaffold_screening/`), plus functional-group statistics (`function_groups/`).
- **`Ligand_preparation/`** — SMILES → MOL2 conversion with MMFF94 minimization and Gasteiger charges, ready for AutoDock Vina / GOLD / Glide docking.

## Repository layout

```
Uni-Flavor/
├── finetune/                   # LoRA fine-tuning of Uni-Mol2
│   ├── unimol_lmdb_preparation.py     # build LMDB from SMILES + labels
│   ├── unimol2_lora_finetune_runner.py# LoRA training entry point
│   ├── lora_modules.py                # LoRA layer definitions
│   └── merge.py                       # merge LoRA weights into base model
│
├── Odor/                       # 138-class odor multi-label prediction
│   ├── unimol_feature_extractor_odor.py  # 768-d Uni-Mol2 embeddings
│   ├── mordred_descriptors.py            # 300-d Mordred descriptors
│   ├── train_odor_model.py               # DLMOF-Net training (5-fold CV)
│   ├── test_odor_model.py                # held-out evaluation
│   ├── odor_interpretability.py          # SHAP-style label attribution
│   └── odor_tag_treemap.py               # tag-frequency treemap plots
│
├── Taste/                      # 6-class taste multi-label prediction
│   ├── unimol_feature_extractor.py
│   ├── mordred_descriptors.py
│   ├── train_taste_model.py              # DLMTF-Net training (5-fold CV)
│   ├── test_taste_model.py
│   └── taste_interpretability.py
│
├── molecule_HTS/               # High-throughput virtual screening
│   ├── filter_coconut.py                 # 7-stage cleaning of COCONUT (~700k mols)
│   ├── toxicity_screener.py              # Ames / carcinogenicity / PAINS / Brenk / ADMET
│   ├── unimol_feature_extractor_hts.py
│   ├── OM/                               # Odor screening + Principal Odor Map
│   │   ├── hts_train_odor.py
│   │   ├── hts_predict_odor.py
│   │   ├── align_features.py             # align HTS features to training space
│   │   └── draw_om_fusion.py             # OM visualization (TriMap/UMAP/t-SNE/PCA + NMF-RGB)
│   └── TM/                               # Taste screening + Taste Map
│       ├── hts_train_taste.py
│       ├── hts_predict_taste.py
│       ├── align_features.py
│       └── draw_tm_fusion.py
│
├── molecule_design/            # Reverse molecule design
│   ├── scaffold_screening/               # scaffold mining + reverse search
│   └── function_groups/                  # functional group analysis
│
└── Ligand_preparation/         # Docking-ready ligand generation
    └── smiles_to_mol2_docking.py
```

## Model architecture

DLMOF-Net (odor) and DLMTF-Net (taste) share the same dual-branch fusion design. The UniMol2 embedding branch (768-d → 512 → H) and Mordred descriptor branch (300-d → 256 → H) are fused via concatenation with their element-wise product (3H → 2H → H). A label-aware query head computes multi-label logits as a bilinear product with learned label embeddings.

Training uses Asymmetric Loss with per-label inverse-frequency weighting, EMA, cosine LR with warmup, and 5-fold stratified CV.

## Installation

Clone Uni-Core and Uni-Mol alongside this repository, then recreate the conda environment:

```bash
git clone https://github.com/dptech-corp/Uni-Core.git
git clone https://github.com/deepmodeling/Uni-Mol.git
git clone <this-repo>.git
conda env create -f environment.yml
conda activate um
```

Key dependencies (as exported in `environment.yml`):

```
python            3.11.5
torch             2.11.0
rdkit             2025.9.6
mordredcommunity  2.0.7
numpy             2.2.6
scikit-learn      1.8.0
lmdb              2.1.1
unimol-tools      0.1.5
tqdm              4.67.3
```

## Typical workflow

`finetune/` → `Odor/` or `Taste/` → `molecule_HTS/{OM,TM}/` → (optional) `molecule_design/` and `Ligand_preparation/`.

Each script is configured at the top of the file (input / output paths, model size, hyper-parameters). Run the relevant script in its own directory; outputs are written to sibling folders such as `model/`, `hts_results/`, `om_results/`.

## Data availability

The high-throughput screening datasets generated and analyzed in this study are publicly available on Zenodo at <https://doi.org/10.5281/zenodo.20071913>. The deposited files include:

- `mordred_output.tar.gz` — molecular descriptor outputs
- `taste_hts_feature.tar.gz` — taste-related HTS features
- `odor_hts_feature.tar.gz` — odor-related HTS features
- `odor_hts_results.tar.gz` — odor HTS prediction results
- `taste_hts_results.tar.gz` — taste HTS prediction results
- `checkpoint_odor_finetune.pt` — fine-tuned Uni-Mol2 checkpoint for odor prediction (place at `Odor/checkpoint/`)
- `checkpoint_taste_finetune.pt` — fine-tuned Uni-Mol2 checkpoint for taste prediction (place at `Taste/checkpoint/`)

## License

MIT. See `LICENSE`.


