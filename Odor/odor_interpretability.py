"""
DLMTF-Net Interpretability — Integrated Gradients with UniMol Atom-level Attribution

For a given SMILES:
  1. Extract UniMol cls_repr (768-dim) and atomic_reprs (N_atoms x 768)
  2. Compute Mordred descriptors -> feature selection -> normalization (300-dim)
  3. Run DLMTF-Net ensemble forward to get label predictions
  4. Compute Integrated Gradients on the UniMol branch (x_embed) per label
  5. Map IG attributions to atom contributions via cosine similarity
  6. Visualize: molecule heatmap + atom contribution bars + label probability bars

Usage:
    python odor_interpretability.py --smiles "CCO"
    python odor_interpretability.py --smiles "CCO" --model_dir <path> --target_label TARGET_01
"""

import os
import sys
import json
import pickle
import importlib
import argparse
import warnings
import io
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings('ignore')


# Config class for pickle compatibility
@dataclass
class ImprovedFastConfig:
    """Config class needed for unpickling transformers.pkl"""
    train_descriptor_path: str = ""
    test_descriptor_path: str = ""
    train_label_path: str = ""
    output_dir: str = ""
    missing_threshold: float = 0.3
    variance_threshold: float = 1e-5
    correlation_threshold: float = 0.95
    n_top_features: int = 300
    selection_method: str = "stability_shap"
    n_bootstrap: int = 50
    bootstrap_fraction: float = 0.8
    stability_threshold: float = 0.4
    stage1_max_features: int = 300
    use_threshold_first: bool = True
    l1_C: float = 0.05
    use_true_shap: bool = True
    shap_sample_size: int = 3000
    tree_n_estimators: int = 200
    tree_max_depth: int = 8
    use_aggregated_label: bool = True
    enable_stability_check: bool = True
    stability_check_seeds: List[int] = None
    random_seed: int = 42
    impute_strategy: str = "median"
    smiles_column: str = "SMILES"
    
    def __post_init__(self):
        if self.stability_check_seeds is None:
            self.stability_check_seeds = [42, 43]

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw, rdDepictor
    from rdkit.Chem.Draw import SimilarityMaps
    from matplotlib.colors import LinearSegmentedColormap
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("WARNING: RDKit not available. Visualization disabled.")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from PIL import Image
except ImportError:
    Image = None

from mordred import Calculator, descriptors as mordred_descriptors

from train_odor_model import EnhancedMultiLabelModel, ensemble_probs


# Default paths
CONFIG = {
    'unimol_checkpoint': r'./checkpoint/checkpoint_odor_finetune.pt',
    'model_size':        '84m',
    'transformers_pkl':  r'./mordred/odor_descriptors/transformers.pkl',
    'model_base_dir':    r'./model/odor_train_finetune',
    'output_dir':        r'./interpretability_results',
    'ig_steps':          50,
}

_ODOR_LABELS = [
    'alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal', 'anisic',
    'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy', 'bergamot',
    'berry', 'bitter', 'black currant', 'brandy', 'burnt', 'buttery', 'cabbage',
    'camphoreous', 'caramellic', 'cedar', 'celery', 'chamomile', 'cheesy', 'cherry',
    'chocolate', 'cinnamon', 'citrus', 'clean', 'clove', 'cocoa', 'coconut',
    'coffee', 'cognac', 'cooked', 'cooling', 'cortex', 'coumarinic', 'creamy',
    'cucumber', 'dairy', 'dry', 'earthy', 'ethereal', 'fatty', 'fermented',
    'fishy', 'floral', 'fresh', 'fruit skin', 'fruity', 'garlic', 'gassy',
    'geranium', 'grape', 'grapefruit', 'grassy', 'green', 'hawthorn', 'hay',
    'hazelnut', 'herbal', 'honey', 'hyacinth', 'jasmin', 'juicy', 'ketonic',
    'lactonic', 'lavender', 'leafy', 'leathery', 'lemon', 'lily', 'malty',
    'meaty', 'medicinal', 'melon', 'metallic', 'milky', 'mint', 'muguet',
    'mushroom', 'musk', 'musty', 'natural', 'nutty', 'odorless', 'oily',
    'onion', 'orange', 'orangeflower', 'orris', 'ozone', 'peach', 'pear',
    'phenolic', 'pine', 'pineapple', 'plum', 'popcorn', 'potato', 'powdery',
    'pungent', 'radish', 'raspberry', 'ripe', 'roasted', 'rose', 'rummy',
    'sandalwood', 'savory', 'sharp', 'smoky', 'soapy', 'solvent', 'sour',
    'spicy', 'strawberry', 'sulfurous', 'sweaty', 'sweet', 'tea', 'terpenic',
    'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable', 'vetiver', 'violet',
    'warm', 'waxy', 'weedy', 'winey', 'woody',
]
LABEL_NAMES = {f'TARGET_{i:02d}': name for i, name in enumerate(_ODOR_LABELS, 1)}


# UniMol
def _load_unimol_checkpoint_manually(model, checkpoint_path: str, use_gpu: bool) -> bool:
    """Manually load a fine-tuned checkpoint into a UniMolRepr model.
    UniMolRepr wraps the actual torch.nn.Module in .model; state_dict lives there.
    We use weights_only=False because the checkpoint contains argparse.Namespace.
    """
    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location=torch.device('cuda' if use_gpu else 'cpu'),
            weights_only=False,
        )
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        inner = model.model if hasattr(model, 'model') else model
        current = inner.state_dict()
        filtered = {k: v for k, v in state_dict.items()
                    if k in current and current[k].shape == v.shape}
        inner.load_state_dict(filtered, strict=False)
        print(f"  Custom checkpoint loaded: {len(filtered)}/{len(current)} params matched")
        return True
    except Exception as e:
        print(f"  Manual checkpoint load failed: {e}")
        return False


def init_unimol(checkpoint_path: str, model_size: str = '84m'):
    """Initialize UniMolRepr model, optionally loading a fine-tuned checkpoint."""
    try:
        from unimol_tools import UniMolRepr
    except ImportError:
        raise ImportError("unimol_tools not installed. Run: pip install unimol_tools")

    use_gpu = torch.cuda.is_available()
    model = UniMolRepr(
        data_type='molecule',
        remove_hs=False,
        model_name='unimolv2',
        model_size=model_size,
        use_gpu=use_gpu,
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  Loading custom checkpoint: {checkpoint_path}")
        _load_unimol_checkpoint_manually(model, checkpoint_path, use_gpu)
    else:
        print(f"  Using default UniMol weights (model_size={model_size})")

    return model


def get_unimol_repr(unimol_model, smiles: str):
    """Returns cls_repr (768,), atomic_reprs (N, 768), rdkit mol."""
    reprs = unimol_model.get_repr([smiles], return_atomic_reprs=True)

    if isinstance(reprs, dict):
        cls_repr    = np.array(reprs['cls_repr'][0],    dtype=np.float32)
        atomic_reprs = np.array(reprs['atomic_reprs'][0], dtype=np.float32)
    else:
        cls_repr    = np.array(reprs[0][0], dtype=np.float32)
        atomic_reprs = np.array(reprs[1][0], dtype=np.float32) if len(reprs) > 1 else None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return cls_repr, atomic_reprs, mol


# Mordred
class _CompatUnpickler(pickle.Unpickler):
    """Redirects __main__ class lookups to this module.
    Needed when transformers.pkl was saved while feature_selection.py was
    run as __main__ but is now loaded inside an imported module (Flask app).
    ImprovedFastConfig is defined above in this file, so no external import needed.
    """
    def find_class(self, module, name):
        if module == '__main__':
            current = sys.modules.get(__name__)
            if current is not None and hasattr(current, name):
                return getattr(current, name)
        return super().find_class(module, name)


def load_transformers(pkl_path: str) -> dict:
    """Load Mordred feature transformers pickle, patching sklearn version differences."""
    with open(pkl_path, 'rb') as f:
        transformers = _CompatUnpickler(f).load()

    # Patch sklearn version incompatibility (saved with 1.6.1, running with newer)
    # _fit_dtype was renamed to _fill_dtype in sklearn >=1.7
    imputer = transformers.get('imputer')
    if imputer is not None:
        if hasattr(imputer, '_fit_dtype') and not hasattr(imputer, '_fill_dtype'):
            imputer._fill_dtype = imputer._fit_dtype

    return transformers


def compute_mordred_features(smiles: str, transformers: dict) -> np.ndarray:
    """Compute Mordred descriptors for one SMILES, apply feature selection + normalization."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    calc = Calculator(mordred_descriptors, ignore_3D=True)
    result = calc.pandas([mol], nproc=1)

    result = result.replace([float('inf'), float('-inf')], float('nan'))
    for col in result.columns:
        if result[col].dtype == object:
            result[col] = pd.to_numeric(result[col], errors='coerce')

    selected_features = transformers['selected_features']
    imputer = transformers['imputer']
    scaler  = transformers['scaler']

    missing = [f for f in selected_features if f not in result.columns]
    if missing:
        raise ValueError(f"Missing Mordred features: {missing[:5]}...")

    X = result[selected_features].values.astype(np.float64)
    X = imputer.transform(X)
    X = scaler.transform(X)

    return X[0].astype(np.float32)


# DLMTF-Net
def load_dlmtf_models(model_dir: str, device: torch.device):
    """Load all fold models. Returns (models, avg_thresholds, config, label_cols)."""
    with open(os.path.join(model_dir, 'config.json')) as f:
        config = json.load(f)
    with open(os.path.join(model_dir, 'label_cols.json')) as f:
        label_cols = json.load(f)

    fold_dirs = sorted([d for d in os.listdir(model_dir) if d.startswith('fold_')])
    models = []
    thresholds_list = []

    for fd in fold_dirs:
        ckpt = torch.load(os.path.join(model_dir, fd, 'model.pt'),
                          map_location=device, weights_only=False)
        model = EnhancedMultiLabelModel(
            embed_dim=config['embed_dim'],
            desc_dim=config['desc_dim'],
            hidden_dim=config['hidden_dim'],
            query_dim=config['query_dim'],
            num_classes=config['num_classes'],
            dropout=config['dropout'],
            use_aux_loss=config['use_aux_loss'],
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        models.append(model)
        thresholds_list.append(ckpt['thresholds'])
        print(f"  Loaded {fd}")

    avg_thresholds = np.mean(thresholds_list, axis=0)
    return models, avg_thresholds, config, label_cols


def run_ensemble(models: list, x_embed_np: np.ndarray, x_desc_np: np.ndarray,
                 config: dict, device: torch.device) -> np.ndarray:
    """Ensemble inference. Returns avg_probs (C,)."""
    x_embed = torch.tensor(x_embed_np[np.newaxis], dtype=torch.float32, device=device)
    x_desc  = torch.tensor(x_desc_np[np.newaxis],  dtype=torch.float32, device=device)

    all_probs = []
    with torch.no_grad():
        for model in models:
            out = model(x_embed, x_desc)
            probs = ensemble_probs(out, config['infer_weights'])
            all_probs.append(probs[0])

    return np.mean(all_probs, axis=0)


# Integrated Gradients
def compute_ig_attributions(model: EnhancedMultiLabelModel,
                             x_embed_np: np.ndarray,
                             x_desc_np: np.ndarray,
                             class_idx: int,
                             steps: int,
                             device: torch.device) -> np.ndarray:
    """
    Integrated Gradients on x_embed (UniMol branch) with x_desc fixed.
    Baseline = zero vector. Returns attributions of shape (embed_dim,).
    """
    baseline  = np.zeros_like(x_embed_np, dtype=np.float32)
    x_desc_t  = torch.tensor(x_desc_np[np.newaxis], dtype=torch.float32, device=device)

    grad_sum = np.zeros_like(x_embed_np, dtype=np.float32)

    model.eval()
    for step in range(1, steps + 1):
        alpha  = step / steps
        interp = baseline + (x_embed_np - baseline) * alpha

        interp_t = torch.tensor(interp[np.newaxis], dtype=torch.float32,
                                 device=device, requires_grad=True)

        out  = model(interp_t, x_desc_t)
        prob = torch.sigmoid(out['logits'][0, class_idx])
        prob.backward()

        grad_sum += interp_t.grad.detach().cpu().numpy()[0]

    avg_grad     = grad_sum / steps
    attributions = (x_embed_np - baseline) * avg_grad
    return attributions


def atom_contributions_from_ig(ig_attributions: np.ndarray,
                                atomic_reprs: np.ndarray,
                                n_rdkit_atoms: int) -> np.ndarray:
    """
    Map 768-dim IG attributions to per-atom scores via cosine similarity.
    Returns importances (n_rdkit_atoms,) normalized to [-1, 1].
    """
    n_unimol  = atomic_reprs.shape[0]
    n_atoms   = min(n_rdkit_atoms, n_unimol)
    attr_norm = np.linalg.norm(ig_attributions)

    importances = np.zeros(n_rdkit_atoms, dtype=np.float32)
    if attr_norm == 0:
        return importances

    for i in range(n_atoms):
        atom_norm = np.linalg.norm(atomic_reprs[i])
        if atom_norm > 0:
            importances[i] = np.dot(atomic_reprs[i], ig_attributions) / (atom_norm * attr_norm)

    # Center if all same sign
    if np.all(importances > 0) or np.all(importances < 0):
        importances = importances - np.mean(importances)

    abs_max = np.max(np.abs(importances))
    if abs_max > 0:
        importances = importances / abs_max

    return importances


def compute_ensemble_ig(models: list,
                         x_embed_np: np.ndarray,
                         x_desc_np: np.ndarray,
                         atomic_reprs: np.ndarray,
                         n_rdkit_atoms: int,
                         class_idx: int,
                         steps: int,
                         device: torch.device) -> np.ndarray:
    """Average atom contributions across all fold models."""
    all_importances = []
    for model in models:
        attrs = compute_ig_attributions(model, x_embed_np, x_desc_np, class_idx, steps, device)
        imp   = atom_contributions_from_ig(attrs, atomic_reprs, n_rdkit_atoms)
        all_importances.append(imp)
    return np.mean(all_importances, axis=0)


# Visualization
def visualize_atom_importance(smiles: str,
                               atom_importances: np.ndarray,
                               avg_probs: np.ndarray,
                               label_cols: List[str],
                               target_class_idx: int,
                               save_path: Optional[str] = None):
    """
    Three-panel figure:
      Left  – molecule heatmap (SimilarityMaps)
      Center – atom contribution bar chart
      Right  – label probability bar chart
    Returns PIL Image.
    """
    if not RDKIT_AVAILABLE or Image is None:
        print("Skipping visualization: RDKit or PIL not available.")
        return None

    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.SetPreferCoordGen(True)
    AllChem.Compute2DCoords(mol, clearConfs=True, canonOrient=True)
    n_atoms = mol.GetNumAtoms()

    weights = list(atom_importances.astype(float))
    if np.all(atom_importances >= 0) or np.all(atom_importances <= 0):
        centered = atom_importances - np.mean(atom_importances)
        weights = list(centered.astype(float))

    cmap = LinearSegmentedColormap.from_list(
        'bwr_custom',
        [(0.0, '#008bc8'), (0.40, '#85c5e4'), (0.5, 'white'),
         (0.60, '#e89a93'), (1.0, '#d65c4d')],
        N=256,
    )

    for atom in mol.GetAtoms():
        atom.SetProp('fontBold', '1')
        atom.SetProp('fontFace', 'Calibri')

    drawer = Draw.MolDraw2DCairo(1200, 1200)
    drawer.drawOptions().bondLineWidth = 10.0
    drawer.drawOptions().padding = 0.08
    drawer.drawOptions().addAtomIndices = True
    try:
        drawer.drawOptions().atomLabelFontSize = 40
    except AttributeError:
        pass
    drawer.drawOptions().clearBackground = False
    drawer.drawOptions().backgroundColour = (1, 1, 1, 1)
    try:
        drawer.drawOptions().fixedBondLength = 20
        drawer.drawOptions().multipleBondOffset = 0.15
    except AttributeError:
        pass
    try:
        with plt.rc_context({'figure.facecolor': 'white',
                             'axes.facecolor':   'white',
                             'savefig.facecolor': 'white'}):
            SimilarityMaps.GetSimilarityMapFromWeights(
                mol, weights, draw2d=drawer, colorMap=cmap,
                contourLines=12, alpha=0.82,
                lineWidthMult=3, atomMapScale=1.6,
                bgColor=(1.0, 1.0, 1.0, 1.0),
            )
    except Exception:
        drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    mol_img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert('RGB')

    label_display = LABEL_NAMES.get(label_cols[target_class_idx], label_cols[target_class_idx])

    atom_symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(n_atoms)]
    order       = np.argsort(atom_importances)
    sorted_imp  = atom_importances[order]
    sorted_lbls = [f'{order[i]} ({atom_symbols[order[i]]})' for i in range(len(order))]
    bar_colors  = ['#008bc8' if v < 0 else '#d65c4d' for v in sorted_imp]
    bar_h = max(0.18, min(0.62, 7.0 / max(n_atoms, 1)))

    top5_idx     = np.argsort(avg_probs)[::-1][:5]
    sorted_probs = avg_probs[top5_idx]
    prob_labels  = [LABEL_NAMES.get(label_cols[i], label_cols[i]) for i in top5_idx]
    pred_colors  = ['#d65c4d' if j < 3 else '#008bc8' for j in range(5)]
    n_display    = 5

    with plt.rc_context({'font.family': 'Calibri'}):
        fig = plt.figure(figsize=(22, 12), dpi=200, facecolor='white')
        gs  = fig.add_gridspec(
            2, 2,
            width_ratios=[1.45, 1.0],
            height_ratios=[max(1.5, n_atoms * 0.13), 1.2],
            wspace=0.28, hspace=0.50,
        )
        ax_mol  = fig.add_subplot(gs[:, 0])
        ax_bar  = fig.add_subplot(gs[0, 1])
        ax_pred = fig.add_subplot(gs[1, 1])

        ax_mol.imshow(mol_img, interpolation='lanczos')
        ax_mol.set_title(f'{label_display}  Molecular Structure',
                         fontsize=17, fontweight='bold', pad=14)
        ax_mol.axis('off')

        ax_bar.barh(range(len(sorted_imp)), sorted_imp, color=bar_colors, height=bar_h)
        ax_bar.set_yticks(range(len(sorted_imp)))
        ax_bar.set_yticklabels(sorted_lbls, fontsize=13, fontweight='bold')
        ax_bar.set_xlabel('Atom Contribution', fontsize=14, fontweight='bold', labelpad=8)
        ax_bar.set_title('Atom Contribution Analysis', fontsize=16, fontweight='bold', pad=12)
        ax_bar.tick_params(axis='x', labelsize=12)
        ax_bar.axvline(0, color='#888888', linewidth=0.9, linestyle='--')
        ax_bar.grid(True, axis='x', alpha=0.3, linestyle='--')
        for sp in ax_bar.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.8); sp.set_edgecolor('black')

        ax_pred.barh(range(n_display), sorted_probs, color=pred_colors, height=0.52)
        ax_pred.set_yticks(range(n_display))
        ax_pred.set_yticklabels(prob_labels, fontsize=13, fontweight='bold')
        ax_pred.set_xlabel('Predicted Probability', fontsize=14, fontweight='bold', labelpad=8)
        ax_pred.set_title('Odor Classification (Top 5)', fontsize=16, fontweight='bold', pad=12)
        ax_pred.set_xlim(0, 1.15)
        ax_pred.tick_params(axis='x', labelsize=12)
        ax_pred.grid(True, axis='x', alpha=0.3, linestyle='--')
        for sp in ax_pred.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.8); sp.set_edgecolor('black')
        for i, p in enumerate(sorted_probs):
            ax_pred.text(p + 0.02, i, f'{p:.4f}', va='center', fontsize=12, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"  Saved: {save_path}")

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
        buf.seek(0)
        result_img = Image.open(buf).copy()
        plt.close()

    return result_img


# Main pipeline
def _resolve_model_dir(model_dir: str) -> str:
    """Auto-detect latest run if model_dir has no config.json."""
    if os.path.exists(os.path.join(model_dir, 'config.json')):
        return model_dir
    latest_txt = os.path.join(model_dir, 'latest.txt')
    if os.path.exists(latest_txt):
        with open(latest_txt) as f:
            return f.read().strip()
    subdirs = sorted([d for d in os.listdir(model_dir)
                      if os.path.isdir(os.path.join(model_dir, d))])
    if not subdirs:
        raise FileNotFoundError(f"No model found in {model_dir}")
    return os.path.join(model_dir, subdirs[-1])


def analyze_molecule(smiles: str,
                      model_dir: Optional[str]      = None,
                      transformers_pkl: Optional[str] = None,
                      checkpoint_path: Optional[str]  = None,
                      model_size: str                  = '84m',
                      target_label: Optional[str]      = None,
                      output_dir: Optional[str]        = None,
                      ig_steps: int                    = 50):
    """
    Full interpretability pipeline for one SMILES.
    Returns list of result dicts (one per analyzed label).
    """
    model_dir        = _resolve_model_dir(model_dir or CONFIG['model_base_dir'])
    transformers_pkl = transformers_pkl or CONFIG['transformers_pkl']
    checkpoint_path  = checkpoint_path  or CONFIG['unimol_checkpoint']
    model_size       = model_size       or CONFIG['model_size']
    output_dir       = output_dir       or CONFIG['output_dir']

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_smi  = ''.join(c if c.isalnum() else '_' for c in smiles)[:25].rstrip('_')
    run_dir   = os.path.join(output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model dir: {model_dir}")

    # Step 1: UniMol features
    print("\n[1] Extracting UniMol features...")
    unimol = init_unimol(checkpoint_path, model_size)
    cls_repr, atomic_reprs, mol = get_unimol_repr(unimol, smiles)
    n_rdkit_atoms = mol.GetNumAtoms()
    n_unimol_atoms = atomic_reprs.shape[0]
    print(f"  cls_repr={cls_repr.shape}, atomic_reprs={atomic_reprs.shape}, "
          f"rdkit_atoms={n_rdkit_atoms}")
    if n_unimol_atoms != n_rdkit_atoms:
        print(f"  NOTE: UniMol atom count ({n_unimol_atoms}) != RDKit heavy atoms "
              f"({n_rdkit_atoms}); using min={min(n_unimol_atoms, n_rdkit_atoms)}")

    unimol_csv = os.path.join(run_dir, f"{safe_smi}_unimol_features.csv")
    pd.DataFrame({'dim': range(len(cls_repr)), 'cls_repr': cls_repr.tolist()}).to_csv(
        unimol_csv, index=False)
    print(f"  UniMol features saved: {unimol_csv}")

    # Step 2: Mordred features
    print("\n[2] Computing Mordred features...")
    transformers = load_transformers(transformers_pkl)
    desc_vec = compute_mordred_features(smiles, transformers)
    print(f"  desc_vec={desc_vec.shape}")

    mordred_csv = os.path.join(run_dir, f"{safe_smi}_mordred_features.csv")
    pd.DataFrame({
        'feature': transformers['selected_features'],
        'value':   desc_vec.tolist(),
    }).to_csv(mordred_csv, index=False)
    print(f"  Mordred features saved: {mordred_csv}")

    # Step 3: Load DLMTF-Net
    print("\n[3] Loading DLMTF-Net models...")
    models, avg_thresholds, config, label_cols = load_dlmtf_models(model_dir, device)

    config['embed_dim'] = int(cls_repr.shape[0])
    config['desc_dim']  = int(desc_vec.shape[0])

    # Step 4: Ensemble prediction
    print("\n[4] Running ensemble prediction...")
    avg_probs   = run_ensemble(models, cls_repr, desc_vec, config, device)
    predictions = (avg_probs >= avg_thresholds).astype(int)

    print("  Top-10 label probabilities:")
    for i in np.argsort(avg_probs)[::-1][:10]:
        col  = label_cols[i]
        name = LABEL_NAMES.get(col, col)
        flag = ' [POSITIVE]' if predictions[i] else ''
        print(f"    {name:<20}: {avg_probs[i]:.4f} (thr={avg_thresholds[i]:.3f}){flag}")

    # Step 5: Determine target labels for IG
    if target_label:
        if target_label not in label_cols:
            raise ValueError(f"Label '{target_label}' not found. Available: {label_cols}")
        target_indices = [label_cols.index(target_label)]
    else:
        target_indices = list(np.argsort(avg_probs)[::-1][:3])

    print(f"\n[5] Computing Integrated Gradients for "
          f"{len(target_indices)} label(s), steps={ig_steps}...")

    # Step 6: IG + visualization per label
    atom_symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(n_rdkit_atoms)]

    results = []
    for class_idx in target_indices:
        label_col  = label_cols[class_idx]
        label_name = LABEL_NAMES.get(label_col, label_col)
        print(f"\n  → {label_name} (prob={avg_probs[class_idx]:.4f})")

        atom_imp = compute_ensemble_ig(
            models, cls_repr, desc_vec, atomic_reprs,
            n_rdkit_atoms, class_idx, ig_steps, device,
        )

        top5_pos = np.argsort(atom_imp)[-5:][::-1]
        top5_neg = np.argsort(atom_imp)[:5]
        print("  Top 5 positive contributors:")
        for idx in top5_pos:
            print(f"    atom {idx:3d} ({atom_symbols[idx]}): {atom_imp[idx]:+.4f}")
        print("  Top 5 negative contributors:")
        for idx in top5_neg:
            print(f"    atom {idx:3d} ({atom_symbols[idx]}): {atom_imp[idx]:+.4f}")

        save_path = os.path.join(run_dir, f"{safe_smi}_{label_col}.png")
        visualize_atom_importance(smiles, atom_imp, avg_probs, label_cols, class_idx, save_path)

        results.append({
            'label':           label_col,
            'label_name':      label_name,
            'probability':     float(avg_probs[class_idx]),
            'predicted':       bool(predictions[class_idx]),
            'atom_symbols':    atom_symbols,
            'atom_importance': atom_imp.tolist(),
            'save_path':       save_path,
        })

    # Save atom contribution CSV
    rows = []
    for r in results:
        for i, (sym, imp) in enumerate(zip(r['atom_symbols'], r['atom_importance'])):
            rows.append({
                'SMILES':       smiles,
                'Label':        r['label'],
                'LabelName':    r['label_name'],
                'Probability':  r['probability'],
                'Predicted':    r['predicted'],
                'AtomIdx':      i,
                'AtomSymbol':   sym,
                'Contribution': imp,
            })

    if rows:
        csv_path = os.path.join(run_dir, f"{safe_smi}_atom_contributions.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\nAtom contribution data saved: {csv_path}")

    print(f"\nDone. {len(results)} visualization(s) in: {run_dir}")
    return results


def main():
    """CLI entry point for SMILES-level interpretability analysis."""
    parser = argparse.ArgumentParser(
        description='DLMTF-Net Integrated Gradients Interpretability'
    )
    parser.add_argument('--smiles',          type=str, required=True,
                        help='SMILES string to analyze')
    parser.add_argument('--model_dir',       type=str, default=None,
                        help='Path to model directory (default: latest in model_base_dir)')
    parser.add_argument('--transformers_pkl', type=str, default=None,
                        help='Path to transformers.pkl for Mordred normalization')
    parser.add_argument('--checkpoint',      type=str, default=None,
                        help='UniMol checkpoint path')
    parser.add_argument('--model_size',      type=str, default='84m',
                        choices=['84m', '164m', '310m', '570m', '1.1B'])
    parser.add_argument('--target_label',    type=str, default=None,
                        help='Specific label for IG (e.g. TARGET_01); '
                             'default: all predicted-positive labels')
    parser.add_argument('--output_dir',      type=str, default=None)
    parser.add_argument('--steps',           type=int, default=50,
                        help='Number of IG integration steps (default: 50)')
    args = parser.parse_args()

    analyze_molecule(
        smiles           = args.smiles,
        model_dir        = args.model_dir,
        transformers_pkl = args.transformers_pkl,
        checkpoint_path  = args.checkpoint,
        model_size       = args.model_size,
        target_label     = args.target_label,
        output_dir       = args.output_dir,
        ig_steps         = args.steps,
    )


if __name__ == '__main__':
    main()
