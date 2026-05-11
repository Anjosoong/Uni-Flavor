#!/usr/bin/env python3
"""
TM (Principal Taste Map) — Fusion Feature Edition

Uses the model's fusion representation h = fusion(cat(h_e, h_d, h_e*h_d))
for dimensionality reduction, and NMF of predicted taste probabilities for
RGB coloring.

Inputs (from hts_predict_taste.py output):
  tm_hts_fusion_features_*.npy    (N, 384)  fusion h
  tm_hts_results_*.csv            id, smiles, prob_*, pred_*

Outputs (in output_dir):
  tm_main_nmf.png            white background TM
  tm_main_nmf_dark.png       dark background TM
  tm_nmf_components.png      RGB channel decomposition
  tm_taste_distributions.png per-taste highlight panels
  tm_fusion_coords_2d.npy    saved 2D coordinates for reuse
  tm_fusion_metadata.json    parameters & stats
"""

import os
import sys
import json
import glob
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import NMF, PCA

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['Calibri', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Config
hts_dir      = Path(r"./taste_hts_results")
output_dir   = Path(r"./tm_results")
fusion_file  = None   # auto-find latest tm_hts_fusion_features_*.npy
results_file = None   # auto-find latest tm_hts_results_*.csv

# Dimensionality reduction
reduce_method = "trimap"          # "trimap" | "umap" | "tsne" | "pca"

# TriMap params
trimap_n_inliers  = 10
trimap_n_outliers = 5
trimap_n_random   = 3
trimap_weight_temp = 0.5

# UMAP params
umap_n_neighbors = 30
umap_min_dist    = 0.0
umap_metric      = "cosine"

# t-SNE params
tsne_perplexity         = 50
tsne_learning_rate      = 200.0
tsne_n_iter             = 1000
tsne_early_exaggeration = 12.0
tsne_metric             = "cosine"

# Sampling (None = use all data)
sample_size       = None
sampling_strategy = "stratified"   # "random" | "stratified"

# Set True to delete cached 2D coordinates and recompute from scratch
force_recompute   = False

# Plot params
dpi             = 300
point_size      = 1.5
alpha           = 0.5
dist_alpha      = 0.75   # higher alpha for prob distribution / heatmap plots
bg_alpha        = 0.15
font_title      = 16
font_label      = 13
font_tick        = 10
random_seed     = 42

# NMF color gamma correction: < 1 brightens colors (reference paper style ~0.45)
nmf_gamma           = 0.45
# Minimum probability to show color in heatmap panels (lower = gray background)
heatmap_threshold   = 0.3

# Taste labels
taste_labels = ["TARGET_01", "TARGET_02", "TARGET_03",
                "TARGET_04", "TARGET_05", "TARGET_06"]
primary_taste_labels = ["TARGET_01", "TARGET_02", "TARGET_03", "TARGET_04", "TARGET_05"]
include_other_in_nmf = True
include_other_in_combined = True

taste_names = {
    "TARGET_01": "Sweet",
    "TARGET_02": "Bitter",
    "TARGET_03": "Umami",
    "TARGET_04": "Sour",
    "TARGET_05": "Salty",
    "TARGET_06": "Secondary Taste",
}

# Column names as written in the CSV (must match hts_predict_taste.py LABEL_NAMES)
label_col_names = {
    "TARGET_01": "Sweet",
    "TARGET_02": "Bitter",
    "TARGET_03": "Umami",
    "TARGET_04": "Sour",
    "TARGET_05": "Salty",
    "TARGET_06": "Other",
}

taste_colors = {
    "TARGET_01": "#FC9898",   # Sweet
    "TARGET_02": "#B2D69A",   # Bitter
    "TARGET_03": "#F3AD7D",   # Umami
    "TARGET_04": "#C8B4C7",   # Sour
    "TARGET_05": "#F9E6A1",   # Salty
    "TARGET_06": "#93BDE3",   # Secondary Taste
}


# Helpers

def auto_find_latest(pattern: str, directory: Path):
    """Find the latest file matching a glob pattern in directory."""
    matches = sorted(glob.glob(str(directory / pattern)))
    return Path(matches[-1]) if matches else None


def prob_col_for_label(label: str) -> str:
    return f"prob_{label_col_names.get(label, label)}"


def pred_col_for_label(label: str, corrected: bool = False) -> str:
    prefix = "corrected_pred" if corrected else "pred"
    return f"{prefix}_{label_col_names.get(label, label)}"


def best_pred_col(df: pd.DataFrame, label: str) -> str:
    """Return corrected_pred_* column if present, else fall back to pred_*."""
    corr = pred_col_for_label(label, corrected=True)
    return corr if corr in df.columns else pred_col_for_label(label, corrected=False)


def load_hts_outputs(hts_dir: Path):
    """Load fusion features and results CSV from HTS output."""
    fusion_path = fusion_file if (fusion_file and fusion_file.exists()) else auto_find_latest("tm_hts_fusion_features_*.npy", hts_dir)
    csv_path    = results_file if (results_file and results_file.exists()) else auto_find_latest("tm_hts_results_*.csv", hts_dir)

    if fusion_path is None:
        raise FileNotFoundError(f"No fusion feature file found in {hts_dir}")
    if csv_path is None:
        raise FileNotFoundError(f"No results CSV found in {hts_dir}")

    print(f"  Fusion features : {fusion_path.name}")
    print(f"  Results CSV     : {csv_path.name}")

    fusion = np.load(str(fusion_path))
    df     = pd.read_csv(str(csv_path))
    return fusion, df


def do_sampling(fusion, df, sample_size, strategy, taste_labels, seed):
    """Optionally sample for faster visualization."""
    N = len(df)
    if sample_size is None or sample_size >= N:
        return fusion, df

    rng = np.random.default_rng(seed)
    print(f"\n  Sampling: {N:,} -> {sample_size:,}  ({strategy})")

    if strategy == "stratified":
        indices = set()
        per_label = sample_size // len(taste_labels)
        for label in taste_labels:
            pred_col = best_pred_col(df, label)
            if pred_col not in df.columns:
                continue
            pos = df.index[df[pred_col] == 1].tolist()
            neg = df.index[df[pred_col] == 0].tolist()
            n_pos = min(len(pos), per_label // 2)
            n_neg = min(len(neg), per_label // 2)
            if pos:
                indices.update(rng.choice(pos, n_pos, replace=False))
            if neg:
                indices.update(rng.choice(neg, n_neg, replace=False))
        # fill remainder randomly
        remaining = sample_size - len(indices)
        if remaining > 0:
            pool = list(set(range(N)) - indices)
            indices.update(rng.choice(pool, min(remaining, len(pool)), replace=False))
        idx = sorted(indices)[:sample_size]
    else:
        idx = sorted(rng.choice(N, sample_size, replace=False))

    return fusion[idx], df.iloc[idx].reset_index(drop=True)


def reduce_2d(features, method, seed, output_dir, force=False):
    """Dimensionality reduction to 2D with caching."""
    coords_file = output_dir / "tm_fusion_coords_2d.npy"

    # Check cache
    if not force and coords_file.exists():
        cached = np.load(str(coords_file))
        if cached.shape[0] == features.shape[0]:
            print(f"  Using cached coordinates: {coords_file.name}")
            return cached
        print(f"  Cache shape mismatch ({cached.shape[0]} vs {features.shape[0]}), recomputing")
    elif force and coords_file.exists():
        coords_file.unlink()
        print(f"  force_recompute=True: deleted cached coordinates")

    # Use features as-is (no StandardScaler — matches original PTM script)
    features_scaled = features
    print(f"  Features shape: {features.shape}  dtype={features.dtype}")

    print(f"  Running {method.upper()} dimensionality reduction...")

    if method == "trimap":
        try:
            import trimap
            coords = trimap.TRIMAP(
                n_dims=2, n_inliers=trimap_n_inliers,
                n_outliers=trimap_n_outliers, n_random=trimap_n_random,
                weight_temp=trimap_weight_temp, verbose=True
            ).fit_transform(features_scaled)
        except ImportError:
            print("  trimap not installed, falling back to UMAP...")
            method = "umap"

    if method == "umap":
        try:
            import umap
            coords = umap.UMAP(
                n_components=2, n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist, metric=umap_metric,
                random_state=seed, verbose=True
            ).fit_transform(features_scaled)
        except ImportError:
            print("  umap not installed, falling back to PCA...")
            method = "pca"

    if method == "tsne":
        try:
            from openTSNE import TSNE as OpenTSNE
            print("  Using OpenTSNE (multi-threaded)")
            coords = np.array(OpenTSNE(
                n_components=2, perplexity=tsne_perplexity,
                learning_rate=tsne_learning_rate, n_iter=tsne_n_iter,
                early_exaggeration=tsne_early_exaggeration,
                metric=tsne_metric, random_state=seed, n_jobs=-1, verbose=True
            ).fit(features_scaled))
        except ImportError:
            from sklearn.manifold import TSNE
            print("  Using sklearn TSNE")
            coords = TSNE(
                n_components=2, perplexity=tsne_perplexity,
                learning_rate=tsne_learning_rate, max_iter=tsne_n_iter,
                random_state=seed, n_jobs=-1, verbose=1
            ).fit_transform(features_scaled)

    if method == "pca":
        pca = PCA(n_components=2, random_state=seed)
        coords = pca.fit_transform(features_scaled)
        print(f"  PCA variance: PC1={pca.explained_variance_ratio_[0]:.3f}, "
              f"PC2={pca.explained_variance_ratio_[1]:.3f}")

    np.save(str(coords_file), coords)
    print(f"  Saved coordinates: {coords_file.name}")
    return coords


def build_nmf_rgb(df, taste_labels):
    """NMF of predicted probabilities → RGB colors."""
    prob_cols = [prob_col_for_label(l) for l in taste_labels]
    missing = [c for c in prob_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing probability columns: {missing}")

    prob_matrix = df[prob_cols].values.astype(np.float64)
    prob_matrix = np.clip(prob_matrix, 0, None)  # NMF needs non-negative

    print(f"  Prob matrix: {prob_matrix.shape}  "
          f"range [{prob_matrix.min():.3f}, {prob_matrix.max():.3f}]")

    nmf = NMF(n_components=3, random_state=random_seed, max_iter=500)
    rgb_raw = nmf.fit_transform(prob_matrix)

    # Normalize each channel to [0, 1]
    rgb = np.zeros_like(rgb_raw)
    for i in range(3):
        lo, hi = rgb_raw[:, i].min(), rgb_raw[:, i].max()
        rgb[:, i] = (rgb_raw[:, i] - lo) / (hi - lo + 1e-12)

    # Gamma correction: boost brightness (values < 1 increase, saturated colors emerge)
    rgb = np.clip(rgb ** nmf_gamma, 0.0, 1.0)

    print(f"  NMF reconstruction error: {nmf.reconstruction_err_:.4f}")
    return rgb, nmf


def compute_dominant_primary_label(df, taste_labels):
    """Return dominant primary-taste label for each molecule based on max probability."""
    prob_cols = [prob_col_for_label(l) for l in taste_labels]
    missing = [c for c in prob_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing probability columns for dominant label plot: {missing}")

    probs = df[prob_cols].values.astype(np.float32)
    best_idx = np.argmax(probs, axis=1)
    return np.array([taste_labels[i] for i in best_idx], dtype=object)


# Plot functions

def _clean_ax(ax, dark=False, frame=False):
    ax.grid(False)
    if frame:
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(1.5)
            sp.set_color("black")
    else:
        for sp in ax.spines.values():
            sp.set_visible(False)
    bg = "black" if dark else "white"
    ax.set_facecolor(bg)


def save_fig(fig, path, dpi=dpi, bg="white"):
    """Save figure (axes frame retained via _clean_ax frame=True)."""
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight",
                facecolor=bg, pad_inches=0.05)
    plt.close(fig)


def _save_bare(fig, ax, path, dpi=dpi, bg="white"):
    """Save a bare version: no title, axis labels, ticks, legend, or spines."""
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.savefig(str(path), dpi=dpi, bbox_inches='tight', pad_inches=0,
                facecolor=bg, edgecolor='none')
    plt.close(fig)


def plot_rgb(coords, rgb, n_mols, output_dir, dark=False):
    """Main TM: all molecules colored by NMF-RGB."""
    suffix = "dark" if dark else "rgb"
    fg     = "white" if dark else "black"
    bg     = "black" if dark else "white"
    _alpha = min(alpha * 1.2, 1.0) if dark else alpha

    fig, ax = plt.subplots(figsize=(14, 10), facecolor=bg)
    ax.scatter(coords[:, 0], coords[:, 1], c=rgb, s=point_size,
              alpha=_alpha, rasterized=True, edgecolors="none")

    ax.set_xlabel(f"{reduce_method.upper()} Dimension 1", fontsize=font_label, fontweight="bold", color=fg)
    ax.set_ylabel(f"{reduce_method.upper()} Dimension 2", fontsize=font_label, fontweight="bold", color=fg)
    ax.set_title(f"Principal Taste Map (TM)\nFusion Feature Space Mapping of {n_mols:,} Molecules",
                 fontsize=font_title, fontweight="bold", pad=20, color=fg)
    ax.tick_params(colors=fg)
    _clean_ax(ax, dark, frame=True)
    plt.tight_layout()

    path = output_dir / ("tm_main_nmf.png" if not dark else "tm_main_nmf_dark.png")
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight", facecolor=bg, pad_inches=0.05)
    print(f"  Saved: {path.name}")

    if not dark:
        bare_path = output_dir / "tm_main_nmf_bare.png"
        _save_bare(fig, ax, bare_path, bg=bg)
        print(f"  Saved: {bare_path.name}")
    else:
        plt.close(fig)


def plot_rgb_channels(coords, rgb, output_dir):
    """RGB channel decomposition (4 panels)."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    cmaps = [None, "Reds", "Greens", "Blues"]
    titles = ["RGB Combined", "R Channel (NMF dim 1)",
              "G Channel (NMF dim 2)", "B Channel (NMF dim 3)"]

    for i, (ax, cmap, title) in enumerate(zip(axes, cmaps, titles)):
        if i == 0:
            ax.scatter(coords[:, 0], coords[:, 1], c=rgb,
                      s=point_size * 0.5, alpha=alpha, rasterized=True, edgecolors="none")
        else:
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=rgb[:, i - 1],
                           s=point_size * 0.5, alpha=alpha, cmap=cmap,
                           rasterized=True, edgecolors="none", vmin=0, vmax=1)
            cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cb.outline.set_visible(False)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Dim 1", fontsize=10)
        if i == 0:
            ax.set_ylabel("Dim 2", fontsize=10)
        _clean_ax(ax)

    plt.suptitle("NMF Color Component Decomposition",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = output_dir / "tm_nmf_components.png"
    save_fig(fig, path)
    print(f"  Saved: {path.name}")


def plot_known_molecules(coords, df, rgb, taste_labels, output_dir):
    """Plot all molecules in gray and predicted-taste molecules in NMF colors."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0], coords[:, 1], c='lightgray', s=point_size * 0.3,
              alpha=0.3, rasterized=True, edgecolors='none', label='All Molecules')

    pred_cols = [best_pred_col(df, label) for label in taste_labels
                 if best_pred_col(df, label) in df.columns]
    has_taste = np.any([df[col].values == 1 for col in pred_cols], axis=0) if pred_cols else np.zeros(len(df), dtype=bool)
    n_known = int(has_taste.sum())

    if n_known > 0:
        ax.scatter(coords[has_taste, 0], coords[has_taste, 1], c=rgb[has_taste],
                  s=point_size * 2, alpha=alpha, rasterized=True,
                  edgecolors='none', label=f'Predicted Taste Molecules (n={n_known:,})')

    ax.set_xlabel(f'{reduce_method.upper()} Dimension 1', fontsize=font_label, fontweight='bold')
    ax.set_ylabel(f'{reduce_method.upper()} Dimension 2', fontsize=font_label, fontweight='bold')
    ax.set_title('Distribution of Predicted Taste Molecules in TM', fontsize=font_title, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    _clean_ax(ax, frame=True)
    plt.tight_layout()
    path = output_dir / 'tm_known_molecules.png'
    fig.savefig(str(path), dpi=dpi, bbox_inches='tight', facecolor='white', pad_inches=0.05)
    print(f"  Saved: {path.name}")

    bare_path = output_dir / 'tm_known_molecules_bare.png'
    _save_bare(fig, ax, bare_path)
    print(f"  Saved: {bare_path.name}")


def plot_per_taste(coords, df, taste_labels, output_dir):
    """Per-taste highlight panels (2×3 grid for 6 labels)."""
    n_labels = len(taste_labels)
    ncols = 3
    nrows = (n_labels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = axes.flatten()

    for i, label in enumerate(taste_labels):
        ax   = axes[i]
        name = taste_names.get(label, label)
        color = taste_colors.get(label, "#1f77b4")
        prob_col = prob_col_for_label(label)
        pred_col = pred_col_for_label(label)

        # Background (all molecules)
        ax.scatter(coords[:, 0], coords[:, 1], c="#e0e0e0",
                  s=point_size * 0.3, alpha=bg_alpha, rasterized=True, edgecolors="none")

        # Positive predictions (prefer corrected if available)
        use_pred_col = best_pred_col(df, label)
        if use_pred_col in df.columns:
            mask = df[use_pred_col].values == 1
            n_pos = int(mask.sum())
            pct   = 100 * n_pos / len(df)

            if n_pos > 0:
                if prob_col in df.columns:
                    sc = ax.scatter(coords[mask, 0], coords[mask, 1], c=df.loc[mask, prob_col].values,
                                   s=point_size * 2, alpha=dist_alpha, rasterized=True,
                                   edgecolors="none", cmap='YlOrRd', vmin=0, vmax=1)
                    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                    cb.outline.set_visible(False)
                else:
                    ax.scatter(coords[mask, 0], coords[mask, 1], c=color,
                              s=point_size * 2, alpha=alpha, rasterized=True,
                              edgecolors="none")

            tag = " [corr]" if use_pred_col.startswith("corrected") else ""
            ax.set_title(f"{name}{tag}\n{n_pos:,} / {len(df):,} ({pct:.2f}%)",
                        fontsize=font_label, fontweight="bold")
        else:
            ax.set_title(name, fontsize=font_label, fontweight="bold")

        ax.set_xlabel('Dimension 1', fontsize=font_tick)
        ax.set_ylabel('Dimension 2', fontsize=font_tick)
        _clean_ax(ax)

    # Hide unused axes
    for j in range(n_labels, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Distribution of Taste Labels in TM',
                 fontsize=font_title, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = output_dir / "tm_taste_distributions.png"
    save_fig(fig, path)
    print(f"  Saved: {path.name}")


def plot_combined(coords, df, taste_labels, output_dir):
    """All tastes overlaid with legend."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Background
    ax.scatter(coords[:, 0], coords[:, 1], c="#e0e0e0",
              s=point_size * 0.3, alpha=bg_alpha, rasterized=True, edgecolors="none")

    for label in taste_labels:
        name     = taste_names.get(label, label)
        color    = taste_colors.get(label, "#1f77b4")
        pred_col = best_pred_col(df, label)
        if pred_col not in df.columns:
            continue
        mask  = df[pred_col].values == 1
        n_pos = int(mask.sum())
        if n_pos > 0:
            ax.scatter(coords[mask, 0], coords[mask, 1], c=color,
                      s=point_size * 4, alpha=alpha, rasterized=True,
                      edgecolors="none", label=f"{name} ({n_pos:,})")

    ax.legend(fontsize=11, markerscale=2, frameon=True, fancybox=True,
              shadow=True, loc="upper right")
    ax.set_xlabel("Dimension 1", fontsize=font_label, fontweight="bold")
    ax.set_ylabel("Dimension 2", fontsize=font_label, fontweight="bold")
    ax.set_title(f"Combined Taste Map — Fusion Feature Space\n"
                 f"{len(df):,} molecules | {reduce_method.upper()}",
                 fontsize=font_title, fontweight="bold", pad=20)
    _clean_ax(ax, frame=True)
    plt.tight_layout()
    path = output_dir / "tm_fusion_combined.png"
    fig.savefig(str(path), dpi=dpi, bbox_inches='tight', facecolor='white', pad_inches=0.05)
    print(f"  Saved: {path.name}")

    bare_path = output_dir / "tm_fusion_combined_bare.png"
    _save_bare(fig, ax, bare_path)
    print(f"  Saved: {bare_path.name}")


def plot_dominant_primary(coords, dominant_labels, output_dir):
    """Plot each molecule once using its dominant primary taste label.

    dominant_labels: array of taste NAME strings (e.g. 'Sweet', 'Bitter')
                     OR TARGET_XX codes — both are handled.
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    for label in taste_labels:
        name  = taste_names.get(label, label)
        color = taste_colors.get(label, "#1f77b4")
        # Accept both TARGET_XX codes and human-readable names
        mask = (dominant_labels == label) | (dominant_labels == name)
        if not np.any(mask):
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color,
                  s=point_size * 4, alpha=alpha, rasterized=True,
                  edgecolors="none", label=f"{name} ({int(mask.sum()):,})")

    ax.legend(fontsize=11, markerscale=2, frameon=True, fancybox=True,
              shadow=True, loc="upper right")
    ax.set_xlabel("Dimension 1", fontsize=font_label, fontweight="bold")
    ax.set_ylabel("Dimension 2", fontsize=font_label, fontweight="bold")
    ax.set_title("Dominant Primary Taste Map — Fusion Feature Space\n"
                 "Each molecule assigned once by max probability among 5 primary tastes",
                 fontsize=font_title, fontweight="bold", pad=20)
    _clean_ax(ax, frame=True)
    plt.tight_layout()
    path = output_dir / "tm_fusion_dominant_primary.png"
    fig.savefig(str(path), dpi=dpi, bbox_inches='tight', facecolor='white', pad_inches=0.05)
    print(f"  Saved: {path.name}")

    bare_path = output_dir / "tm_fusion_dominant_primary_bare.png"
    _save_bare(fig, ax, bare_path)
    print(f"  Saved: {bare_path.name}")


def plot_prob_heatmap(coords, df, taste_labels, output_dir):
    """Per-taste probability heatmap panels."""
    n_labels = len(taste_labels)
    ncols = 3
    nrows = (n_labels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = axes.flatten()

    for i, label in enumerate(taste_labels):
        ax   = axes[i]
        name = taste_names.get(label, label)
        prob_col = prob_col_for_label(label)
        if prob_col not in df.columns:
            continue

        probs = df[prob_col].values
        ax.scatter(coords[:, 0], coords[:, 1], c="#e0e0e0",
                  s=point_size * 0.3, alpha=bg_alpha, rasterized=True, edgecolors="none")
        mask = probs > heatmap_threshold
        if mask.sum() > 0:
            sc = ax.scatter(coords[mask, 0], coords[mask, 1], c=probs[mask],
                           cmap="YlOrRd", s=point_size * 2, alpha=dist_alpha,
                           rasterized=True, edgecolors="none",
                           vmin=0, vmax=1)
            cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cb.outline.set_visible(False)
        ax.set_title(f"{name} probability", fontsize=font_label, fontweight="bold")
        ax.set_xlabel("Dim 1", fontsize=font_tick)
        ax.set_ylabel("Dim 2", fontsize=font_tick)
        _clean_ax(ax)

    for j in range(n_labels, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Taste Probability Heatmaps — Fusion Feature Space",
                 fontsize=font_title, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = output_dir / "tm_fusion_prob_heatmap.png"
    save_fig(fig, path)
    print(f"  Saved: {path.name}")


# Main

def main():
    print("TM — Fusion Feature Visualization")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print("\n[1/5] Loading HTS outputs...")
    fusion, df = load_hts_outputs(hts_dir)
    print(f"  Fusion shape: {fusion.shape}")
    print(f"  Results rows: {len(df):,}")

    assert fusion.shape[0] == len(df), \
        f"Row mismatch: fusion={fusion.shape[0]}, csv={len(df)}"

    # 2. Sampling
    fusion, df = do_sampling(fusion, df, sample_size, sampling_strategy,
                             taste_labels, random_seed)
    N = len(df)
    print(f"  Final sample: {N:,} molecules")

    # 3. Dimensionality reduction
    print(f"\n[2/5] Dimensionality reduction ({reduce_method.upper()})...")
    coords = reduce_2d(fusion, reduce_method, random_seed, output_dir, force=force_recompute)
    print(f"  Coords shape: {coords.shape}")

    # 4. NMF -> RGB
    print(f"\n[3/5] NMF of taste probabilities → RGB...")
    nmf_labels = taste_labels if include_other_in_nmf else primary_taste_labels
    rgb, nmf_model = build_nmf_rgb(df, nmf_labels)

    # Use precomputed dominant_primary_taste column from hts_predict_taste.py if available
    if "dominant_primary_taste" in df.columns:
        dominant_primary = df["dominant_primary_taste"].values
        print(f"  Using precomputed dominant_primary_taste column from CSV")
    else:
        dominant_primary = compute_dominant_primary_label(df, taste_labels)
        print(f"  Computed dominant_primary_taste locally from prob columns")

    # 5. Draw plots
    print(f"\n[4/5] Drawing plots...")
    plot_rgb(coords, rgb, N, output_dir, dark=False)
    plot_known_molecules(coords, df, rgb, taste_labels, output_dir)
    plot_per_taste(coords, df, taste_labels, output_dir)
    plot_rgb_channels(coords, rgb, output_dir)
    combined_labels = taste_labels if include_other_in_combined else primary_taste_labels
    plot_combined(coords, df, combined_labels, output_dir)
    plot_dominant_primary(coords, dominant_primary, output_dir)
    plot_prob_heatmap(coords, df, taste_labels, output_dir)

    # 6. Save metadata
    print(f"\n[5/5] Saving metadata...")
    meta = {
        "n_molecules":     N,
        "fusion_dim":      int(fusion.shape[1]),
        "reduce_method":   reduce_method,
        "sample_size":     sample_size,
        "nmf_recon_error": float(nmf_model.reconstruction_err_),
        "nmf_labels":      nmf_labels,
        "include_other_in_nmf": include_other_in_nmf,
        "include_other_in_combined": include_other_in_combined,
        "nmf_components":  {
            "R": nmf_model.components_[0].tolist(),
            "G": nmf_model.components_[1].tolist(),
            "B": nmf_model.components_[2].tolist(),
        },
        "taste_labels":    taste_labels,
        "taste_names":     taste_names,
    }
    meta_path = output_dir / "tm_fusion_metadata.json"
    with open(str(meta_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {meta_path.name}")

    # Summary
    print(f"\nDone! Fusion TM plots saved to: {output_dir}")
    output_files = sorted(output_dir.glob("tm_*"))
    for f in output_files:
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
