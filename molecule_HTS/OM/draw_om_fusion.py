#!/usr/bin/env python3
"""
OM (Principal Odor Map) — Fusion Feature Edition

Uses the model's fusion representation h = fusion(cat(h_e, h_d, h_e*h_d))
for dimensionality reduction, and NMF of predicted odor probabilities for
RGB coloring. Per-label plots show only the top N odor labels by positive
prediction count.

Inputs (from hts_predict_odor.py output):
  om_hts_fusion_features_*.npy    (N, 384)  fusion h
  om_hts_results_*.csv            id, smiles, prob_TARGET_XX, pred_TARGET_XX

Outputs (in output_dir):
  om_main_nmf.png              white background OM
  om_nmf_components.png        RGB channel decomposition
  om_known_molecules.png       predicted-odor molecules highlighted
  om_odor_distributions.png    per-odor highlight panels (top N)
  om_odor_combined.png         top-N odors overlaid with legend
  om_odor_dominant.png         dominant odor assignment per molecule
  om_fusion_coords_2d.npy      saved 2D coordinates for reuse
  om_odor_metadata.json        parameters & stats
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
label_mapping_file = Path(r"./label_mapping.txt")
hts_dir            = Path(r"./odor_hts_results")
output_dir         = Path(r"./om_results")
fusion_file        = None   # auto-find latest om_odor_hts_fusion_features_*.npy
results_file       = None   # auto-find latest om_odor_hts_results_*.csv

# Number of top odor labels to plot (by positive prediction count)
top_n_odor = 6

# Color palette for top-N odor labels
odor_palette = ["#FC9898", "#B2D69A", "#F3AD7D", "#C8B4C7", "#F9E6A1", "#A8D8EA",
                "#93BDE3", "#AADDCC", "#FFD580", "#D4A0C8", "#A0C4FF"]

# Dimensionality reduction
reduce_method      = "trimap"          # "trimap" | "umap" | "tsne" | "pca"

# TriMap params
trimap_n_inliers   = 10
trimap_n_outliers  = 5
trimap_n_random    = 3
trimap_weight_temp = 0.5

# UMAP params
umap_n_neighbors   = 30
umap_min_dist      = 0.0
umap_metric        = "cosine"

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
dpi         = 300
point_size  = 1.5
alpha       = 0.5
dist_alpha  = 0.75   # higher alpha for prob distribution / heatmap plots
bg_alpha    = 0.15
font_title  = 16
font_label  = 13
font_tick   = 10
random_seed = 42

# NMF color gamma correction: < 1 brightens colors (reference paper style ~0.45)
nmf_gamma         = 0.45
# Minimum probability to show color in heatmap panels (lower = gray background)
heatmap_threshold = 0.3


# Helpers

def load_label_mapping(path: Path) -> dict:
    """Load TARGET_XX → title-cased display name from label_mapping.txt."""
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                mapping[parts[0].strip()] = parts[1].strip().title()
    return mapping


def auto_find_latest(pattern: str, directory: Path):
    """Find the latest file matching a glob pattern in directory."""
    matches = sorted(glob.glob(str(directory / pattern)))
    return Path(matches[-1]) if matches else None


def prob_col_for_label(label: str) -> str:
    return f"prob_{label}"


def pred_col_for_label(label: str, corrected: bool = False) -> str:
    prefix = "corrected_pred" if corrected else "pred"
    return f"{prefix}_{label}"


def best_pred_col(df: pd.DataFrame, label: str) -> str:
    """Return corrected_pred_* column if present, else fall back to pred_*."""
    corr = pred_col_for_label(label, corrected=True)
    return corr if corr in df.columns else pred_col_for_label(label, corrected=False)


def get_all_labels(df: pd.DataFrame) -> list:
    """Extract all TARGET_XX labels from prob_ columns in CSV."""
    return sorted([c[len("prob_"):] for c in df.columns if c.startswith("prob_TARGET_")])


def get_top_n_labels(df: pd.DataFrame, all_labels: list, n: int = 5) -> list:
    """Return top N labels by positive prediction count."""
    counts = {}
    for label in all_labels:
        pc = pred_col_for_label(label)
        if pc in df.columns:
            counts[label] = int(df[pc].sum())
    return sorted(counts, key=lambda l: counts[l], reverse=True)[:n]


def load_hts_outputs(hts_dir: Path):
    """Load fusion features and results CSV from HTS output."""
    fusion_path = (fusion_file if (fusion_file and fusion_file.exists())
                   else auto_find_latest("om_odor_hts_fusion_features_*.npy", hts_dir))
    csv_path    = (results_file if (results_file and results_file.exists())
                   else auto_find_latest("om_odor_hts_results_*.csv", hts_dir))

    if fusion_path is None:
        raise FileNotFoundError(f"No fusion feature file found in {hts_dir}")
    if csv_path is None:
        raise FileNotFoundError(f"No results CSV found in {hts_dir}")

    print(f"  Fusion features : {fusion_path.name}")
    print(f"  Results CSV     : {csv_path.name}")

    fusion = np.load(str(fusion_path))
    df     = pd.read_csv(str(csv_path))
    return fusion, df


def do_sampling(fusion, df, sample_size, strategy, top_labels, seed):
    """Optionally sample for faster visualization."""
    N = len(df)
    if sample_size is None or sample_size >= N:
        return fusion, df

    rng = np.random.default_rng(seed)
    print(f"\n  Sampling: {N:,} -> {sample_size:,}  ({strategy})")

    if strategy == "stratified":
        indices = set()
        per_label = sample_size // max(len(top_labels), 1)
        for label in top_labels:
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
    coords_file = output_dir / "om_fusion_coords_2d.npy"

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


def compute_dominant_label(df, top_labels):
    """Return dominant label for each molecule based on max probability among top labels."""
    prob_cols = [prob_col_for_label(l) for l in top_labels]
    missing   = [c for c in prob_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing probability columns for dominant label: {missing}")

    probs    = df[prob_cols].values.astype(np.float32)
    best_idx = np.argmax(probs, axis=1)
    return np.array([top_labels[i] for i in best_idx], dtype=object)



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


def plot_rgb(coords, rgb, n_mols, output_dir, dark=False, active_mask=None):
    """Main OM: all molecules colored by NMF-RGB.

    If active_mask is provided, inactive molecules are drawn as gray background
    and only active molecules receive NMF-RGB color.
    """
    fg     = "white" if dark else "black"
    bg     = "black" if dark else "white"
    _alpha = min(alpha * 1.2, 1.0) if dark else alpha

    fig, ax = plt.subplots(figsize=(14, 10), facecolor=bg)

    if active_mask is not None:
        gray = "#a0a0a0" if not dark else "#505050"
        ax.scatter(coords[~active_mask, 0], coords[~active_mask, 1],
                  c=gray, s=point_size * 2, alpha=0.55,
                  rasterized=True, edgecolors="none")
        ax.scatter(coords[active_mask, 0], coords[active_mask, 1],
                  c=rgb[active_mask], s=point_size,
                  alpha=_alpha, rasterized=True, edgecolors="none")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], c=rgb, s=point_size,
                  alpha=_alpha, rasterized=True, edgecolors="none")

    ax.set_xlabel(f"{reduce_method.upper()} Dimension 1", fontsize=font_label, fontweight="bold", color=fg)
    ax.set_ylabel(f"{reduce_method.upper()} Dimension 2", fontsize=font_label, fontweight="bold", color=fg)
    ax.set_title(f"Principal Odor Map (OM)\nFusion Feature Space Mapping of {n_mols:,} Molecules",
                 fontsize=font_title, fontweight="bold", pad=20, color=fg)
    ax.tick_params(colors=fg)
    _clean_ax(ax, dark, frame=True)
    plt.tight_layout()

    path = output_dir / ("om_main_nmf.png" if not dark else "om_main_nmf_dark.png")
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight", facecolor=bg, pad_inches=0.05)
    print(f"  Saved: {path.name}")

    if not dark:
        bare_path = output_dir / "om_main_nmf_bare.png"
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

    plt.suptitle("NMF Color Component Decomposition (Top Odor Labels)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = output_dir / "om_nmf_components.png"
    save_fig(fig, path)
    print(f"  Saved: {path.name}")


def plot_known_molecules(coords, df, rgb, top_labels, output_dir):
    """Plot all molecules in gray and predicted-odor molecules in NMF colors."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0], coords[:, 1], c='lightgray', s=point_size * 0.3,
              alpha=0.3, rasterized=True, edgecolors='none', label='All Molecules')

    pred_cols = [best_pred_col(df, l) for l in top_labels if best_pred_col(df, l) in df.columns]
    has_odor  = np.any([df[c].values == 1 for c in pred_cols], axis=0) if pred_cols else np.zeros(len(df), dtype=bool)
    n_known   = int(has_odor.sum())

    if n_known > 0:
        idx     = np.where(has_odor)[0]
        sub_rgb = rgb[idx]
        sub_xy  = coords[idx]

        # Draw order: blue-dominant first (background) → red → green last (foreground)
        b_dom = (sub_rgb[:, 2] >= sub_rgb[:, 0]) & (sub_rgb[:, 2] >= sub_rgb[:, 1])
        r_dom = (~b_dom) & (sub_rgb[:, 0] >= sub_rgb[:, 1])
        g_dom = ~b_dom & ~r_dom

        for grp in [b_dom, r_dom, g_dom]:
            if grp.any():
                ax.scatter(sub_xy[grp, 0], sub_xy[grp, 1], c=sub_rgb[grp],
                          s=point_size * 2, alpha=alpha, rasterized=True,
                          edgecolors='none')

    ax.scatter([], [], c='lightgray', s=20, label='All Molecules')
    if n_known > 0:
        ax.scatter([], [], c='#8888cc', s=20,
                  label=f'Predicted Odor Molecules (n={n_known:,})')
    ax.set_xlabel(f'{reduce_method.upper()} Dimension 1', fontsize=font_label, fontweight='bold')
    ax.set_ylabel(f'{reduce_method.upper()} Dimension 2', fontsize=font_label, fontweight='bold')
    ax.set_title('Distribution of Predicted Odor Molecules in OM', fontsize=font_title, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    _clean_ax(ax, frame=True)
    plt.tight_layout()
    path = output_dir / 'om_known_molecules.png'
    fig.savefig(str(path), dpi=dpi, bbox_inches='tight', facecolor='white', pad_inches=0.05)
    print(f"  Saved: {path.name}")

    bare_path = output_dir / 'om_known_molecules_bare.png'
    _save_bare(fig, ax, bare_path)
    print(f"  Saved: {bare_path.name}")


def plot_per_odor_top5(coords, df, top_labels, label_names, output_dir):
    """Per-odor highlight panels for top N labels."""
    n_labels = len(top_labels)
    ncols    = min(n_labels, 3)
    nrows    = (n_labels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for i, label in enumerate(top_labels):
        ax       = axes[i]
        name     = label_names.get(label, label)
        prob_col = prob_col_for_label(label)

        ax.scatter(coords[:, 0], coords[:, 1], c="#e0e0e0",
                  s=point_size * 0.3, alpha=bg_alpha, rasterized=True, edgecolors="none")

        use_pred_col = best_pred_col(df, label)
        if use_pred_col in df.columns:
            mask  = df[use_pred_col].values == 1
            n_pos = int(mask.sum())
            pct   = 100 * n_pos / len(df)

            if n_pos > 0 and prob_col in df.columns:
                sc = ax.scatter(coords[mask, 0], coords[mask, 1],
                               c=df.loc[mask, prob_col].values,
                               s=point_size * 2, alpha=dist_alpha, rasterized=True,
                               edgecolors="none", cmap='YlOrRd', vmin=0, vmax=1)
                cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                cb.outline.set_visible(False)
            elif n_pos > 0:
                ax.scatter(coords[mask, 0], coords[mask, 1],
                          c=odor_palette[i % len(odor_palette)],
                          s=point_size * 2, alpha=alpha, rasterized=True, edgecolors="none")

            ax.set_title(f"{name}\n{n_pos:,} / {len(df):,} ({pct:.2f}%)",
                        fontsize=font_label, fontweight="bold")
        else:
            ax.set_title(name, fontsize=font_label, fontweight="bold")

        ax.set_xlabel('Dimension 1', fontsize=font_tick)
        ax.set_ylabel('Dimension 2', fontsize=font_tick)
        _clean_ax(ax)

    for j in range(n_labels, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'Top {n_labels} Odor Labels \u2014 Distribution in OM',
                 fontsize=font_title, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = output_dir / "om_odor_distributions.png"
    save_fig(fig, path)
    print(f"  Saved: {path.name}")


def plot_combined(coords, df, top_labels, label_names, output_dir):
    """Top-N odors overlaid with legend."""
    fig, ax = plt.subplots(figsize=(14, 10))

    ax.scatter(coords[:, 0], coords[:, 1], c="#e0e0e0",
              s=point_size * 0.3, alpha=bg_alpha, rasterized=True, edgecolors="none")

    for i, label in enumerate(top_labels):
        name     = label_names.get(label, label)
        color    = odor_palette[i % len(odor_palette)]
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
    ax.set_title(f"Combined Odor Map \u2014 Fusion Feature Space\n"
                 f"{len(df):,} molecules | {reduce_method.upper()}",
                 fontsize=font_title, fontweight="bold", pad=20)
    _clean_ax(ax, frame=True)
    plt.tight_layout()
    path = output_dir / "om_odor_combined.png"
    fig.savefig(str(path), dpi=dpi, bbox_inches='tight', facecolor='white', pad_inches=0.05)
    print(f"  Saved: {path.name}")

    bare_path = output_dir / "om_odor_combined_bare.png"
    _save_bare(fig, ax, bare_path)
    print(f"  Saved: {bare_path.name}")


def plot_dominant(coords, dominant_labels, top_labels, label_names, output_dir):
    """Plot each molecule once using its dominant top-odor label."""
    fig, ax = plt.subplots(figsize=(14, 10))

    for i, label in enumerate(top_labels):
        name  = label_names.get(label, label)
        color = odor_palette[i % len(odor_palette)]
        mask  = dominant_labels == label
        if not np.any(mask):
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color,
                  s=point_size * 4, alpha=alpha, rasterized=True,
                  edgecolors="none", label=f"{name} ({int(mask.sum()):,})")

    ax.legend(fontsize=11, markerscale=2, frameon=True, fancybox=True,
              shadow=True, loc="upper right")
    ax.set_xlabel("Dimension 1", fontsize=font_label, fontweight="bold")
    ax.set_ylabel("Dimension 2", fontsize=font_label, fontweight="bold")
    ax.set_title("Dominant Odor Map \u2014 Fusion Feature Space\n"
                 "Each molecule assigned once by max probability among top odors",
                 fontsize=font_title, fontweight="bold", pad=20)
    _clean_ax(ax, frame=True)
    plt.tight_layout()
    path = output_dir / "om_odor_dominant.png"
    fig.savefig(str(path), dpi=dpi, bbox_inches='tight', facecolor='white', pad_inches=0.05)
    print(f"  Saved: {path.name}")

    bare_path = output_dir / "om_odor_dominant_bare.png"
    _save_bare(fig, ax, bare_path)
    print(f"  Saved: {bare_path.name}")


def plot_prob_heatmap(coords, df, top_labels, label_names, output_dir):
    """Per-odor probability heatmap panels (top N labels)."""
    n_labels = len(top_labels)
    ncols    = min(n_labels, 3)
    nrows    = (n_labels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for i, label in enumerate(top_labels):
        ax       = axes[i]
        name     = label_names.get(label, label)
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

    plt.suptitle("Odor Probability Heatmaps — Fusion Feature Space",
                 fontsize=font_title, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = output_dir / "om_odor_prob_heatmap.png"
    save_fig(fig, path)
    print(f"  Saved: {path.name}")


# Main

def main():
    print("OM — Fusion Feature Visualization")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load label mapping
    print(f"\n[0/6] Loading label mapping from {label_mapping_file.name}...")
    label_names = load_label_mapping(label_mapping_file)
    print(f"  {len(label_names)} labels loaded")

    # Load data
    print("\n[1/6] Loading HTS outputs...")
    fusion, df = load_hts_outputs(hts_dir)
    print(f"  Fusion shape: {fusion.shape}")
    print(f"  Results rows: {len(df):,}")

    assert fusion.shape[0] == len(df), \
        f"Row mismatch: fusion={fusion.shape[0]}, csv={len(df)}"

    # Detect labels and top-N
    all_labels = get_all_labels(df)
    top_labels = get_top_n_labels(df, all_labels, n=top_n_odor)
    print(f"\n  Top {top_n_odor} odor labels by positive prediction count:")
    for label in top_labels:
        pc    = pred_col_for_label(label)
        n_pos = int(df[pc].sum()) if pc in df.columns else 0
        name  = label_names.get(label, label)
        print(f"    {label}  {name:<20}  {n_pos:>8,}")

    # Sampling
    fusion, df = do_sampling(fusion, df, sample_size, sampling_strategy,
                             top_labels, random_seed)
    N = len(df)
    print(f"  Final sample: {N:,} molecules")

    # Dimensionality reduction
    print(f"\n[2/6] Dimensionality reduction ({reduce_method.upper()})...")
    coords = reduce_2d(fusion, reduce_method, random_seed, output_dir, force=force_recompute)
    print(f"  Coords shape: {coords.shape}")

    # NMF → RGB
    print(f"\n[3/6] NMF of odor probabilities → RGB (top {top_n_odor} labels)...")
    rgb, nmf_model = build_nmf_rgb(df, top_labels)
    dominant       = compute_dominant_label(df, top_labels)

    # Draw plots
    print(f"\n[4/6] Drawing plots...")
    # Active mask: any positive prediction among top labels
    pred_cols_top  = [pred_col_for_label(l) for l in top_labels if pred_col_for_label(l) in df.columns]
    active_mask    = np.any(np.stack([df[c].values == 1 for c in pred_cols_top]), axis=0) if pred_cols_top else np.ones(N, dtype=bool)
    n_active       = int(active_mask.sum())
    print(f"  Active molecules (top-{top_n_odor} labels): {n_active:,} / {N:,} ({100*n_active/N:.1f}%)")

    plot_rgb(coords, rgb, N, output_dir, dark=False, active_mask=active_mask)
    plot_rgb_channels(coords, rgb, output_dir)
    plot_known_molecules(coords, df, rgb, top_labels, output_dir)
    plot_per_odor_top5(coords, df, top_labels, label_names, output_dir)
    plot_combined(coords, df, top_labels, label_names, output_dir)
    plot_dominant(coords, dominant, top_labels, label_names, output_dir)
    plot_prob_heatmap(coords, df, top_labels, label_names, output_dir)

    # Save metadata
    print(f"\n[5/6] Saving metadata...")
    meta = {
        "n_molecules":     N,
        "fusion_dim":      int(fusion.shape[1]),
        "reduce_method":   reduce_method,
        "sample_size":     sample_size,
        "top_n_odor":      top_n_odor,
        "top_labels":      top_labels,
        "top_label_names": {l: label_names.get(l, l) for l in top_labels},
        "nmf_recon_error": float(nmf_model.reconstruction_err_),
        "nmf_components":  {
            "R": nmf_model.components_[0].tolist(),
            "G": nmf_model.components_[1].tolist(),
            "B": nmf_model.components_[2].tolist(),
        },
    }
    meta_path = output_dir / "om_odor_metadata.json"
    with open(str(meta_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {meta_path.name}")

    # Summary
    print(f"\nDone! OM plots saved to: {output_dir}")
    for f in sorted(output_dir.glob("om_*")):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
