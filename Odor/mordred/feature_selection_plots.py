#!/usr/bin/env python3
"""
feature_selection_plots.py
Standalone visualization module for each step of the feature selection pipeline.

Style: Calibri Bold | Colormap: #85B4DF (low) → #ffb497 (mid) → #EA6852 (high)
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path


# Shared style

SHAP_CMAP = LinearSegmentedColormap.from_list(
    'shap_custom', ['#7DAFDD', '#C7AAD6', '#F27A83']
)

_BAR_COLOR = '#7DAFDD'
_THRESH_COLOR = '#F27A83'

_RC = {
    'font.family': ['Calibri', 'DejaVu Sans'],
    'font.weight': 'bold',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
}


def _apply_style():
    """Apply shared rcParams style to the current matplotlib session."""
    plt.rcParams.update(_RC)


def _bold(ax):
    """Set Calibri Bold on all text elements of an Axes."""
    items = (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    )
    for item in items:
        try:
            item.set_fontfamily('Calibri')
            item.set_fontweight('bold')
        except Exception:
            pass


def _save(fig, path):
    """Save figure to a 'picture/' subdirectory next to the given path."""
    path = Path(path)
    pic_dir = path.parent / 'picture'
    pic_dir.mkdir(parents=True, exist_ok=True)
    final_path = pic_dir / path.name
    plt.tight_layout()
    plt.savefig(final_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    logging.info(f'Plot saved → {final_path}')


# Step 1a: Missing rate distribution

def plot_missing_rates(df: pd.DataFrame, feature_cols: list, output_dir,
                       missing_threshold: float = 0.3):
    """Histogram of per-feature missing rates; bars above threshold are highlighted."""
    _apply_style()
    features = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    missing_rate = features.isnull().mean().values

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 51)
    counts, edges, patches = ax.hist(missing_rate, bins=bins,
                                     color=_BAR_COLOR, edgecolor='white', linewidth=0.3)
    for patch, left in zip(patches, edges[:-1]):
        if left >= missing_threshold:
            patch.set_facecolor(_THRESH_COLOR)
    ax.axvline(missing_threshold, color=_THRESH_COLOR, linewidth=1.8,
               linestyle='--', label=f'Threshold = {missing_threshold}')

    n_kept = int((missing_rate <= missing_threshold).sum())
    ax.set_xlabel('Missing Rate')
    ax.set_ylabel('Number of Features')
    ax.set_title(f'Step 1a · Missing Rate Distribution  (kept {n_kept} / {len(feature_cols)})')
    ax.legend(prop={'family': 'Calibri', 'weight': 'bold', 'size': 9})
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    _bold(ax)
    _save(fig, Path(output_dir) / 'step1a_missing_rates.png')


# Step 1b: Variance distribution

def plot_variance_distribution(df: pd.DataFrame, feature_cols: list, output_dir,
                                var_threshold: float = 1e-5):
    """Histogram of log10(variance); bars below threshold are highlighted."""
    _apply_style()
    features = (df[feature_cols]
                .apply(pd.to_numeric, errors='coerce')
                .replace([np.inf, -np.inf], np.nan))
    variances = features.var().values.clip(min=1e-15)

    fig, ax = plt.subplots(figsize=(8, 4))
    log_var = np.log10(variances)
    log_thr = np.log10(var_threshold)
    counts, edges, patches = ax.hist(log_var, bins=50,
                                     color=_BAR_COLOR, edgecolor='white', linewidth=0.3)
    for patch, left in zip(patches, edges[:-1]):
        if left < log_thr:
            patch.set_facecolor(_THRESH_COLOR)
    ax.axvline(log_thr, color=_THRESH_COLOR, linewidth=1.8,
               linestyle='--', label=f'Threshold = {var_threshold:.0e}')

    n_removed = int((variances < var_threshold).sum())
    ax.set_xlabel('log₁₀(Variance)')
    ax.set_ylabel('Number of Features')
    ax.set_title(f'Step 1b · Variance Distribution  (removed {n_removed} low-variance)')
    ax.legend(prop={'family': 'Calibri', 'weight': 'bold', 'size': 9})
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    _bold(ax)
    _save(fig, Path(output_dir) / 'step1b_variance_distribution.png')


# Step 2: Pairwise correlation

def plot_correlation_distribution(df: pd.DataFrame, feature_cols: list, output_dir,
                                   corr_threshold: float = 0.95):
    """Histogram of upper-triangle absolute Pearson correlations (sampled to ≤400 cols)."""
    _apply_style()
    features = (df[feature_cols]
                .apply(pd.to_numeric, errors='coerce')
                .replace([np.inf, -np.inf], np.nan))

    rng = np.random.default_rng(42)
    sample_cols = (feature_cols if len(feature_cols) <= 400
                   else list(rng.choice(feature_cols, 400, replace=False)))
    corr = features[sample_cols].corr().abs()
    upper = corr.values[np.triu_indices_from(corr.values, k=1)]

    fig, ax = plt.subplots(figsize=(8, 4))
    counts, edges, patches = ax.hist(upper, bins=50,
                                     color=_BAR_COLOR, edgecolor='white', linewidth=0.3)
    for patch, left in zip(patches, edges[:-1]):
        if left >= corr_threshold:
            patch.set_facecolor(_THRESH_COLOR)
    ax.axvline(corr_threshold, color=_THRESH_COLOR, linewidth=1.8,
               linestyle='--', label=f'Threshold = {corr_threshold}')

    n_high = int((upper >= corr_threshold).sum())
    ax.set_xlabel('Absolute Pearson Correlation')
    ax.set_ylabel('Feature Pairs')
    ax.set_title(f'Step 2 · Pairwise Correlation  ({n_high} pairs ≥ threshold, sampled ≤400 cols)')
    ax.legend(prop={'family': 'Calibri', 'weight': 'bold', 'size': 9})
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    _bold(ax)
    _save(fig, Path(output_dir) / 'step2_correlation_distribution.png')


# Step 4: Stability selection frequencies

def plot_stability_frequencies(selection_frequency: dict, threshold: float, output_dir):
    """Two-panel plot: frequency histogram (left) and top-30 horizontal bar (right)."""
    _apply_style()
    feat_names = list(selection_frequency.keys())
    freqs = np.array([selection_frequency[f] for f in feat_names])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: overall histogram
    ax = axes[0]
    counts, edges, patches = ax.hist(freqs, bins=40,
                                     color=_BAR_COLOR, edgecolor='white', linewidth=0.3)
    for patch, left in zip(patches, edges[:-1]):
        if left >= threshold:
            patch.set_facecolor(_THRESH_COLOR)
    ax.axvline(threshold, color='#555555', linewidth=1.5, linestyle='--',
               label=f'Threshold = {threshold}')
    n_above = int((freqs >= threshold).sum())
    ax.set_xlabel('Selection Frequency')
    ax.set_ylabel('Number of Features')
    ax.set_title(f'Step 4a · Stability Frequencies  ({n_above} above threshold)')
    ax.legend(prop={'family': 'Calibri', 'weight': 'bold', 'size': 9})
    ax.set_facecolor('white')
    _bold(ax)

    # Right: top-30 horizontal bar
    ax2 = axes[1]
    top_idx = np.argsort(freqs)[::-1][:30][::-1]   # ascending for barh
    top_freqs = freqs[top_idx]
    top_names = [feat_names[i] for i in top_idx]
    norm = mcolors.Normalize(vmin=freqs.min(), vmax=freqs.max())
    colors = [SHAP_CMAP(norm(f)) for f in top_freqs]
    ax2.barh(range(len(top_names)), top_freqs, color=colors,
             edgecolor='white', linewidth=0.3)
    ax2.set_yticks(range(len(top_names)))
    ax2.set_yticklabels(top_names, fontfamily='Calibri', fontweight='bold', fontsize=7)
    ax2.axvline(threshold, color='#555555', linewidth=1.2, linestyle='--')
    ax2.set_xlabel('Selection Frequency')
    ax2.set_title('Step 4b · Top 30 Features by Frequency')
    ax2.set_facecolor('white')
    _bold(ax2)

    fig.patch.set_facecolor('white')
    _save(fig, Path(output_dir) / 'step4_stability_frequencies.png')


# Step 5a: SHAP beeswarm

def _beeswarm_y(sv_col: np.ndarray, row_center: float,
                row_height: float = 0.38, n_bins: int = 80) -> np.ndarray:
    """Bin-and-stack beeswarm y-positions for a single feature row."""
    if len(sv_col) == 0:
        return np.array([])
    order = np.argsort(sv_col)
    sv_s = sv_col[order]
    span = sv_s[-1] - sv_s[0]
    if span < 1e-10:
        return np.full(len(sv_col), row_center)
    bins = np.linspace(sv_s[0] - 1e-10, sv_s[-1] + 1e-10, n_bins + 1)
    bin_idx = np.clip(np.digitize(sv_s, bins) - 1, 0, n_bins - 1)
    y = np.zeros(len(sv_col))
    for b in np.unique(bin_idx):
        m = bin_idx == b
        cnt = int(m.sum())
        step = row_height / max(cnt, 2)
        offsets = np.linspace(-(cnt - 1) / 2 * step, (cnt - 1) / 2 * step, cnt)
        y[order[m]] = row_center + offsets
    return y


def plot_shap_beeswarm(shap_values: np.ndarray, feature_data: pd.DataFrame,
                       top_n: int = 10, output_path: str = None, output_dir=None):
    """
    SHAP beeswarm plot using shap.summary_plot (original beautiful layout).
    Colormap: #93BDE3 (low fval) → #D7C9D6 (mid) → #FFB1B0 (high fval).
    Font: Calibri Bold.
    """
    try:
        import shap
    except ImportError:
        logging.warning("SHAP not available, skipping beeswarm plot")
        return

    _apply_style()
    sv = np.asarray(shap_values)
    feat_df = feature_data.reset_index(drop=True)

    mean_abs = np.mean(np.abs(sv), axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    top_idx_sorted = top_idx[::-1]

    sv_top = sv[:, top_idx_sorted]
    feat_top = feat_df.iloc[:, top_idx_sorted]

    fig, ax = plt.subplots(figsize=(10, 0.65 * top_n + 2.5))

    shap.summary_plot(
        sv_top,
        feat_top,
        plot_type="dot",
        max_display=top_n,
        show=False,
        color_bar_label="Feature value",
        plot_size=None,
        cmap=SHAP_CMAP,
        alpha=0.7,
        dot_size=20
    )

    ax = plt.gca()
    
    # Add vertical jitter to spread dense clusters; fixed seed for reproducibility
    for collection in ax.collections:
        if hasattr(collection, 'get_offsets'):
            offsets = collection.get_offsets()
            if len(offsets) > 0:
                np.random.seed(42)
                offsets[:, 1] += np.random.uniform(-0.12, 0.12, len(offsets))
                collection.set_offsets(offsets)
                collection.set_sizes([25] * len(offsets))
                collection.set_alpha(0.65)
    ax.set_xlabel('SHAP value', fontfamily='Calibri', fontweight='bold', fontsize=13)
    ax.set_ylabel('Molecular descriptor', fontfamily='Calibri', fontweight='bold', fontsize=13)
    ax.tick_params(axis='both', labelsize=11)
    for label in ax.get_yticklabels():
        label.set_fontfamily('Calibri')
        label.set_fontweight('bold')
        label.set_fontsize(12)
    for label in ax.get_xticklabels():
        label.set_fontfamily('Calibri')
        label.set_fontweight('bold')
        label.set_fontsize(11)
    
    # Horizontal reference lines at each feature row
    for i in range(top_n):
        ax.axhline(i, color='#cccccc', linewidth=0.8, linestyle='--', alpha=0.5, zorder=0)
    
    ax.axvline(0, color='gray', linewidth=0.9, linestyle='-')
    ax.set_facecolor('white')
    fig = plt.gcf()
    fig.patch.set_facecolor('white')

    if output_path is None:
        output_path = str(Path(output_dir) / f'step5a_shap_beeswarm_top{top_n}.png')
    _save(fig, output_path)


# Step 5b: SHAP importance bar

def plot_shap_importance_bar(shap_values: np.ndarray, feature_data: pd.DataFrame,
                              top_n: int = 10, output_dir=None):
    """Horizontal bar of mean |SHAP| value, colored by importance magnitude."""
    _apply_style()
    sv = np.asarray(shap_values)
    mean_abs = np.mean(np.abs(sv), axis=0)
    feat_names = list(feature_data.columns)

    top_idx = np.argsort(mean_abs)[::-1][:top_n][::-1]   # ascending for barh
    top_vals = mean_abs[top_idx]
    top_names = [feat_names[i] for i in top_idx]

    norm = mcolors.Normalize(vmin=top_vals.min(), vmax=top_vals.max())
    colors = [SHAP_CMAP(norm(v)) for v in top_vals]

    fig, ax = plt.subplots(figsize=(9, max(5.0, 0.50 * top_n + 2.0)))
    ax.barh(range(top_n), top_vals, color=colors, edgecolor='white', linewidth=0.4)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names, fontfamily='Calibri', fontweight='bold', fontsize=11)
    ax.set_xlabel('Mean |SHAP value|', fontfamily='Calibri', fontweight='bold', fontsize=13)
    ax.set_title(f'Step 5b · SHAP Feature Importance (Top {top_n})',
                 fontfamily='Calibri', fontweight='bold', fontsize=13)
    ax.set_facecolor('#f9f9f9')
    fig.patch.set_facecolor('white')
    _bold(ax)
    _save(fig, Path(output_dir) / f'step5b_shap_importance_bar_top{top_n}.png')
