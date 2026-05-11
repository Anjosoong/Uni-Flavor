#!/usr/bin/env python3
import logging
import pickle
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from feature_selection_plots import (
    plot_missing_rates,
    plot_variance_distribution,
    plot_correlation_distribution,
    plot_stability_frequencies,
    plot_shap_beeswarm,
    plot_shap_importance_bar,
)

try:
    import shap
    shap_available = True
except ImportError:
    shap_available = False


@dataclass
class ImprovedFastConfig:
    """Improved fast stable configuration"""
    train_descriptor_path: str = r"./mordred_descriptors/train_descriptors.npy"
    test_descriptor_path: str = r"./mordred_descriptors/test_descriptors.npy"
    train_label_path: str = r"../Data/processed_data/train_data.csv" 
    output_dir: str = r"./mordred/taste_descriptors"
    
    # Feature filtering parameters
    missing_threshold: float = 0.3
    variance_threshold: float = 1e-5
    correlation_threshold: float = 0.95
    
    # Feature selection parameters
    n_top_features: int = 300
    selection_method: str = "stability_shap"  # "stability_only", "stability_shap"
    
    # Stage 1: Improved stability selection
    n_bootstrap: int = 50  # Increased for better stability
    bootstrap_fraction: float = 0.8
    stability_threshold: float = 0.4  # More principled threshold
    stage1_max_features: int = 300
    use_threshold_first: bool = True  # Use threshold before top-K
    
    # L1 regularization (more sparse)
    l1_C: float = 0.05
    
    # Stage 2: True TreeSHAP parameters
    use_true_shap: bool = True
    shap_sample_size: int = 3000  # Increased for better diversity
    tree_n_estimators: int = 200
    tree_max_depth: int = 8
    use_aggregated_label: bool = True  # Use y_any instead of individual labels
    
    # Stability check
    enable_stability_check: bool = True
    stability_check_seeds: List[int] = None  # Will default to [42, 43]
    
    # Other parameters
    random_seed: int = 42
    impute_strategy: str = "median"
    smiles_column: str = "SMILES"
    
    def __post_init__(self):
        if self.stability_check_seeds is None:
            self.stability_check_seeds = [42, 43]


class ImprovedFastProcessor:
    """Improved fast stable processor"""

    def __init__(self, cfg: ImprovedFastConfig):
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.selected_features = None
        self.imputer = None
        self.scaler = None
        self.rng = np.random.default_rng(cfg.random_seed)
        self._shap_values_matrix = None   # (n_samples, n_features) for beeswarm plot
        self._shap_feature_data = None    # corresponding feature DataFrame
        
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    def _to_binary_int(self, y: np.ndarray) -> np.ndarray:
        """Convert labels to binary integers"""
        y = np.asarray(y)
        if y.dtype.kind in "fc" and np.isnan(y).any():
            y = np.nan_to_num(y, nan=0.0)
        return (y > 0).astype(int)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load data"""
        logging.info("Loading data...")
        train_labels = pd.read_csv(self.cfg.train_label_path)

        def _read_desc(path, smiles_series=None):
            if path.endswith('.npy'):
                names = pd.read_csv(path[:-4] + '.csv', nrows=1).select_dtypes(include='number').columns.tolist()
                df = pd.DataFrame(np.load(path), columns=names)
                if smiles_series is not None:
                    df.insert(0, self.cfg.smiles_column, smiles_series.values)
                return df
            return pd.read_csv(path)

        train_desc = _read_desc(self.cfg.train_descriptor_path, train_labels[self.cfg.smiles_column])
        test_label_path = self.cfg.train_label_path.replace('train_data', 'test_data')
        test_smiles = pd.read_csv(test_label_path)[self.cfg.smiles_column] if self.cfg.test_descriptor_path.endswith('.npy') else None
        test_desc  = _read_desc(self.cfg.test_descriptor_path, test_smiles)
        
        logging.info(f"Train descriptors: {train_desc.shape}")
        logging.info(f"Test descriptors: {test_desc.shape}")
        logging.info(f"Train labels: {train_labels.shape}")
        
        return train_desc, test_desc, train_labels

    def filter_features(self, df: pd.DataFrame, exclude_cols: List[str]) -> List[str]:
        """Basic feature filtering with inf handling"""
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        features = df[feature_cols].apply(pd.to_numeric, errors='coerce')
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Missing rate filter
        missing_rate = features.isnull().mean()
        valid_features = missing_rate[missing_rate <= self.cfg.missing_threshold].index.tolist()
        logging.info(f"After missing filter: {len(valid_features)}/{len(feature_cols)}")
        
        # Variance filter
        variance = features[valid_features].var()
        valid_features = variance[variance >= self.cfg.variance_threshold].index.tolist()
        logging.info(f"After variance filter: {len(valid_features)}/{len(feature_cols)}")
        
        # Single-value filter
        nunique = features[valid_features].nunique()
        valid_features = nunique[nunique > 1].index.tolist()
        logging.info(f"After single-value filter: {len(valid_features)}/{len(feature_cols)}")
        
        return valid_features

    def remove_correlated(self, df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
        """Remove correlated features with proper imputation"""
        features_subset = df[feature_cols].copy()
        features_subset = features_subset.replace([np.inf, -np.inf], np.nan)
        
        temp_imputer = SimpleImputer(strategy='median')
        features_imputed = pd.DataFrame(
            temp_imputer.fit_transform(features_subset),
            columns=feature_cols,
            index=features_subset.index
        )
        
        corr_matrix = features_imputed.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [col for col in upper.columns if any(upper[col] > self.cfg.correlation_threshold)]
        valid_features = [c for c in feature_cols if c not in to_drop]
        
        logging.info(f"After correlation filter: {len(valid_features)}/{len(feature_cols)}")
        return valid_features

    def select_features_improved_stability(self, x: pd.DataFrame, y: pd.DataFrame, 
                                          feature_cols: List[str]) -> Tuple[List[str], Dict]:
        """Improved stability selection with principled threshold"""
        logging.info(f"Improved stability selection with {self.cfg.n_bootstrap} bootstraps...")
        logging.info(f"Using threshold-first approach: {self.cfg.use_threshold_first}")
        
        n_features = len(feature_cols)
        selection_counts = np.zeros(n_features)
        total_selections = 0
        
        for bootstrap in tqdm(range(self.cfg.n_bootstrap), desc="Stability bootstrap"):
            # Bootstrap sampling without replacement (more conservative)
            n_samples = len(x)
            bootstrap_size = int(self.cfg.bootstrap_fraction * n_samples)
            bootstrap_indices = self.rng.choice(n_samples, size=bootstrap_size, replace=False)
            
            x_bootstrap = x.iloc[bootstrap_indices]
            y_bootstrap = y.iloc[bootstrap_indices]
            
            effective_labels = 0
            
            for i, label in enumerate(y_bootstrap.columns):
                y_binary = self._to_binary_int(y_bootstrap.iloc[:, i].values)
                
                if len(np.unique(y_binary)) < 2:
                    continue
                
                effective_labels += 1
                
                try:
                    # Sparse L1 logistic regression
                    lr = LogisticRegression(
                        penalty='l1',
                        solver='liblinear',
                        C=self.cfg.l1_C,
                        random_state=self.cfg.random_seed + bootstrap,
                        max_iter=1000,
                        class_weight='balanced'
                    )
                    lr.fit(x_bootstrap[feature_cols], y_binary)
                    
                    # Count non-zero coefficients
                    non_zero_features = np.abs(lr.coef_[0]) > 1e-4
                    selection_counts += non_zero_features.astype(int)
                    
                except Exception as e:
                    logging.warning(f"Bootstrap {bootstrap}, label {i} failed: {e}")
                    continue
            
            total_selections += effective_labels
        
        # Calculate selection frequency
        if total_selections > 0:
            selection_frequency = selection_counts / total_selections
        else:
            selection_frequency = np.zeros(n_features)
        
        # Principled feature selection
        if self.cfg.use_threshold_first:
            # Method B: Threshold first, then top-K if needed
            stable_mask = selection_frequency >= self.cfg.stability_threshold
            stable_indices = np.where(stable_mask)[0]
            
            if len(stable_indices) > self.cfg.stage1_max_features:
                # Too many, select top by frequency
                sorted_indices = stable_indices[np.argsort(selection_frequency[stable_indices])[::-1]]
                selected_indices = sorted_indices[:self.cfg.stage1_max_features]
                logging.info(f"Threshold {self.cfg.stability_threshold} gave {len(stable_indices)} features, truncated to {self.cfg.stage1_max_features}")
            elif len(stable_indices) < self.cfg.stage1_max_features:
                # Too few, supplement with next best
                remaining_indices = np.where(~stable_mask)[0]
                remaining_sorted = remaining_indices[np.argsort(selection_frequency[remaining_indices])[::-1]]
                n_supplement = self.cfg.stage1_max_features - len(stable_indices)
                selected_indices = np.concatenate([stable_indices, remaining_sorted[:n_supplement]])
                logging.info(f"Threshold {self.cfg.stability_threshold} gave {len(stable_indices)} features, supplemented to {self.cfg.stage1_max_features}")
            else:
                selected_indices = stable_indices
                logging.info(f"Threshold {self.cfg.stability_threshold} gave exactly {len(stable_indices)} features")
        else:
            # Method A: Fixed top-K
            selected_indices = np.argsort(selection_frequency)[::-1][:self.cfg.stage1_max_features]
            logging.info(f"Selected top {self.cfg.stage1_max_features} features by frequency")
        
        selected_features = [feature_cols[i] for i in selected_indices]
        
        # Statistics
        freq_stats = {
            'min': float(np.min(selection_frequency)),
            'median': float(np.median(selection_frequency)),
            'max': float(np.max(selection_frequency)),
            'mean': float(np.mean(selection_frequency)),
            'q25': float(np.percentile(selection_frequency, 25)),
            'q75': float(np.percentile(selection_frequency, 75)),
            'threshold_used': self.cfg.stability_threshold,
            'above_threshold': int(np.sum(selection_frequency >= self.cfg.stability_threshold))
        }
        
        results = {
            'selection_frequency': {feat: selection_frequency[i] for i, feat in enumerate(feature_cols)},
            'frequency_stats': freq_stats,
            'n_stable': len(selected_features),
            'avg_frequency': float(np.mean(selection_frequency[selected_indices])) if len(selected_indices) > 0 else 0.0,
            'total_selections': total_selections
        }
        
        logging.info(f"Stability selection: {len(selected_features)} features selected")
        logging.info(f"Frequency stats - Min: {freq_stats['min']:.3f}, Median: {freq_stats['median']:.3f}, Max: {freq_stats['max']:.3f}")
        logging.info(f"Above threshold {self.cfg.stability_threshold}: {freq_stats['above_threshold']} features")
        logging.info(f"Selected features avg frequency: {results['avg_frequency']:.3f}")
        
        return selected_features, results

    def select_features_true_shap(self, x: pd.DataFrame, y: pd.DataFrame, 
                                  feature_cols: List[str]) -> Tuple[List[str], Dict]:
        """True TreeSHAP refinement with aggregated label and balanced sampling"""
        if not shap_available:
            logging.warning("SHAP not available, falling back to tree importance")
            return self._select_features_tree_importance_fallback(x, y, feature_cols)
        
        logging.info(f"True TreeSHAP refinement from {len(feature_cols)} to {self.cfg.n_top_features} features...")
        
        if self.cfg.use_aggregated_label:
            # Create aggregated label using median split strategy
            # For multi-label tasks where all samples have at least one label,
            # use the median number of labels as threshold
            label_counts = y.sum(axis=1).values
            median_count = np.median(label_counts)
            
            # If median is 0 (sparse labels), use "any label" strategy
            if median_count == 0:
                y_any = (label_counts > 0).astype(int)
                logging.info(f"Using 'any label' strategy (median=0)")
            else:
                # Use median split: samples with >= median labels are "positive"
                y_any = (label_counts >= median_count).astype(int)
                logging.info(f"Using median split strategy (median={median_count:.1f})")
            
            # Balanced sampling for SHAP
            neg_idx = np.where(y_any == 0)[0]
            pos_idx = np.where(y_any == 1)[0]
            
            logging.info(f"Aggregated label distribution: {len(pos_idx)} pos, {len(neg_idx)} neg")
            
            if len(neg_idx) == 0 or len(pos_idx) == 0:
                # Median split gave only one class (e.g. all samples have same label count)
                # Retry with 'any label' strategy before giving up
                logging.warning("Median split gave single class, retrying with 'any label' strategy")
                y_any = (label_counts > 0).astype(int)
                neg_idx = np.where(y_any == 0)[0]
                pos_idx = np.where(y_any == 1)[0]
                if len(neg_idx) == 0 or len(pos_idx) == 0:
                    logging.warning("'any label' also single class, falling back to tree importance")
                    return self._select_features_tree_importance_fallback(x, y, feature_cols)
                logging.info(f"Retry 'any label': {len(pos_idx)} pos, {len(neg_idx)} neg")
            
            # Balanced sampling strategy
            if len(x) > self.cfg.shap_sample_size:
                # Ensure at least 200 samples per class, or use all if fewer
                min_per_class = min(200, len(neg_idx), len(pos_idx))
                max_per_class = self.cfg.shap_sample_size // 2
                n_per_class = min(max_per_class, max(min_per_class, len(neg_idx), len(pos_idx)))
                
                # Sample from each class
                pos_take = self.rng.choice(pos_idx, size=min(len(pos_idx), n_per_class), replace=False)
                neg_take = self.rng.choice(neg_idx, size=min(len(neg_idx), n_per_class), replace=False)
                
                sample_indices = np.concatenate([pos_take, neg_take])
                self.rng.shuffle(sample_indices)  # Shuffle to avoid ordering bias
                
                x_sample = x.iloc[sample_indices]
                y_any_sample = y_any[sample_indices]
                
                logging.info(f"Balanced sampling: {len(sample_indices)} rows ({np.sum(y_any_sample)} pos, {len(y_any_sample) - np.sum(y_any_sample)} neg)")
            else:
                x_sample = x
                y_any_sample = y_any
                logging.info(f"Using full dataset: {len(x)} rows ({np.sum(y_any_sample)} pos, {len(y_any_sample) - np.sum(y_any_sample)} neg)")
            
            # Train single model on aggregated label
            model = ExtraTreesClassifier(
                n_estimators=self.cfg.tree_n_estimators,
                max_depth=self.cfg.tree_max_depth,
                random_state=self.cfg.random_seed,
                n_jobs=-1,
                class_weight='balanced'
            )
            model.fit(x_sample[feature_cols], y_any_sample)
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x_sample[feature_cols])
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Binary classification: [class0, class1]
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            shap_values = np.asarray(shap_values)
            
            # Ensure shap_values is 2D: (n_samples, n_features)
            if shap_values.ndim == 3:
                # (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
                if shap_values.shape[1] == len(feature_cols):
                    shap_values = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]
                elif shap_values.shape[2] == len(feature_cols):
                    shap_values = shap_values[1, :, :] if shap_values.shape[0] > 1 else shap_values[0, :, :]
            
            # Save for beeswarm plot
            self._shap_values_matrix = shap_values
            self._shap_feature_data = x_sample[feature_cols].reset_index(drop=True)

            # Aggregate absolute SHAP values -> (n_features,)
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Ensure 1D
            if mean_abs_shap.ndim > 1:
                mean_abs_shap = mean_abs_shap.ravel()
            
            shap_importance = mean_abs_shap
            effective_labels = 1
            
        else:
            # Original multi-label approach (with balanced sampling per label)
            shap_importance = np.zeros(len(feature_cols))
            effective_labels = 0
            
            # Sample once for all labels
            if len(x) > self.cfg.shap_sample_size:
                sample_indices = self.rng.choice(len(x), size=self.cfg.shap_sample_size, replace=False)
                x_sample = x.iloc[sample_indices]
                y_sample = y.iloc[sample_indices]
                logging.info(f"Sampled {self.cfg.shap_sample_size} rows for multi-label SHAP")
            else:
                x_sample = x
                y_sample = y
                logging.info(f"Using full dataset ({len(x)} rows) for multi-label SHAP")
            
            for i, label in enumerate(y_sample.columns):
                y_binary = self._to_binary_int(y_sample.iloc[:, i].values)
                
                if len(np.unique(y_binary)) < 2:
                    continue
                
                effective_labels += 1
                
                try:
                    model = ExtraTreesClassifier(
                        n_estimators=self.cfg.tree_n_estimators,
                        max_depth=self.cfg.tree_max_depth,
                        random_state=self.cfg.random_seed + i,
                        n_jobs=-1,
                        class_weight='balanced'
                    )
                    model.fit(x_sample[feature_cols], y_binary)
                    
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(x_sample[feature_cols])
                    
                    # Handle different formats
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                    
                    shap_values = np.asarray(shap_values)
                    
                    # Ensure 2D
                    if shap_values.ndim == 3:
                        if shap_values.shape[1] == len(feature_cols):
                            shap_values = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]
                        elif shap_values.shape[2] == len(feature_cols):
                            shap_values = shap_values[1, :, :] if shap_values.shape[0] > 1 else shap_values[0, :, :]
                    
                    # Save first valid label's matrix for beeswarm plot
                    if self._shap_values_matrix is None:
                        self._shap_values_matrix = shap_values
                        self._shap_feature_data = x_sample[feature_cols].reset_index(drop=True)
                    
                    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                    
                    # Ensure 1D
                    if mean_abs_shap.ndim > 1:
                        mean_abs_shap = mean_abs_shap.ravel()
                    
                    shap_importance += mean_abs_shap
                    
                except Exception as e:
                    logging.warning(f"SHAP calculation failed for label {i}: {e}")
                    continue
            
            if effective_labels > 0:
                shap_importance /= effective_labels
            else:
                logging.warning("No effective labels for SHAP, falling back to tree importance")
                return self._select_features_tree_importance_fallback(x, y, feature_cols)
        
        # Ensure shap_importance is 1D before sorting
        shap_importance = np.asarray(shap_importance).ravel()
        
        # Select top features by SHAP importance
        top_indices = np.argsort(shap_importance)[::-1][:self.cfg.n_top_features]
        
        # Use numpy array indexing
        feature_cols_array = np.array(feature_cols)
        selected_features = feature_cols_array[top_indices].tolist()
        
        # Calculate statistics
        shap_stats = {
            'min': float(np.min(shap_importance)),
            'median': float(np.median(shap_importance)),
            'max': float(np.max(shap_importance)),
            'mean': float(np.mean(shap_importance)),
            'q25': float(np.percentile(shap_importance, 25)),
            'q75': float(np.percentile(shap_importance, 75))
        }
        
        # Top 20 features for inspection
        top20_indices = top_indices[:20] if len(top_indices) >= 20 else top_indices
        top20_features = {
            feature_cols_array[i]: float(shap_importance[i]) 
            for i in top20_indices
        }
        
        results = {
            'shap_importance': {feat: float(shap_importance[i]) for i, feat in enumerate(feature_cols)},
            'shap_stats': shap_stats,
            'top20_features': top20_features,
            'effective_labels': effective_labels,
            'top_shap_score': float(shap_importance[top_indices[0]]) if len(top_indices) > 0 else 0.0,
            'used_aggregated_label': self.cfg.use_aggregated_label,
            'sample_size': len(x_sample) if self.cfg.use_aggregated_label else len(x_sample)
        }
        
        logging.info(f"TreeSHAP: SHAP matrix saved ({self._shap_values_matrix.shape if self._shap_values_matrix is not None else 'n/a'}) for beeswarm plot")
        logging.info(f"TreeSHAP refinement: selected {len(selected_features)} features")
        logging.info(f"SHAP stats - Min: {shap_stats['min']:.6f}, Median: {shap_stats['median']:.6f}, Max: {shap_stats['max']:.6f}")
        logging.info(f"Top SHAP importance: {results['top_shap_score']:.6f}")
        logging.info(f"Used {'aggregated' if self.cfg.use_aggregated_label else 'individual'} label(s)")
        
        return selected_features, results

    def _select_features_tree_importance_fallback(self, x: pd.DataFrame, y: pd.DataFrame, 
                                                 feature_cols: List[str]) -> Tuple[List[str], Dict]:
        """Fallback: ExtraTrees feature importance when SHAP unavailable"""
        logging.info("Using ExtraTrees feature importance as fallback...")
        
        if self.cfg.use_aggregated_label:
            # Use aggregated label
            y_any = (y.sum(axis=1) > 0).astype(int)
            
            if len(np.unique(y_any)) < 2:
                logging.error("Aggregated label has only one class!")
                return feature_cols[:self.cfg.n_top_features], {}
            
            model = ExtraTreesClassifier(
                n_estimators=self.cfg.tree_n_estimators,
                max_depth=self.cfg.tree_max_depth,
                random_state=self.cfg.random_seed,
                n_jobs=-1,
                class_weight='balanced'
            )
            model.fit(x[feature_cols], y_any)
            feature_importance = model.feature_importances_
            effective_labels = 1
            
        else:
            # Multi-label approach
            feature_importance = np.zeros(len(feature_cols))
            effective_labels = 0
            
            for i, label in enumerate(y.columns):
                y_binary = self._to_binary_int(y.iloc[:, i].values)
                
                if len(np.unique(y_binary)) < 2:
                    continue
                
                effective_labels += 1
                
                model = ExtraTreesClassifier(
                    n_estimators=self.cfg.tree_n_estimators,
                    max_depth=self.cfg.tree_max_depth,
                    random_state=self.cfg.random_seed + i,
                    n_jobs=-1,
                    class_weight='balanced'
                )
                model.fit(x[feature_cols], y_binary)
                feature_importance += model.feature_importances_
            
            if effective_labels > 0:
                feature_importance /= effective_labels
        
        # Select top features
        top_indices = np.argsort(feature_importance)[::-1][:self.cfg.n_top_features]
        # Convert to list to avoid numpy indexing issues
        feature_cols_list = list(feature_cols) if not isinstance(feature_cols, list) else feature_cols
        selected_features = [feature_cols_list[int(i)] for i in top_indices]
        
        results = {
            'tree_importance': {feat: feature_importance[i] for i, feat in enumerate(feature_cols)},
            'effective_labels': effective_labels,
            'used_aggregated_label': self.cfg.use_aggregated_label
        }
        
        logging.info(f"Tree importance fallback: selected {len(selected_features)} features")
        
        return selected_features, results

    def check_stability_across_seeds(self, train_desc: pd.DataFrame, train_labels: pd.DataFrame) -> Dict:
        """Check feature selection stability across different random seeds"""
        if not self.cfg.enable_stability_check:
            return {}
        
        logging.info("=" * 60)
        logging.info("STABILITY CHECK: Testing across different seeds")
        logging.info("=" * 60)
        
        all_selected_features = {}
        
        for seed in self.cfg.stability_check_seeds:
            logging.info(f"Running with seed {seed}...")
            
            # Create temporary processor with different seed
            temp_cfg = self.cfg
            temp_cfg.random_seed = seed
            temp_processor = ImprovedFastProcessor(temp_cfg)
            
            # Run feature selection (without saving files)
            temp_processor._run_feature_selection_only(train_desc, train_labels)
            all_selected_features[seed] = set(temp_processor.selected_features)
        
        # Calculate overlaps
        seeds = list(all_selected_features.keys())
        overlaps = {}
        
        for i, seed1 in enumerate(seeds):
            for seed2 in seeds[i+1:]:
                set1 = all_selected_features[seed1]
                set2 = all_selected_features[seed2]
                overlap = len(set1 & set2) / len(set1 | set2)  # Jaccard similarity
                overlaps[f"{seed1}_vs_{seed2}"] = {
                    'overlap_ratio': overlap,
                    'intersection_size': len(set1 & set2),
                    'union_size': len(set1 | set2),
                    'seed1_unique': len(set1 - set2),
                    'seed2_unique': len(set2 - set1)
                }
        
        # Overall stability metrics
        overlap_ratios = [v['overlap_ratio'] for v in overlaps.values()]
        stability_metrics = {
            'mean_overlap': float(np.mean(overlap_ratios)),
            'min_overlap': float(np.min(overlap_ratios)),
            'max_overlap': float(np.max(overlap_ratios)),
            'std_overlap': float(np.std(overlap_ratios))
        }
        
        results = {
            'selected_features_by_seed': {k: list(v) for k, v in all_selected_features.items()},
            'pairwise_overlaps': overlaps,
            'stability_metrics': stability_metrics
        }
        
        logging.info(f"Stability check results:")
        logging.info(f"  Mean overlap: {stability_metrics['mean_overlap']:.3f}")
        logging.info(f"  Min overlap: {stability_metrics['min_overlap']:.3f}")
        logging.info(f"  Max overlap: {stability_metrics['max_overlap']:.3f}")
        
        if stability_metrics['mean_overlap'] > 0.7:
            logging.info("  STABLE: High consistency across seeds")
        elif stability_metrics['mean_overlap'] > 0.4:
            logging.info("  MODERATE: Consider increasing bootstrap or regularization")
        else:
            logging.info("  UNSTABLE: Low consistency, need parameter tuning")
        
        return results

    def _run_feature_selection_only(self, train_desc: pd.DataFrame, train_labels: pd.DataFrame):
        """Run feature selection without saving files (for stability check)"""
        label_cols = [c for c in train_labels.columns if c != self.cfg.smiles_column]
        
        # Merge data
        train_df = pd.merge(
            train_desc,
            train_labels[[self.cfg.smiles_column] + label_cols],
            on=self.cfg.smiles_column,
            how="inner"
        )
        
        exclude_cols = [self.cfg.smiles_column] + label_cols
        
        # Basic filtering
        valid_features = self.filter_features(train_df, exclude_cols)
        valid_features = self.remove_correlated(train_df, valid_features)
        
        # Prepare data
        train_df[valid_features] = train_df[valid_features].apply(pd.to_numeric, errors="coerce")
        train_df[valid_features] = train_df[valid_features].replace([np.inf, -np.inf], np.nan)
        
        temp_imputer = SimpleImputer(strategy=self.cfg.impute_strategy)
        temp_scaler = StandardScaler()
        
        X_processed = pd.DataFrame(
            temp_scaler.fit_transform(temp_imputer.fit_transform(train_df[valid_features])),
            columns=valid_features,
            index=train_df.index
        )
        
        # Feature selection
        stable_features, _ = self.select_features_improved_stability(
            X_processed, train_df[label_cols], valid_features
        )
        
        if self.cfg.selection_method == "stability_shap":
            self.selected_features, _ = self.select_features_true_shap(
                X_processed, train_df[label_cols], stable_features
            )
        else:
            self.selected_features = stable_features[:self.cfg.n_top_features]

    def fit_transform_train(self, train_desc: pd.DataFrame, train_labels: pd.DataFrame):
        """Process training data with improved feature selection"""
        logging.info("=" * 60)
        logging.info("Processing TRAIN set with IMPROVED FAST STABLE selection")
        logging.info("=" * 60)
        
        label_cols = [c for c in train_labels.columns if c != self.cfg.smiles_column]
        logging.info(f"Detected {len(label_cols)} label columns")
        
        # Stability check first (if enabled)
        stability_results = self.check_stability_across_seeds(train_desc, train_labels)
        
        # Merge data
        train_df = pd.merge(
            train_desc,
            train_labels[[self.cfg.smiles_column] + label_cols],
            on=self.cfg.smiles_column,
            how="inner"
        )
        
        exclude_cols = [self.cfg.smiles_column] + label_cols
        all_feature_cols = [c for c in train_df.columns if c not in exclude_cols]

        # Basic filtering
        logging.info("Step 1: Basic filtering...")
        valid_features = self.filter_features(train_df, exclude_cols)
        plot_missing_rates(train_df, all_feature_cols, self.output_dir,
                           missing_threshold=self.cfg.missing_threshold)
        plot_variance_distribution(train_df, all_feature_cols, self.output_dir,
                                   var_threshold=self.cfg.variance_threshold)

        logging.info("Step 2: Removing correlated features...")
        valid_features_pre_corr = list(valid_features)
        valid_features = self.remove_correlated(train_df, valid_features)
        plot_correlation_distribution(train_df, valid_features_pre_corr, self.output_dir,
                                      corr_threshold=self.cfg.correlation_threshold)
        
        # Prepare data
        logging.info("Step 3: Preparing data...")
        train_df[valid_features] = train_df[valid_features].apply(pd.to_numeric, errors="coerce")
        train_df[valid_features] = train_df[valid_features].replace([np.inf, -np.inf], np.nan)
        
        temp_imputer = SimpleImputer(strategy=self.cfg.impute_strategy)
        temp_scaler = StandardScaler()
        
        X_processed = pd.DataFrame(
            temp_scaler.fit_transform(temp_imputer.fit_transform(train_df[valid_features])),
            columns=valid_features,
            index=train_df.index
        )
        
        # Feature selection
        logging.info("Step 4: Improved stability selection...")
        stable_features, stability_results = self.select_features_improved_stability(
            X_processed, train_df[label_cols], valid_features
        )
        plot_stability_frequencies(
            stability_results['selection_frequency'],
            self.cfg.stability_threshold,
            self.output_dir
        )
        
        if self.cfg.selection_method == "stability_shap":
            logging.info("Step 5: True TreeSHAP refinement...")
            self.selected_features, shap_results = self.select_features_true_shap(
                X_processed, train_df[label_cols], stable_features
            )
            logging.info("Step 5: Generating SHAP visualizations...")
            if self._shap_values_matrix is not None:
                plot_shap_beeswarm(
                    self._shap_values_matrix, self._shap_feature_data,
                    top_n=10, output_dir=self.output_dir
                )
                plot_shap_importance_bar(
                    self._shap_values_matrix, self._shap_feature_data,
                    top_n=10, output_dir=self.output_dir
                )
        else:
            logging.info("Step 5: Using stability features only...")
            self.selected_features = stable_features[:self.cfg.n_top_features]
            shap_results = {}
        
        # Final processing
        logging.info("Step 6: Final processing...")
        train_df = pd.merge(
            train_desc,
            train_labels[[self.cfg.smiles_column] + label_cols],
            on=self.cfg.smiles_column,
            how="inner"
        )
        
        train_df[self.selected_features] = train_df[self.selected_features].apply(pd.to_numeric, errors="coerce")
        train_df[self.selected_features] = train_df[self.selected_features].replace([np.inf, -np.inf], np.nan)
        
        self.imputer = SimpleImputer(strategy=self.cfg.impute_strategy)
        self.scaler = StandardScaler()
        
        train_df[self.selected_features] = self.imputer.fit_transform(train_df[self.selected_features])
        train_df[self.selected_features] = self.scaler.fit_transform(train_df[self.selected_features])
        
        # Save results
        output_cols = [self.cfg.smiles_column] + self.selected_features + label_cols
        train_processed = train_df[output_cols]
        train_processed.to_csv(self.output_dir / "train_processed.csv", index=False)
        
        desc_only = train_processed[[self.cfg.smiles_column] + self.selected_features]
        desc_only.to_csv(self.output_dir / "train_descriptors_only.csv", index=False)
        
        with open(self.output_dir / "selected_features.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(self.selected_features))
        
        with open(self.output_dir / "transformers.pkl", "wb") as f:
            pickle.dump({
                "imputer": self.imputer,
                "scaler": self.scaler,
                "selected_features": self.selected_features,
                "config": self.cfg,
                "random_seed": self.cfg.random_seed
            }, f)
        
        # Save comprehensive report
        report = {
            "processing_config": {
                "selection_method": self.cfg.selection_method,
                "n_bootstrap": self.cfg.n_bootstrap,
                "stability_threshold": self.cfg.stability_threshold,
                "stage1_max_features": self.cfg.stage1_max_features,
                "use_threshold_first": self.cfg.use_threshold_first,
                "use_aggregated_label": self.cfg.use_aggregated_label,
                "use_true_shap": self.cfg.use_true_shap,
                "random_seed": self.cfg.random_seed,
                "n_top_features": self.cfg.n_top_features
            },
            "stability_results": stability_results,
            "shap_results": shap_results,
            "stability_check": stability_results if self.cfg.enable_stability_check else {},
            "selected_features_list": self.selected_features
        }
        
        with open(self.output_dir / "processing_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Train processed: {train_processed.shape}")
        logging.info(f"Selected {len(self.selected_features)} improved stable features")
        
        return train_processed
    def transform_test(self, test_desc: pd.DataFrame):
        """Transform test data"""
        logging.info("Processing TEST set...")
        
        if self.selected_features is None or self.scaler is None:
            raise ValueError("Must fit on train set first!")
        
        test_df = test_desc[[self.cfg.smiles_column] + self.selected_features].copy()
        
        test_df[self.selected_features] = test_df[self.selected_features].apply(pd.to_numeric, errors="coerce")
        test_df[self.selected_features] = test_df[self.selected_features].replace([np.inf, -np.inf], np.nan)
        test_df[self.selected_features] = self.imputer.transform(test_df[self.selected_features])
        test_df[self.selected_features] = self.scaler.transform(test_df[self.selected_features])
        
        test_df.to_csv(self.output_dir / "test_processed.csv", index=False)
        
        logging.info(f"Test processed: {test_df.shape}")
        
        return test_df

    def run(self):
        """Run the improved pipeline"""
        train_desc, test_desc, train_labels = self.load_data()
        train_processed = self.fit_transform_train(train_desc, train_labels)
        test_processed = self.transform_test(test_desc)
        
        logging.info("=" * 60)
        logging.info("IMPROVED FAST STABLE feature selection completed!")
        logging.info("=" * 60)
        logging.info(f"Output directory: {self.output_dir}")
        logging.info(f"Train: {train_processed.shape}")


def main():
    """Main function"""
    cfg = ImprovedFastConfig(
        # All paths use defaults from class definition
        n_top_features=300,
        selection_method="stability_shap",  # or "stability_only"
        
        # Improved Stage 1 parameters
        n_bootstrap=50,  # Increased for better stability
        bootstrap_fraction=0.8,
        stability_threshold=0.4,  # More principled threshold
        stage1_max_features=300,
        use_threshold_first=True,  # Use threshold before top-K
        l1_C=0.05,  # More sparse
        
        # True TreeSHAP parameters
        use_true_shap=True,
        shap_sample_size=3000,  # Increased for better diversity
        tree_n_estimators=200,
        tree_max_depth=8,
        use_aggregated_label=False,  # Use multi-label for odor data (138 labels)
        
        # Stability check
        enable_stability_check=False,  # Disabled for faster testing
        stability_check_seeds=[42, 43],  # Test consistency
        
        random_seed=42
    )
    
    processor = ImprovedFastProcessor(cfg)
    processor.run()
    
    logging.info("=" * 60)
    logging.info("Summary:")
    logging.info(f"  Method: Improved {cfg.selection_method}")
    logging.info(f"  Bootstrap samples: {cfg.n_bootstrap}")
    logging.info(f"  Stability threshold: {cfg.stability_threshold}")
    logging.info(f"  True TreeSHAP: {cfg.use_true_shap}")
    logging.info(f"  Aggregated label: {cfg.use_aggregated_label}")
    logging.info(f"  Stability check: {cfg.enable_stability_check}")
    logging.info(f"  Final features: {cfg.n_top_features}")
    logging.info("  ✓ Principled, stable, interpretable, universal")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()