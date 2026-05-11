#!/usr/bin/env python3
"""
Uni-Mol HTS Feature Extractor

High-throughput extraction of CLS-token molecular features using Uni-Mol.
Uses sort-by-size batch inference to minimise intra-batch padding, with
periodic checkpoint/resume support and automatic per-molecule fallback on
batch failures.  Designed for large-scale virtual screening datasets.

Usage:
  python unimol_feature_extractor_hts.py --input molecules.csv --output ./out
  python unimol_feature_extractor_hts.py --help
"""

import os
import sys
import json
import types
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from tqdm import tqdm

try:
    from unimol_tools import UniMolRepr
except ImportError:
    print("Error: unimol_tools not installed. Please run: pip install unimol_tools")
    sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


CONFIG = {
    'input_file': r'./input/molecules.csv',
    'output_dir': r'./output/hts_run',
    'smiles_column': 'smiles',
    'model_size': '84m',
    'checkpoint_path': None,  # None=use default weights; set to a .pt path to load a custom checkpoint
    'use_gpu': None,          # None=auto-detect, True=force GPU, False=force CPU
    'remove_hs': False,
    'batch_size': 64,
    'validate_smiles': False,
    'save_invalid': True,
    'save_failed': True,
}


class UniMolHTSExtractor:
    VALID_MODEL_SIZES = ['84m', '164m', '310m', '570m', '1.1B']

    def __init__(
        self,
        model_size: str = '84m',
        checkpoint_path: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        remove_hs: bool = False,
    ):
        if model_size not in self.VALID_MODEL_SIZES:
            raise ValueError(f"Invalid model_size: {model_size}")

        if checkpoint_path and not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.use_gpu = self._resolve_device(use_gpu)
        self.remove_hs = remove_hs

        logger.info(f"Initializing Uni-Mol model (size: {model_size})...")
        self.model = UniMolRepr(
            data_type='molecule',
            remove_hs=remove_hs,
            model_name='unimolv2',
            model_size=model_size,
            use_gpu=self.use_gpu,
            batch_size=32,
        )
        # Fix UniMolV2 padding bug: attn_bias is init to 0 for both valid and padding positions,
        # so attn_mask provides no masking. Padding atoms at (0,0,0) corrupt 3D attention biases.
        self._patch_collate_fn(self.model.model)

        if checkpoint_path:
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            self.model.model.load_pretrained_weights(path=checkpoint_path)
            logger.info("Checkpoint loaded successfully")

    @staticmethod
    def _patch_collate_fn(unimol_model) -> None:
        """Patch batch_collate_fn to fix UniMolV2 padding bug.

        Root cause: attn_bias is initialised to 0 for valid pairs AND padding pairs
        (pad_idx=0 == valid value), so attn_mask never blocks padding atoms.
        Padding atoms at coord (0,0,0) then contribute spurious 3D attention biases,
        completely corrupting CLS repr when molecules of different sizes share a batch
        (cosine similarity ≈ 0.5 vs single-molecule processing).

        Fix:
          - attn_bias  padding → -1e4  : attn_mask becomes very negative for padding
                                          positions → softmax ≈ 0  ✓
          - src_coord  padding → 1e4   : distance from real atoms to padding atoms
                                          becomes ≫ 0 → se3_invariant_kernel ≈ 0  ✓
        """
        from unimol_tools.utils import pad_coords, pad_1d_tokens, pad_2d
        pad_idx = unimol_model.padding_idx  # 0

        def fixed_batch_collate_fn(self, samples):
            batch = {}
            for k in samples[0][0].keys():
                if k == 'atom_feat':
                    v = pad_coords([torch.tensor(s[0][k]) for s in samples],
                                   pad_idx=pad_idx, dim=8)
                elif k == 'atom_mask':
                    v = pad_1d_tokens([torch.tensor(s[0][k]) for s in samples],
                                      pad_idx=pad_idx)
                elif k == 'edge_feat':
                    v = pad_2d([torch.tensor(s[0][k]) for s in samples],
                               pad_idx=pad_idx, dim=3)
                elif k == 'shortest_path':
                    v = pad_2d([torch.tensor(s[0][k]) for s in samples],
                               pad_idx=pad_idx)
                elif k == 'degree':
                    v = pad_1d_tokens([torch.tensor(s[0][k]) for s in samples],
                                      pad_idx=pad_idx)
                elif k == 'pair_type':
                    v = pad_2d([torch.tensor(s[0][k]) for s in samples],
                               pad_idx=pad_idx, dim=2)
                elif k == 'attn_bias':
                    v = pad_2d([torch.tensor(s[0][k]) for s in samples],
                               pad_idx=-1e4)  # was 0: now very negative → blocks padding in softmax
                elif k == 'src_tokens':
                    v = pad_1d_tokens([torch.tensor(s[0][k]) for s in samples],
                                      pad_idx=pad_idx)
                elif k == 'src_coord':
                    v = pad_coords([torch.tensor(s[0][k]) for s in samples],
                                   pad_idx=1e4)  # was 0: now far → 3D kernel output ≈ 0
                else:
                    continue
                batch[k] = v
            try:
                label = torch.tensor([s[1] for s in samples])
            except Exception:
                label = None
            return batch, label

        unimol_model.batch_collate_fn = types.MethodType(fixed_batch_collate_fn, unimol_model)
        logger.info("batch_collate_fn patched: attn_bias padding=-1e4, src_coord padding=1e4")

    @staticmethod
    def _resolve_device(use_gpu: Optional[bool]) -> bool:
        """Resolve actual GPU usage: auto-detect CUDA or honour the use_gpu override."""
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            if use_gpu is None or use_gpu:
                gpu_name = torch.cuda.get_device_name(0)
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"Using GPU: {gpu_name} ({total_mem:.1f} GB)")
                return True
            logger.info("Using CPU (GPU available but forced off)")
            return False

        if use_gpu:
            logger.warning("GPU requested but CUDA not available, falling back to CPU")
        else:
            logger.info("Using CPU")
        return False

    @staticmethod
    def _get_atom_counts(smiles_list: List[str], remove_hs: bool = False) -> List[int]:
        """Return total atom count for each SMILES (incl. H if remove_hs=False).

        Used to sort molecules before batching so consecutive molecules have
        similar sizes, keeping intra-batch padding near zero and eliminating
        the OPM normalisation error caused by N_total vs N_valid mismatch.
        """
        counts = []
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    counts.append(0)
                elif remove_hs:
                    counts.append(mol.GetNumAtoms())
                else:
                    counts.append(Chem.AddHs(mol).GetNumAtoms())
            except Exception:
                counts.append(0)
        return counts

    def _save_checkpoint(
        self,
        ckpt_dir: Path,
        features_sorted: List[np.ndarray],
        sorted_success: List[int],
        sorted_failed: List[int],
        n_batches_done: int,
        n_total: int,
        batch_size: int,
    ) -> None:
        """Persist current extraction progress to disk for later resumption."""
        if features_sorted:
            np.save(ckpt_dir / 'features.npy',
                    np.asarray(features_sorted, dtype=np.float32))
        np.save(ckpt_dir / 'success.npy', np.array(sorted_success, dtype=np.int64))
        np.save(ckpt_dir / 'failed.npy',  np.array(sorted_failed,  dtype=np.int64))
        state = {
            'n_batches_done': n_batches_done,
            'n_features':     len(features_sorted),
            'n_total':        n_total,
            'batch_size':     batch_size,
            'timestamp':      datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(ckpt_dir / 'state.json', 'w') as f:
            json.dump(state, f)
        n_batches_total = (n_total + batch_size - 1) // batch_size
        pct = 100.0 * n_batches_done / n_batches_total
        logger.info(f"Checkpoint saved: {n_batches_done}/{n_batches_total} batches "
                    f"({pct:.1f}%), {len(features_sorted):,} features")

    def _load_checkpoint(
        self,
        ckpt_dir: Path,
        n_total: int,
        batch_size: int,
    ) -> Tuple[Optional[np.ndarray], List[np.ndarray], List[int], List[int], int]:
        """
        Reload extraction state from a previous checkpoint.

        Returns empty state (sort_idx=None, start_batch=0) if no checkpoint
        exists or if n_total / batch_size no longer matches the saved run.
        """
        state_file    = ckpt_dir / 'state.json'
        sort_idx_file = ckpt_dir / 'sort_idx.npy'
        feat_file     = ckpt_dir / 'features.npy'
        success_file  = ckpt_dir / 'success.npy'
        failed_file   = ckpt_dir / 'failed.npy'

        if not state_file.exists() or not sort_idx_file.exists():
            return None, [], [], [], 0

        with open(state_file) as f:
            state = json.load(f)

        if state.get('n_total') != n_total or state.get('batch_size') != batch_size:
            logger.warning("Checkpoint metadata mismatch (n_total or batch_size changed). "
                           "Starting fresh.")
            return None, [], [], [], 0

        sort_idx      = np.load(sort_idx_file)
        n_batches_done = state['n_batches_done']
        n_features    = state['n_features']

        features_sorted = list(np.load(feat_file)) if feat_file.exists() and n_features > 0 else []
        sorted_success  = np.load(success_file).tolist() if success_file.exists() else []
        sorted_failed   = np.load(failed_file).tolist()  if failed_file.exists()  else []

        n_batches_total = (n_total + batch_size - 1) // batch_size
        logger.info(f"Resuming from checkpoint: {len(features_sorted):,} features, "
                    f"batch {n_batches_done}/{n_batches_total} "
                    f"({100.0 * n_batches_done / n_batches_total:.1f}% done)")
        return sort_idx, features_sorted, sorted_success, sorted_failed, n_batches_done

    @staticmethod
    def validate_smiles(smiles_list: List[str]) -> Tuple[List[int], List[int]]:
        valid_indices, invalid_indices = [], []
        for i, smi in enumerate(tqdm(smiles_list, desc='Validating SMILES')):
            if isinstance(smi, str) and smi and Chem.MolFromSmiles(smi):
                valid_indices.append(i)
            else:
                invalid_indices.append(i)
        logger.info(f"Valid SMILES: {len(valid_indices)}/{len(smiles_list)}")
        if invalid_indices:
            logger.warning(f"Invalid SMILES: {len(invalid_indices)}")
        return valid_indices, invalid_indices

    def extract_hts(
        self,
        smiles_list: List[str],
        batch_size: int = 64,
        return_atomic_reprs: bool = False,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_every: int = 200,
        resume: bool = True,
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        n = len(smiles_list)

        # ── Checkpoint resume ────────────────────────────────────────────────
        start_batch:    int                  = 0
        features_sorted: List[np.ndarray]   = []
        sorted_success:  List[int]           = []
        sorted_failed:   List[int]           = []
        sort_idx:        Optional[np.ndarray] = None

        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if resume:
                sort_idx, features_sorted, sorted_success, sorted_failed, start_batch = \
                    self._load_checkpoint(checkpoint_dir, n, batch_size)

        # ── Sort by atom count ───────────────────────────────────────────────
        if sort_idx is None:
            logger.info("Computing atom counts for sort-by-size batching...")
            atom_counts = self._get_atom_counts(smiles_list, self.remove_hs)
            sort_idx = np.argsort(atom_counts, kind='stable')  # sort_idx[k] → original index
            if checkpoint_dir:
                np.save(checkpoint_dir / 'sort_idx.npy', sort_idx)

        smiles_sorted = [smiles_list[i] for i in sort_idx]
        n_batches = (n + batch_size - 1) // batch_size

        if start_batch > 0:
            logger.info(f"Resuming: {start_batch * batch_size:,} / {n:,} molecules already done")

        # ── Extraction loop ──────────────────────────────────────────────────
        with torch.no_grad():
            for b in tqdm(range(start_batch, n_batches), desc='Extracting features',
                          initial=start_batch, total=n_batches):
                start = b * batch_size
                end   = min(start + batch_size, n)
                batch_smiles = smiles_sorted[start:end]

                try:
                    out = self.model.get_repr(batch_smiles, return_atomic_reprs=return_atomic_reprs)
                    cls_repr = out['cls_repr'] if isinstance(out, dict) else out

                    if isinstance(cls_repr, torch.Tensor):
                        cls_repr = cls_repr.detach().cpu().numpy()
                    else:
                        cls_repr = np.asarray(cls_repr)

                    if cls_repr.ndim == 1:
                        cls_repr = cls_repr.reshape(1, -1)

                    if cls_repr.shape[0] != len(batch_smiles):
                        raise RuntimeError(
                            f"Batch output rows mismatch: got {cls_repr.shape[0]}, "
                            f"expected {len(batch_smiles)}"
                        )

                    for j in range(len(batch_smiles)):
                        features_sorted.append(cls_repr[j])
                        sorted_success.append(start + j)

                except Exception as e:
                    logger.warning(f"Batch {b} failed ({start}-{end}): {e}")
                    for j, smi in enumerate(batch_smiles):
                        sp = start + j
                        try:
                            out1 = self.model.get_repr([smi], return_atomic_reprs=return_atomic_reprs)
                            cls1 = out1['cls_repr'] if isinstance(out1, dict) else out1
                            if isinstance(cls1, torch.Tensor):
                                cls1 = cls1.detach().cpu().numpy()
                            else:
                                cls1 = np.asarray(cls1)
                            if cls1.ndim == 2:
                                cls1 = cls1[0]
                            features_sorted.append(cls1)
                            sorted_success.append(sp)
                        except Exception:
                            sorted_failed.append(sp)

                if self.use_gpu:
                    torch.cuda.empty_cache()

                # ── Periodic checkpoint ──────────────────────────────────────
                if checkpoint_dir and (b + 1) % checkpoint_every == 0:
                    self._save_checkpoint(checkpoint_dir, features_sorted, sorted_success,
                                          sorted_failed, b + 1, n, batch_size)

        # ── Final checkpoint then clean state ────────────────────────────────
        if checkpoint_dir:
            self._save_checkpoint(checkpoint_dir, features_sorted, sorted_success,
                                  sorted_failed, n_batches, n, batch_size)

        if not features_sorted:
            return np.array([]), [], sorted(int(sort_idx[k]) for k in range(n))

        # ── Restore original order ───────────────────────────────────────────
        pairs = sorted(
            zip((int(sort_idx[k]) for k in sorted_success), features_sorted),
            key=lambda x: x[0],
        )
        success_indices = [p[0] for p in pairs]
        feature_array   = np.asarray([p[1] for p in pairs], dtype=np.float32)
        failed_indices  = sorted(int(sort_idx[k]) for k in sorted_failed)

        logger.info(f"Successfully extracted: {len(success_indices):,} molecules, "
                    f"shape: {feature_array.shape}")
        if failed_indices:
            logger.warning(f"Failed extractions: {len(failed_indices):,}")
        return feature_array, success_indices, failed_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Extract molecular features using Uni-Mol (HTS)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input', type=str, default=None, help='Input CSV/Excel file')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--smiles-column', type=str, default=None, help='SMILES column name')
    parser.add_argument('--model-size', type=str, default=None, choices=UniMolHTSExtractor.VALID_MODEL_SIZES)
    parser.add_argument('--checkpoint', type=str, default=None, help='Custom checkpoint path')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size for extraction')
    parser.add_argument('--use-gpu', action='store_true', help='Force GPU')
    parser.add_argument('--use-cpu', action='store_true', help='Force CPU')
    parser.add_argument('--remove-hs', action='store_true', help='Remove hydrogens')
    parser.add_argument('--validate-smiles', action='store_true', help='Run RDKit validation first')
    parser.add_argument('--checkpoint-every', type=int, default=200,
                        help='Save checkpoint every N extraction batches (default: 200)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Ignore existing checkpoint and start fresh')
    return parser.parse_args()


def load_dataframe(input_path: str) -> pd.DataFrame:
    """Load a CSV/TXT or Excel file into a DataFrame."""
    ext = Path(input_path).suffix.lower()
    if ext in ['.csv', '.txt']:
        return pd.read_csv(input_path)
    if ext in ['.xlsx', '.xls']:
        return pd.read_excel(input_path)
    raise ValueError(f"Unsupported input file format: {ext}")


def main() -> None:
    args = parse_args()

    cfg = CONFIG.copy()
    if args.input:
        cfg['input_file'] = args.input
    if args.output:
        cfg['output_dir'] = args.output
    if args.smiles_column:
        cfg['smiles_column'] = args.smiles_column
    if args.model_size:
        cfg['model_size'] = args.model_size
    if args.checkpoint:
        cfg['checkpoint_path'] = args.checkpoint
    if args.batch_size:
        cfg['batch_size'] = args.batch_size
    if args.remove_hs:
        cfg['remove_hs'] = True
    if args.validate_smiles:
        cfg['validate_smiles'] = True
    cfg['checkpoint_every'] = args.checkpoint_every
    cfg['resume'] = not args.no_resume
    if args.use_gpu and args.use_cpu:
        raise ValueError('Cannot set both --use-gpu and --use-cpu')
    if args.use_gpu:
        cfg['use_gpu'] = True
    if args.use_cpu:
        cfg['use_gpu'] = False

    input_file = cfg['input_file']
    output_dir = Path(cfg['output_dir'])
    smiles_col = cfg['smiles_column']

    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / f"{Path(input_file).stem}_extraction.log"
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(fh)

    logger.info('Uni-Mol Molecular Feature Extractor (HTS) — starting')

    df = load_dataframe(input_file)
    logger.info(f"Loaded data: {input_file} (shape: {df.shape})")
    if smiles_col not in df.columns:
        raise KeyError(f"SMILES column not found: {smiles_col}")

    smiles_list = df[smiles_col].astype(str).tolist()
    logger.info(f"Total molecules: {len(smiles_list)}")

    if cfg['validate_smiles']:
        valid_indices, invalid_indices = UniMolHTSExtractor.validate_smiles(smiles_list)
        valid_smiles = [smiles_list[i] for i in valid_indices]
    else:
        valid_indices = list(range(len(smiles_list)))
        invalid_indices = []
        valid_smiles = smiles_list
        logger.info('SMILES validation skipped')

    extractor = UniMolHTSExtractor(
        model_size=cfg['model_size'],
        checkpoint_path=cfg['checkpoint_path'],
        use_gpu=cfg['use_gpu'],
        remove_hs=cfg['remove_hs'],
    )

    stem = Path(input_file).stem
    ckpt_dir = output_dir / f'.ckpt_{stem}'

    feature_array, success_indices_valid, failed_indices_valid = extractor.extract_hts(
        valid_smiles,
        batch_size=cfg['batch_size'],
        return_atomic_reprs=False,
        checkpoint_dir=ckpt_dir,
        checkpoint_every=cfg['checkpoint_every'],
        resume=cfg['resume'],
    )

    success_indices = [valid_indices[i] for i in success_indices_valid]
    failed_indices = [valid_indices[i] for i in failed_indices_valid]

    feature_path = output_dir / f'{stem}_molecular_features.npy'
    indices_path = output_dir / f'{stem}_success_indices.npy'
    filtered_path = output_dir / f'{stem}_filtered_data.csv'
    config_path = output_dir / f'{stem}_extraction_config.json'

    np.save(feature_path, feature_array)
    np.save(indices_path, np.array(success_indices, dtype=np.int64))

    filtered_df = df.iloc[success_indices].reset_index(drop=True)
    filtered_df.to_csv(filtered_path, index=False)

    run_info: Dict[str, object] = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'input_file': input_file,
        'output_dir': str(output_dir),
        'smiles_column': smiles_col,
        'model_size': cfg['model_size'],
        'checkpoint_path': cfg['checkpoint_path'],
        'use_gpu': extractor.use_gpu,
        'remove_hs': cfg['remove_hs'],
        'batch_size': cfg['batch_size'],
        'validate_smiles': cfg['validate_smiles'],
        'total_molecules': len(smiles_list),
        'valid_smiles': len(valid_indices),
        'invalid_smiles': len(invalid_indices),
        'successful_extractions': len(success_indices),
        'failed_extractions': len(failed_indices),
        'feature_dim': int(feature_array.shape[1]) if feature_array.size else 0,
        'files': {
            'features': str(feature_path),
            'success_indices': str(indices_path),
            'filtered_data': str(filtered_path),
        },
    }

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    if cfg['save_invalid'] and invalid_indices:
        invalid_path = output_dir / f'{stem}_invalid_smiles.csv'
        df.iloc[invalid_indices].to_csv(invalid_path, index=False)
        logger.info(f"Invalid SMILES saved: {invalid_path} ({len(invalid_indices)} rows)")

    if cfg['save_failed'] and failed_indices:
        failed_path = output_dir / f'{stem}_failed_extractions.csv'
        df.iloc[failed_indices].to_csv(failed_path, index=False)
        logger.info(f"Failed extractions saved: {failed_path} ({len(failed_indices)} rows)")

    logger.info(f"Features saved: {feature_path}")
    logger.info(f"Indices saved: {indices_path}")
    logger.info(f"Filtered data saved: {filtered_path} (shape: {filtered_df.shape})")
    logger.info(f"Configuration saved: {config_path}")
    logger.info('Extraction completed successfully.')


if __name__ == '__main__':
    main()
