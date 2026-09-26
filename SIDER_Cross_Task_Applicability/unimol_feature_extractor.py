#!/usr/bin/env python3
"""
Uni-Mol Molecular Feature Extraction Script
Extracts molecular CLS-token representations from SMILES strings using Uni-Mol v2.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import json

import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm

try:
    from unimol_tools import UniMolRepr
except ImportError:
    print("Error: unimol_tools not installed")
    print("Please install: pip install unimol_tools")
    sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# Configuration
CONFIG = {
    'input_file': r'./SIDER2/test_data.csv',
    'output_dir': r'./unimol_feature/finetune_feature/test_output',
    'smiles_column': 'SMILES',
    'model_size': '84m',  # 84m, 164m, 310m, 570m, 1.1B
    'checkpoint_path': r'./checkpoint/merged_checkpoint.pt',  # optional custom checkpoint path
    'use_gpu': None,  # None=auto-detect, True=force GPU, False=force CPU
    'remove_hs': False,
    'batch_size': 1,
    'save_invalid': True,
    'save_failed': True,
    'verbose': False,
}


class UniMolFeatureExtractor:
    """Uni-Mol molecular feature extractor."""
    
    VALID_MODEL_SIZES = ['84m', '164m', '310m', '570m', '1.1B']
    
    def __init__(
        self,
        model_size: str = '84m',
        checkpoint_path: Optional[str] = None,
        use_gpu: bool = None,
        remove_hs: bool = False
    ):
        if model_size not in self.VALID_MODEL_SIZES:
            raise ValueError(f"Invalid model_size: {model_size}")
        
        if checkpoint_path and not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.model_size = model_size
        self.checkpoint_path = checkpoint_path
        self.remove_hs = remove_hs
        self.use_gpu = self._detect_and_configure_gpu(use_gpu)
        
        logger.info(f"Initializing Uni-Mol model (size: {model_size})...")
        
        self.model = UniMolRepr(
            data_type='molecule',
            remove_hs=remove_hs,
            model_name='unimolv2',
            model_size=model_size,
            use_gpu=self.use_gpu
        )
        
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
    
    def _detect_and_configure_gpu(self, use_gpu: Optional[bool]) -> bool:
        """Detect available hardware and configure GPU usage."""
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            if use_gpu is None or use_gpu:
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"Using GPU: {gpu_name}")
                return True
            else:
                logger.info("Using CPU (GPU available but not used)")
                return False
        else:
            if use_gpu:
                logger.warning("GPU requested but CUDA not available, falling back to CPU")
            else:
                logger.info("Using CPU")
            return False
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a custom checkpoint into the model."""
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        try:
            self.model.model.load_pretrained_weights(path=checkpoint_path)
            logger.info("Checkpoint loaded successfully")
        except Exception as e:
            logger.debug(f"Built-in loading failed: {e}")
            logger.info("Attempting manual loading...")
            
            if not self._load_checkpoint_manually(checkpoint_path):
                raise RuntimeError(f"Failed to load checkpoint: {checkpoint_path}")
    
    def _load_checkpoint_manually(self, checkpoint_path: str) -> bool:
        """Manually load checkpoint weights with shape-matched filtering."""
        try:
            state_dict = torch.load(
                checkpoint_path,
                map_location=torch.device('cuda' if self.use_gpu else 'cpu'),
                weights_only=False
            )
            
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            model_state_dict = self.model.state_dict()
            
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k in model_state_dict and model_state_dict[k].shape == v.shape:
                    filtered_state_dict[k] = v
            
            self.model.load_state_dict(filtered_state_dict, strict=False)
            logger.info(f"Checkpoint loaded (manual): {len(filtered_state_dict)} parameters")
            
            return True
            
        except Exception as e:
            logger.error(f"Manual loading failed: {e}")
            return False

    @staticmethod
    def validate_smiles(smiles_list: List[str]) -> Tuple[List[int], List[int]]:
        """Validate SMILES strings via RDKit. Returns (valid_indices, invalid_indices)."""
        valid_indices = []
        invalid_indices = []
        
        for i, smi in enumerate(tqdm(smiles_list, desc="Validating SMILES")):
            if isinstance(smi, str) and smi and Chem.MolFromSmiles(smi):
                valid_indices.append(i)
            else:
                invalid_indices.append(i)
                logger.debug(f"Invalid SMILES at index {i}: {smi}")
        
        logger.info(f"Valid SMILES: {len(valid_indices)}/{len(smiles_list)}")
        if invalid_indices:
            logger.warning(f"Invalid SMILES: {len(invalid_indices)}")
        
        return valid_indices, invalid_indices
    
    def extract_features(
        self,
        smiles_list: List[str],
        batch_size: int = 16,
        return_atomic_reprs: bool = False
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        """Extract CLS-token molecular features from a list of SMILES strings."""
        cls_repr_list = []
        success_indices = []
        failed_indices = []
        
        for idx, smi in enumerate(tqdm(smiles_list, desc="Extracting features")):
            try:
                unimol_repr = self.model.get_repr(
                    [smi],
                    return_atomic_reprs=return_atomic_reprs
                )
                
                # Handle both return formats across unimol_tools versions:
                # Normalize mapping and array outputs from Uni-Mol tools.
                if isinstance(unimol_repr, dict):
                    cls_repr = unimol_repr['cls_repr']
                else:
                    cls_repr = unimol_repr
                
                cls_repr_list.append(cls_repr[0])
                success_indices.append(idx)
                
            except Exception as e:
                logger.debug(f"Failed at index {idx}, SMILES: {smi}, Error: {e}")
                failed_indices.append(idx)
        
        if cls_repr_list:
            feature_array = np.array(cls_repr_list)
            logger.info(f"Successfully extracted: {len(success_indices)} molecules, shape: {feature_array.shape}")
        else:
            feature_array = np.array([])
            logger.error("No features extracted!")
        
        if failed_indices:
            logger.warning(f"Failed extractions: {len(failed_indices)}")
        
        return feature_array, success_indices, failed_indices


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract molecular features using Uni-Mol',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input', '-i', type=str, required=False, help='Input CSV/Excel file')
    parser.add_argument('--output', '-o', type=str, default='./output', help='Output directory')
    parser.add_argument('--smiles-column', type=str, default='SMILES', help='SMILES column name')
    parser.add_argument('--model-size', type=str, default='84m', 
                       choices=['84m', '164m', '310m', '570m', '1.1B'], help='Model size')
    parser.add_argument('--checkpoint', type=str, default=None, help='Custom checkpoint path')
    parser.add_argument('--use-gpu', action='store_true', default=None, help='Use GPU')
    parser.add_argument('--use-cpu', action='store_true', help='Force CPU')
    parser.add_argument('--remove-hs', action='store_true', help='Remove hydrogen atoms')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--save-invalid', action='store_true', help='Save invalid SMILES')
    parser.add_argument('--save-failed', action='store_true', help='Save failed extractions')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')
    
    return parser.parse_args()


def save_results(
    output_dir: Path,
    input_filename: str,
    df: pd.DataFrame,
    feature_array: np.ndarray,
    success_indices: List[int],
    valid_indices: List[int],
    invalid_indices: List[int],
    failed_indices: List[int],
    config: Dict,
    save_invalid: bool = True,
    save_failed: bool = True
) -> None:
    """Save extraction results: features, indices, filtered data, and run config."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save feature matrix
    feature_path = output_dir / f'{input_filename}_molecular_features.npy'
    np.save(feature_path, feature_array)
    logger.info(f"Features saved: {feature_path}")
    
    # Save success indices
    indices_path = output_dir / f'{input_filename}_success_indices.npy'
    np.save(indices_path, np.array(success_indices))
    logger.info(f"Indices saved: {indices_path}")
    
    # Save filtered data rows corresponding to successful extractions
    filtered_df = df.iloc[[valid_indices[i] for i in success_indices]].reset_index(drop=True)
    filtered_path = output_dir / f'{input_filename}_filtered_data.csv'
    filtered_df.to_csv(filtered_path, index=False)
    logger.info(f"Filtered data saved: {filtered_path} (shape: {filtered_df.shape})")
    
    # Save invalid SMILES rows
    if save_invalid and invalid_indices:
        invalid_df = df.iloc[invalid_indices]
        invalid_path = output_dir / f'{input_filename}_invalid_smiles.csv'
        invalid_df.to_csv(invalid_path, index=False)
        logger.info(f"Invalid SMILES saved: {invalid_path} ({len(invalid_indices)} rows)")
    
    # Save rows for which feature extraction failed
    if save_failed and failed_indices:
        failed_orig_indices = [valid_indices[i] for i in failed_indices]
        failed_df = df.iloc[failed_orig_indices]
        failed_path = output_dir / f'{input_filename}_failed_extractions.csv'
        failed_df.to_csv(failed_path, index=False)
        logger.info(f"Failed extractions saved: {failed_path} ({len(failed_indices)} rows)")
    
    # Save run configuration and statistics
    config_path = output_dir / f'{input_filename}_extraction_config.json'
    run_config = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'input_file': str(config['input_file']),
        'smiles_column': config['smiles_column'],
        'model_size': config['model_size'],
        'checkpoint_path': config.get('checkpoint_path'),
        'use_gpu': config.get('use_gpu', False),
        'gpu_info': {
            'available': torch.cuda.is_available(),
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A'
        } if config.get('use_gpu') else None,
        'remove_hs': config.get('remove_hs', False),
        'statistics': {
            'total_molecules': len(df),
            'valid_smiles': len(valid_indices),
            'invalid_smiles': len(invalid_indices),
            'successful_extractions': len(success_indices),
            'failed_extractions': len(failed_indices),
            'feature_dimension': int(feature_array.shape[1]) if len(feature_array) > 0 else 0,
            'success_rate': f"{len(success_indices)/len(valid_indices)*100:.2f}%" if valid_indices else "0%"
        },
        'output_directory': str(output_dir)
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(run_config, f, indent=4, ensure_ascii=False)
    logger.info(f"Configuration saved: {config_path}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Use in-script CONFIG when no CLI arguments are provided
    if len(sys.argv) == 1 or args.input is None:
        logger.info("Using in-script configuration (CONFIG)")
        input_path = Path(CONFIG['input_file'])
        output_dir = Path(CONFIG['output_dir'])
        smiles_column = CONFIG['smiles_column']
        model_size = CONFIG['model_size']
        checkpoint_path = CONFIG['checkpoint_path']
        use_gpu = CONFIG['use_gpu']
        remove_hs = CONFIG['remove_hs']
        batch_size = CONFIG.get('batch_size', 16)
        save_invalid = CONFIG['save_invalid']
        save_failed = CONFIG['save_failed']
        verbose = CONFIG['verbose']
        quiet = False
    else:
        logger.info("Using command line arguments")
        input_path = Path(args.input)
        output_dir = Path(args.output)
        smiles_column = args.smiles_column
        model_size = args.model_size
        checkpoint_path = args.checkpoint
        remove_hs = args.remove_hs
        batch_size = args.batch_size
        save_invalid = args.save_invalid
        save_failed = args.save_failed
        verbose = args.verbose
        quiet = args.quiet
        
        if args.use_cpu:
            use_gpu = False
        elif args.use_gpu:
            use_gpu = True
        else:
            use_gpu = None
    
    # Configure log verbosity
    if verbose:
        logger.setLevel(logging.DEBUG)
    elif quiet:
        logger.setLevel(logging.ERROR)
    
    # Validate input file
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    input_filename = input_path.stem
    
    # Attach a file handler for persistent logging
    log_file = output_dir / f'{input_filename}_extraction.log'
    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    
    try:
        logger.info("Uni-Mol Molecular Feature Extractor")
        
        # Load input data
        if input_path.suffix.lower() == '.csv':
            df = pd.read_csv(input_path)
        elif input_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        
        logger.info(f"Loaded data: {input_path} (shape: {df.shape})")
        
        # Verify SMILES column exists
        if smiles_column not in df.columns:
            raise ValueError(
                f"SMILES column '{smiles_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        
        smiles_list = df[smiles_column].tolist()
        logger.info(f"Total molecules: {len(smiles_list)}")
        
        # Validate SMILES
        valid_indices, invalid_indices = UniMolFeatureExtractor.validate_smiles(smiles_list)
        valid_smiles = [smiles_list[i] for i in valid_indices]
        
        if not valid_smiles:
            logger.error("No valid SMILES found. Exiting.")
            sys.exit(1)
        
        # Initialize extractor
        extractor = UniMolFeatureExtractor(
            model_size=model_size,
            checkpoint_path=checkpoint_path,
            use_gpu=use_gpu,
            remove_hs=remove_hs
        )
        
        # Extract features
        feature_array, success_indices, failed_indices = extractor.extract_features(
            valid_smiles,
            batch_size=batch_size
        )
        
        # Save results
        config = {
            'input_file': input_path,
            'smiles_column': smiles_column,
            'model_size': model_size,
            'checkpoint_path': checkpoint_path,
            'use_gpu': extractor.use_gpu,
            'remove_hs': remove_hs
        }
        
        save_results(
            output_dir=output_dir,
            input_filename=input_filename,
            df=df,
            feature_array=feature_array,
            success_indices=success_indices,
            valid_indices=valid_indices,
            invalid_indices=invalid_indices,
            failed_indices=failed_indices,
            config=config,
            save_invalid=save_invalid,
            save_failed=save_failed
        )
        
        logger.info("Extraction Summary")
        logger.info(f"Total molecules:          {len(df)}")
        logger.info(f"Valid SMILES:             {len(valid_indices)}")
        logger.info(f"Invalid SMILES:           {len(invalid_indices)}")
        logger.info(f"Successful extractions:   {len(success_indices)}")
        logger.info(f"Failed extractions:       {len(failed_indices)}")
        logger.info(f"Success rate:             {len(success_indices)/len(valid_indices)*100:.2f}%")
        logger.info(f"Feature dimension:        {feature_array.shape[1] if len(feature_array) > 0 else 0}")
        logger.info(f"Output directory:         {output_dir}")
        logger.info("Extraction completed successfully.")
        
    except Exception as e:
        logger.error(f"Error during extraction: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
