#!/usr/bin/env python3
"""
Uni-Mol2 LoRA Fine-tuning Runner

This script implements parameter-efficient fine-tuning of Uni-Mol2 models using
LoRA (Low-Rank Adaptation). LoRA significantly reduces memory requirements and
training time while maintaining competitive performance.

"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

import torch
import torch.nn as nn

# Resolve project paths from this file so the script is independent of the
# caller's current working directory. Uni-Core and Uni-Mol are expected to be
# cloned beside the Uni-Flavor repository, as documented in the root README.
SCRIPT_DIR = Path(__file__).resolve().parent
SIDER_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SIDER_ROOT.parent
DEPENDENCY_ROOT = REPO_ROOT.parent
UNICORE_DIR = DEPENDENCY_ROOT / 'Uni-Core'
UNIMOL2_DIR = DEPENDENCY_ROOT / 'Uni-Mol' / 'unimol2'

for dependency_path in (SCRIPT_DIR, UNICORE_DIR, UNIMOL2_DIR):
    if str(dependency_path) not in sys.path:
        sys.path.insert(0, str(dependency_path))

from lora_modules import (
    apply_lora_to_model,
    print_trainable_parameters,
    LoRALinear,
)


def get_args():
    """
    Parse and validate command line arguments for LoRA fine-tuning.
    
    Returns:
        argparse.Namespace: Parsed arguments with validation and defaults applied
    """
    parser = argparse.ArgumentParser(
        description='Uni-Mol2 LoRA Fine-tuning Runner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Task configuration
    task_group = parser.add_argument_group('Task Configuration')
    task_group.add_argument('--task_name', type=str, default='lmdb_preparation_sider',
                           help='Name of the fine-tuning task')
    task_group.add_argument('--arch_name', type=str, default='84M',
                           choices=['84M', '164M', '310M', '570M', '1100M'],
                           help='Model architecture size')
    task_group.add_argument('--num_classes', type=int, default=27,
                           help='Number of output classes for classification')
    task_group.add_argument('--loss_func', type=str, default='multi_task_BCE',
                           help='Loss function for training')
    
    # Path configuration
    path_group = parser.add_argument_group('Path Configuration')
    path_group.add_argument('--data_path', type=str,
                           default=str(SCRIPT_DIR / 'lmdb_preparation_sider'),
                           help='Path to training data directory')
    path_group.add_argument('--weight_path', type=str,
                           default=str(SCRIPT_DIR / 'weights' / '84M' /
                                       'checkpoint.pt'),
                           help='Path to pretrained model weights')
    path_group.add_argument('--save_dir', type=str,
                           default=str(SCRIPT_DIR / 'result' /
                                       'lora_finetune_sider'),
                           help='Directory for saving training outputs')
    path_group.add_argument('--user_dir', type=str,
                           default=str(UNIMOL2_DIR / 'unimol2'),
                           help='Uni-Mol2 user directory path')
    path_group.add_argument('--unicore_dir', type=str,
                           default=str(UNICORE_DIR),
                           help='UniCore framework directory path')
    
    # LoRA configuration
    lora_group = parser.add_argument_group('LoRA Configuration')
    lora_group.add_argument('--lora_rank', type=int, default=16,
                           help='LoRA rank parameter (8, 16, 32). r=16 balances expressivity and cost')
    lora_group.add_argument('--lora_alpha', type=float, default=16.0,
                           help='LoRA scaling = alpha/rank. alpha=rank gives scaling=1.0 (stable default)')
    lora_group.add_argument('--lora_dropout', type=float, default=0.1,
                           help='Dropout probability for LoRA layers')
    lora_group.add_argument('--lora_target_modules', type=str,
                           default='linear_q,linear_k,linear_v,linear_o,linear_1,linear_2',
                           help='Comma-separated list of modules for LoRA adaptation. '
                                'Uni-Mol2 FFN uses Transition(linear_1, linear_2), NOT fc1/fc2. '
                                'linear_1/linear_2 also exist in pair_ffn; use include_paths to restrict.')
    lora_group.add_argument('--lora_include_paths', type=str,
                           default='self_attn.,.ffn.',
                           help='Restrict LoRA to paths containing these substrings. '
                                '".ffn." matches encoder.layers.N.ffn.* (atom FFN) '
                                'but NOT encoder.layers.N.pair_ffn.* (pair FFN), '
                                'because "pair_ffn" contains "_ffn" not ".ffn".')
    
    # Training hyperparameters
    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument('--lr', type=float, default=1e-4,
                            help='Learning rate for optimization')
    train_group.add_argument('--batch_size', type=int, default=8,
                            help='Batch size (reduced for memory efficiency)')
    train_group.add_argument('--epochs', type=int, default=100,
                            help='Number of training epochs')
    train_group.add_argument('--dropout', type=float, default=0.1,
                            help='Dropout rate for regularization')
    train_group.add_argument('--warmup', type=float, default=0.06,
                            help='Warmup ratio for learning rate scheduling')
    train_group.add_argument('--seed', type=int, default=0,
                            help='Random seed for reproducibility')
    train_group.add_argument('--conf_size', type=int, default=16,
                            help='Number of molecular conformers')
    train_group.add_argument('--update_freq', type=int, default=2,
                            help='Gradient accumulation steps')
    train_group.add_argument('--num_workers', type=int, default=0,
                            help='Number of data loader workers')
    train_group.add_argument('--log_interval', type=int, default=100,
                            help='Logging interval (steps)')
    train_group.add_argument('--patience', type=int, default=10,
                            help='Early stopping patience (epochs)')
    train_group.add_argument('--weight_decay', type=float, default=0.01,
                            help='L2 regularization weight decay for LoRA parameters')
    train_group.add_argument('--clip_norm', type=float, default=1.0,
                            help='Gradient clipping norm')
    
    # Additional parameters
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument('--drop_feat_prob', type=float, default=0.0,
                           help='Feature dropout probability')
    misc_group.add_argument('--use_2d_pos_prob', type=float, default=0.0,
                           help='Probability of using 2D positional encoding')
    misc_group.add_argument('--max_atoms', type=int, default=400,
                           help='Maximum number of atoms per molecule')
    misc_group.add_argument('--valid_subsets', type=str, default='valid',
                           help='Validation subset names')
    misc_group.add_argument('--use_gpu', action='store_true', default=True,
                           help='Enable GPU acceleration')
    misc_group.add_argument('--gpu_id', type=str, default='0',
                           help='GPU device ID')
    misc_group.add_argument('--suppress_warnings', action='store_true', default=True,
                           help='Suppress warning messages')
    misc_group.add_argument('--log_level', type=str, default='INFO',
                           choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                           help='Logging verbosity level')
    
    args = parser.parse_args()
    
    # Configure architecture-specific settings
    architecture_configs = {
        '84M': {'arch': 'unimol2_84M', 'layers': 12},
        '164M': {'arch': 'unimol2_164M', 'layers': 12},
        '310M': {'arch': 'unimol2_310M', 'layers': 18},
        '570M': {'arch': 'unimol2_570M', 'layers': 24},
        '1100M': {'arch': 'unimol2_1100M', 'layers': 36},
    }
    
    if args.arch_name in architecture_configs:
        config = architecture_configs[args.arch_name]
        args.arch = config['arch']
        print(f"Architecture: {args.arch} ({config['layers']} encoder layers)")
    else:
        raise ValueError(f"Unsupported architecture: {args.arch_name}")
    
    return args


def setup_lora_task(user_dir: str) -> None:
    """
    Setup custom LoRA fine-tuning task for Uni-Mol2.
    
    This function creates the necessary task files and imports for LoRA-based
    fine-tuning within the Uni-Mol2 framework.
    
    Args:
        user_dir: Path to Uni-Mol2 user directory where tasks are defined
    """
    tasks_dir = os.path.join(user_dir, 'tasks')
    os.makedirs(tasks_dir, exist_ok=True)
    
    # Copy the task-local LoRA implementation into Uni-Mol2's task package.
    import shutil
    lora_modules_src = str(SCRIPT_DIR / 'lora_modules.py')
    lora_modules_dst = os.path.join(tasks_dir, 'lora_modules.py')
    
    if os.path.exists(lora_modules_src):
        shutil.copy2(lora_modules_src, lora_modules_dst)
        print(f"✓ Copied lora_modules.py to: {lora_modules_dst}")
    else:
        raise FileNotFoundError(
            f"Cannot locate task-local lora_modules.py: {lora_modules_src}")
    
    # Create LoRA fine-tuning task
    lora_task_content = '''# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unicore.tasks import register_task
from ..tasks.unimol_finetune import UniMolFinetuneTask

# Import LoRA modules from same directory
from .lora_modules import apply_lora_to_model, print_trainable_parameters


@register_task('mol_finetune_lora')
class UniMolLoRAFinetuneTask(UniMolFinetuneTask):
    """
    LoRA-based fine-tuning task for Uni-Mol2.
    
    This task extends the standard fine-tuning task to support LoRA
    (Low-Rank Adaptation) for parameter-efficient training.
    """
    
    @staticmethod
    def add_args(parser):
        """Add LoRA-specific command line arguments."""
        # Add parent task arguments
        UniMolFinetuneTask.add_args(parser)
        
        # Add LoRA-specific arguments
        parser.add_argument('--lora-rank', type=int, default=8,
                          help='LoRA rank parameter (lower = less memory)')
        parser.add_argument('--lora-alpha', type=float, default=16.0,
                          help='LoRA scaling parameter')
        parser.add_argument('--lora-dropout', type=float, default=0.1,
                          help='LoRA dropout probability')
        parser.add_argument('--lora-target-modules', type=str,
                          default='linear_q,linear_k,linear_v,linear_o',
                          help='Comma-separated list of modules for LoRA adaptation')
        parser.add_argument('--lora-include-paths', type=str,
                          default='self_attn.',
                          help='Apply LoRA only to paths containing these strings')
        parser.add_argument('--pretrained-model-path', type=str, default=None,
                          help='Path to pretrained model (loaded before LoRA application)')
    
    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the LoRA fine-tuning task."""
        return cls(args, **kwargs)
    
    def build_model(self, args):
        """
        Build model with LoRA adaptations.
        
        This method:
        1. Builds the base Uni-Mol2 model
        2. Loads pretrained weights
        3. Applies LoRA adaptations to specified modules
        4. Freezes base parameters and unfreezes task-specific heads
        """
        import torch
        
        # Build base model using parent class
        model = super().build_model(args)
        
        # Step 1: Load pretrained weights before applying LoRA
        pretrain_path = getattr(args, 'pretrained_model_path', None)
        if pretrain_path:
            print("\\n" + "="*60)
            print("LOADING PRETRAINED WEIGHTS")
            print("="*60)
            print(f"Loading from: {pretrain_path}")
            
            checkpoint = torch.load(pretrain_path, map_location='cpu')
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            
            # Load with strict=False to handle classification head mismatches
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            print(f"✓ Pretrained weights loaded successfully")
            print(f"  Missing keys: {len(missing_keys)} (expected: classification head)")
            print(f"  Unexpected keys: {len(unexpected_keys)} (expected: pretraining head)")
            
            # Display examples for debugging
            if missing_keys:
                print(f"  Example missing: {missing_keys[:3]}")
            if unexpected_keys:
                print(f"  Example unexpected: {unexpected_keys[:3]}")
        
        # Step 2: Apply LoRA adaptations
        print("\\n" + "="*60)
        print("APPLYING LORA ADAPTATIONS")
        print("="*60)
        
        # Extract LoRA configuration from arguments
        lora_rank = getattr(args, 'lora_rank', 8)
        lora_alpha = getattr(args, 'lora_alpha', 16.0)
        lora_dropout = getattr(args, 'lora_dropout', 0.1)
        lora_targets = getattr(args, 'lora_target_modules', 'linear_q,linear_k,linear_v,linear_o')
        lora_includes = getattr(args, 'lora_include_paths', 'self_attn.')
        
        target_modules = [t.strip() for t in lora_targets.split(',')]
        include_paths = [p.strip() for p in lora_includes.split(',')] if lora_includes else None
        
        print(f"LoRA Configuration:")
        print(f"  Rank: {lora_rank}")
        print(f"  Alpha: {lora_alpha}")
        print(f"  Dropout: {lora_dropout}")
        print(f"  Target modules: {target_modules}")
        print(f"  Include paths: {include_paths}")
        print(f"  Expected adaptations (Uni-Mol2-84M): 48 (12 layers × 4 projections)")
        print()
        
        # Step 3: Freeze all model parameters
        print("Freezing all model parameters...")
        for param in model.parameters():
            param.requires_grad = False
        
        # Step 4: Apply LoRA to encoder
        if hasattr(model, 'encoder'):
            model.encoder = apply_lora_to_model(
                model.encoder,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=target_modules,
                include_paths=include_paths,
            )
        
        # Step 5: Unfreeze classification head for task-specific training
        print("\\nUnfreezing classification head...")
        if hasattr(model, 'classification_heads'):
            for name, param in model.classification_heads.named_parameters():
                param.requires_grad = True
                print(f"  Unfroze: {name}")
        
        # Display final parameter statistics
        print_trainable_parameters(model)
        
        return model
'''
    
    # Write LoRA task file
    lora_task_file = os.path.join(tasks_dir, 'mol_finetune_lora.py')
    with open(lora_task_file, 'w', encoding='utf-8') as f:
        f.write(lora_task_content)
    
    # Update __init__.py to include LoRA task
    init_file = os.path.join(tasks_dir, '__init__.py')
    
    # Read existing content or create empty file
    if os.path.exists(init_file):
        with open(init_file, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        content = ''
    
    # Add LoRA task import if not already present
    if 'mol_finetune_lora' not in content:
        with open(init_file, 'a', encoding='utf-8') as f:
            if content and not content.endswith('\\n'):
                f.write('\\n')
            f.write('from .mol_finetune_lora import *\\n')
    
    print(f"LoRA task created: {lora_task_file}")


def run_lora_training(args) -> None:
    """
    Execute LoRA fine-tuning with comprehensive validation and monitoring.
    
    This function performs pre-flight checks, configures the environment,
    and launches the training process with proper logging.
    
    Args:
        args: Parsed command line arguments containing training configuration
    """
    import platform
    import warnings
    import logging
    
    is_windows = platform.system().lower() == 'windows'
    
    # Configure logging system
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level, 
        format='%(asctime)s | %(levelname)s | %(message)s'
    )
    
    # Suppress warnings if requested
    if args.suppress_warnings:
        warnings.filterwarnings('ignore')
        logging.getLogger('unicore.trainer').setLevel(logging.CRITICAL)
        logging.getLogger('unicore').setLevel(logging.ERROR)
        logging.getLogger('torch.cuda').setLevel(logging.ERROR)
    
    # Pre-flight validation checks
    print("\\n" + "="*60)
    print("PRE-FLIGHT VALIDATION")
    print("="*60)
    
    # Check 1: Training script availability
    train_script = f"{args.unicore_dir}/unicore_cli/train.py"
    if not os.path.exists(train_script):
        raise FileNotFoundError(f"Training script not found: {train_script}")
    print(f"✓ Training script found: {train_script}")
    
    # Check 2: LoRA modules import capability
    try:
        if str(SCRIPT_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPT_DIR))
        from lora_modules import LoRALayer
        print(f"✓ LoRA modules can be imported successfully")
    except ImportError as e:
        raise ImportError(f"Cannot import lora_modules: {e}. "
                         f"Ensure lora_modules.py is in: {SCRIPT_DIR}")
    
    # Check 3: Data directory validation
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path not found: {args.data_path}")
    print(f"✓ Data path exists: {args.data_path}")
    
    # Check 4: Pretrained weights validation
    if not os.path.exists(args.weight_path):
        raise FileNotFoundError(f"Pretrained weights not found: {args.weight_path}")
    print(f"✓ Pretrained weights found: {args.weight_path}")
    
    # Check 5: User directory validation
    if not os.path.exists(args.user_dir):
        raise FileNotFoundError(f"User directory not found: {args.user_dir}")
    print(f"✓ User directory exists: {args.user_dir}")
    
    print("="*60 + "\\n")
    
    # Environment configuration
    if not is_windows:
        current_pythonpath = os.environ.get('PYTHONPATH', '')
        if current_pythonpath:
            os.environ['PYTHONPATH'] = f"{args.unicore_dir}:{current_pythonpath}"
        else:
            os.environ['PYTHONPATH'] = args.unicore_dir
    
    if args.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Construct training command arguments
    train_args = [
        args.data_path,
        '--user-dir', args.user_dir,
        '--task', 'mol_finetune_lora',
        '--arch', args.arch,
        '--loss', args.loss_func,
        '--train-subset', 'train',
        '--valid-subset', args.valid_subsets,
        '--task-name', args.task_name,
        '--classification-head-name', args.task_name,
        '--num-classes', str(args.num_classes),
        '--conf-size', str(args.conf_size),
        '--reg',
        '--drop-feat-prob', str(args.drop_feat_prob),
        '--use-2d-pos-prob', str(args.use_2d_pos_prob),
        '--max-atoms', str(args.max_atoms),
        '--num-workers', str(0 if is_windows else args.num_workers),
        '--ddp-backend', 'c10d',
        '--optimizer', 'adam',
        '--adam-betas', '0.9,0.999',
        '--adam-eps', '1e-8',
        '--weight-decay', str(args.weight_decay),
        '--clip-norm', str(args.clip_norm),
        '--lr-scheduler', 'polynomial_decay',
        '--warmup-ratio', str(args.warmup),
        '--lr', str(args.lr),
        '--max-epoch', str(args.epochs),
        '--seed', str(args.seed),
        '--pooler-dropout', str(args.dropout),
        '--batch-size', str(args.batch_size),
        '--update-freq', str(args.update_freq),
        '--required-batch-size-multiple', '8',
        '--log-interval', str(args.log_interval),
        '--log-format', 'simple',
        '--validate-interval', '1',
        '--save-dir', args.save_dir,
        '--keep-last-epochs', '5',
        '--best-checkpoint-metric', 'valid_agg_auc',
        '--patience', str(args.patience),
        '--fp16',
        '--fp16-init-scale', '4',
        '--fp16-scale-window', '256',
        '--maximize-best-checkpoint-metric',
        '--pretrained-model-path', args.weight_path,
        # LoRA-specific arguments
        '--lora-rank', str(args.lora_rank),
        '--lora-alpha', str(args.lora_alpha),
        '--lora-dropout', str(args.lora_dropout),
        '--lora-target-modules', args.lora_target_modules,
        '--lora-include-paths', args.lora_include_paths,
    ]
    
    train_script = f"{args.unicore_dir}/unicore_cli/train.py"
    cmd = ['python', train_script] + train_args
    
    log_file = os.path.join(args.save_dir, "training.log")
    
    # Display training configuration
    print(f"{'='*60}")
    print(f"STARTING LORA FINE-TUNING")
    print(f"{'='*60}")
    print(f"Data: {args.data_path}")
    print(f"Save: {args.save_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size: {args.batch_size} (effective: {args.batch_size * args.update_freq})")
    print(f"LoRA Rank: {args.lora_rank}")
    print(f"LoRA Alpha: {args.lora_alpha}")
    print(f"Log File: {log_file}")
    print(f"{'='*60}\\n")
    
    # Execute training with logging
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output to both console and log file
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                f.flush()
            
            process.wait()
            
            # Report training completion status
            if process.returncode == 0:
                print(f"\\nLoRA fine-tuning completed successfully")
                
                best_checkpoint = os.path.join(args.save_dir, 'checkpoint_best.pt')
                if os.path.exists(best_checkpoint):
                    size_mb = os.path.getsize(best_checkpoint) / (1024 * 1024)
                    print(f"Best checkpoint: {best_checkpoint} ({size_mb:.1f} MB)")
            else:
                print(f"\\nTraining failed with exit code: {process.returncode}")
                
    except Exception as e:
        error_msg = f"\\nExecution error: {e}"
        print(error_msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(error_msg + "\\n")


def main():
    """
    Main entry point for Uni-Mol2 LoRA fine-tuning.
    
    This function orchestrates the complete LoRA fine-tuning workflow:
    1. Parse command line arguments
    2. Display configuration summary
    3. Setup LoRA task integration
    4. Execute training process
    """
    args = get_args()
    
    # Display configuration summary
    print(f"\\n{'='*60}")
    print(f"UNI-MOL2 LORA FINE-TUNING")
    print(f"{'='*60}")
    print(f"Task: {args.task_name}")
    print(f"Architecture: {args.arch}")
    print(f"Save Directory: {args.save_dir}")
    print(f"LoRA Rank: {args.lora_rank}")
    print(f"LoRA Alpha: {args.lora_alpha}")
    print(f"{'='*60}\\n")
    
    # Setup LoRA task integration
    setup_lora_task(args.user_dir)
    
    # Execute training
    run_lora_training(args)


if __name__ == '__main__':
    main()
