#!/usr/bin/env python3
"""
Merge a LoRA fine-tuned checkpoint into a standard Uni-Mol2 checkpoint.

Training configuration (unimol2_lora_finetune_runner.py defaults):
  --lora_rank              16
  --lora_alpha             16.0   (scaling = alpha/rank = 1.0)
  --lora_target_modules    linear_q,linear_k,linear_v,linear_o,linear_1,linear_2
  --lora_include_paths     self_attn.,.ffn.

Key format after merging:
  encoder.layers.N.self_attn.linear_q.linear.weight  ->  encoder.layers.N.self_attn.linear_q.weight
  encoder.layers.N.ffn.linear_1.linear.weight        ->  encoder.layers.N.ffn.linear_1.weight
  (all .lora.lora_A / .lora.lora_B keys are removed)

Usage:
  python merge.py --lora-checkpoint checkpoint_best.pt --output merged.pt
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, '/Uni-Core')
sys.path.insert(0, '/Uni-Mol/unimol2')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lora_modules import LoRALinear


def merge_lora_weights_in_state_dict(state_dict, lora_rank=16, lora_alpha=16.0):
    """
    Absorb LoRA adaptations into base weights at state_dict level.
    W_merged = W_base + (lora_B @ lora_A) * (alpha / rank)

    Args:
        state_dict: full model state_dict from a LoRA fine-tuned checkpoint
        lora_rank:  LoRA rank used during training (default 16)
        lora_alpha: LoRA alpha used during training (default 16.0, scaling = 1.0)
    """
    
    merged_state_dict = {}

    # Group all LoRA keys by module prefix
    lora_modules = {}
    for key in state_dict.keys():
        if '.lora.' not in key:
            continue
        parts = key.split('.lora.')
        module_prefix = parts[0]
        if module_prefix not in lora_modules:
            lora_modules[module_prefix] = {}
        lora_modules[module_prefix][parts[1]] = state_dict[key]  # 'lora_A' or 'lora_B'

    print(f"LoRA modules detected: {len(lora_modules)}")
    
    # Validate rank against the actual lora_A shape
    first_module = next(iter(lora_modules.values()))
    detected_rank = first_module['lora_A'].shape[0] if 'lora_A' in first_module else lora_rank
    if detected_rank != lora_rank:
        print(f"[WARNING] rank mismatch: arg={lora_rank}, detected={detected_rank}, using detected")
        lora_rank = detected_rank
    print(f"LoRA config: rank={lora_rank}, alpha={lora_alpha}, scaling={lora_alpha/lora_rank:.4f}")
    
    # Merge each LoRA module: W' = W + (B @ A) * scaling
    merged_count = 0
    for module_prefix, lora_params in lora_modules.items():
        base_key      = f"{module_prefix}.linear.weight"
        base_bias_key = f"{module_prefix}.linear.bias"

        if base_key not in state_dict:
            print(f"[WARNING] base weight not found: {base_key}")
            continue
        if 'lora_A' not in lora_params or 'lora_B' not in lora_params:
            print(f"[WARNING] {module_prefix}: missing lora_A or lora_B")
            continue

        lora_A = lora_params['lora_A']
        lora_B = lora_params['lora_B']
        scaling = lora_alpha / lora_A.shape[0]

        merged_weight = state_dict[base_key].clone() + (lora_B @ lora_A) * scaling
        merged_state_dict[f"{module_prefix}.weight"] = merged_weight

        if base_bias_key in state_dict:
            merged_state_dict[f"{module_prefix}.bias"] = state_dict[base_bias_key]

        merged_count += 1

    print(f"modules merged: {merged_count}")

    # Build exact set of consumed keys to avoid accidentally dropping non-LoRA .linear.weight keys
    consumed_keys = set()
    for prefix in lora_modules.keys():
        consumed_keys.update([
            f"{prefix}.lora.lora_A",
            f"{prefix}.lora.lora_B",
            f"{prefix}.linear.weight",
            f"{prefix}.linear.bias",
        ])

    # Pass through all remaining parameters unchanged
    for key, value in state_dict.items():
        if key not in consumed_keys:
            merged_state_dict[key] = value
    
    return merged_state_dict


# Default paths — update as needed
DEFAULT_LORA_CHECKPOINT = '/data_finetune/odor/result/checkpoint_best.pt'
DEFAULT_OUTPUT          = '/data_finetune/odor/result/merged_checkpoint.pt'
DEFAULT_LORA_RANK       = 16     # must match training
DEFAULT_LORA_ALPHA      = 16.0   # scaling = alpha/rank = 1.0


def main():
    parser = argparse.ArgumentParser(
        description='Merge a LoRA fine-tuned checkpoint into a standard Uni-Mol2 checkpoint')
    parser.add_argument('--lora-checkpoint', type=str, default=DEFAULT_LORA_CHECKPOINT,
                        help='path to LoRA fine-tuned checkpoint')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help='output path for merged checkpoint')
    parser.add_argument('--lora-rank', type=int, default=DEFAULT_LORA_RANK,
                        help='LoRA rank used during training (must match)')
    parser.add_argument('--lora-alpha', type=float, default=DEFAULT_LORA_ALPHA,
                        help='LoRA alpha used during training (must match)')
    args = parser.parse_args()
    
    print(f"input : {args.lora_checkpoint}")
    print(f"output: {args.output}")
    print(f"LoRA  : rank={args.lora_rank}, alpha={args.lora_alpha}, scaling={args.lora_alpha/args.lora_rank:.4f}")

    checkpoint = torch.load(args.lora_checkpoint, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    print(f"keys (before): {len(state_dict)}")

    merged_state_dict = merge_lora_weights_in_state_dict(state_dict, args.lora_rank, args.lora_alpha)
    print(f"keys (after) : {len(merged_state_dict)}")

    # Retain checkpoint metadata; drop optimizer state
    output_checkpoint = {'model': merged_state_dict}
    if isinstance(checkpoint, dict):
        for key in ['args', 'optimizer_history', 'extra_state']:
            if key in checkpoint:
                output_checkpoint[key] = checkpoint[key]

    torch.save(output_checkpoint, args.output)

    input_size  = os.path.getsize(args.lora_checkpoint) / (1024 * 1024)
    output_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"saved : {args.output}  ({input_size:.1f} MB -> {output_size:.1f} MB)")


if __name__ == '__main__':
    main()
