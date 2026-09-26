#!/usr/bin/env python3
"""Merge a SIDER LoRA checkpoint into a standard Uni-Mol2 checkpoint.

Training configuration (unimol2_lora_finetune_runner.py defaults):
  --lora_rank              16
  --lora_alpha             16.0   (scaling = alpha/rank = 1.0)
  --lora_target_modules    linear_q,linear_k,linear_v,linear_o,linear_1,linear_2
  --lora_include_paths     self_attn.,.ffn.

Key format after merging:
  encoder.layers.N.self_attn.linear_q.linear.weight
    -> encoder.layers.N.self_attn.linear_q.weight
  encoder.layers.N.ffn.linear_1.linear.weight
    -> encoder.layers.N.ffn.linear_1.weight

All LoRA A and B keys are consumed during the merge.

Usage:
  python merge.py
  python merge.py --lora-checkpoint checkpoint_best.pt --output merged.pt
"""

import argparse
import os
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
SIDER_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SIDER_ROOT.parent
DEPENDENCY_ROOT = REPO_ROOT.parent
UNICORE_DIR = DEPENDENCY_ROOT / 'Uni-Core'
UNIMOL2_DIR = DEPENDENCY_ROOT / 'Uni-Mol' / 'unimol2'

for dependency_path in (SCRIPT_DIR, UNICORE_DIR, UNIMOL2_DIR):
    if str(dependency_path) not in sys.path:
        sys.path.insert(0, str(dependency_path))

from lora_modules import LoRALinear  # noqa: E402,F401


DEFAULT_LORA_CHECKPOINT = (
    SCRIPT_DIR / 'result' / 'lora_finetune_sider' / 'checkpoint_best.pt'
)
DEFAULT_OUTPUT = SIDER_ROOT / 'checkpoint' / 'merged_checkpoint.pt'
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 16.0


def merge_lora_weights_in_state_dict(
        state_dict, lora_rank=DEFAULT_LORA_RANK,
        lora_alpha=DEFAULT_LORA_ALPHA):
    """Absorb LoRA adaptations into base weights at state-dict level.

    The merge follows W_merged = W_base + (B @ A) * (alpha / rank).
    Non-LoRA parameters are copied without modification.
    """
    merged_state_dict = {}

    # Group LoRA tensors by the wrapped linear-module prefix.
    lora_modules = {}
    for key in state_dict:
        if '.lora.' not in key:
            continue
        module_prefix, lora_parameter = key.split('.lora.', maxsplit=1)
        lora_modules.setdefault(module_prefix, {})[lora_parameter] = (
            state_dict[key]
        )

    print(f"LoRA modules detected: {len(lora_modules)}")
    if not lora_modules:
        raise ValueError(
            "The checkpoint contains no '.lora.' parameters to merge.")

    # Validate the command-line rank against the tensors stored in the model.
    first_module = next(iter(lora_modules.values()))
    detected_rank = (
        first_module['lora_A'].shape[0]
        if 'lora_A' in first_module else lora_rank
    )
    if detected_rank != lora_rank:
        print(
            f"[WARNING] rank mismatch: arg={lora_rank}, "
            f"detected={detected_rank}; using detected rank"
        )
        lora_rank = detected_rank
    print(
        f"LoRA config: rank={lora_rank}, alpha={lora_alpha}, "
        f"scaling={lora_alpha / lora_rank:.4f}"
    )

    # Merge each wrapped linear layer into the original Uni-Mol2 key format.
    merged_count = 0
    for module_prefix, lora_parameters in lora_modules.items():
        base_key = f"{module_prefix}.linear.weight"
        base_bias_key = f"{module_prefix}.linear.bias"

        if base_key not in state_dict:
            print(f"[WARNING] base weight not found: {base_key}")
            continue
        if ('lora_A' not in lora_parameters or
                'lora_B' not in lora_parameters):
            print(f"[WARNING] {module_prefix}: missing lora_A or lora_B")
            continue

        lora_a = lora_parameters['lora_A']
        lora_b = lora_parameters['lora_B']
        scaling = lora_alpha / lora_a.shape[0]
        merged_weight = (
            state_dict[base_key].clone() + (lora_b @ lora_a) * scaling
        )
        merged_state_dict[f"{module_prefix}.weight"] = merged_weight

        if base_bias_key in state_dict:
            merged_state_dict[f"{module_prefix}.bias"] = (
                state_dict[base_bias_key]
            )
        merged_count += 1

    print(f"Modules merged: {merged_count}")

    # Exclude only tensors consumed by a LoRA wrapper. All unrelated
    # parameters, including ordinary linear weights, pass through.
    consumed_keys = set()
    for module_prefix in lora_modules:
        consumed_keys.update({
            f"{module_prefix}.lora.lora_A",
            f"{module_prefix}.lora.lora_B",
            f"{module_prefix}.linear.weight",
            f"{module_prefix}.linear.bias",
        })

    for key, value in state_dict.items():
        if key not in consumed_keys:
            merged_state_dict[key] = value

    return merged_state_dict


def main():
    """Parse paths, merge LoRA tensors, and save a standard checkpoint."""
    parser = argparse.ArgumentParser(
        description=(
            'Merge a SIDER LoRA checkpoint into a standard Uni-Mol2 '
            'checkpoint.'
        )
    )
    parser.add_argument(
        '--lora-checkpoint', type=str, default=str(DEFAULT_LORA_CHECKPOINT),
        help='Path to the LoRA fine-tuned checkpoint.'
    )
    parser.add_argument(
        '--output', type=str, default=str(DEFAULT_OUTPUT),
        help='Output path for the merged checkpoint.'
    )
    parser.add_argument(
        '--lora-rank', type=int, default=DEFAULT_LORA_RANK,
        help='LoRA rank used during training.'
    )
    parser.add_argument(
        '--lora-alpha', type=float, default=DEFAULT_LORA_ALPHA,
        help='LoRA alpha used during training.'
    )
    args = parser.parse_args()

    if args.lora_rank <= 0:
        parser.error('--lora-rank must be positive.')

    print(f"Input : {args.lora_checkpoint}")
    print(f"Output: {args.output}")
    print(
        f"LoRA  : rank={args.lora_rank}, alpha={args.lora_alpha}, "
        f"scaling={args.lora_alpha / args.lora_rank:.4f}"
    )

    checkpoint = torch.load(args.lora_checkpoint, map_location='cpu')
    state_dict = (
        checkpoint['model']
        if isinstance(checkpoint, dict) and 'model' in checkpoint
        else checkpoint
    )
    print(f"Keys before merge: {len(state_dict)}")

    merged_state_dict = merge_lora_weights_in_state_dict(
        state_dict, args.lora_rank, args.lora_alpha
    )
    print(f"Keys after merge : {len(merged_state_dict)}")

    # Retain model metadata while intentionally omitting optimizer tensors.
    output_checkpoint = {'model': merged_state_dict}
    if isinstance(checkpoint, dict):
        for key in ('args', 'optimizer_history', 'extra_state'):
            if key in checkpoint:
                output_checkpoint[key] = checkpoint[key]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, output_path)

    input_size = os.path.getsize(args.lora_checkpoint) / (1024 * 1024)
    output_size = output_path.stat().st_size / (1024 * 1024)
    print(
        f"Saved: {output_path} "
        f"({input_size:.1f} MB -> {output_size:.1f} MB)"
    )


if __name__ == '__main__':
    main()
