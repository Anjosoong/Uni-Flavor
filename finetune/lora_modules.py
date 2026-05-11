#!/usr/bin/env python3
"""
LoRA (Low-Rank Adaptation) Implementation for Uni-Mol2

This module implements LoRA for parameter-efficient fine-tuning of molecular 
representation models. LoRA reduces memory usage by adding trainable low-rank 
matrices to frozen pretrained weights.

"""

import os
import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer implementation.
    
    LoRA decomposes weight updates into two low-rank matrices A and B:
    ΔW = B @ A, where A ∈ R^(r×d_in) and B ∈ R^(d_out×r)
    
    The forward pass computes: h = W₀x + ΔWx = W₀x + B(Ax)
    where W₀ is the frozen pretrained weight matrix.
    
    Args:
        in_features (int): Input dimension
        out_features (int): Output dimension  
        rank (int): Rank of decomposition (r). Lower rank = fewer parameters
        alpha (float): Scaling hyperparameter, typically 2×rank
        dropout (float): Dropout probability applied to LoRA path
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices: A (down-projection) and B (up-projection)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Initialize matrices following LoRA paper recommendations:
        # A: kaiming_uniform (non-zero gradient at init)
        # B: zeros (ensures adaptation output = 0 at initialization)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA adaptation.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            LoRA adaptation of shape (..., out_features)
        """
        x = self.dropout(x)
        
        # Compute low-rank adaptation: B @ (A @ x)
        result = F.linear(x, self.lora_A)      # (..., rank)
        result = F.linear(result, self.lora_B)  # (..., out_features)
        
        # Apply scaling factor
        return result * self.scaling


class LoRALinear(nn.Module):
    """
    Linear layer augmented with LoRA adaptation.
    
    This module combines a frozen pretrained linear layer with trainable
    LoRA parameters, enabling parameter-efficient fine-tuning.
    
    Args:
        linear (nn.Linear): Pretrained linear layer to be adapted
        rank (int): LoRA rank parameter
        alpha (float): LoRA scaling parameter
        dropout (float): Dropout probability for LoRA path
    """
    
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Freeze original pretrained weights
        self.linear = linear
        for param in self.linear.parameters():
            param.requires_grad = False
        
        # Add trainable LoRA adaptation
        self.lora = LoRALayer(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining frozen weights with LoRA adaptation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor: W₀x + ΔWx
        """
        # Frozen pretrained path
        base_output = self.linear(x)
        
        # Trainable LoRA adaptation
        lora_output = self.lora(x)
        
        return base_output + lora_output


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    include_paths: Optional[List[str]] = None,
    exclude_paths: Optional[List[str]] = None,
) -> nn.Module:
    """
    Apply LoRA adaptation to specified modules in a neural network.
    
    This function identifies target linear layers and replaces them with
    LoRA-augmented versions for parameter-efficient fine-tuning.
    
    Args:
        model: Neural network model to adapt
        rank: LoRA rank (lower values = fewer parameters)
        alpha: LoRA scaling factor (typically 2×rank)
        dropout: Dropout probability for LoRA layers
        target_modules: List of module names to target (default: Uni-Mol2 attention)
        include_paths: Only apply LoRA to modules containing these path substrings
        exclude_paths: Skip modules containing these path substrings
    
    Returns:
        Modified model with LoRA adaptations applied
    """
    
    # Default targeting for Uni-Mol2 attention mechanisms
    if target_modules is None:
        target_modules = ['linear_q', 'linear_k', 'linear_v', 'linear_o']
    
    # Collect modules for replacement (avoid modification during iteration)
    modules_to_replace = []
    
    for module_path, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        
        # Apply path filtering
        if include_paths and not any(path in module_path for path in include_paths):
            continue
        
        if exclude_paths and any(path in module_path for path in exclude_paths):
            continue
        
        # Check if module should receive LoRA adaptation
        module_name = module_path.split('.')[-1]
        should_adapt = any(
            target == module_name or target in module_path 
            for target in target_modules
        )
        
        if should_adapt:
            parent_path = '.'.join(module_path.split('.')[:-1])
            parent_module = model.get_submodule(parent_path) if parent_path else model
            modules_to_replace.append((module_path, parent_module, module_name, module))
    
    # Apply LoRA adaptations
    num_adaptations = 0
    for full_path, parent, attr_name, linear_module in modules_to_replace:
        # Skip if already adapted
        if isinstance(getattr(parent, attr_name), LoRALinear):
            continue
        
        # Create LoRA-adapted version
        lora_linear = LoRALinear(
            linear=linear_module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        
        setattr(parent, attr_name, lora_linear)
        num_adaptations += 1
    
    print(f"LoRA adaptations applied: {num_adaptations}")
    
    # Sanity check for Uni-Mol2-84M attention-only configuration
    if 'linear_q' in target_modules and include_paths and 'self_attn' in str(include_paths):
        expected_adaptations = 48  # 12 layers x 4 projections (q,k,v,o)
        if num_adaptations != expected_adaptations:
            print(f"[WARNING] expected ~{expected_adaptations} adaptations for Uni-Mol2-84M, "
                  f"applied {num_adaptations}. Verify target_modules and include_paths.")
    
    return model


def count_parameters(model: nn.Module) -> dict:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: Neural network model to analyze
        
    Returns:
        Dictionary containing parameter statistics:
        - total: Total number of parameters
        - trainable: Number of trainable parameters  
        - frozen: Number of frozen parameters
        - trainable_percent: Percentage of parameters that are trainable
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params,
        'trainable_percent': 100 * trainable_params / total_params if total_params > 0 else 0,
    }


def print_trainable_parameters(model: nn.Module) -> None:
    """
    Display parameter statistics for a model.
    
    Args:
        model: Neural network model to analyze
    """
    stats = count_parameters(model)
    
    print("\n" + "="*60)
    print("PARAMETER STATISTICS")
    print("="*60)
    print(f"Total parameters:      {stats['total']:,}")
    print(f"Trainable parameters:  {stats['trainable']:,}")
    print(f"Frozen parameters:     {stats['frozen']:,}")
    print(f"Trainable percentage:  {stats['trainable_percent']:.2f}%")
    print("="*60 + "\n")


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge LoRA adaptations into base model weights for inference.
    
    This function combines the frozen pretrained weights with learned LoRA
    adaptations to create a single weight matrix: W' = W₀ + BA
    
    After merging, the model has no computational overhead and can be
    used for inference without LoRA-specific code.
    
    Args:
        model: Model containing LoRA adaptations
        
    Returns:
        Model with merged weights (LoRA adaptations integrated)
    """
    modules_to_merge = []
    
    # Collect LoRA modules for merging
    for module_path, module in model.named_modules():
        if isinstance(module, LoRALinear):
            parent_path = '.'.join(module_path.split('.')[:-1])
            attr_name = module_path.split('.')[-1]
            parent_module = model.get_submodule(parent_path) if parent_path else model
            modules_to_merge.append((module_path, parent_module, attr_name, module))
    
    # Perform weight merging
    for full_path, parent, attr_name, lora_module in modules_to_merge:
        with torch.no_grad():
            # Compute LoRA weight update: ΔW = B @ A
            lora_delta = lora_module.lora.lora_B @ lora_module.lora.lora_A
            lora_delta = lora_delta * lora_module.lora.scaling
            
            # Merge into base weights: W' = W₀ + ΔW
            lora_module.linear.weight.data.add_(lora_delta)
        
        # Replace LoRALinear with merged standard Linear layer
        setattr(parent, attr_name, lora_module.linear)
    
    return model


def save_lora_weights(model: nn.Module, path: str) -> None:
    """
    Save LoRA adaptation weights to disk.
    
    This saves only the LoRA parameters (not the full model), resulting
    in much smaller checkpoint files.
    
    Args:
        model: Model containing LoRA adaptations
        path: File path for saving LoRA weights
    """
    lora_state_dict = {}
    lora_config = {}
    
    for module_path, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Save LoRA matrices (detached and moved to CPU)
            lora_state_dict[f"{module_path}.lora_A"] = module.lora.lora_A.detach().cpu()
            lora_state_dict[f"{module_path}.lora_B"] = module.lora.lora_B.detach().cpu()
            
            # Save configuration (only once)
            if 'rank' not in lora_config:
                lora_config.update({
                    'rank': module.lora.rank,
                    'alpha': module.lora.alpha,
                    'scaling': module.lora.scaling
                })
    
    # Create checkpoint with weights and configuration
    checkpoint = {
        'lora_state_dict': lora_state_dict,
        'lora_config': lora_config,
    }
    
    torch.save(checkpoint, path)
    
    # Report save statistics
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"LoRA weights saved: {path}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Configuration: rank={lora_config.get('rank')}, alpha={lora_config.get('alpha')}")


def load_lora_weights(model: nn.Module, path: str) -> None:
    """
    Load LoRA adaptation weights from disk.
    
    Args:
        model: Model with LoRA structure (must match saved weights)
        path: File path containing LoRA weights
    """
    checkpoint = torch.load(path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'lora_state_dict' in checkpoint:
        lora_state_dict = checkpoint['lora_state_dict']
        lora_config = checkpoint.get('lora_config', {})
        print(f"Loading LoRA weights with configuration: {lora_config}")
    else:
        lora_state_dict = checkpoint
    
    # Load weights into model
    for module_path, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_a_key = f"{module_path}.lora_A"
            lora_b_key = f"{module_path}.lora_B"
            
            if lora_a_key in lora_state_dict:
                module.lora.lora_A.data.copy_(lora_state_dict[lora_a_key])
                module.lora.lora_B.data.copy_(lora_state_dict[lora_b_key])
    
    print(f"LoRA weights loaded: {path}")


if __name__ == "__main__":
    """
    Example usage demonstrating LoRA application to a simple model.
    
    This example shows how to:
    1. Create a model with attention-like linear layers
    2. Apply LoRA adaptations to specific modules
    3. Compare parameter counts before and after LoRA
    """
    
    class ExampleModel(nn.Module):
        """Simple model mimicking attention mechanism structure."""
        
        def __init__(self, hidden_size: int = 768, intermediate_size: int = 3072):
            super().__init__()
            # Attention projections (typical LoRA targets)
            self.q_proj = nn.Linear(hidden_size, hidden_size)
            self.k_proj = nn.Linear(hidden_size, hidden_size)
            self.v_proj = nn.Linear(hidden_size, hidden_size)
            self.out_proj = nn.Linear(hidden_size, hidden_size)
            
            # Feed-forward layers (usually not targeted by LoRA)
            self.fc1 = nn.Linear(hidden_size, intermediate_size)
            self.fc2 = nn.Linear(intermediate_size, hidden_size)
        
        def forward(self, x):
            return x
    
    # Demonstrate LoRA application
    print("LoRA Module Example")
    print("=" * 50)
    
    # Create example model
    model = ExampleModel()
    print("Original model parameter statistics:")
    print_trainable_parameters(model)
    
    # Apply LoRA to attention projections only
    model = apply_lora_to_model(
        model,
        rank=8,
        alpha=16.0,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj']
    )
    
    print("After applying LoRA:")
    print_trainable_parameters(model)
