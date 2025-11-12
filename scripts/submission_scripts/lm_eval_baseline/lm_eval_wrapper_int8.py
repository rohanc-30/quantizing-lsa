#!/usr/bin/env python
"""
Wrapper script to load INT8 quantized models for lm_eval.
This ensures CUDA is initialized before fla imports and handles quantized model loading.
"""

import sys
import os
import torch

# For INT8 quantized models, we run on CPU (FBGEMM backend), so CUDA is optional
# However, we still try to initialize it gracefully in case fla needs it
try:
    import triton
    # Try to initialize CUDA if available, but don't fail if it's not
    if torch.cuda.is_available():
        try:
            # Create a tensor on CUDA to ensure CUDA context is initialized
            _ = torch.zeros(1).cuda()
            print(f"[CUDA init] CUDA initialized successfully on device {torch.cuda.current_device()}", file=sys.stderr)
        except Exception as e:
            print(f"[CUDA init] CUDA available but initialization failed (this is OK for quantized models): {e}", file=sys.stderr)
    else:
        print("[CUDA init] CUDA is not available (this is OK for quantized models)", file=sys.stderr)
except ImportError:
    print("[CUDA init] Triton not available (this is OK for quantized models)", file=sys.stderr)

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import quantization utilities
from transformers import AutoTokenizer, AutoModelForCausalLM
from quant.mapping_gpt2 import build_qmap_for_gpt2
from quant.pipeline import prepare_model, convert_to_int8
from pathlib import Path

# Save the original from_pretrained BEFORE we monkey-patch it
# This allows load_quantized_model to use it without infinite recursion
_original_from_pretrained = AutoModelForCausalLM.from_pretrained


def load_quantized_model(checkpoint_path, device="cpu"):
    """
    Load a quantized INT8 model from checkpoint.
    
    The checkpoint should contain:
    - pytorch_model_int8_full.pt (full quantized model) - PREFERRED
    - OR pytorch_model_int8.bin (quantized state_dict) - fallback
    - config.json, modeling files, tokenizer files
    
    Note: Quantized models with FBGEMM backend run on CPU.
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path), trust_remote_code=True)
    
    # Try to load the full quantized model first (simplest approach)
    full_model_path = checkpoint_path / "pytorch_model_int8_full.pt"
    if full_model_path.exists():
        print(f"[Loading] Loading full quantized model from {full_model_path}", file=sys.stderr)
        int8_model = torch.load(full_model_path, map_location="cpu", weights_only=False)
        int8_model = int8_model.eval()
        
        # CRITICAL: Quantized models with FBGEMM backend MUST run on CPU
        # Override any device placement attempts - quantized ops only work on CPU
        int8_model = int8_model.to("cpu")
        
        # Register a hook to prevent moving to CUDA
        original_to = int8_model.to
        def force_cpu(device=None, *args, **kwargs):
            print(f"[Warning] Attempted to move quantized model to {device}, forcing CPU instead", file=sys.stderr)
            return original_to("cpu", *args, **kwargs)
        int8_model.to = force_cpu
        
        # Wrap forward method to ensure inputs are on CPU
        # lm_eval may pass CUDA tensors, but quantized ops only work on CPU
        original_forward = int8_model.forward
        def forward_with_cpu_inputs(*args, **kwargs):
            # Move all tensor inputs to CPU
            cpu_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    cpu_args.append(arg.cpu() if arg.is_cuda else arg)
                else:
                    cpu_args.append(arg)
            
            cpu_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    cpu_kwargs[key] = value.cpu() if value.is_cuda else value
                elif isinstance(value, dict):
                    # Handle nested dicts (like attention_mask)
                    cpu_dict = {}
                    for k, v in value.items():
                        if isinstance(v, torch.Tensor):
                            cpu_dict[k] = v.cpu() if v.is_cuda else v
                        else:
                            cpu_dict[k] = v
                    cpu_kwargs[key] = cpu_dict
                else:
                    cpu_kwargs[key] = value
            
            return original_forward(*cpu_args, **cpu_kwargs)
        
        int8_model.forward = forward_with_cpu_inputs
        
        print(f"[Done] Quantized model loaded successfully on CPU (FBGEMM backend)", file=sys.stderr)
        print(f"[Done] Forward method wrapped to automatically move CUDA inputs to CPU", file=sys.stderr)
        return int8_model, tokenizer
    
    # Fallback: if full model doesn't exist, use the old method
    # (This is for backwards compatibility with existing checkpoints)
    print(f"[Loading] Full model not found, using state_dict loading method...", file=sys.stderr)
    raise FileNotFoundError(
        f"Full quantized model not found at {full_model_path}. "
        f"Please rerun ptq_calibrate_gpt.py to generate the full model file. "
        f"This is much simpler than the state_dict approach."
    )


# Monkey-patch AutoModelForCausalLM.from_pretrained to detect INT8 checkpoints
# This allows lm_eval to use the checkpoint path directly
# Note: _original_from_pretrained is already defined above
def _from_pretrained_int8_wrapper(cls_or_self, pretrained_model_name_or_path, *args, **kwargs):
    """
    Custom from_pretrained that detects INT8 checkpoints and loads them properly.
    Works as both classmethod and instance method.
    """
    checkpoint_path = Path(pretrained_model_name_or_path)
    
    # Check if this is an INT8 checkpoint
    int8_weights_path = checkpoint_path / "pytorch_model_int8.bin"
    if int8_weights_path.exists():
        print(f"[INT8] Detected INT8 checkpoint at {checkpoint_path}", file=sys.stderr)
        # Use our custom loader
        # Extract device from kwargs if present, but quantized models run on CPU
        device = kwargs.pop("device", "cpu")
        model, _ = load_quantized_model(checkpoint_path, device=device)
        return model
    else:
        # Fall back to original behavior
        return _original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

# Patch the classmethod - create a classmethod wrapper
import types
AutoModelForCausalLM.from_pretrained = classmethod(_from_pretrained_int8_wrapper)

# Now it's safe to import fla and run lm_eval
# The fla import will happen when the model is loaded, but CUDA is already initialized

# Run lm_eval by calling it as a module with all original arguments
if __name__ == "__main__":
    # Replace this script's name with 'lm_eval' in argv so argparse works correctly
    sys.argv[0] = "lm_eval"
    # Now import and run lm_eval's main entry point
    from lm_eval.__main__ import cli_evaluate
    cli_evaluate()

