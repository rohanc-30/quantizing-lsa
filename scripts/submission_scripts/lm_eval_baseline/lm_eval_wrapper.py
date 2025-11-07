#!/usr/bin/env python
"""
Wrapper script to ensure CUDA is initialized before fla imports.
This is critical because fla detects the backend (CUDA vs CPU) at import time.
"""

import sys
import os

# Initialize CUDA and Triton BEFORE any fla imports
import torch
import triton

# Force CUDA initialization
if torch.cuda.is_available():
    # Create a tensor on CUDA to ensure CUDA context is initialized
    _ = torch.zeros(1).cuda()
    print(f"[CUDA init] CUDA initialized successfully on device {torch.cuda.current_device()}", file=sys.stderr)
else:
    print("[CUDA init] WARNING: CUDA is not available!", file=sys.stderr)

# Now it's safe to import fla and run lm_eval
# The fla import will happen when the model is loaded, but CUDA is already initialized

# Run lm_eval by calling it as a module with all original arguments
if __name__ == "__main__":
    # Replace this script's name with 'lm_eval' in argv so argparse works correctly
    sys.argv[0] = "lm_eval"
    # Now import and run lm_eval's main entry point
    from lm_eval.__main__ import cli_evaluate
    cli_evaluate()

