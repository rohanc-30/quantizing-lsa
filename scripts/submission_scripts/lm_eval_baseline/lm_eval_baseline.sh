#!/bin/bash

source /etc/profile.d/modules.sh

module load cudnn/9.10.2
module load cuda/12.9

source /home/rcherukuri/quantizing-lsa/.venv/bin/activate


nvcc --version
/usr/bin/nvidia-smi


export HF_HOME=/home/rcherukuri/hf
export TRANSFORMERS_CACHE=/home/rcherukuri/hf/transformers
export HF_DATASETS_CACHE=/home/rcherukuri/hf/datasets
export HF_HUB_CACHE=/home/rcherukuri/hf/hub

export HOME=/home/rcherukuri
export PATH="/lustre/home/rcherukuri/quantizing-lsa/.venv/bin:/home/rcherukuri/.local/bin:/home/rcherukuri/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/usr/lib/jvm/java-8-oracle/bin:/usr/lib/jvm/java-8-oracle/db/bin:/usr/lib/jvm/java-8-oracle/jre/bin:${PATH}"

which python
which nvcc
which ptxas
which basename
which nvidia-smi
which triton

# Ensure CUDA is visible and initialized before any fla imports
# This is critical because fla detects the backend at import time
python - <<'EOF'
import torch
import os

# Print CUDA info first
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Torch version:", torch.__version__)
print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))

# Force CUDA initialization by accessing a device
if torch.cuda.is_available():
    _ = torch.zeros(1).cuda()
    print("CUDA device initialized successfully")
    print("Current CUDA device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("WARNING: CUDA is not available!")
EOF

# Now check fla backend detection
python - <<'EOF'
import torch
# Import triton first to ensure it initializes with CUDA context
import triton
# Force CUDA access before fla import
if torch.cuda.is_available():
    _ = torch.zeros(1).cuda()

import fla
print("CUDA available:", torch.cuda.is_available())
print("Triton backend:", fla.utils.device_torch_lib)
print("fla compiled with CUDA?", hasattr(fla.utils, "device_torch_lib") and fla.utils.device_torch_lib is torch.cuda)
print("fla.utils.custom_device_ctx:")
import inspect
print(inspect.getsource(fla.utils.custom_device_ctx))
EOF

python - <<'PY'
import torch, triton, os
print("Torch CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
print("Torch version:", torch.__version__)
print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
import triton.runtime.jit as jit
print("Triton available:", hasattr(jit, "JITFunction"))
# Check if Triton can see CUDA - this needs to happen before fla imports
try:
    # Try to access Triton's CUDA backend
    if hasattr(triton, 'runtime') and hasattr(triton.runtime, 'driver'):
        print("Triton runtime driver:", triton.runtime.driver)
    # Try to check if CUDA is available to Triton
    if torch.cuda.is_available():
        # Force Triton to initialize with CUDA by accessing a CUDA device
        _ = torch.zeros(1).cuda()
        print("Successfully moved tensor to CUDA - Triton should see CUDA")
except Exception as e:
    print(f"Warning: Could not verify Triton CUDA setup: {e}")
PY

# Use wrapper script to ensure CUDA is initialized before fla imports
python /home/rcherukuri/quantizing-lsa/scripts/submission_scripts/lm_eval_baseline/lm_eval_wrapper.py \
  --model hf \
  --model_args "pretrained=$1,tokenizer=$1,dtype=float32,trust_remote_code=True" \
  --tasks wikitext,piqa,lambada_openai \
  --batch_size 8 \
  --limit 500 \
  --output_path $2
