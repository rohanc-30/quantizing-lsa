# Quantizing LSA (Linear State-space Attention)

This repository provides tools for quantizing Linear State-space Attention (LSA) models, specifically GLA (Gated Linear Attention) models, using Post-Training Quantization (PTQ) with PyTorch's quantization framework. The project supports both FP32 and INT8 quantized models, with evaluation capabilities using `lm-eval`.

## Overview

The repository implements:
- **Model Export**: Download and save GLA and GPT-2 models as HuggingFace checkpoints
- **Post-Training Quantization (PTQ)**: Calibrate and convert FP32 models to INT8 using FBGEMM backend
- **Model Evaluation**: Evaluate models using `lm-eval` on standard benchmarks (wikitext, piqa, lambada_openai)

The repository assumes jobs will be run on a compute cluster using HTCondor. It contains `.sub` files for you to submit as jobs written as per HTCondor's specs. Terminal commands to run certain functionalities will be given as condor submissions. If you wish to avoid using HTCondor, you can read the corresponding `.sub` and `.sh` files to the job provided to find the `.py` files they invoke, and then run those from the terminal with your own custom arguments.

Furthermore, in order to use GLA, you will need CUDA access. Thus, even if you choose to use the `.py` scripts instead of the `.sub` files with your own arguments, you must ensure your device has CUDA enabled when running anything adjacent to GLA.

## Requirements

- Python >= 3.10
- PyTorch >= 2.9.0
- CUDA 12.9 (for FP32 models)
- Flash Linear Attention (`flash-linear-attention`)
- `lm-eval[hf]>=0.4.9.1`
- Additional dependencies listed in `pyproject.toml`

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd quantizing-lsa
```

2. Install dependencies using `uv` (or your preferred package manager):
```bash
uv sync
```

3. Activate the virtual environment:
```bash
source .venv/bin/activate
```

4. Set up HuggingFace cache directories (necessary on clusters/systems where filelock is disabled):
```bash
export HF_HOME=/path/to/hf
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
```

## Usage

### 1. Export Models

Export pre-trained models from HuggingFace Hub to local checkpoints, enabling for more fine-grained control of model architectures.

#### Export GLA Model
```bash
condor_submit scripts/submission_scripts/export_gla/export_gla.sub
```

This downloads the `fla-hub/GLA-2.7B-100B` model and saves it to `checkpoints/gla_mod`. The job will be submitted to HTCondor with GPU resources. Check logs in `logs/export_gla/`.

#### Export GPT-2 Model
```bash
condor_submit scripts/submission_scripts/export_gpt/export_gpt.sub
```

This downloads the GPT-2 model and saves it to `checkpoints/gpt2_mod`. Check logs in `logs/export_gpt/`.

### 2. Quantize Models (PTQ Calibration)

Quantize FP32 models to INT8 using Post-Training Quantization. The quantization process:
1. Prepares the model by inserting observers
2. Calibrates using validation data from lm-eval tasks
3. Converts to INT8 quantized format (assuming FBGEMM/CPU backend)
4. Saves the quantized model and activation histogram information

#### Quantize GLA Model
```bash
condor_submit scripts/submission_scripts/quant_observers/quant_obs_gla.sub
```

This quantizes the GLA model from `checkpoints/gla_mod` to `checkpoints/gla_mod_int8_fbgemm`. The script uses default calibration settings (wikitext and lambada tasks, 2000 max docs, batch size 8). Check logs in `logs/quant_observers/`.

#### Quantize GPT-2 Model

**Note:** There is currently no HTCondor submission script for GPT-2 quantization. You can either:

1. **Create a submission script** similar to `quant_obs_gla.sub`, or
2. **Run directly** (if you have local GPU access):
```bash
python scripts/ptq_calibrate_gpt.py \
    --ckpt-in checkpoints/gpt2_mod \
    --ckpt-out checkpoints/gpt2_mod_int8_fbgemm \
    --tasks wikitext lambada \
    --max-docs 2000 \
    --batch-size 8 \
    --histo-out outs/out_gpt2_mod/activation_histos_gpt2_mod.npz \
    --histo-out-pngs outs/out_gpt2_mod/histo_pngs
```

**Arguments:**
- `--ckpt-in`: Input FP32 checkpoint directory
- `--ckpt-out`: Output directory for INT8 quantized model
- `--tasks`: lm-eval tasks to use for calibration (default: `wikitext`, `lambada`)
- `--max-docs`: Maximum number of documents for calibration (default: 2000)
- `--batch-size`: Batch size for calibration (default: 8)
- `--seq-len`: Sequence length for calibration (default: 512)
- `--histo-out`: Path to save activation histograms as NPZ (optional)
- `--histo-out-pngs`: Directory to save histogram PNGs (optional)
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda` if available)

**Note:** Quantized models use the FBGEMM backend and run on CPU. The script automatically handles device placement.

### 3. Evaluate Models

Evaluate models using `lm-eval` on standard benchmarks.

#### Evaluate FP32 GLA Model
```bash
condor_submit scripts/submission_scripts/lm_eval_baseline/lm_eval_baseline_gla.sub
```

This evaluates the GLA model from `checkpoints/gla_mod` and saves results to `outs/lm_eval_baseline/gla_mod_eval.json`. Check logs in `logs/lm_eval_baseline/`.

#### Evaluate FP32 GPT-2 Model
```bash
condor_submit scripts/submission_scripts/lm_eval_baseline/lm_eval_baseline_gpt.sub
```

This evaluates the GPT-2 model from `checkpoints/gpt2_mod` and saves results to `outs/lm_eval_baseline/gpt2_mod_eval.json`.

#### Evaluate INT8 Quantized Model
**NOTE: Does NOT work due to quantization kernels being written for CPU backend, but GLA requiring CUDA device**
```bash
condor_submit scripts/submission_scripts/lm_eval_baseline/lm_eval_baseline_gla_int8.sub
```

This evaluates the INT8 quantized GLA model from `checkpoints/gla_mod_int8_fbgemm` and saves results to `outs/lm_eval_baseline/gla_mod_int8_fbgemm_eval.json`.

The script automatically detects INT8 checkpoints (by checking for `pytorch_model_int8.bin`) and uses the appropriate wrapper.

**Evaluation Tasks:**
- `wikitext`: Language modeling perplexity
- `piqa`: Physical interaction QA
- `lambada_openai`: Language modeling with context

## Submission Scripts

The repository includes HTCondor submission scripts for cluster environments. These are located in `scripts/submission_scripts/`:

- **Export scripts**: `export_gla/`, `export_gpt/`
- **Evaluation scripts**: `lm_eval_baseline/`
- **Quantization observer scripts**: `quant_observers/`

Each directory contains:
- `.sh`: Bash script with environment setup (CUDA modules, virtual environment, etc.)
- `.sub`: HTCondor submission file with resource requirements

### Submitting Jobs

To submit a job, use:
```bash
condor_submit scripts/submission_scripts/<script_name>/<script_name>.sub
```

To check job status:
```bash
condor_q
```

To view job logs:
- Standard output: `logs/<script_name>/<script_name>.out`
- Standard error: `logs/<script_name>/<script_name>.err`
- HTCondor log: `logs/<script_name>/<script_name>.log`

## Project Structure

```
quantizing-lsa/
├── checkpoints/          # Model checkpoints (FP32 and INT8)
│   ├── gla_mod/          # FP32 GLA model
│   ├── gla_mod_int8_fbgemm/  # INT8 quantized GLA model
│   └── gpt2_mod/         # FP32 GPT-2 model
├── evals/                # Evaluation utilities
├── logs/                 # Log files from submission scripts
├── models/               # Custom model implementations
│   └── GLA/              # GLA model with quantization support
├── outs/                 # Output files (histograms, evaluation results)
├── quant/                # Quantization framework
│   ├── fx_compat.py      # FX compatibility layer
│   ├── mapping_gla.py    # Quantization mapping for GLA
│   ├── mapping_gpt2.py   # Quantization mapping for GPT-2
│   ├── observers.py       # Observer configuration
│   └── pipeline.py        # PTQ pipeline (prepare, calibrate, convert)
├── scripts/              # Main scripts
│   ├── export_gla.py     # Export GLA model
│   ├── export_gpt.py     # Export GPT-2 model
│   ├── ptq_calibrate_gpt.py  # PTQ calibration script
│   └── submission_scripts/   # Cluster submission scripts
└── pyproject.toml        # Project dependencies
```

## Quantization Details

### Quantization Strategy

- **Weights**: Per-channel symmetric INT8 quantization (using `PerChannelMinMaxObserver`)
- **Activations**: Per-tensor affine UINT8 quantization (using `HistogramObserver`)
- **Embeddings**: Float-params weight-only quantization (required by PyTorch)
- **Output Layer**: FP32 (typically `lm_head` is kept in FP32 for accuracy)

### Backend

- **FP32 Models**: Run on CUDA with Triton backend (via `flash-linear-attention`)
- **INT8 Models**: Run on CPU with FBGEMM backend (PyTorch's quantized operations)

### Calibration Data

Calibration uses validation splits from lm-eval tasks. The default tasks are `wikitext` and `lambada`, but you can specify custom tasks via the `--tasks` argument.

## Troubleshooting

### CUDA Initialization Issues

For FP32 models, CUDA must be initialized before importing `fla` (flash-linear-attention). The evaluation scripts handle this automatically. If you encounter issues:

1. Ensure CUDA modules are loaded (if using a cluster):
```bash
module load cudnn/9.10.2
module load cuda/12.9
```

2. Verify CUDA availability:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### INT8 Model Loading

Presently, INT8 models must run on CPU due to the backends available in torch.ao. The wrapper scripts automatically handle device placement. If you see errors about moving quantized models to CUDA, ensure you're using the INT8 wrapper (`lm_eval_wrapper_int8.py`).

### Model Loading Errors

If you encounter errors loading quantized models:
- Ensure `pytorch_model_int8_full.pt` exists in the checkpoint directory
- Verify all required files are present: `config.json`, `modeling_*.py`, `tokenizer.json`, etc.
- Check that the model was quantized with the same PyTorch version
