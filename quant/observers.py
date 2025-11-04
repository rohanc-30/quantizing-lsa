# quant/observers.py
"""
Observer policy for PTQ with torch.ao (FBGEMM backend).

Exports:
  - qconfig : torch.ao.quantization.QConfig (activation + weight observer factories)
  - act_obs : Activation observer factory (HistogramObserver, per-tensor affine uint8)
  - w_obs   : Weight observer factory (PerChannelMinMaxObserver, per-channel symmetric int8)

Usage (elsewhere):
  from quant.observers import qconfig
  # ... build QConfigMapping and call prepare_fx(model, qconfig_mapping, example_inputs)
"""

import torch
from torch.ao.quantization import (
    HistogramObserver,
    PerChannelMinMaxObserver,
    QConfig,
)

# --- Backend choice: x86/AVX CPUs (server/desktop). Set this BEFORE prepare_fx/convert_fx. ---
# Try to set fbgemm backend, fall back to qnnpack if not available
try:
    torch.backends.quantized.engine = "fbgemm"
except RuntimeError:
    # Fall back to qnnpack if fbgemm is not available
    torch.backends.quantized.engine = "qnnpack"

# --- Activation observer: per-tensor histogram (uint8, affine) ---
# Good PTQ default for Transformer activations; robust to heavy tails/outliers.
act_obs = HistogramObserver.with_args(
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine,
    bins=1024,          # tune 512–2048 based on calibration budget
    reduce_range=False, # full 8-bit range
    eps=1e-8,
)

# --- Weight observer: per-channel min/max (int8, symmetric) ---
# One scale per output channel (row) for nn.Linear; improves accuracy significantly.
w_obs = PerChannelMinMaxObserver.with_args(
    dtype=torch.qint8,
    qscheme=torch.per_channel_symmetric,
    ch_axis=0,          # rows/out_features for nn.Linear; out_channels for Conv
    reduce_range=False,
    eps=1e-8,
    # quant_min=-128, quant_max=127,  # (optional) expose for research sweeps
)

# --- QConfig ties the two factories together (how to observe). ---
qconfig = QConfig(
    activation=act_obs,
    weight=w_obs,
)

__all__ = ["qconfig", "act_obs", "w_obs"]
