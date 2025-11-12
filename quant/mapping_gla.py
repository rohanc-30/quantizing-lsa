import torch.nn as nn
from torch.ao.quantization import QConfigMapping, float_qparams_weight_only_qconfig
from .observers import qconfig  # uses the qconfig you just wrote

def build_qmap_for_gla():
    """
    Build quantization mapping for GLA models.
    
    GLA models have the same structure as GPT2 (Linear layers and Embeddings),
    so we use the same quantization strategy:
    - Quantize all Linear layers
    - Quantize Embeddings with float_qparams_weight_only_qconfig (required by PyTorch)
    - Keep lm_head in FP32 for accuracy
    """
    return (
        QConfigMapping()
        .set_global(qconfig)                   # default: quantize with qconfig
        .set_object_type(nn.Linear, qconfig)   # quantize all nn.Linear ops
        .set_object_type(nn.Embedding, float_qparams_weight_only_qconfig)  # required config for embedding quantization
        .set_module_name("lm_head", None)          # often FP32 for accuracy
    )

