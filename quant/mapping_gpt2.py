import torch.nn as nn
from torch.ao.quantization import QConfigMapping
from .observers import qconfig  # uses the qconfig you just wrote

def build_qmap_for_gpt2():
    # Quantize all Linear layers; keep embeddings + lm_head in FP32 (common baseline)
    return (
        QConfigMapping()
        .set_global(qconfig)                   # default: quantize with qconfig
        .set_object_type(nn.Linear, qconfig)   # quantize all nn.Linear ops
        .set_module_name("transformer.wte", None)  # word embeddings
        .set_module_name("transformer.wpe", None)  # position embeddings
        .set_module_name("lm_head", None)          # often FP32 for accuracy
    )