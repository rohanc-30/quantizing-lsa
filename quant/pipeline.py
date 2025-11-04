import torch
import numpy as np

from .fx_compat import prepare_fx_compat, convert_fx_compat

import os, re
import matplotlib.pyplot as plt

def prepare_model(model: torch.nn.Module, qconfig_mapping, example_inputs):
    """
    FX 'prepare' pass: inserts observers (and fake-quant for QAT) according to qconfig_mapping.
    Model still runs in FP32; this only wires up instrumentation.
    """
    model.eval()
    prepared = prepare_fx_compat(model, qconfig_mapping, example_inputs)
    return prepared

@torch.inference_mode()
def calibrate(prepared_model: torch.nn.Module, batch_iter, device: str = "cuda"):
    """
    Forward-only loop to populate observers on the prepared model.
    Expects batch_iter to yield dicts with keys: 'input_ids' and optional 'attention_mask'.
    """
    prepared_model.eval().to(device)
    for batch in batch_iter:
        ids = batch["input_ids"].to(device)
        mask = batch.get("attention_mask")
        mask = None if mask is None else mask.to(device)
        pos  = torch.arange(ids.size(1), device=device).unsqueeze(0).expand(ids.size(0), -1)  # [B, T]
        prepared_model(ids, attention_mask=mask)
    return prepared_model

def convert_to_int8(prepared_model: torch.nn.Module) -> torch.nn.Module:
    """
    FX 'convert' pass: removes observers, computes qparams, packs weights,
    and swaps supported ops for backend int8 kernels (FBGEMM/QNNPACK).
    """
    prepared_model = prepared_model.to("cpu")
    int8_model = convert_fx_compat(prepared_model)
    return int8_model

@torch.inference_mode()
def collect_activation_histograms(prepared_model: torch.nn.Module):
    """
    Extracts activation histograms/min/max from each inserted HistogramObserver
    after calibration. Returns a list of dicts for easy saving/plotting.
    """
    records = []
    for name, mod in prepared_model.named_modules():
        if hasattr(mod, "activation_post_process"):
            obs = mod.activation_post_process
            hist = getattr(obs, "histogram", None)
            min_v = getattr(obs, "min_val", None)
            max_v = getattr(obs, "max_val", None)
            if hist is None or min_v is None or max_v is None or hist.numel() == 0:
                continue
            nbins = int(hist.numel())
            edges = torch.linspace(float(min_v), float(max_v), steps=nbins + 1)
            records.append(
                dict(
                    module=name,
                    bins=nbins,
                    counts=hist.detach().cpu().numpy(),
                    edges=edges.detach().cpu().numpy(),
                    min=float(min_v),
                    max=float(max_v),
                )
            )
    return records

def save_histograms_npz(records, path: str):
    """
    Saves histogram records to a single .npz (compact, fast to load).
    """
    npz_payload = {}
    for i, r in enumerate(records):
        p = f"r{i}"
        npz_payload[f"{p}_module"] = np.string_(r["module"])
        npz_payload[f"{p}_bins"] = np.int32(r["bins"])
        npz_payload[f"{p}_counts"] = r["counts"].astype(np.int64)
        npz_payload[f"{p}_edges"] = r["edges"].astype(np.float64)
        npz_payload[f"{p}_min"] = np.float64(r["min"])
        npz_payload[f"{p}_max"] = np.float64(r["max"])
    np.savez(path, **npz_payload)

def save_histograms_png(records, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for i, r in enumerate(records):
        counts = r["counts"]; edges = r["edges"]; name = r["module"]
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths  = edges[1:] - edges[:-1]
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)
        plt.figure()
        plt.bar(centers, counts, width=widths, align="center")
        plt.title(name); plt.xlabel("Activation value"); plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"r{i}_{safe}.png"), dpi=150)
        plt.close()
