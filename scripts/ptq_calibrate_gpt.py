import os
import argparse
import sys
import shutil
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval.tasks import get_task_dict
from pathlib import Path

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from quant.mapping_gpt2 import build_qmap_for_gpt2
from quant.pipeline import (
    prepare_model,
    calibrate,
    collect_activation_histograms,
    save_histograms_npz,
    save_histograms_png,
    convert_to_int8,
)

def iter_lmeval_batches(tokenizer, task_names=("wikitext", "lambada"),
                        split="validation", max_docs=2000, batch_size=8, seq_len=512):
    """
    Streams doc_to_text from lm-eval tasks and yields padded HF-style batches:
    dict(input_ids, attention_mask). Adjust task_names/split to mirror your short run.
    """
    task_dict = get_task_dict(list(task_names))
    seen = 0
    buf_ids, buf_masks = [], []
    for task_name, task in task_dict.items():
        # Handle cases where task is a tuple (group_name, task_object)
        if isinstance(task, tuple):
            group_name, actual_task = task
        else:
            actual_task = task
            
        # Try to get docs from the appropriate split
        docs = None
        if split == "validation" and hasattr(actual_task, "validation_docs"):
            try:
                docs = actual_task.validation_docs()
            except Exception as e:
                print(f"Warning: Failed to get validation_docs for {task_name}: {e}")
        elif hasattr(actual_task, "test_docs"):
            try:
                docs = actual_task.test_docs()
            except Exception as e:
                print(f"Warning: Failed to get test_docs for {task_name}: {e}")
        
        # Handle case where docs is None (task has no data or data not loaded)
        if docs is None:
            print(f"Warning: {task_name} has no {split} docs available, skipping...")
            continue
            
        for doc in docs:
            text = actual_task.doc_to_text(doc)
            if not text:
                continue
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
            # Ensure ids is 1D for pad_sequence
            ids = enc["input_ids"].squeeze(0)
            if ids.dim() == 0:
                ids = ids.unsqueeze(0)  # Handle edge case of single token
            # Get or create attention mask, ensure it's 1D
            mask = enc.get("attention_mask", None)
            if mask is None:
                mask = torch.ones_like(ids)
            else:
                mask = mask.squeeze(0)  # Ensure 1D
            if mask.dim() == 0:
                mask = mask.unsqueeze(0)  # Handle edge case of single token
            # Verify ids and mask have same length
            if ids.shape[0] != mask.shape[0]:
                mask = torch.ones_like(ids)  # Fallback: create matching mask
            buf_ids.append(ids); buf_masks.append(mask)
            if len(buf_ids) == batch_size:
                yield {
                    "input_ids": pad_sequence(buf_ids, batch_first=True),
                    "attention_mask": pad_sequence(buf_masks, batch_first=True),
                }
                buf_ids.clear(); buf_masks.clear()
                seen += 1
                if seen >= max_docs:
                    return
    if buf_ids:
        yield {
            "input_ids": pad_sequence(buf_ids, batch_first=True),
            "attention_mask": pad_sequence(buf_masks, batch_first=True),
        }

class PTQWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # disable KV-cache during tracing
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        # Force inputs_embeds to be None so the HF guard doesn't trip under FX
        return self.model(input_ids=input_ids,
                          attention_mask=None,
                          position_ids=position_ids,
                          inputs_embeds=None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-in",  default="checkpoints/gpt2_mod",
                    help="FP32 HF checkpoint folder to load from")
    ap.add_argument("--ckpt-out", default="checkpoints/gpt2_mod_int8_fbgemm",
                    help="Folder to save INT8 weights/state_dict")
    ap.add_argument("--histo-out", default="outs/out_gpt2_mod/activation_histos_gpt2_mod.npz",
                    help="NPZ path to save activation histograms (set empty to skip)")
    ap.add_argument("--histo-out-pngs", default="outs/out_gpt2_mod/histo_pngs",
                    help="Directory to put histogram PNGs")
    ap.add_argument("--tasks", nargs="+", default=["wikitext", "lambada"],
                    help="lm-eval tasks to stream for calibration")
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--max-docs", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.ckpt_out, exist_ok=True)
    if args.histo_out:
        os.makedirs(os.path.dirname(args.histo_out), exist_ok=True)

    # 1) Load tokenizer + FP32 model
    tok = AutoTokenizer.from_pretrained(args.ckpt_in, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(args.ckpt_in, trust_remote_code=True).eval().to(args.device)

    print(torch.__version__)

    '''
    try:
        base_model.set_attn_implementation("eager")      # transformers >= ~4.36
    except Exception:
        try:
            base_model.config._attn_implementation = "eager"  # fallback knob in many versions
        except Exception:
            pass
    # optional: ensure no sliding window surprises
    if hasattr(base_model.config, "sliding_window"):
        base_model.config.sliding_window = None
    '''

    fp32 = base_model

    # 2) Build mapping (what to observe/quantize)
    qmap = build_qmap_for_gpt2()

    # 3) Prepare (insert observers). Example inputs just need correct shapes/dtypes.
    seq = 16
    ex_ids = torch.randint(0, tok.vocab_size, (1, seq), device=args.device)
    ex_pos = torch.arange(seq, device=args.device).unsqueeze(0)  # [1, seq]
    
    # Wrap the model to handle FX tracing issues
    wrapped_model = PTQWrapper(fp32)
    prepared = prepare_model(wrapped_model, qmap, (ex_ids,)).to(args.device).eval()

    # 4) Calibrate via short lm-eval subset
    print(f"[Calibration] Starting calibration with {args.max_docs} docs...")
    calib_iter = iter_lmeval_batches(
        tok, task_names=tuple(args.tasks), split=args.split,
        max_docs=args.max_docs, batch_size=args.batch_size, seq_len=args.seq_len
    )
    calibrate(prepared, calib_iter, device=args.device)
    
    # Verify calibration: count how many observers have been run
    observer_count = 0
    calibrated_count = 0
    for name, mod in prepared.named_modules():
        if hasattr(mod, "activation_post_process"):
            observer_count += 1
            obs = mod.activation_post_process
            # Check if observer has been run (has min_val/max_val or histogram)
            if hasattr(obs, "min_val") and obs.min_val is not None:
                calibrated_count += 1
            elif hasattr(obs, "histogram") and obs.histogram is not None and obs.histogram.numel() > 0:
                calibrated_count += 1
    print(f"[Calibration] Found {observer_count} observers, {calibrated_count} were calibrated")

    # 5) (Optional) Save activation histograms BEFORE convert (observers are removed by convert)
    if args.histo_out:
        histos = collect_activation_histograms(prepared)
        save_histograms_npz(histos, args.histo_out)
        save_histograms_png(histos, args.histo_out_pngs)
        print(f"[info] saved {len(histos)} activation histograms → {args.histo_out}")

    # 6) Convert to INT8 and save the FULL model (not just state_dict)
    # This allows direct loading without reconstruction
    int8_model = convert_to_int8(prepared)
    
    # Unwrap from PTQWrapper if needed (the wrapper adds 'model.' prefix)
    if hasattr(int8_model, 'model'):
        int8_model = int8_model.model
    
    # Save the full quantized model
    torch.save(int8_model, os.path.join(args.ckpt_out, "pytorch_model_int8_full.pt"))
    # Also save state_dict for compatibility
    torch.save(int8_model.state_dict(), os.path.join(args.ckpt_out, "pytorch_model_int8.bin"))
    
    # Save tokenizer alongside for convenience
    tok.save_pretrained(args.ckpt_out)
    
    # 7) Copy necessary files from input checkpoint to make it a valid HF checkpoint
    # This includes config.json, modeling files, etc.
    ckpt_in_path = Path(args.ckpt_in)
    ckpt_out_path = Path(args.ckpt_out)
    
    # Files to copy that are needed for loading the model
    files_to_copy = [
        "config.json",
        "generation_config.json",
        "modeling_gla.py",  # GLA-specific
        "configuration_gla.py",  # GLA-specific
        # "modeling_gpt2.py",  # GPT2-specific (if exists)
        #"configuration_gpt2.py",  # GPT2-specific (if exists)
    ]
    
    for file_name in files_to_copy:
        src_file = ckpt_in_path / file_name
        if src_file.exists():
            dst_file = ckpt_out_path / file_name
            shutil.copy2(src_file, dst_file)
            print(f"[info] Copied {file_name} to output checkpoint")
    
    # Also copy any Python files that might be needed (modeling, configuration)
    for py_file in ckpt_in_path.glob("*.py"):
        if py_file.name not in [f.name for f in ckpt_out_path.glob("*.py")]:
            shutil.copy2(py_file, ckpt_out_path / py_file.name)
            print(f"[info] Copied {py_file.name} to output checkpoint")
    
    print(f"[done] INT8 state_dict saved → {args.ckpt_out}/pytorch_model_int8.bin")
    print(f"[done] Checkpoint files copied to make {args.ckpt_out} a valid HF checkpoint")

if __name__ == "__main__":
    main()