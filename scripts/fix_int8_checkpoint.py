#!/usr/bin/env python
"""
Utility script to fix existing INT8 quantized checkpoints by copying
necessary files (config.json, modeling files) from the original checkpoint.

Usage:
    python scripts/fix_int8_checkpoint.py --ckpt-int8 checkpoints/gla_mod_int8_fbgemm --ckpt-original checkpoints/gla_mod
"""

import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description="Fix INT8 checkpoint by copying necessary files from original checkpoint"
    )
    ap.add_argument(
        "--ckpt-int8",
        required=True,
        help="Path to INT8 quantized checkpoint (missing config files)"
    )
    ap.add_argument(
        "--ckpt-original",
        required=True,
        help="Path to original FP32 checkpoint (has all files)"
    )
    args = ap.parse_args()
    
    ckpt_int8_path = Path(args.ckpt_int8)
    ckpt_original_path = Path(args.ckpt_original)
    
    if not ckpt_int8_path.exists():
        raise FileNotFoundError(f"INT8 checkpoint not found: {ckpt_int8_path}")
    if not ckpt_original_path.exists():
        raise FileNotFoundError(f"Original checkpoint not found: {ckpt_original_path}")
    
    # Files to copy that are needed for loading the model
    files_to_copy = [
        "config.json",
        "generation_config.json",
        "modeling_gla.py",
        "configuration_gla.py",
        "modeling_gpt2.py",
        "configuration_gpt2.py",
    ]
    
    copied_count = 0
    for file_name in files_to_copy:
        src_file = ckpt_original_path / file_name
        if src_file.exists():
            dst_file = ckpt_int8_path / file_name
            shutil.copy2(src_file, dst_file)
            print(f"[Copied] {file_name}")
            copied_count += 1
        else:
            print(f"[Skipped] {file_name} (not found in original checkpoint)")
    
    # Also copy any Python files that might be needed
    for py_file in ckpt_original_path.glob("*.py"):
        if py_file.name not in [f.name for f in ckpt_int8_path.glob("*.py")]:
            shutil.copy2(py_file, ckpt_int8_path / py_file.name)
            print(f"[Copied] {py_file.name}")
            copied_count += 1
    
    print(f"\n[Done] Copied {copied_count} file(s) to {ckpt_int8_path}")
    print(f"INT8 checkpoint should now be loadable as a regular HuggingFace checkpoint")


if __name__ == "__main__":
    main()

