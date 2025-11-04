import math
import random
import os
import time
import datetime as dt
# import matplotlib.pyplot as plt
# import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import tqdm
import importlib
from pathlib import Path
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer
from itertools import chain
import pickle
import sys
import os
import argparse
import pathlib

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import importlib
importlib.reload(importlib.import_module("models.GLA.GLATransformer"))
from models.GLA.GLATransformer import QuantizableGLAModel, QuantizableGLAConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", default="checkpoints/gla_mod")
    args = p.parse_args()

    cfg = QuantizableGLAConfig.from_pretrained("fla-hub/GLA-2.7B-100B")
    model = QuantizableGLAModel.from_pretrained("fla-hub/GLA-2.7B-100B", config=cfg)
    tok = AutoTokenizer.from_pretrained("fla-hub/GLA-2.7B-100B")

    pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.save_dir)
    tok.save_pretrained(args.save_dir)
    print(f"Saved HF checkpoint to {args.save_dir}")

if __name__ == "__main__":
    main()
