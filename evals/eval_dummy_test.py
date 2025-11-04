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

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import importlib
importlib.reload(importlib.import_module("models.GLA.GLATransformer"))
from models.GLA.GLATransformer import QuantizableGPT2LMHeadModel, QuantizableGPT2Config


if __name__ == "__main__":
    model = QuantizableGPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print(model)

    input_prompt = "This is a story all about how my life got flipped, turned upside down."
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=128)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])