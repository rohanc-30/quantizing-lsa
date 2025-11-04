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
from transformers import GPT2LMHeadModel, GPT2Config
import fla
from transformers import AutoTokenizer, AutoModelForCausalLM
from fla.models import GLAConfig, GLAForCausalLM

# Build a class equivalent to hf's fla-hub/GLA
class QuantizableGLAConfig(GLAConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class QuantizableGLAModel(GLAForCausalLM):
    config_class = QuantizableGLAConfig
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

class QuantizableGPT2Config(GPT2Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class QuantizableGPT2LMHeadModel(GPT2LMHeadModel):
    config_class = QuantizableGPT2Config

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
