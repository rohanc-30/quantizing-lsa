#!/bin/bash

source /etc/profile.d/modules.sh

module load cudnn/9.10.2
module load cuda/12.9

source /home/rcherukuri/quantizing-lsa/.venv/bin/activate

export HF_HOME=/home/rcherukuri/hf
export TRANSFORMERS_CACHE=/home/rcherukuri/hf/transformers
export HF_DATASETS_CACHE=/home/rcherukuri/hf/datasets
export HF_HUB_CACHE=/home/rcherukuri/hf/hub

export HOME=/home/rcherukuri

python scripts/export_gla.py --save_dir $1