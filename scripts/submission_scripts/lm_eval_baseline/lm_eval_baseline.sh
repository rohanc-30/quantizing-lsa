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

python -m lm_eval \
  --model hf \
  --model_args "pretrained=$1,tokenizer=$1,dtype=float32,trust_remote_code=True" \
  --tasks wikitext,piqa,lambada_openai \
  --device cpu \
  --batch_size 8 \
  --limit 500 \
  --output_path $2