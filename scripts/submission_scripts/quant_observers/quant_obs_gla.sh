#!/bin/bash

source /etc/profile.d/modules.sh

module load cudnn/9.10.2
module load cuda/12.9

source /home/rcherukuri/quantizing-lsa/.venv/bin/activate


nvcc --version
/usr/bin/nvidia-smi


export HF_HOME=/home/rcherukuri/hf
export TRANSFORMERS_CACHE=/home/rcherukuri/hf/transformers
export HF_DATASETS_CACHE=/home/rcherukuri/hf/datasets
export HF_HUB_CACHE=/home/rcherukuri/hf/hub

export HOME=/home/rcherukuri
export PATH="/lustre/home/rcherukuri/quantizing-lsa/.venv/bin:/home/rcherukuri/.local/bin:/home/rcherukuri/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/usr/lib/jvm/java-8-oracle/bin:/usr/lib/jvm/java-8-oracle/db/bin:/usr/lib/jvm/java-8-oracle/jre/bin:${PATH}"



# Use wrapper script to ensure CUDA is initialized before fla imports
python /home/rcherukuri/quantizing-lsa/scripts/ptq_calibrate_gpt.py --ckpt-in $1 --ckpt-out $2 --batch-size 8 --histo-out $3 --histo-out-pngs $4