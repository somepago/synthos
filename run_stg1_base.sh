#!/bin/bash
# Stage 1: Train SigLip projection with Z-Image Base (50-step flow matching, cfg=4.0)
# Run inside screen: screen -S stg1b bash run_stg1_base.sh

set -euo pipefail

cd /home/gnan/projects/synthos
source venv/bin/activate

export HF_HOME=/home/gnan/.cache/huggingface
export DIFFSYNTH_MODEL_BASE_PATH=/home/gnan/projects/synthos/models
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

python train_stage1_projection.py \
    --model base \
    --data_dir /home/gnan/projects/data/datasets/relaion-art-lowres/ \
    --lr 1e-4 \
    --lr_warmup_steps 100 \
    --lr_min_ratio 0.01 \
    --grad_accum_steps 8 \
    --steps 5000 \
    --eval_every 100 \
    --save_every 500 \
    --log_every 10 \
    --n_val 32 \
    --n_eval_images 12 \
    --max_size 768 \
    --image_size 512 \
    --output_dir outputs/stage1_base
