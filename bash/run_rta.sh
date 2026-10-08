#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Override these paths when running on another machine.
DATA_DIR="${DATA_DIR:-.data}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-./weights/sam_vit_h_4b8939.pth}"
GPU_ID="${GPU_ID:-0}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python main.py \
  --adapt \
  --method rta \
  --prompt_dir prompts.yaml \
  --dataset "${DATASET:-COCOStuffDataset}" \
  --data_dir "${DATA_DIR}" \
  --batch_size 1 \
  --workers "${WORKERS:-4}" \
  --steps 10 \
  --trials 1 \
  --ovss_type naclip \
  --ovss_backbone ViT-L/14 \
  --use_sam_refinement \
  --sam_model_type vit_h \
  --sam_checkpoint "${SAM_CHECKPOINT}" \
  --percentile_low 40 \
  --percentile_high 85 \
  --prompt_type point \
  --num_sample_points 10 \
  --local_cache_sample_stride 3 \
  --local_cache_capacity 10 \
  --local_cache_beta 5.0 \
  --save_dir "${SAVE_DIR:-./runs/rta}" \
  --no_shuffle_data
