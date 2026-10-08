#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# GPU Configuration
GPU_ID="${GPU_ID:-0}"

# Dataset Configuration
DATASET="${DATASET:-PascalVOC20Dataset}"
DATA_DIR="${DATA_DIR:-.data/VOC2012/}"
INIT_RESIZE="${INIT_RESIZE:-224 224}"
WORKERS="${WORKERS:-4}"

# Method and OVSS Model Configuration
METHOD="${METHOD:-rta}"
OUT_VISION="${OUT_VISION:--1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18}"
PROMPT_DIR="${PROMPT_DIR:-prompts.yaml}"
ALPHA_CLS="${ALPHA_CLS:-1.0}"
OVSS_TYPE="${OVSS_TYPE:-naclip}"
OVSS_BACKBONE="${OVSS_BACKBONE:-ViT-L/14}"

# SAM Configuration
SAM_CHECKPOINT="${SAM_CHECKPOINT:-./weights/sam_vit_h_4b8939.pth}"
SAM_MODEL_TYPE="${SAM_MODEL_TYPE:-vit_h}"

# RTA Hyperparameters
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-0.001}"
STEPS="${STEPS:-10}"
TRIALS="${TRIALS:-3}"
P_REL="${P_REL:-40}"
P_UNC="${P_UNC:-85}"
LOCAL_CACHE_CAPACITY="${LOCAL_CACHE_CAPACITY:-10}"
LOCAL_CACHE_SAMPLE_STRIDE="${LOCAL_CACHE_SAMPLE_STRIDE:-3}"
SAMPLES_PER_MASK="${SAMPLES_PER_MASK:-3}"
LOCAL_CACHE_BETA="${LOCAL_CACHE_BETA:-5.0}"

# Output
SAVE_DIR="${SAVE_DIR:-.save/${DATASET}/${METHOD}/}"

read -r -a INIT_RESIZE_ARGS <<< "$INIT_RESIZE"
read -r -a OUT_VISION_ARGS <<< "$OUT_VISION"

CUDA_VISIBLE_DEVICES="$GPU_ID" python main.py \
  --adapt \
  --method "$METHOD" \
  --prompt_dir "$PROMPT_DIR" \
  --vision_outputs "${OUT_VISION_ARGS[@]}" \
  --alpha_cls "$ALPHA_CLS" \
  --ovss_type "$OVSS_TYPE" \
  --ovss_backbone "$OVSS_BACKBONE" \
  --save_dir "$SAVE_DIR" \
  --data_dir "$DATA_DIR" \
  --dataset "$DATASET" \
  --workers "$WORKERS" \
  --init_resize "${INIT_RESIZE_ARGS[@]}" \
  --patch_size 224 224 \
  --patch_stride 112 \
  --lr "$LR" \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --trials "$TRIALS" \
  --class_extensions \
  --use_sam_refinement \
  --sam_checkpoint "$SAM_CHECKPOINT" \
  --sam_model_type "$SAM_MODEL_TYPE" \
  --semantic_weight 0.3 \
  --geo_weight 0.7 \
  --cov_exp 1.0 \
  --prompt_type point \
  --post_process nms \
  --topk_num 1 \
  --nms_iou_thresh 0.9 \
  --score_thresh 0.1 \
  --response_thresh_ratio 0.5 \
  --min_mask_area 100 \
  --num_sample_points 10 \
  --temperature 15.0 \
  --local_cache_capacity "$LOCAL_CACHE_CAPACITY" \
  --local_cache_beta "$LOCAL_CACHE_BETA" \
  --local_cache_sample_stride "$LOCAL_CACHE_SAMPLE_STRIDE" \
  --samples_per_mask "$SAMPLES_PER_MASK" \
  --percentile_low "$P_REL" \
  --percentile_high "$P_UNC" \
  --no_shuffle_data
