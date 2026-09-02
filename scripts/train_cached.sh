#!/usr/bin/env bash
# ==============================================================================
# Step 2 of 2 for the cached workflow: train the student from the cache written
# by precompute_cache.sh. The teacher is not loaded, so this fits in much less
# memory and you can sweep temperature / alpha / lr cheaply.
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

ACCEL_CONFIG=${ACCEL_CONFIG:-configs/accelerate_1gpu.yaml}
STUDENT=${STUDENT:-Qwen/Qwen2.5-3B}
TEACHER=${TEACHER:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}   # tokenizer only
VOCAB_MODE=${VOCAB_MODE:-shared}
K=${K:-64}
CACHE_DIR=${CACHE_DIR:-./experiments/topk_cache_k${K}}
TEMPERATURE=${TEMPERATURE:-2.0}
ALPHA=${ALPHA:-0.9}
MAX_LENGTH=${MAX_LENGTH:-2048}
BATCH_SIZE=${BATCH_SIZE:-16}
PER_DEVICE=${PER_DEVICE:-1}
EPOCHS=${EPOCHS:-3}
LR=${LR:-5e-4}
OUTPUT_DIR=${OUTPUT_DIR:-./experiments/models/soft_cached_k${K}_T${TEMPERATURE}_a${ALPHA}}
SAVE_STEPS=${SAVE_STEPS:-100}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false

EXTRA=()
[[ -n "${STUDENT_TOKENIZER:-}" ]] && EXTRA+=(--student_tokenizer "$STUDENT_TOKENIZER")
[[ -n "${TEACHER_REVISION:-}" ]] && EXTRA+=(--teacher_revision "$TEACHER_REVISION")
[[ -n "${STUDENT_REVISION:-}" ]] && EXTRA+=(--student_revision "$STUDENT_REVISION")

accelerate launch --config_file "$ACCEL_CONFIG" soft_distill.py \
    --mode train \
    --teacher_logits cached \
    --vocab_mode "$VOCAB_MODE" \
    --teacher "$TEACHER" \
    --student "$STUDENT" \
    --cache_dir "$CACHE_DIR" \
    --top_k "$K" \
    --temperature "$TEMPERATURE" \
    --alpha "$ALPHA" \
    --kl_variant tail_bucket \
    --max_length "$MAX_LENGTH" \
    --batch_size "$BATCH_SIZE" \
    --per_device_batch_size "$PER_DEVICE" \
    --num_epochs "$EPOCHS" \
    --lr "$LR" \
    --gradient_checkpointing \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 3 \
    --resume_from_checkpoint auto \
    --lora --lora_r 128 --lora_alpha 128 \
    --output_dir "$OUTPUT_DIR" \
    "${EXTRA[@]}"

echo "Student written to ${OUTPUT_DIR}/final"
