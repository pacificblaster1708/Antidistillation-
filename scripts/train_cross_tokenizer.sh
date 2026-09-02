#!/usr/bin/env bash
# ==============================================================================
# Cross-tokenizer distillation: Qwen-vocabulary teacher -> Llama-vocabulary
# student. Positions are aligned by character offset and teacher token ids are
# mapped to student ids by surface form; whatever cannot be aligned falls back
# to hard-label CE.
#
# This is an APPROXIMATION. Read the two coverage numbers the run prints:
#   "... carry a teacher top-K target"  -> position alignment
#   "Cross-vocab map: ... map to a single student token" -> id mapping
# If either is low, prefer a Qwen student and scripts/train_shared_online.sh.
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

ACCEL_CONFIG=${ACCEL_CONFIG:-configs/accelerate_1gpu.yaml}
TEACHER=${TEACHER:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}
STUDENT=${STUDENT:-meta-llama/Llama-3.2-3B}
STUDENT_TOKENIZER=${STUDENT_TOKENIZER:-meta-llama/Llama-3.2-3B-Instruct}
TRAIN_TRACES=${TRAIN_TRACES:?set TRAIN_TRACES to your training traces directory}
EVAL_TRACES=${EVAL_TRACES:-}
K=${K:-64}
TEMPERATURE=${TEMPERATURE:-2.0}
ALPHA=${ALPHA:-0.7}     # lower than the shared-vocab default: the soft signal is noisier
MAX_LENGTH=${MAX_LENGTH:-2048}
BATCH_SIZE=${BATCH_SIZE:-16}
PER_DEVICE=${PER_DEVICE:-1}
EPOCHS=${EPOCHS:-3}
LR=${LR:-5e-4}
OUTPUT_DIR=${OUTPUT_DIR:-./experiments/models/soft_cross_k${K}}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false

EXTRA=()
[[ -n "$EVAL_TRACES" ]] && EXTRA+=(--eval_traces "$EVAL_TRACES")
[[ -n "${TEACHER_REVISION:-}" ]] && EXTRA+=(--teacher_revision "$TEACHER_REVISION")
[[ -n "${STUDENT_REVISION:-}" ]] && EXTRA+=(--student_revision "$STUDENT_REVISION")

accelerate launch --config_file "$ACCEL_CONFIG" soft_distill.py \
    --mode train \
    --teacher_logits online \
    --vocab_mode cross \
    --teacher "$TEACHER" \
    --student "$STUDENT" \
    --student_tokenizer "$STUDENT_TOKENIZER" \
    --train_traces "$TRAIN_TRACES" \
    --top_k "$K" \
    --temperature "$TEMPERATURE" \
    --alpha "$ALPHA" \
    --kl_variant student_full \
    --max_length "$MAX_LENGTH" \
    --batch_size "$BATCH_SIZE" \
    --per_device_batch_size "$PER_DEVICE" \
    --num_epochs "$EPOCHS" \
    --lr "$LR" \
    --gradient_checkpointing --save_steps 100 --resume_from_checkpoint auto \
    --lora --lora_r 128 --lora_alpha 128 \
    --output_dir "$OUTPUT_DIR" \
    "${EXTRA[@]}"

echo "Student written to ${OUTPUT_DIR}/final"
