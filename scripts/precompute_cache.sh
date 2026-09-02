#!/usr/bin/env bash
# ==============================================================================
# Step 1 of 2 for the cached workflow: run the teacher once over the dataset and
# write its top-K logits to disk. After this the teacher is never loaded again,
# so every later training run is faster and needs far less GPU memory.
#
# Rebuild the cache if you change K, the dataset, --max_length, the tokenizers
# or --vocab_mode. Changing temperature or alpha does NOT need a rebuild --
# raw logits are cached, not probabilities.
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

ACCEL_CONFIG=${ACCEL_CONFIG:-configs/accelerate_1gpu.yaml}
TEACHER=${TEACHER:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}
STUDENT=${STUDENT:-Qwen/Qwen2.5-3B}
VOCAB_MODE=${VOCAB_MODE:-shared}
TRAIN_TRACES=${TRAIN_TRACES:?set TRAIN_TRACES to your training traces directory}
K=${K:-64}
TEMPERATURE=${TEMPERATURE:-2.0}
MAX_LENGTH=${MAX_LENGTH:-2048}
PRECOMPUTE_BS=${PRECOMPUTE_BS:-1}
CACHE_DIR=${CACHE_DIR:-./experiments/topk_cache_k${K}}
TRACE_COLNAME=${TRACE_COLNAME:-auto}
PROBLEM_COLNAME=${PROBLEM_COLNAME:-auto}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false

EXTRA=()
[[ -n "${STUDENT_TOKENIZER:-}" ]] && EXTRA+=(--student_tokenizer "$STUDENT_TOKENIZER")
[[ -n "${TEACHER_REVISION:-}" ]] && EXTRA+=(--teacher_revision "$TEACHER_REVISION")
[[ -n "${STUDENT_REVISION:-}" ]] && EXTRA+=(--student_revision "$STUDENT_REVISION")
[[ "${OVERWRITE_CACHE:-false}" == "true" ]] && EXTRA+=(--overwrite_cache)

accelerate launch --config_file "$ACCEL_CONFIG" soft_distill.py \
    --mode precompute \
    --vocab_mode "$VOCAB_MODE" \
    --teacher "$TEACHER" \
    --student "$STUDENT" \
    --train_traces "$TRAIN_TRACES" \
    --top_k "$K" \
    --temperature "$TEMPERATURE" \
    --max_length "$MAX_LENGTH" \
    --trace_colname "$TRACE_COLNAME" \
    --problem_colname "$PROBLEM_COLNAME" \
    --precompute_batch_size "$PRECOMPUTE_BS" \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$CACHE_DIR" \
    "${EXTRA[@]}"

echo "Cache written to ${CACHE_DIR}"
du -sh "${CACHE_DIR}" 2>/dev/null || true
