#!/usr/bin/env bash
#SBATCH --job-name=top64_cache
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --account=faculty
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --requeue

set -Eeuo pipefail
trap 'status=$?; echo "[top64 cache] failed at line ${LINENO} (exit ${status})" >&2; exit "${status}"' ERR

REPO_DIR=${REPO_DIR:?set REPO_DIR}
TRAIN_TRACES=${TRAIN_TRACES:?set TRAIN_TRACES}
VENV_DIR=${VENV_DIR:-/raid/mtanveer/antidistill_env}
RUN_ROOT=${RUN_ROOT:-/raid/mtanveer/top64_soft_distillation}
HF_HOME=${HF_HOME:-/raid/mtanveer/model_cache/huggingface}
TMPDIR=${TMPDIR:-/raid/mtanveer/ads_tmp}
K=${K:-64}
TEMPERATURE=${TEMPERATURE:-2.0}
MAX_LENGTH=${MAX_LENGTH:-2048}
CACHE_DIR=${CACHE_DIR:-${RUN_ROOT}/cache_k${K}_t${TEMPERATURE}_l${MAX_LENGTH}}

source "${VENV_DIR}/bin/activate"
cd "${REPO_DIR}"
mkdir -p "${RUN_ROOT}" "${TMPDIR}" "${HF_HOME}"
export HF_HOME HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "[top64 cache] host=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

EXTRA=()
[[ -n "${TEACHER_REVISION:-}" ]] && EXTRA+=(--teacher_revision "${TEACHER_REVISION}")
[[ -n "${STUDENT_REVISION:-}" ]] && EXTRA+=(--student_revision "${STUDENT_REVISION}")

if [[ -f "${CACHE_DIR}/cache_meta.json" && -d "${CACHE_DIR}/data" ]]; then
    echo "[top64 cache] validating existing cache: ${CACHE_DIR}"
    if python soft_distill.py \
        --mode validate_cache \
        --vocab_mode shared \
        --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --student Qwen/Qwen2.5-3B \
        --train_traces "${TRAIN_TRACES}" \
        --trace_colname "${TRACE_COLNAME:-auto}" \
        --problem_colname "${PROBLEM_COLNAME:-auto}" \
        --top_k "${K}" \
        --temperature "${TEMPERATURE}" \
        --max_length "${MAX_LENGTH}" \
        --cache_dir "${CACHE_DIR}" \
        "${EXTRA[@]}"; then
        echo "[top64 cache] existing cache is valid"
        exit 0
    fi
    echo "[top64 cache] existing cache is stale or incomplete; rebuilding it"
    EXTRA+=(--overwrite_cache)
elif [[ -d "${CACHE_DIR}" ]] && [[ -n "$(find "${CACHE_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "[top64 cache] incomplete cache found; rebuilding it"
    EXTRA+=(--overwrite_cache)
fi

python soft_distill.py \
    --mode precompute \
    --vocab_mode shared \
    --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --student Qwen/Qwen2.5-3B \
    --train_traces "${TRAIN_TRACES}" \
    --trace_colname "${TRACE_COLNAME:-auto}" \
    --problem_colname "${PROBLEM_COLNAME:-auto}" \
    --top_k "${K}" \
    --temperature "${TEMPERATURE}" \
    --max_length "${MAX_LENGTH}" \
    --precompute_batch_size 1 \
    --teacher_topk_chunk_size 128 \
    --cache_dir "${CACHE_DIR}" \
    --output_dir "${RUN_ROOT}/cache_job" \
    "${EXTRA[@]}"

echo "[top64 cache] complete: ${CACHE_DIR}"
