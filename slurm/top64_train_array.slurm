#!/usr/bin/env bash
#SBATCH --job-name=top64_train
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --account=faculty
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --requeue

set -Eeuo pipefail
trap 'status=$?; echo "[top64 train] failed at line ${LINENO} (exit ${status})" >&2; exit "${status}"' ERR

REPO_DIR=${REPO_DIR:?set REPO_DIR}
TRAIN_TRACES=${TRAIN_TRACES:?set TRAIN_TRACES}
VENV_DIR=${VENV_DIR:-/raid/mtanveer/antidistill_env}
RUN_ROOT=${RUN_ROOT:-/raid/mtanveer/top64_soft_distillation}
HF_HOME=${HF_HOME:-/raid/mtanveer/model_cache/huggingface}
TMPDIR=${TMPDIR:-/raid/mtanveer/ads_tmp}
K=${K:-64}
TEMPERATURE=${TEMPERATURE:-2.0}
MAX_LENGTH=${MAX_LENGTH:-2048}
ALPHA=${ALPHA:-0.9}
CACHE_DIR=${CACHE_DIR:-${RUN_ROOT}/cache_k${K}_t${TEMPERATURE}_l${MAX_LENGTH}}
SEEDS_TEXT=${SEEDS:-"42 123 456"}
read -r -a SEED_LIST <<< "${SEEDS_TEXT}"
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= ${#SEED_LIST[@]} )); then
    echo "Invalid array task ${TASK_ID}; configured seeds: ${SEED_LIST[*]}" >&2
    exit 2
fi
SEED=${SEED_LIST[$TASK_ID]}
OUTPUT_DIR=${RUN_ROOT}/models/top64_seed${SEED}

source "${VENV_DIR}/bin/activate"
cd "${REPO_DIR}"
mkdir -p "${RUN_ROOT}/models" "${TMPDIR}" "${HF_HOME}"
export HF_HOME HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

[[ -f "${CACHE_DIR}/cache_meta.json" && -d "${CACHE_DIR}/data" ]] || {
    echo "Verified cache not found: ${CACHE_DIR}" >&2
    exit 2
}

echo "[top64 train] host=$(hostname) job=${SLURM_JOB_ID:-none} task=${TASK_ID} seed=${SEED}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

EXTRA=()
[[ -n "${TEACHER_REVISION:-}" ]] && EXTRA+=(--teacher_revision "${TEACHER_REVISION}")
[[ -n "${STUDENT_REVISION:-}" ]] && EXTRA+=(--student_revision "${STUDENT_REVISION}")

python soft_distill.py \
    --mode train \
    --teacher_logits cached \
    --vocab_mode shared \
    --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --student Qwen/Qwen2.5-3B \
    --train_traces "${TRAIN_TRACES}" \
    --cache_dir "${CACHE_DIR}" \
    --top_k "${K}" \
    --temperature "${TEMPERATURE}" \
    --alpha "${ALPHA}" \
    --kl_variant tail_bucket \
    --max_length "${MAX_LENGTH}" \
    --batch_size 16 \
    --per_device_batch_size 1 \
    --num_epochs 3 \
    --lr 5e-4 \
    --lora --lora_r 128 --lora_alpha 128 \
    --gradient_checkpointing \
    --save_steps 100 \
    --save_total_limit 3 \
    --resume_from_checkpoint auto \
    --seed "${SEED}" \
    --output_dir "${OUTPUT_DIR}" \
    "${EXTRA[@]}"

echo "[top64 train] complete: ${OUTPUT_DIR}/final"
