#!/usr/bin/env bash
# =============================================================================
# Antidistillation Sampling — SLURM job script
# Target: 1 × 40 GB A100, mixed precision
#
# Memory budget per GPU:
#   Stage 1 (Holdout)   : teacher 7B  (BF16)        = ~14 GB
#   Stage 2 (Gradients) : proxy 3B    (FP32)         = ~12 GB
#   Stage 3 (ADS Gen)   : teacher(BF16) + 2×proxy(FP16) = ~26 GB
#   Stage 4 (Distill)   : student 3B  (BF16 + LoRA)  = ~10 GB
#
# Cleanup strategy (runs only on full pipeline success):
#   PRESERVE : results.json, config.json, SLURM logs, sha256 checksums,
#              success marker
#   DELETE   : model checkpoints, optimizer/scheduler states, saved gradients,
#              generated traces, tokenized caches, temp files
#   ON FAIL  : no cleanup — retain everything for debugging
#
# Select mode:
#   MODE=normal  →  ADS=false NORMAL=true
#   MODE=ads     →  ADS=true  NORMAL=false
# =============================================================================

#SBATCH --job-name=antidistil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_antidistil.out
#SBATCH --error=logs/%j_antidistil.err

# ── adjust to your cluster ────────────────────────────────────────────────────
#SBATCH --partition=gpu
# #SBATCH --account=my_project
# #SBATCH --qos=high

# Do NOT use set -e here; we need to capture run.py's exit code manually
# so that failures skip cleanup and preserve artifacts for debugging.
set -uo pipefail

# =============================================================================
# Configuration — edit here
# =============================================================================
MODE=normal        # normal | ads
EXP_DIR=experiments

TEACHER=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
STUDENT=meta-llama/Llama-3.2-3B
STUDENT_TOK=meta-llama/Llama-3.2-3B-Instruct
PROXY_STUDENT=Qwen/Qwen2.5-3B   # must share teacher tokenizer (Qwen family)

DATASET=gsm8k
SEED=42

# ── batch sizes ───────────────────────────────────────────────────────────────
if [ "$MODE" = "ads" ]; then
    GEN_BATCH=8      # teacher + 2× proxy active simultaneously
    GRAD_BATCH=1
else
    GEN_BATCH=16     # teacher only
    GRAD_BATCH=2
fi

# Single GPU: effective batch = per_device × grad_accum
TRAIN_BATCH=4
PER_DEVICE_BATCH=2

# =============================================================================
# Environment
# =============================================================================
mkdir -p logs

module purge
module load cuda/12.1      # adjust to your cluster's CUDA module

source .venv/bin/activate  # or: conda activate antidistil

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# =============================================================================
# Derive paths from run.py / config.py naming conventions
# =============================================================================
if [ "$MODE" = "ads" ]; then
    MODE_FLAGS="--ads=true --normal=false"
    # run_name = ads_tau{tau}_lmin{lam_min}_lmax{lam_max}_eps{eps}
    RUN_NAME="ads_tau1_lmin0.01_lmax0.075_eps0.01"
else
    MODE_FLAGS="--ads=false --normal=true"
    RUN_NAME="normal_tau1"
fi

RUN_DIR="${EXP_DIR}/${RUN_NAME}"
RESULTS_FILE="${RUN_DIR}/results.json"
CONFIG_FILE="${RUN_DIR}/config.json"
STUDENT_FINAL="${RUN_DIR}/student/final"
GRAD_FILE="${EXP_DIR}/proxy_student_grads.pt"
TRACES_DIR="${EXP_DIR}/traces"
CACHE_DIR="${EXP_DIR}/.cache"

# Where preserved artifacts land
PRESERVE_DIR="${RUN_DIR}/preserved"

# =============================================================================
# Helper: preserve_artifacts
#   Copies results + config, checksums them, copies SLURM logs, writes marker.
#   Returns non-zero if any required file is missing.
# =============================================================================
preserve_artifacts() {
    echo "[cleanup] Preserving artifacts to ${PRESERVE_DIR} ..."
    mkdir -p "${PRESERVE_DIR}"

    # -- required files --------------------------------------------------------
    local missing=0
    for f in "${RESULTS_FILE}" "${CONFIG_FILE}"; do
        if [ ! -f "$f" ]; then
            echo "[cleanup] MISSING required file: $f" >&2
            missing=1
        fi
    done
    [ "$missing" -ne 0 ] && return 1

    cp "${RESULTS_FILE}"  "${PRESERVE_DIR}/results.json"
    cp "${CONFIG_FILE}"   "${PRESERVE_DIR}/config.json"

    # -- SLURM logs ------------------------------------------------------------
    LOG_OUT="logs/${SLURM_JOB_ID}_antidistil.out"
    LOG_ERR="logs/${SLURM_JOB_ID}_antidistil.err"
    [ -f "$LOG_OUT" ] && cp "$LOG_OUT" "${PRESERVE_DIR}/"
    [ -f "$LOG_ERR" ] && cp "$LOG_ERR" "${PRESERVE_DIR}/"

    # -- checksums -------------------------------------------------------------
    sha256sum \
        "${PRESERVE_DIR}/results.json" \
        "${PRESERVE_DIR}/config.json" \
        > "${PRESERVE_DIR}/checksums.sha256"
    echo "[cleanup] Checksums written:"
    cat "${PRESERVE_DIR}/checksums.sha256"

    # -- success marker --------------------------------------------------------
    echo "job_id=${SLURM_JOB_ID} mode=${MODE} run_name=${RUN_NAME} \
seed=${SEED} teacher=${TEACHER} student=${STUDENT} \
completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > "${PRESERVE_DIR}/SUCCESS"

    echo "[cleanup] Preservation complete."
}

# =============================================================================
# Helper: delete_large_artifacts
#   Removes everything that can be reproduced from config + seed.
#   Only called after preserve_artifacts succeeds.
# =============================================================================
delete_large_artifacts() {
    echo "[cleanup] Deleting large reproducible artifacts ..."

    # Intermediate LoRA checkpoints (optimizer + scheduler states live here)
    if [ -d "${RUN_DIR}/student" ]; then
        find "${RUN_DIR}/student" -maxdepth 1 -name "checkpoint-*" -type d \
            -exec rm -rf {} + 2>/dev/null || true
        echo "[cleanup] Removed intermediate checkpoints."
    fi

    # Final merged student weights (large; re-generate from config + seed)
    if [ -d "${STUDENT_FINAL}" ]; then
        rm -rf "${STUDENT_FINAL}"
        echo "[cleanup] Removed final student weights."
    fi

    # Proxy-student gradients / perturbation directions
    if [ -f "${GRAD_FILE}" ]; then
        rm -f "${GRAD_FILE}"
        echo "[cleanup] Removed proxy gradient file."
    fi

    # Generated traces (training, holdout, evaluation)
    if [ -d "${TRACES_DIR}" ]; then
        rm -rf "${TRACES_DIR}"
        echo "[cleanup] Removed traces directory."
    fi

    # Tokenized dataset caches
    if [ -d "${CACHE_DIR}" ]; then
        rm -rf "${CACHE_DIR}"
        echo "[cleanup] Removed tokenized cache."
    fi

    echo "[cleanup] Done. Disk reclaimed."
}

# =============================================================================
# Launch pipeline
# =============================================================================
echo "============================================================"
echo "Job ID   : ${SLURM_JOB_ID}"
echo "Node     : $(hostname)"
echo "GPU      : 1 × 40 GB A100"
echo "Mode     : ${MODE}  (${RUN_NAME})"
echo "dtype    : auto (BF16 teacher / FP16 proxy / FP32 grads)"
echo "============================================================"

python run.py \
    $MODE_FLAGS \
    --attn_impl=flash_attention_2 \
    \
    --teacher=$TEACHER \
    --proxy_student=$PROXY_STUDENT \
    --student=$STUDENT \
    --student_tokenizer=$STUDENT_TOK \
    \
    --dataset=$DATASET \
    --seed=$SEED \
    --exp_dir=$EXP_DIR \
    \
    --tau=1.0 \
    --top_p=0.95 \
    --eps=1e-2 \
    --gen_batch_size=$GEN_BATCH \
    --max_new_tokens=1024 \
    --max_prompt_length=512 \
    --answer_force=true \
    \
    --grad_batch_size=$GRAD_BATCH \
    \
    --lora=true \
    --lora_r=128 \
    --lora_alpha=128 \
    --lr=5e-4 \
    --train_batch_size=$TRAIN_BATCH \
    --per_device_batch_size=$PER_DEVICE_BATCH \
    --num_epochs=3 \
    --train_max_length=4096 \
    \
    --num_gpus=1 \
    --launcher=python \
    --eval_teacher=true
PIPELINE_EXIT=$?

# =============================================================================
# Post-run: preserve then clean, or bail out on failure
# =============================================================================
if [ "$PIPELINE_EXIT" -ne 0 ]; then
    echo ""
    echo "================================================================"
    echo "PIPELINE FAILED (exit code ${PIPELINE_EXIT}). Skipping cleanup."
    echo "Artifacts retained in ${EXP_DIR}/ for debugging."
    echo "================================================================"
    exit "$PIPELINE_EXIT"
fi

echo ""
echo "================================================================"
echo "Pipeline succeeded. Starting preservation + cleanup ..."
echo "================================================================"

if preserve_artifacts; then
    delete_large_artifacts
    echo ""
    echo "================================================================"
    echo "Run complete. Preserved artifacts: ${PRESERVE_DIR}/"
    echo "================================================================"
else
    echo ""
    echo "================================================================"
    echo "Preservation FAILED — skipping deletion to avoid data loss."
    echo "Check ${PRESERVE_DIR}/ and ${EXP_DIR}/ manually."
    echo "================================================================"
    exit 1
fi
