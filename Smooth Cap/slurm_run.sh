#!/bin/bash
#SBATCH --job-name=antidistil
#SBATCH --partition=gpu
#SBATCH --qos=iitdgx
#SBATCH --account=faculty
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=UNLIMITED
#SBATCH --requeue
#SBATCH --output=/path/to/logs/run_%j.out
#SBATCH --error=/path/to/logs/run_%j.err

set -u

source /path/to/virtual_environment/bin/activate
cd /path/to/repository

export HF_HOME=/raid/username/model_cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TMPDIR=/raid/username/tmp
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =============================================================================
# Configuration
# =============================================================================
MODE=ads           # normal | ads
EXP_DIR=experiments

TEACHER=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
STUDENT=meta-llama/Llama-3.2-3B
STUDENT_TOK=meta-llama/Llama-3.2-3B-Instruct
PROXY_STUDENT=Qwen/Qwen2.5-3B

DATASET=gsm8k
SEED=42

if [ "$MODE" = "ads" ]; then
    MODE_FLAGS="--ads=true --normal=false"
    RUN_NAME="ads_tau1_lmin0.01_lmax0.075_eps0.01"
    GEN_BATCH=8
    GRAD_BATCH=1
else
    MODE_FLAGS="--ads=false --normal=true"
    RUN_NAME="normal_tau1"
    GEN_BATCH=16
    GRAD_BATCH=2
fi

RUN_DIR="${EXP_DIR}/${RUN_NAME}"
PRESERVE_DIR="${RUN_DIR}/preserved"

# =============================================================================
# Preserve key artifacts, write checksums and success marker
# =============================================================================
preserve_artifacts() {
    echo "[cleanup] Preserving artifacts to ${PRESERVE_DIR} ..."
    mkdir -p "${PRESERVE_DIR}"

    local missing=0
    for f in "${RUN_DIR}/results.json" "${RUN_DIR}/config.json"; do
        if [ ! -f "$f" ]; then
            echo "[cleanup] MISSING: $f" >&2
            missing=1
        fi
    done
    [ "$missing" -ne 0 ] && return 1

    cp "${RUN_DIR}/results.json" "${PRESERVE_DIR}/results.json"
    cp "${RUN_DIR}/config.json"  "${PRESERVE_DIR}/config.json"

    # SLURM logs (written by the scheduler, may still be flushing)
    for ext in out err; do
        log="/path/to/logs/run_${SLURM_JOB_ID}.${ext}"
        [ -f "$log" ] && cp "$log" "${PRESERVE_DIR}/"
    done

    sha256sum \
        "${PRESERVE_DIR}/results.json" \
        "${PRESERVE_DIR}/config.json" \
        > "${PRESERVE_DIR}/checksums.sha256"

    printf "job_id=%s mode=%s run_name=%s seed=%s teacher=%s student=%s completed_at=%s\n" \
        "$SLURM_JOB_ID" "$MODE" "$RUN_NAME" "$SEED" "$TEACHER" "$STUDENT" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${PRESERVE_DIR}/SUCCESS"

    echo "[cleanup] Preserved: results.json, config.json, logs, checksums, SUCCESS"
}

# =============================================================================
# Delete large reproducible artifacts (only after preservation succeeds)
# =============================================================================
delete_large_artifacts() {
    echo "[cleanup] Deleting large reproducible artifacts ..."

    # Intermediate LoRA checkpoints (optimizer / scheduler / scaler states)
    find "${RUN_DIR}/student" -maxdepth 1 -name "checkpoint-*" -type d \
        -exec rm -rf {} + 2>/dev/null || true

    # Final merged student weights
    rm -rf "${RUN_DIR}/student/final"

    # Proxy-student gradients / perturbation directions
    rm -f "${EXP_DIR}/proxy_student_grads.pt"

    # Generated traces (training, holdout, evaluation)
    rm -rf "${EXP_DIR}/traces"

    # Tokenized dataset caches and temporary files
    rm -rf "${EXP_DIR}/.cache"

    echo "[cleanup] Done."
}

# =============================================================================
# Run pipeline
# =============================================================================
python run.py \
    $MODE_FLAGS \
    --attn_impl=flash_attention_2 \
    --teacher=$TEACHER \
    --proxy_student=$PROXY_STUDENT \
    --student=$STUDENT \
    --student_tokenizer=$STUDENT_TOK \
    --dataset=$DATASET \
    --seed=$SEED \
    --exp_dir=$EXP_DIR \
    --tau=1.0 \
    --top_p=0.95 \
    --eps=1e-2 \
    --gen_batch_size=$GEN_BATCH \
    --max_new_tokens=1024 \
    --max_prompt_length=512 \
    --answer_force=true \
    --grad_batch_size=$GRAD_BATCH \
    --lora=true \
    --lora_r=128 \
    --lora_alpha=128 \
    --lr=5e-4 \
    --train_batch_size=4 \
    --per_device_batch_size=2 \
    --num_epochs=3 \
    --train_max_length=4096 \
    --num_gpus=1 \
    --launcher=python \
    --eval_teacher=true
PIPELINE_EXIT=$?

# =============================================================================
# Post-run: preserve then clean on success; retain everything on failure
# =============================================================================
if [ "$PIPELINE_EXIT" -ne 0 ]; then
    echo "Pipeline FAILED (exit ${PIPELINE_EXIT}). Retaining all artifacts for debugging."
    exit "$PIPELINE_EXIT"
fi

if preserve_artifacts; then
    delete_large_artifacts
    echo "Run complete. Preserved artifacts: ${PRESERVE_DIR}/"
else
    echo "Preservation FAILED — skipping deletion to avoid data loss."
    exit 1
fi
