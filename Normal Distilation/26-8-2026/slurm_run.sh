#!/usr/bin/env bash
# =============================================================================
# Antidistillation Sampling — SLURM job script
# Target: 40 GB A100 nodes, fp32
#
# Memory budget per GPU (Mixed Precision):
#   Stage 1 (Holdout)   : teacher 7B (BF16) = ~14 GB
#   Stage 2 (Gradients) : proxy 3B (FP32) = ~12.4 GB (Teacher/Student not loaded)
#   Stage 3 (ADS Gen)   : teacher 7B (BF16) + 2× proxy 3B (FP16) = 14+6+6 = ~26.4 GB
#   Stage 4 (Distill)   : student 3B (BF16, LoRA) = fits easily
#
# Select mode by changing the MODE variable below:
#   MODE=normal   →  ADS=false NORMAL=true
#   MODE=ads      →  ADS=true  NORMAL=false
# =============================================================================

#SBATCH --job-name=antidistil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # one process per GPU
#SBATCH --gres=gpu:a100_40gb:4       # 4 × 40 GB A100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_antidistil.out
#SBATCH --error=logs/%j_antidistil.err

# ── adjust to your cluster ────────────────────────────────────────────────────
#SBATCH --partition=gpu
# #SBATCH --account=my_project       # uncomment if required
# #SBATCH --qos=high                 # uncomment if required

set -euo pipefail

# =============================================================================
# Configuration — edit here
# =============================================================================
MODE=normal        # normal | ads
EXP_DIR=experiments

TEACHER=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
STUDENT=meta-llama/Llama-3.2-3B
STUDENT_TOK=meta-llama/Llama-3.2-3B-Instruct

# Proxy must share the teacher tokenizer (Qwen family).
PROXY_STUDENT=Qwen/Qwen2.5-3B

DATASET=gsm8k
SEED=42

# ── batch sizes ───────────────────────────────────────────────────────────────
if [ "$MODE" = "ads" ]; then
    GEN_BATCH=8           # teacher + 2× proxy active simultaneously
    GRAD_BATCH=1
else
    GEN_BATCH=16          # teacher only
    GRAD_BATCH=2
fi

# Global train batch 8 across 4 GPUs → per_device=2, accum=1
TRAIN_BATCH=8
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
# Mode flags
# =============================================================================
if [ "$MODE" = "ads" ]; then
    MODE_FLAGS="--ads=true --normal=false"
else
    MODE_FLAGS="--ads=false --normal=true"
fi

# =============================================================================
# Launch
# =============================================================================
echo "============================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "GPUs     : $SLURM_GPUS_ON_NODE  (40 GB A100 × 4)"
echo "Mode     : $MODE"
echo "dtype    : auto (mixed precision)"
echo "============================================================"

srun python run.py \
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
    --num_gpus=4 \
    --launcher=accelerate \
    --eval_teacher=true
