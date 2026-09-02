#!/usr/bin/env bash
# Submit one cache job, then a three-seed training array with at most two GPUs.
set -Eeuo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
TRAIN_TRACES=${TRAIN_TRACES:?set TRAIN_TRACES to the normal teacher-training traces}
RUN_ROOT=${RUN_ROOT:-/raid/mtanveer/top64_soft_distillation}
VENV_DIR=${VENV_DIR:-/raid/mtanveer/antidistill_env}
SEEDS=${SEEDS:-"42 123 456"}
mkdir -p "${RUN_ROOT}/logs"

"${VENV_DIR}/bin/python" "${REPO_DIR}/scripts/preflight.py" \
    --repo_dir "${REPO_DIR}" --traces "${TRAIN_TRACES}" --run_root "${RUN_ROOT}"

COMMON_EXPORT="ALL,REPO_DIR=${REPO_DIR},TRAIN_TRACES=${TRAIN_TRACES},RUN_ROOT=${RUN_ROOT},VENV_DIR=${VENV_DIR},SEEDS=${SEEDS}"
CACHE_JOB=$(sbatch --parsable \
    --output="${RUN_ROOT}/logs/top64_cache_%j.out" \
    --error="${RUN_ROOT}/logs/top64_cache_%j.err" \
    --export="${COMMON_EXPORT}" \
    "${REPO_DIR}/slurm/top64_precompute.slurm")

TRAIN_JOB=$(sbatch --parsable \
    --dependency="afterok:${CACHE_JOB}" \
    --array=0-2%2 \
    --output="${RUN_ROOT}/logs/top64_train_%A_%a.out" \
    --error="${RUN_ROOT}/logs/top64_train_%A_%a.err" \
    --export="${COMMON_EXPORT}" \
    "${REPO_DIR}/slurm/top64_train_array.slurm")

echo "Cache job: ${CACHE_JOB}"
echo "Training array: ${TRAIN_JOB} (three seeds, maximum two concurrent GPUs)"
echo "Monitor: squeue -j ${CACHE_JOB},${TRAIN_JOB}"
