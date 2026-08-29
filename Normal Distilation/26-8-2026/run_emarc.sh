#!/usr/bin/env bash
set -euo pipefail

# Pass any model/data/hardware overrides after this script. Example:
#   ./run_emarc.sh --dataset=gsm8k --tau=1 --emarc_horizon_steps=1
python run.py \
  --emarc=true \
  --ads=false \
  --normal=false \
  "$@"
