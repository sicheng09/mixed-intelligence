#!/usr/bin/env bash
set -euo pipefail
# Explicit matched-driver-budget branch. Set CUDA_VISIBLE_DEVICES externally.
mode="${1:?mode: uniform, fixed or adaptive}"
output_root="${2:?provide a new output root}"
case "$mode" in uniform|fixed|adaptive) ;; *) exit 2 ;; esac
if [ -e "$output_root/$mode" ] || [ -e "$output_root/$mode.log" ]; then
  echo 'Output already exists; choose a new output root. This command starts a fresh branch.' >&2
  exit 2
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PUFFER_DISABLE_VIDEO=1
export DRIVE_BINARIES_DATA_ROOT="${DRIVE_BINARIES_DATA_ROOT:-$PWD/data/binaries_full}"
mkdir -p "$output_root"
# 96 full batches: 50,331,648 driving transitions per policy, fresh optimizers.
.venv/bin/python -m pufferlib.pufferl train puffer_drive \
  --config pufferlib/config/ocean/drive_population_main.ini \
  --train.population-mode "$mode" --train.total-timesteps 100663296 \
  --train.data-dir "$output_root/$mode" --env.num-maps 10000 \
  --train.mix-ppo-policy-paths experiments/puffer_drive_178958541908/model_policy_0_puffer_drive_002000.pt,experiments/puffer_drive_178958541908/model_policy_1_puffer_drive_002000.pt \
  --eval.num-maps 20 > "$output_root/$mode.log" 2>&1
