#!/usr/bin/env bash
set -euo pipefail
# Engineering integration test only: not the efficacy or matched-budget pilot.
mode="${1:-adaptive}"
output_root="${2:-.logs/New2/experiments/main_smoke_910}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PUFFER_DISABLE_VIDEO=1
export DRIVE_BINARIES_DATA_ROOT="${DRIVE_BINARIES_DATA_ROOT:-$PWD/data/binaries_full}"
mkdir -p "$output_root"
resume_args=()
if [ -n "${3:-}" ]; then resume_args=(--load-model-path "$3"); fi
.venv/bin/python -m pufferlib.pufferl train puffer_drive \
  --config pufferlib/config/ocean/drive_population_main.ini "${resume_args[@]}" \
  --train.population-mode "$mode" --train.total-timesteps 262144 \
  --train.batch-size 8192 --train.minibatch-size 2048 --train.max-minibatch-size 2048 \
  --train.mix-ppo-policy-minibatch-sizes 2048,2048 --train.mix-ppo-policy-update-steps 2,2 \
  --train.population-phase-interval 8 --train.population-phase-samples 16 \
  --train.population-phase-epochs 4 --train.data-dir "$output_root/$mode" \
  --train.mix-ppo-policy-paths experiments/puffer_drive_178958541908/model_policy_0_puffer_drive_002000.pt,experiments/puffer_drive_178958541908/model_policy_1_puffer_drive_002000.pt \
  --env.num-agents 128 --env.num-maps 20 --env.resample-frequency 910 \
  --vec.num-envs 2 --vec.num-workers 2 --vec.batch-size 2 --eval.num-maps 2 \
  > "$output_root/$mode.log" 2>&1
