#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
config_path="${repo_root}/pufferlib/config/ocean/drive_try1_mix_ppo.ini"

python -m pufferlib.pufferl train puffer_drive \
  --config "${config_path}" \
  "$@"
