#!/usr/bin/env bash
# Install isolated environments; this does not download data or start training.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo/scripts/apex_env.sh" paths
bootstrap="$APEX_WORKSPACE/.tools/uv-venv"
if [ ! -x "$bootstrap/bin/uv" ]; then
    bootstrap_python="${APEX_BOOTSTRAP_PYTHON:-python3}"
    if [ -x /opt/conda/bin/python ] && [ -z "${APEX_BOOTSTRAP_PYTHON:-}" ]; then
        bootstrap_python=/opt/conda/bin/python
    fi
    "$bootstrap_python" -m venv "$bootstrap"
    "$bootstrap/bin/python" -m pip install uv
fi
uv="$bootstrap/bin/uv"
"$uv" python install --no-bin 3.11 3.10
for mode in train data; do
    py=3.11
    if [ "$mode" = data ]; then py=3.10; fi
    if [ ! -x "$APEX_WORKSPACE/envs/apex-$mode/bin/python" ]; then
        "$uv" venv --python "$py" "$APEX_WORKSPACE/envs/apex-$mode"
    fi
done
train="$APEX_WORKSPACE/envs/apex-train/bin/python"
data="$APEX_WORKSPACE/envs/apex-data/bin/python"
"$uv" pip install --python "$train" torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
train_requirements="$repo/requirements/apex-train.txt"
data_requirements="$repo/requirements/apex-data.txt"
if [ -f "$repo/requirements/apex-train.lock.txt" ]; then train_requirements="$repo/requirements/apex-train.lock.txt"; fi
if [ -f "$repo/requirements/apex-data.lock.txt" ]; then data_requirements="$repo/requirements/apex-data.lock.txt"; fi
"$uv" pip install --python "$train" -r "$train_requirements" 'torch==2.5.1+cu118' \
    -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
"$uv" pip install --python "$train" --no-build torch-cluster==1.6.3 torch-scatter==2.1.2 \
    -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
"$uv" pip install --python "$data" -r "$data_requirements"
cd "$repo"
export PUFFER_DRIVE_ONLY=1
# PyTorch detects visible GPU architectures unless TORCH_CUDA_ARCH_LIST is set.
"$uv" pip install --python "$train" --no-build-isolation --no-deps -e .
git submodule update --init -- external/pyxodr
"$uv" pip install --python "$train" -e external/pyxodr
"$uv" pip check --python "$train"
"$uv" pip check --python "$data"
echo "Ready: source scripts/apex_env.sh train   (or data)"
