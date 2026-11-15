#!/usr/bin/env bash
# Source this file: source scripts/apex_env.sh train|data
_apex_mode="${1:-train}"
case "$_apex_mode" in train|data|paths) ;; *) echo 'Expected train or data' >&2; return 2 ;; esac
export APEX_WORKSPACE="${APEX_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export XDG_CACHE_HOME="$APEX_WORKSPACE/.apex/cache"
export XDG_CONFIG_HOME="$APEX_WORKSPACE/.apex/config"
export XDG_STATE_HOME="$APEX_WORKSPACE/.apex/state"
export TMPDIR="$APEX_WORKSPACE/.apex/tmp"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export UV_PYTHON_INSTALL_DIR="$APEX_WORKSPACE/.apex/python"
export UV_PYTHON_BIN_DIR="$APEX_WORKSPACE/.tools/bin"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_HUB_DISABLE_XET=1
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache"
export WANDB_DIR="$APEX_WORKSPACE/runs"
export WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb"
export WANDB_CONFIG_DIR="$XDG_CONFIG_HOME/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"
export DRIVE_DATA_ROOT="$APEX_WORKSPACE/datasets/gpudrive-medium"
export DRIVE_BINARIES_DATA_ROOT="$APEX_WORKSPACE/datasets/drive-binaries"
export PUFFER_DISABLE_VIDEO=1
export MAX_JOBS="${MAX_JOBS:-4}"
if [ -z "${APEX_PROXY_URL:-}" ] && [ -r "$APEX_WORKSPACE/.apex/proxy-url" ]; then
    IFS= read -r APEX_PROXY_URL < "$APEX_WORKSPACE/.apex/proxy-url" || true
    export APEX_PROXY_URL
fi
if [ -n "${APEX_PROXY_URL:-}" ]; then
    export http_proxy="$APEX_PROXY_URL" https_proxy="$APEX_PROXY_URL"
fi
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_STATE_HOME" "$TMPDIR" "$WANDB_DIR" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR"
if [ -d /usr/local/cuda-11.8 ]; then
    export CUDA_HOME=/usr/local/cuda-11.8
    export PATH="$CUDA_HOME/bin:$PATH"
fi
if [ "$_apex_mode" != paths ]; then
    source "$APEX_WORKSPACE/envs/apex-$_apex_mode/bin/activate"
fi
unset _apex_mode
