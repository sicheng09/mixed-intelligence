# Environment and data setup

## Workspace

By default, `APEX_WORKSPACE` is the parent directory of this repository. The
installer and activation script resolve it automatically. Set `APEX_WORKSPACE`
before installation to use a different directory on your data volume.

```bash
cd mixed-intelligence
bash scripts/setup_apex.sh
```

The setup uses two isolated environments:

| Environment | Python | Stack | Use |
|---|---|---|---|
| `apex-train` | 3.11 | PyTorch 2.5.1 + CUDA 11.8, PyG 2.7 | Simulator, training, binary conversion, evaluation |
| `apex-data` | 3.10 | TensorFlow 2.11, Waymo SDK 1.6.1 | Raw WOMD processing |

Activate the appropriate environment:

```bash
source scripts/apex_env.sh train
# For raw-data utilities:
source scripts/apex_env.sh data
```

The environments are stored in `$APEX_WORKSPACE/envs`. Package caches and temporary
files use `$APEX_WORKSPACE/.apex`; outputs use `$APEX_WORKSPACE/runs`. For sites
requiring a proxy, set `APEX_PROXY_URL` or use the workspace-local `.apex/proxy-url`.
Network configuration is separate from the repository.

## Build and dependencies

The separate environments avoid incompatibilities between Waymo's legacy
TensorFlow/NumPy stack and the modern PyTorch trainer. Dependency specifications
and validated Linux package locks are under `requirements/`.

The training build requires a C/C++ compiler, the CUDA 11.8 toolkit, and a
CUDA-capable GPU supported by the pinned PyTorch/CUDA stack. It builds Drive and
the advantage kernel with `PUFFER_DRIVE_ONLY=1`. By default, PyTorch selects CUDA
architectures from the GPUs visible during the build. When building without a
visible GPU or for different target GPUs, set `TORCH_CUDA_ARCH_LIST` to the target
compute capabilities supported by the toolkit before running the installer.
The pinned `external/pyxodr` submodule is installed by the setup script.

## Training data

The medium dataset is the 10,000-scene `training/group_0` subset of
[EMERGE-lab/GPUDrive](https://huggingface.co/datasets/EMERGE-lab/GPUDrive), fixed at
revision `96fdbc298733daca450c42f7fad441d21467a717`.

```bash
source scripts/apex_env.sh train
python scripts/prepare_apex_medium.py --download --convert --workers 4
```

This command downloads missing files, retries interrupted transfers, verifies the
10,000-scene count, and creates `.bin` maps under
`$APEX_WORKSPACE/datasets/drive-binaries/training`. A source manifest records the
revision and JSON-to-binary correspondence. Existing binary outputs are checked
before reuse; a partial or mismatched output is retained for inspection rather
than overwritten.

The medium JSON-to-binary step uses the training environment. Raw Waymo utilities
use the data environment. The validation pipeline can invoke the latter through
`--tf-python`, as shown in [BENCHMARK.md](../BENCHMARK.md).

## Training and benchmark

APEX and PPO share the training data, binary-conversion pipeline and evaluation
splits described above. [TRAINING.md](../TRAINING.md) contains their launch
commands, aligned hyperparameters and checkpoint handling. APEX uses
`first_episode` feedback by default; the optional `legacy_window` feedback and
its episode weighting are described in the same document.

Use [BENCHMARK.md](../BENCHMARK.md) to prepare the common held-out validation
splits and [EVALUATION.md](../EVALUATION.md) for planner and traffic settings.
