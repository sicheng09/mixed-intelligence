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

## Training configurations

- `drive_apex_medium.ini`: 10k, view-cap APEX configuration with first-exposure
  population feedback by default.
- `drive_try1_mix_ppo.ini`: historical Gaussian population path with different
  checked-in first-exposure/view defaults.
- `drive_population_main.ini`: separate uniform/fixed/adaptive anchored-mixture
  experiments.

## Population feedback modes

The default `population_reward_mode = first_episode` uses only the focal agents'
first-episode value-return differences after sampling a new population descriptor.
The two driving policies continue learning through later episodes in the window.
This mode follows the manuscript's first-exposure objective and centering analysis.

For a denser feedback signal across repeated interactions, select
`population_reward_mode = legacy_window`. It accumulates per-episode value-return
differences within a descriptor's map window, weighted by episode start time.
`population_episode_decay = 0.8` emphasizes earlier episodes; this parameter is
ignored by `first_episode`. This is an optional multi-exposure training objective.
Enable it by appending the following overrides to the training command:

```text
--train.population-reward-mode legacy_window --train.population-episode-decay 0.8
```

The provided `apex_focal_seed42_4b.pt` contains focal-policy weights from the
multi-exposure variant reported in the manuscript. The simplified filename does
not change its parameters or training provenance. A complete training resume also
needs the traffic-policy and optimizer state. See [TRAINING.md](../TRAINING.md)
and [EVALUATION.md](../EVALUATION.md).
