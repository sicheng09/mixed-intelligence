# APEX and PPO baseline training

APEX and the PPO baseline share the WOMD medium data, simulator, focal-policy
architecture, driving reward and benchmark. APEX adds a separately trained
behavior-conditioned traffic policy and a population constructor. The default
APEX feedback mode is `first_episode`.

## Shared data and evaluation

Use [APEX_SETUP.md](docs/APEX_SETUP.md) for installation and the common
10,000-scene medium training-data preparation. Both methods read
`$DRIVE_BINARIES_DATA_ROOT/training`. Prepare the held-out WOMD validation data
and Interactive1k / Random1k splits once using [BENCHMARK.md](BENCHMARK.md);
both methods use those same files and manifests.

The commands below use 20 validation maps for training-time evaluation.
Place them under `$DRIVE_BINARIES_DATA_ROOT/validation` before training.
Final comparisons use all 1,000 scenes in each benchmark split, one fixed
focal checkpoint per method, and the same traffic controllers and metric
settings. See [EVALUATION.md](EVALUATION.md).

## Start training

Run from the repository root and select an available GPU with
`CUDA_VISIBLE_DEVICES`. Configuration precedence is command line, selected
INI, then `pufferlib/config/default.ini`.

```bash
source scripts/apex_env.sh train

# Common medium-data, reproducibility and validation settings.
common=(
  --train.seed 42
  --train.reproducibility True
  --env.num-maps 10000
  --vec.num-envs 16 --vec.num-workers 16
  --eval.split validation --eval.num-maps 20
  --eval.wosac-realism-eval False --eval.human-replay-eval False
)

# APEX: 4B total agent steps, nominally 2B per driving policy.
puffer train puffer_drive \
  --config pufferlib/config/ocean/drive_apex_medium.ini \
  "${common[@]}" \
  --train.name apex_medium \
  --train.data-dir "$APEX_WORKSPACE/runs/apex_medium" \
  --train.population-monitor-dir "$APEX_WORKSPACE/runs/apex_medium/population"

# PPO baseline: 2B agent steps for one policy.
puffer train puffer_drive \
  --config pufferlib/config/ocean/drive.ini \
  "${common[@]}" \
  --train.name ppo_baseline_medium \
  --train.data-dir "$APEX_WORKSPACE/runs/ppo_baseline_medium"
```

Choose a new output directory for each independent run. The activation script
keeps environments, caches, temporary files and outputs under `APEX_WORKSPACE`.
`reproducibility=True` seeds the Python/NumPy/PyTorch paths and enables
reproducible Drive/vector execution; it requires a C binding exporting
`SEEDED_SHARED=1`. The supplied build includes that support.

## Driving-policy parameter alignment

These values are the effective settings of the commands above. APEX-specific
values come from `drive_apex_medium.ini`; PPO uses `drive.ini` with the shared
CLI overrides. Both inherit the optimizer and precision from `default.ini`.

| Parameter | PPO baseline | APEX |
|---|---|---|
| Training scenarios / split | 10,000 / training | Same |
| Seed / reproducibility | 42 / True | Same |
| Policies | One shared `Drive` | Focal `Drive` + traffic `DriveBehaviorConditioned` |
| Policy allocation | 100% shared policy | 50% focal / 50% traffic, stratified |
| Total agent steps | 2,000,000,000 | 4,000,000,000 |
| Agent steps per driving policy | 2B | Approximately 2B each |
| Rollout batch, `train.batch_size` | 524,288 | 1,048,576 combined; 524,288 per policy |
| `minibatch_size` / `max_minibatch_size` | 32,768 / 32,768 | Same, separately for each policy |
| `update_epochs` | 1 | 1 per policy |
| Optimizer steps per rollout | 16 | 16 per policy (`mix_ppo_policy_update_steps=16,16`) |
| `bptt_horizon` | 32 | Same |
| Encoder embedding / hidden size | 64 / 256 | Same for both policies |
| Recurrent model | LSTM, input 256 / hidden 256 | Separate LSTM per policy, same dimensions |
| `optimizer` / `learning_rate` | Muon / 0.003 | Independent Muon optimizer per policy / 0.003 |
| `adam_beta1`, `adam_beta2`, `adam_eps` | 0.9, 0.999, 1e-8 | Same; these config keys are also passed to Muon |
| `anneal_lr` | True, cosine schedule | Same, independently per policy |
| `gamma` / `gae_lambda` | 0.98 / 0.95 | Same |
| `clip_coef` / `vf_clip_coef` | 0.2 / 0.2 | Same |
| `vf_coef` / `ent_coef` | 2.0 / 0.005 | Same |
| `max_grad_norm` | 1.0 | Same |
| `prio_alpha` / `prio_beta0` | 0.85 / 0.85 | Same |
| `vtrace_rho_clip` / `vtrace_c_clip` | 1.0 / 1.0 | Same |
| Precision / compilation | float32 / `compile=False` | Same |
| Checkpoint / evaluation interval | 1,000 trainer updates | Same |
| Warm start / opponent pool | None / disabled | Same |

An agent step is one agent transition, summed across the vector environments.
The APEX total includes both driving policies. Stratified sampling keeps each
policy's rollout size and optimization budget aligned with the single-policy
baseline.

### Shared simulator and focal reward

| Parameter | Value for both methods |
|---|---|
| `vec.num_envs` / `vec.num_workers` / `vec.batch_size` | 16 / 16 / 4 |
| `env.num_agents` | 1,024 agent slots per vector environment |
| Dynamics / actions | Classic bicycle / 91 discrete actions (7 acceleration x 13 steering) |
| `dt` / `episode_length` / `resample_frequency` | 0.1 s / 91 steps / 910 steps |
| `control_mode` / `init_mode` / `init_steps` | control_vehicles / create_all_valid / 0 |
| `collision_shrink` | 0.7 |
| `reward_vehicle_collision` / `reward_offroad_collision` | -0.5 / -0.5 |
| `reward_goal` / `reward_goal_post_respawn` | 1.0 / 0.25 |
| `reward_l_align_vel` | 0.5 |
| Other configured driving-reward coefficients | 0 |
| `goal_radius` / `goal_speed` / `goal_target_distance` | 2 m / 100 / 30 m |
| `goal_behavior` / `collision_behavior` / `offroad_behavior` | 3 (remove) / 2 (remove) / 2 (remove) |
| `termination_mode` | 1 (all agents done) |
| Training-time scripted traffic | Disabled: `mix_traffic=False`, `idm_others=False` |

`num_envs=16` is the total number of vector environments, with one per worker
at these settings. `vec.batch_size=4` is the number received in a vector batch.
Together they allocate 16,384 agent slots.

The classic base observation has 1,151 floats: 7 ego features, 31 partners x 8
features and 128 road segments x 7 features. APEX appends six condition values:
the traffic policy uses them and masks partner observations according to its
sampled perception conditions; the focal `Drive` policy uses the base features.

## APEX population constructor

The traffic condition contains four reward coordinates and two perception
coordinates. A 12-dimensional Gaussian latent descriptor parameterizes six
Beta distributions. A new descriptor is sampled per simulator subenvironment
for each 910-step population window.

| Parameter | Value |
|---|---|
| `behavior_conditioning` / `behavior_conditioned_policy_id` | True / 1 (traffic) |
| Radius sampling / maximum | continuous / 50 m; sampled radius in [20, 50] m |
| View sampling / maximum | discrete_caps / 360 degrees; caps 90, 180, 270, 360 degrees |
| `population_adversary` | True |
| `population_reward_mode` / `population_gamma` | first_episode / 0.98 |
| `population_epsilon` | 0.02 |
| `population_rho_min` / `population_rho_max` | 0.02 / 0.5 |
| `population_std_start` / `population_std_end` | 0.5 / 0.2 |
| `population_std_decay_samples` | 600,000 descriptor records |
| `population_batch_size` | 256 minimum records per constructor update |
| `population_minibatch_size` / `population_update_epochs` | 64 / 4 |
| `population_clip_coef` / `population_target_kl` | 0.2 / 0.03 |
| Optimizer / `population_learning_rate` | Adam / 0.0003 |
| `population_max_grad_norm` | 1.0 |

### Feedback modes

**first_episode (default).** For each new descriptor, the constructor uses the
first focal episode group's mean critic-centered signal:
`V_start - G_bootstrapped`. Returns use discount 0.98; time-limit truncations
include the continuation value. Later episodes in the same window continue
training both driving policies. The constructor centers and standardizes its
record batch before the clipped population-policy update.

**legacy_window (optional).** Accumulate the critic-centered signals from
successive episode groups in the same descriptor window. An episode group
starting at `start_step` receives weight
`population_episode_decay ** (start_step / episode_length)`.
The default decay 0.8 emphasizes earlier episodes. This gives the constructor
feedback from repeated interactions under the same descriptor.

Append these options to the APEX command to select it:

```text
--train.population-reward-mode legacy_window --train.population-episode-decay 0.8
```

`population_episode_decay` applies to `legacy_window`. Population records and
update summaries are written to the `population-monitor-dir` set above.
The PPO baseline has no population constructor.

## Checkpoints and continuation

Training saves `model_puffer_drive_NNNNNN.pt`. APEX also saves
`model_policy_0_puffer_drive_NNNNNN.pt` (focal),
`model_policy_1_puffer_drive_NNNNNN.pt` (traffic), and a combined `mix_model`.
The suffix is the trainer update counter. The provided evaluation files are
`weights/apex_focal_medium_epoch3815.pt` and
`weights/ppo_baseline_medium_epoch3815.pt`.

To initialize a policy from weights, use `--load-model-path`. For full training
continuation, retain the complete run directory, `trainer_state.pt`, both APEX
policy checkpoints and optimizer state; the exported focal file is used for
standalone evaluation.

## Training entry points

| Entry point | Configuration and use |
|---|---|
| Explicit APEX command above | `drive_apex_medium.ini`: first_episode, 10k maps, 360-degree view caps |
| Explicit PPO command above | `drive.ini` with `--env.num-maps 10000` and common validation/reproducibility overrides |
| `scripts/slurm/start_try1_mix_ppo.sh` | Selects `drive_try1_mix_ppo.ini`: 79,000 maps, 180-degree maximum view, continuous view sampling by default; accepts CLI overrides |
| `scripts/slurm/run_train.sh` | Cluster-specific Slurm wrapper around `puffer train`; takes config and overrides through `"$@"` |
| `scripts/population/main_train.sh` | Separate uniform/fixed/adaptive population experiments using `drive_population_main.ini` |

For the medium APEX/PPO comparison, use the explicit commands above. They select
the data count, reproducibility settings, validation split and APEX monitoring
directory directly.
