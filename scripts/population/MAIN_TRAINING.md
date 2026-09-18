# Population sampling in the main MixPPO trainer

Use the explicit `drive_population_main.ini` config. The historical
`drive_try1_mix_ppo.ini` retains its legacy Gaussian adversary path.

| `train.population_mode` | Driver training population | q optimization |
|---|---|---|
| `uniform` | Exact original six independent uniforms | Disabled |
| `fixed` | Five equal-probability anchors + Gaussian + Beta | Disabled |
| `adaptive` | Same anchors, learned probabilities with floor 0.05 | Short phases on current frozen driver snapshots |

All modes train both drivers. The first adaptive version deliberately uses
no actor context or scene baseline. It reuses the tested phase learner and KL
backtracking from `population_experiment.py`; the learner is not reimplemented.

The adaptive runtime runs synchronously after `population_phase_interval`
driver updates. It copies both current drivers to CPU, collects
`population_phase_samples` first episodes on randomly selected **training**
maps, and updates only q. Maps with fewer than two controlled agents are skipped
by eligibility, not by return. It verifies frozen source/snapshot hashes.
True terminals have zero continuation; time limits include bootstrap. These
extra trajectories never enter the driving PPO buffer. CPU torch/NumPy RNGs are
isolated, and q has separate sampling generators. The published probabilities,
descriptor sigma and version use a single-writer sequence-checked shared buffer;
workers apply them only on their next map window. Ongoing windows do not change.
The first CPU sampler is deliberately simple, not a throughput-optimized final
implementation. q scene role assignment is the evaluator's alternating local
assignment, while main worker roles remain the existing stable slot assignment.

The driver window is still 910 steps. q's separate first-episode sampler uses
an explicit reset and the configured episode horizon minus one. Main driver
rollout recurrent behavior is unchanged; the auxiliary phase starts recurrent
state at zero and does not reproduce intra-rollout training-state resets.

New dashboard fields show all five actual selection probabilities, q version,
challenge, KL, and extra q agent-steps. Environment info also reports completed
window versions/component fractions. `population_phases.jsonl` next to model
checkpoints stores complete phase records, driver hashes, counts and diagnostics.
q counters are separate from driver `global_step`. Stock end-of-run evaluation
rollouts are additional simulator work, not additional driver optimization.

## Engineering verification

From the repository root, using an available GPU:

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/population/main_smoke.sh adaptive
CUDA_VISIBLE_DEVICES=1 bash scripts/population/main_smoke.sh fixed
CUDA_VISIBLE_DEVICES=1 bash scripts/population/main_smoke.sh uniform
```

Each smoke uses 32 driver updates / 262144 shared training transitions, 2
workers, and the 910-step window. Adaptive phases are deliberately tiny
(16 samples every 8 updates); changing probabilities here is **not** an efficacy
result. The early 64-step-window attempt is separate and was stopped during
stock terminal evaluation because reset prevented sufficient episode reporting.

## Planned matched-budget pilot (not automatically started)

First lock a common evaluation protocol: fixed opponent checkpoint(s), the same
development maps/seeds for every trained SimplePPO, task completion alongside
collision/offroad rates, and separate stress groups. Never compare training
returns against each group's changing opponent. Keep confirmation maps sealed.
Then use new output directories:

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/population/main_train.sh uniform .logs/New2/experiments/main_pilot
CUDA_VISIBLE_DEVICES=1 bash scripts/population/main_train.sh fixed .logs/New2/experiments/main_pilot
CUDA_VISIBLE_DEVICES=1 bash scripts/population/main_train.sh adaptive .logs/New2/experiments/main_pilot
```

All three commands branch from the same epoch-2000 driver weights with fresh
optimizers, the same LR schedule, 16 workers and 96 full batches (50,331,648
transitions per policy). This is continued training, not training from scratch.
Report the adaptive sampler's extra simulation and wall time separately.
The scripts are launch templates; they do not yet run or summarize the final
fixed-opponent efficacy evaluation automatically.

## Checkpoints and limitations

New-mode checkpoints save both driver optimizers/schedulers and RNG state,
plus q optimizer/generators, version, schedule counters and sampling totals.
`--load-model-path` resumes a matching new-mode run, restoring both driver files
named in `trainer_state.pt`; mode/budget mismatches fail explicitly. Use
`mix_ppo_policy_paths` for a fresh driver-only branch from legacy weights.
Simulator state and worker RNGs are **not** checkpointed: resume is an explicit
global reset, not bitwise trajectory continuation. Full q phases are synchronous
and represented in the driver checkpoint; a crash before a checkpoint can leave
extra audit JSONL entries, which are not authoritative for restoration.

This path currently targets the audited two-policy recurrent discrete Drive
setup, single trainer rank, no IDM/mixed-controller traffic. Context transfer,
learned anchor residuals, conditioned reward scaling, broad architecture support
and final SimplePPO robustness confirmation remain separate work.
