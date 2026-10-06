# Reproducible population publication

`mixed-intelligence` already includes `b805e027`, which seeds Python, NumPy,
Torch, CUDA and Drive map selection when `train.reproducibility=True`.
That flag defaults to `False`. Drive requires a rebuilt C binding exposing
`SEEDED_SHARED`; an older binding fails explicitly when reproducibility is enabled.

Seeding alone did not make adaptive population training reproducible. The parent
previously overwrote every worker's shared mean/std or mixture immediately.
A worker could be replacing maps inside its current STEP at that moment, so
process timing determined which published descriptor it sampled. Ordered result
reception did not order these concurrent reads and writes.

In reproducible Multiprocessing runs, publication now makes a private copy in the
parent. `send()` copies the latest snapshot into only the workers receiving the
next STEP, after their previous result has returned. `async_reset()` first waits
for outstanding commands, publishes all rows, and then sends RESET. Worker host
RNGs are also seeded before constructing environments, as well as at reset.

The population distribution is still sampled at its existing map-window boundary.
This patch changes delivery ordering; it keeps the reward definition, bootstrap,
PPO objective, update frequency and sampling transformations. With reproducibility
disabled, publication keeps its immediate behavior. Serial and Ray publication
are not changed by this patch.

## Validation on the original mixed-intelligence functionality

The migration was tested against `5f9648b3`, with the full 12-coordinate descriptor,
without the view-only, truncated-normal or counterfactual-score experiments.
Fresh processes used seed 42, 4 workers, 4 native environments, receive batch 2,
32 agent slots per environment, a 10,000-map training pool, 16,384 training steps
and 32 driver updates. Map replacement every 12 simulation steps and std decay
over 256 records make the original race observable in a short run. A paired run
delays worker 44 by 150 ms at map replacement without consuming RNG.

| Comparison | Outcome |
| --- | --- |
| Original CPU implementation, with/without delay | Records, transitions, policies and q diverged |
| Fixed CPU implementation, with/without delay | All 150 records, worker transitions, all 70 driving-policy tensors and q matched |
| Fixed GPU 4 / Muon, with/without delay | All 150 records, worker transitions, all 70 driving-policy tensors and q matched |
| Plain PPO, original versus fixed | Transitions, every update hash and all 32 final policy tensors matched |
| Fixed implementation, seed 42 versus 43 | Initial policies, sampled maps/conditions and final parameters differed |

The regression suite covers population objectives, KL stopping, first-episode
rewards, bootstrap, phase runtime, frozen rollouts and summaries. New real-process
tests cover in-flight publication, caller array mutation, worker constructor RNGs,
integer/slice/list worker selection, multiple environments per worker, reset
publication and the disabled option. Two pre-existing population test fixtures
were completed with their required monitor/update fields; production trainer
code was not changed for that fixture repair.

The retained commands, configs, source hashes, traces, checkpoints and comparison
results are in `.logs/New1/Try4/seed_mainline_audit_20261006/`. The tests can be run
with:

```bash
python -m pytest -q tests/test_population_publication.py tests/test_drive_reproducibility.py tests/test_try1_mix_ppo.py tests/test_main_population.py tests/test_population_experiment.py tests/test_frozen_population_rollout.py tests/test_anchored_population.py tests/test_population_summary.py
```

## Scope

These are fresh-start checks on the same software/hardware setup and worker
layout. They establish the command-publication fix and short-run repeatability,
not bitwise identity for a complete 4B run or across hardware/library versions.
Legacy checkpoint resume omits complete worker environment/RNG state and does not
reproduce uninterrupted training. Existing processes keep their loaded code.

The descriptor and per-agent conditions still consume their original shared RNG
stream. Different reset counts can therefore alter later descriptor draws; that
coupling alone does not show a failure under identical deterministic inputs.
Changing the RNG streams or critic bootstrap would change training sequences or
targets and is outside this verified scheduling repair.
