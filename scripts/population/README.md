# Frozen-driver population experiments

> Current project priority (2026-09-17): the full frozen-driver matrix below is
> deferred. Follow the execution plan at the top of
> [new1.md](../../.logs/New2/new1.md): integrate optional uniform / fixed-mixture /
> adaptive-population paths into MixPPO and measure the **trained SimplePPO**.
> The frozen efficacy gate below remains the historical standalone experiment
> protocol; it is not a prerequisite for a small main-training comparison.
> These tools remain available for targeted diagnostics. No experiment results
> or prespecified configuration have been changed by this planning update.

This is the opt-in frozen-driver research path. Its first-episode evaluator and
phase learner are now reused by the explicit main-training integration described
in [MAIN_TRAINING.md](MAIN_TRAINING.md). The frozen experiment CLI remains
separate; it does not train driving policies. The historical Gaussian training
configuration is not switched automatically.
Run commands from the repository root with the existing virtual environment.

## Prespecified comparison

| Variant | Actor input | Minimum probability per component | Scene advantage baseline |
|---|---|---:|---|
| `global` | None | 0 | None |
| `floor` | None | 0.05 | None |
| `baseline` | None | 0.05 | Previous phase's scene predictor |
| `context` | Six scene features | 0.05 | Previous phase's scene predictor |

All variants start with uniform component probabilities and the same five
anchors. Every sampled component receives Gaussian descriptor noise, followed
by the environment's existing Beta condition sampling. Anchors and descriptor
sigma stay fixed. PPO uses the actual floored categorical probabilities:
`p = (1 - 5 * floor) * softmax(logits) + floor`. The Gaussian factor of the
recorded joint action cancels from the ratio.

`design.json` declares 80 training maps (`0:20,60:108,109:120,160`), 40 development maps
(`20:60`), and 40 reserved confirmation maps (`120:160`), all in the validation
split for this frozen-checkpoint research. Map 40..59 has already been inspected;
it is deliberately classified as development. Range endpoints are exclusive.
The runner rejects overlap and duplicate map IDs. Confirmation data is not
opened or hashed during development runs.

Initialization preflight found only one controllable agent on map 108. It was
replaced by map 160 based only on eligibility for the two-policy experiment,
without evaluating returns. All 80 training and 40 development maps passed
initial-context checks; the reserved confirmation range is unchanged.

There are three q training seeds and four variants. Each run uses 12 phases of
320 rollouts: every training map occurs four times per phase, in a deterministic
shuffled order shared across variants. The full matrix costs 46,080 training
rollouts and 8,640 development evaluation rollouts. Each diagnostic rollout lasts
at most 90 environment steps. The suite runs serially on CPU and uses a bounded
environment cache. `smoke.json` costs only 192 training and 96 evaluation
rollouts; it verifies operation and is not an efficacy experiment.

## Scene information and reward timing

Before a q action is drawn, the evaluator resets the scene with a neutral uniform
descriptor. Without taking a physical step, it extracts:

1. `log(1 + agent_count) / log(129)`;
2. mean and standard deviation of SimplePPO initial absolute speed, divided by 20 m/s;
3. mean number of observed partners within 30 m, divided by partner slots;
4. mean and standard deviation of the frozen SimplePPO initial value.

The partner statistic is capped by observation slots and is not an uncapped
physical density. It excludes padded partner entries using vehicle dimensions.
No map ID, sampled condition, previous action, or outcome is an input. The schema
uses Drive's documented-in-source `speed / 100`, `relative_position * .02`, and
eight-feature partner layout. The experiment rejects unsupported observation
configurations instead of silently adapting checkpoint architecture.

After q samples `(component, descriptor)`, the evaluator resets the same scene
and seed with that descriptor. It verifies the initial physical observation hash,
context and frozen SimplePPO baseline are unchanged. This two-reset procedure is
an experimental adapter, not a production worker sampling hook. A descriptor is
published only at reset, never in the middle of a rollout.

The raw reward always remains the first-episode average `V_frozen(start) - G`.
True terminal states have no bootstrap; environment truncations and surviving
agents at the diagnostic horizon include the discounted continuation value.
SimplePPO rewards retain the previous `[-1, 1]` clipping and gamma=.98. The
experiment verifies both driving policies and critics remain frozen.

For scene-baseline variants, each record stores the prediction made *before*
the current action. Actor advantages use `raw_challenge - saved_prediction`,
followed by the same batch normalization in every variant. Only after the actor
update is the scene predictor fitted to this phase's raw rewards; those fitted
weights are first used in the next phase. First-phase predictions are zero.
The predictor is action-independent but not guaranteed to reduce variance:
pre-fit MSE and residual variance, not post-fit training MSE, diagnose its value.

Both optional networks have one 32-unit tanh hidden layer and zero-initialized
output layers. The context actor has global logits plus a context residual.
All actor parameter groups use the declared learning rate; scene fitting uses
its separate declared learning rate. There is no shared optimizer or parameter
with either driver critic.

## Phase update and artifacts

One whole phase shares a policy/baseline version and sigma. Records save context,
component, latent, full behavior probabilities, component log probability,
baseline prediction, map/action seed and first-episode reward decomposition.
Stale versions, changed anchor/network fingerprints, inconsistent probabilities
and non-finite data are rejected before updating.

Unlike the previous diagnostic's post-step KL early stop, this new path applies
the same bounded mean behavior-KL update to all four variants. If an Adam step
exceeds the budget, both parameters and optimizer moments are restored and the
same gradient is retried with halved learning rates, up to eight halvings. A
rejected proposal leaves no update. The budget is the empirical mean KL over
collected contexts, not a bound for every possible scene; maximum observed
context KL and backtracking counts are logged separately.

Each phase logs component counts, probabilities, entropy, phases without a new
sample, challenge statistics, map summaries and baseline prediction error.
Evaluation records store scene-specific probabilities. Reports multiply each
scene's component returns by that scene's weights, not by globally averaged
probabilities. Comparisons include exact uniform and the initial equal mixture.
`collision_indicator_rate` is the observed SimplePPO collision feature during
its first episode, not a vehicle/offroad collision subtype or success rate.

Every run has an exclusive output directory:

- `results.json`: complete records, phase diagnostics, and evaluation rows;
- `state.pt`: authoritative phase checkpoint, including actor/baseline Adam
  states, q RNG, version, component sampling history and the JSON document;
- the suite stores a separate `<variant>_<seed>.log` next to run directories.

Writes use temporary files and atomic replacement. Resume requires identical
config, code, driver checkpoints and previously used map contents. A phase
interrupted before its checkpoint is recollected from the last committed phase.
Evaluation resumes at completed map/seed pairs. Existing output directories are
never silently overwritten. Source hashes are intended for reproducibility, not
as a claim that git HEAD represents the inherited working tree.

## Commands

Verify the declared training/development scenes before a full run:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.population.preflight \
  --config scripts/population/design.json \
  --output .logs/New2/experiments/phase5_design_preflight.json
```

Inspect the full matrix without opening maps or loading drivers:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.population.suite \
  --config scripts/population/design.json \
  --output-root .logs/New2/experiments/phase5_full --dry-run
```

Run a new smoke matrix, then summarize it:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.population.suite \
  --config scripts/population/smoke.json \
  --output-root .logs/New2/experiments/phase5_smoke
PYTHONPATH=. .venv/bin/python -m scripts.population.summarize \
  --root .logs/New2/experiments/phase5_smoke
```

For the full experiment, use the first command without `--dry-run`. To resume a
matrix, add `--resume`. To resume one run or evaluate its already-trained model:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.population.run \
  --config scripts/population/design.json --variant context --seed 917 \
  --output .logs/New2/experiments/phase5_full/context_917 --resume
```

After development selection is complete, confirmation evaluation uses the same
saved model with `--resume --evaluation-set confirmation`. It does not retrain a
completed model. Confirmation reports use the summarizer's matching flag.

## Interpretation and remaining work

Use paired development comparisons `floor-global`, `baseline-floor` and
`context-baseline` to isolate each addition. All variants share scene/action
seeds and evaluation descriptor noise. Component rollout equality is checked
across result files before reporting paired comparisons. Primary intervals
bootstrap maps, averaging action seeds within each map, separately for each
trained q; they do not quantify uncertainty over the population of q training
seeds. Reports identify an incomplete matrix instead of silently calling it full.

Before opening confirmation maps, select one variant by its average development
return across all three prespecified q seeds, keeping all three checkpoints.
Do not select a lucky training seed. The proposed confirmation gate is negative
return differences versus the initial equal mixture for all three seeds, with
each seed's map interval below zero. Report uniform and global-q controls too.
This is a demanding prespecified research gate, not evidence already obtained;
only smoke verification has been run when introducing these modules.

Still outside this change: a production worker context/publication hook,
alternating driver/q scheduling, learned anchor offsets, ConditionedPPO reward
scaling and inactive-dimension ablations, and final SimplePPO robustness tests.
Those should follow the frozen-q efficacy gate, with separate configurations and
tests. No existing training path is switched to this experiment automatically.
