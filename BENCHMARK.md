# BehaviorBench validation splits

Validation preparation follows the original
[BehaviorBench benchmark guide](https://github.com/boschresearch/behavior-bench/blob/main-behavior-bench/BENCHMARK.md).
The benchmark definitions and split manifests are inherited from that project.

**Interactive1k** and **Random1k** each select 1,000 scenarios from the full
Waymo Open Motion Dataset (WOMD) validation set. They are separate from the
10,000-scene medium training subset.

## 1. Obtain the full validation inputs

Download the authorized WOMD validation **scenario TFRecords** from the
[Waymo Open Dataset](https://waymo.com/open/data/motion/). Prepare a corresponding
GPUDrive-style JSON for every validation scenario using
[ScenarioMax](https://github.com/valeoai/ScenarioMax), following its conversion
instructions and the upstream BehaviorBench guide.

Both inputs are needed: JSONs provide trajectories and map geometry; raw
TFRecords provide lane connectivity. The prebuilt GPUDrive collection contains
only a subset of WOMD validation data and cannot substitute for the complete
validation inputs required by the upstream manifests.

Expected layout:

```text
$APEX_WORKSPACE/datasets/
  waymo-raw/validation/             # WOMD scenario TFRecords
  waymo-full-json/validation/       # one GPUDrive-style JSON per scenario
```

Preserve the upstream scenario naming and conversion order: the manifests refer
to the original `map_NNNNNN.bin` filenames from the full validation conversion.

## 2. Extract connectivity and generate binaries

Run the main pipeline in `apex-train`; `--tf-python` selects `apex-data` for the
TensorFlow/Waymo subprocesses:

```bash
source scripts/apex_env.sh train
python data_utils/womd/create_training_binaries.py \
  --tf-python "$APEX_WORKSPACE/envs/apex-data/bin/python" \
  --tfrecord-root "$APEX_WORKSPACE/datasets/waymo-raw" \
  --json-root "$APEX_WORKSPACE/datasets/waymo-full-json" \
  --enriched-root "$APEX_WORKSPACE/datasets/waymo-connectivity-json" \
  --output-root "$DRIVE_BINARIES_DATA_ROOT" \
  --splits validation --workers 4
```

The pipeline extracts lane connectivity, enriches the JSONs, and writes binaries
to `$DRIVE_BINARIES_DATA_ROOT/validation`. If JSONs already include the required
connectivity, the JSON-to-binary conversion can be run directly with
`process_all_maps()` from `pufferlib/ocean/drive/drive.py`, as described upstream.

## 3. Apply the benchmark manifests

The original manifests are checked in at:

- `pufferlib/resources/splits/interactive1k.csv`
- `pufferlib/resources/splits/random1k.csv`

```bash
for split in interactive1k random1k; do
  python scripts/remap_files.py \
    --source-path "$DRIVE_BINARIES_DATA_ROOT/validation" \
    --csv-path "pufferlib/resources/splits/$split.csv" \
    --target-path "$DRIVE_BINARIES_DATA_ROOT"
done
```

The script copies each manifest-selected original file to its new benchmark
filename and preserves the CSV as `manifest.csv`. Confirm **1,000 copied maps and
zero missing sources** for each split before evaluation.

```text
$DRIVE_BINARIES_DATA_ROOT/
  training/                        # independent 10,000-scene training set
  validation/                      # full validation conversion
  interactive1k/                   # 1,000 maps + manifest.csv
  random1k/                        # 1,000 maps + manifest.csv
```

## 4. Evaluate

```bash
python pufferlib/ocean/benchmark/eval.py \
  --eval.split interactive1k --map-ids all \
  --planner.type ppo \
  --planner.ppo.weights-path weights/apex_focal_seed42_4b.pt \
  --traffic.type idm
```

Repeat with `--eval.split random1k`. See [EVALUATION.md](EVALUATION.md) for traffic
controllers and metrics. Use the same traffic checkpoints across policy
comparisons, and keep the held-out validation data distinct from training.
