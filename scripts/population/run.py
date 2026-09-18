"""Run an isolated, resumable frozen-driver ablation from a JSON design."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from pufferlib.population_adversary import initial_population_mean
from pufferlib.population_experiment import CONTEXT_NAMES, VARIANTS, FrozenPhaseLearner


def map_ids(specification):
    """Comma-separated IDs or Python-style half-open ranges, e.g. 0:20,60:120."""
    result = []
    for part in specification.split(","):
        if ":" in part:
            start, end = map(int, part.split(":"))
            if end <= start:
                raise ValueError("map ranges must be increasing")
            result.extend(range(start, end))
        else:
            result.append(int(part))
    if not result or min(result) < 0 or len(result) != len(set(result)):
        raise ValueError("map IDs must be nonnegative, nonempty and unique")
    return result


def validate_config(config):
    for key in ("batch", "stages", "horizon", "epochs", "baseline_epochs", "env_cache_size"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["batch"] < 2:
        raise ValueError("batch must be at least two")
    for key in ("std", "lr", "baseline_lr", "target_kl"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not 0 < config["gamma"] <= 1:
        raise ValueError("gamma must be in (0, 1]")
    groups = [set(map_ids(config[key])) for key in ("train_maps", "development_maps", "confirmation_maps")]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("training, development and confirmation maps must be disjoint")
    for key in ("seeds", "evaluation_seeds"):
        seeds = config[key]
        if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(s, int) or s < 0 or s >= 2**31 for s in seeds):
            raise ValueError(f"{key} must contain unique seeds in [0, 2**31)")
    if set(config["variants"]) - set(VARIANTS) or len(set(config["variants"])) != len(config["variants"]):
        raise ValueError("invalid variants")
    if len(config["checkpoints"]) != 2:
        raise ValueError("two frozen driver checkpoints are required")


def phase_scenes(config, seed, version):
    maps = map_ids(config["train_maps"])
    ordered = [maps[(version * config["batch"] + j) % len(maps)] for j in range(config["batch"])]
    rng = np.random.default_rng(np.random.SeedSequence([seed, version, 721]))
    rng.shuffle(ordered)
    seeds = rng.integers(0, 2**31 - 1, size=len(ordered))
    return list(zip(ordered, map(int, seeds)))


def manifest(config, evaluation_set):
    from pufferlib.ocean.drive.drive import binding
    sources = ["pufferlib/population_experiment.py", "pufferlib/population_adversary.py",
               "scripts/population/run.py", "scripts/population/frozen.py",
               "pufferlib/ocean/drive/drive.py", "pufferlib/ocean/torch.py", binding.__file__]
    files = list(config["checkpoints"]) + [config["env_config"]] + sources
    # Confirmation files are deliberately not opened by a development run.
    maps = map_ids(config["train_maps"])
    if evaluation_set != "none":
        maps += map_ids(config[evaluation_set + "_maps"])
    files += [str(Path(config["data_root"]) / config["split"] / f"map_{m:06d}.bin") for m in maps]
    return {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in files}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def save(output, document, learner, generator):
    # Checkpoint is authoritative on resume; JSON is the human-readable view.
    temporary = output / "state.pt.tmp"
    torch.save(dict(document=document, learner=learner.state_dict(), rng=generator.get_state()), temporary)
    temporary.replace(output / "state.pt")
    write_json(output / "results.json", document)


def run(config, variant, seed, output, resume=False, evaluation_set="development"):
    from scripts.population.frozen import FrozenEvaluator
    validate_config(config)
    if variant not in config["variants"] or seed not in config["seeds"]:
        raise ValueError("variant/seed must be declared in the experiment design")
    torch.set_num_threads(1)
    learner = FrozenPhaseLearner(variant, seed, lr=config["lr"], epochs=config["epochs"],
        target_kl=config["target_kl"], baseline_lr=config["baseline_lr"],
        baseline_epochs=config["baseline_epochs"])
    generator = torch.Generator().manual_seed(seed)
    hashes = manifest(config, evaluation_set)
    if resume:
        state = torch.load(output / "state.pt", map_location="cpu", weights_only=True)
        document = state["document"]
        if document["config"] != config or document["variant"] != variant or document["seed"] != seed:
            raise ValueError("resume configuration differs from checkpoint")
        if any(path in hashes and hashes[path] != value for path, value in document["manifest"].items()):
            raise ValueError("source, driver checkpoint or map contents changed since checkpoint")
        learner.load_state_dict(state["learner"])
        generator.set_state(state["rng"])
        document["manifest"].update(hashes)
    else:
        output.mkdir(parents=True, exist_ok=False)
        document = dict(schema_version=1, config=config, variant=variant, seed=seed,
                        context_names=CONTEXT_NAMES, component_names=learner.policy.names,
                        manifest=hashes, records=[], history=[], evaluations={}, training_complete=False)
        save(output, document, learner, generator)
    evaluator = FrozenEvaluator(config)
    try:
        while learner.version < config["stages"]:
            version = learner.version
            snapshot = learner.snapshot()
            batch = []
            for map_id, action_seed in phase_scenes(config, seed, version):
                # This neutral reset and context snapshot precede the categorical draw.
                prepared = evaluator.prepare(map_id, action_seed)
                context = torch.tensor([prepared["context"]], dtype=torch.float32)
                with torch.no_grad():
                    prediction = float(learner.predict_baseline(context)[0])
                    k, latent, logprob, probabilities = learner.policy.sample(context, config["std"], generator)
                record = evaluator.run(map_id, action_seed, latent[0].numpy(), prepared)
                record.update(snapshot, component=int(k[0]), latent=latent[0].tolist(),
                    old_logprob=float(logprob[0]), old_probs=probabilities[0].tolist(),
                    std=config["std"], scene_baseline=prediction)
                batch.append(record)
            evaluator.assert_frozen()
            diagnostics = learner.update(batch, config["std"])
            document["records"].extend(batch)
            document["history"].append(diagnostics)
            document["training_complete"] = learner.version == config["stages"]
            save(output, document, learner, generator)
            print(json.dumps(dict(variant=variant, seed=seed, version=version,
                probabilities=diagnostics["mean_probabilities"], kl=diagnostics["update_kl"],
                raw_variance=diagnostics["raw_challenge_variance"],
                residual_variance=diagnostics["residual_variance"])), flush=True)
        if evaluation_set != "none":
            evaluation = document["evaluations"].setdefault(evaluation_set, [])
            done = {(r["map"], r["seed"]) for r in evaluation}
            for map_id in map_ids(config[evaluation_set + "_maps"]):
                for action_seed in config["evaluation_seeds"]:
                    if (map_id, action_seed) in done:
                        continue
                    prepared = evaluator.prepare(map_id, action_seed)
                    context = torch.tensor([prepared["context"]], dtype=torch.float32)
                    with torch.no_grad():
                        probabilities = learner.policy.distribution(context).probs[0].tolist()
                    # Independent of training draws, variant and q seed; shared across all comparisons.
                    rng = np.random.default_rng(np.random.SeedSequence([map_id, action_seed, 831]))
                    noise = rng.normal(0, config["std"], 12).astype(np.float32)
                    components = []
                    for k in range(len(learner.policy.names)):
                        latent = learner.policy.anchors[k].numpy() + noise
                        result = evaluator.run(map_id, action_seed, latent, prepared)
                        components.append(dict(component=k, latent=latent.tolist(), **result))
                    uniform = evaluator.run(map_id, action_seed, initial_population_mean(), prepared)
                    evaluation.append(dict(map=map_id, seed=action_seed, context=prepared["context"],
                                           probabilities=probabilities, components=components, uniform=uniform))
                evaluator.assert_frozen()
                save(output, document, learner, generator)
                print(f"{variant} seed={seed} {evaluation_set} map={map_id}", flush=True)
        evaluator.assert_frozen()
        document["drivers_verified_frozen"] = True
        save(output, document, learner, generator)
    finally:
        evaluator.close()
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluation-set", choices=("none", "development", "confirmation"), default="development")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    if args.dry_run:
        print(json.dumps(dict(variant=args.variant, seed=args.seed, training_rollouts=config["stages"] * config["batch"],
            training_maps=map_ids(config["train_maps"]),
            evaluation_maps=[] if args.evaluation_set == "none" else map_ids(config[args.evaluation_set + "_maps"])), indent=2))
        return
    run(config, args.variant, args.seed, args.output, args.resume, args.evaluation_set)


if __name__ == "__main__":
    main()
