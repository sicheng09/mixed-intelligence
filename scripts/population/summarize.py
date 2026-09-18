"""Paired map-cluster intervals; keep training seeds separate in primary results."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.population.run import map_ids, write_json


def comparison(values, reference, bootstrap_indices):
    delta = np.asarray(values) - np.asarray(reference)
    return dict(mean_return=float(np.mean(values)), reference_return=float(np.mean(reference)),
                delta=float(delta.mean()),
                map_bootstrap_95=np.quantile(delta[bootstrap_indices].mean(-1), [.025, .975]).tolist())


def summarize(paths, evaluation_set="development", bootstrap_samples=10000):
    documents = [json.loads(path.read_text()) for path in paths]
    if not documents:
        raise ValueError("no result files found")
    config = documents[0]["config"]
    maps = map_ids(config[evaluation_set + "_maps"])
    seeds = config["evaluation_seeds"]
    expected_pairs = {(m, s) for m in maps for s in seeds}
    bootstrap_indices = np.random.default_rng(9301).integers(len(maps), size=(bootstrap_samples, len(maps)))
    runs, values_by_run = [], {}
    reference_components = None
    for document in documents:
        if document["config"] != config or not document["training_complete"]:
            raise ValueError("runs must be complete and use the same experiment design")
        if document["variant"] not in config["variants"] or document["seed"] not in config["seeds"]:
            raise ValueError("run variant/seed is absent from the declared matrix")
        rows = document["evaluations"].get(evaluation_set, [])
        if len(rows) != len(expected_pairs) or {(r["map"], r["seed"]) for r in rows} != expected_pairs:
            raise ValueError("evaluation is incomplete or has duplicate map/seed pairs")
        indexed = {(r["map"], r["seed"]): r for r in rows}
        learned, initial, uniform, collision = [], [], [], []
        component_signature = []
        probabilities = []
        for map_id in maps:
            map_learned, map_initial, map_uniform, map_collision = [], [], [], []
            for seed in seeds:
                row = indexed[map_id, seed]
                if [r["component"] for r in row["components"]] != list(range(5)):
                    raise ValueError("missing or unordered component evaluations")
                returns = np.asarray([r["bootstrapped_return"] for r in row["components"]])
                probs = np.asarray(row["probabilities"])
                if not np.isfinite(probs).all() or (probs < 0).any() or not np.isclose(probs.sum(), 1):
                    raise ValueError("invalid evaluation probabilities")
                probs = probs / probs.sum()  # Float32 saved probabilities can sum to 1 +/- 1e-7.
                reference_return = row["uniform"]["bootstrapped_return"]
                # An exactly unchanged component return must produce exactly zero effect.
                differences = returns - reference_return
                map_learned.append(float(reference_return + differences @ probs))
                map_initial.append(float(reference_return + differences.mean()))
                map_uniform.append(reference_return)
                map_collision.append(float(probs @ np.asarray([r["collision_indicator_rate"] for r in row["components"]])))
                probabilities.append(probs)
                component_signature.append(dict(map=map_id, seed=seed, context=row["context"],
                    components=row["components"], uniform=row["uniform"]))
            learned.append(np.mean(map_learned))
            initial.append(np.mean(map_initial))
            uniform.append(np.mean(map_uniform))
            collision.append(np.mean(map_collision))
        # All policies are evaluated on identical component rollouts, including their noise.
        # Context weights remain scene-specific; never replace them by global average weights.
        if reference_components is None:
            reference_components = component_signature
        elif component_signature != reference_components:
            raise ValueError("component rollouts differ across runs; paired comparison would be confounded")
        key = (document["variant"], document["seed"])
        if key in values_by_run:
            raise ValueError("duplicate variant/seed")
        values_by_run[key] = np.asarray(learned)
        runs.append(dict(variant=key[0], seed=key[1],
            vs_uniform=comparison(learned, uniform, bootstrap_indices),
            vs_initial_mixture=comparison(learned, initial, bootstrap_indices),
            mean_probabilities=np.mean(probabilities, axis=0).tolist(),
            mean_collision_indicator_rate=float(np.mean(collision))))
    paired = []
    for improved, reference in (("floor", "global"), ("baseline", "floor"), ("context", "baseline")):
        for seed in config["seeds"]:
            if (improved, seed) in values_by_run and (reference, seed) in values_by_run:
                paired.append(dict(variant=improved, reference=reference, seed=seed,
                    **comparison(values_by_run[improved, seed], values_by_run[reference, seed], bootstrap_indices)))
    expected_runs = {(v, s) for v in config["variants"] for s in config["seeds"]}
    return dict(evaluation_set=evaluation_set, maps=maps, evaluation_seeds=seeds,
        complete_matrix=set(values_by_run) == expected_runs,
        interval_scope="Map-cluster intervals conditional on each trained q; no pooled training-seed confidence claim.",
        runs=runs, paired_ablations=paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evaluation-set", choices=("development", "confirmation"), default="development")
    args = parser.parse_args()
    result = summarize(sorted(args.root.glob("*/results.json")), args.evaluation_set)
    write_json(args.root / f"summary_{args.evaluation_set}.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
