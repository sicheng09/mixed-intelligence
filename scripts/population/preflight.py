"""Check training/development map eligibility without rollout or q training."""

import argparse
import json
from pathlib import Path

import torch

from scripts.population.frozen import FrozenEvaluator
from scripts.population.run import map_ids, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    torch.set_num_threads(1)
    rows, invalid = [], []
    evaluator = FrozenEvaluator(config)
    try:
        for group in ("train_maps", "development_maps"):
            for map_id in map_ids(config[group]):
                try:
                    context = evaluator.prepare(map_id, config["evaluation_seeds"][0])
                    rows.append(dict(group=group, map=map_id,
                        num_agents=evaluator.envs[map_id].num_agents, **context))
                except ValueError as error:
                    invalid.append(dict(group=group, map=map_id, error=str(error)))
        evaluator.assert_frozen()
    finally:
        evaluator.close()
    result = dict(config=config, valid=not invalid, checked_maps=len(rows) + len(invalid),
                  invalid=invalid, rows=rows, confirmation_maps_opened=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(json.dumps(dict(valid=not invalid, checked_maps=result["checked_maps"], invalid=invalid)))
    if invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
