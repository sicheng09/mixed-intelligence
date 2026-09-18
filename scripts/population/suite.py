"""Run a declared ablation matrix serially, preserving separate run artifacts."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from scripts.population.run import validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluation-set", choices=("none", "development", "confirmation"), default="development")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    for seed in config["seeds"]:
        for variant in config["variants"]:
            name = f"{variant}_{seed}"
            output = args.output_root / name
            command = [sys.executable, "-m", "scripts.population.run", "--config", str(args.config),
                       "--variant", variant, "--seed", str(seed), "--output", str(output),
                       "--evaluation-set", args.evaluation_set]
            if args.resume and (output / "state.pt").exists():
                command.append("--resume")
            print(shlex.join(command), flush=True)
            if args.dry_run:
                continue
            args.output_root.mkdir(parents=True, exist_ok=True)
            with (args.output_root / f"{name}.log").open("a" if args.resume else "x") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
