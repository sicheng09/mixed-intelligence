#!/usr/bin/env python3
"""Download the official 10k training group and/or build Drive binaries.

Run download in either environment; conversion requires apex-train's C binding.
The source revision and per-map source filenames are recorded for reproducibility.
Does not create or substitute the independent BehaviorBench evaluation splits.
"""
import argparse
import json
import os
import time
from pathlib import Path

REPO = "EMERGE-lab/GPUDrive"
REVISION = "96fdbc298733daca450c42f7fad441d21467a717"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(os.environ.get("APEX_WORKSPACE", Path(__file__).resolve().parents[2])))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--convert", action="store_true")
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--download-attempts", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not (args.download or args.convert):
        parser.error("select --download, --convert, or both")
    if min(args.download_workers, args.download_attempts, args.workers) < 1:
        parser.error("worker and attempt counts must be positive")
    root = args.workspace / "datasets/gpudrive-medium"
    if args.download:
        import requests
        from huggingface_hub import snapshot_download
        for attempt in range(1, args.download_attempts + 1):
            try:
                print(f"Download attempt {attempt}/{args.download_attempts}; "
                      f"complete JSONs: {len(list((root / 'training/group_0').glob('*.json')))}",
                      flush=True)
                snapshot_download(repo_id=REPO, repo_type="dataset", revision=REVISION,
                                  allow_patterns=["training/group_0/*.json"], local_dir=root,
                                  max_workers=args.download_workers)
                break
            except (OSError, requests.exceptions.RequestException) as error:
                cause = error
                status = None
                retry_after = 0
                while cause is not None:
                    response = getattr(cause, "response", None)
                    if response is not None:
                        status = response.status_code
                        try:
                            retry_after = float(response.headers.get("Retry-After", 0))
                        except ValueError:
                            pass
                    cause = cause.__cause__
                if status in (401, 403) or attempt == args.download_attempts:
                    raise
                delay = max(min(10 * 2 ** (attempt - 1), 60), retry_after)
                print(f"Download interrupted ({type(error).__name__}, HTTP {status}); "
                      f"resuming cached files after {delay:g}s", flush=True)
                time.sleep(delay)
    files = sorted((root / "training/group_0").glob("*.json"))
    if len(files) != 10000:
        raise RuntimeError(f"Expected exactly 10000 training JSONs, found {len(files)}")
    print("Verified exactly 10000 training JSONs", flush=True)
    manifest = {"repo_id": REPO, "revision": REVISION, "training_scenarios": len(files),
                "pattern": "training/group_0/*.json", "lane_connectivity": "as provided by source JSONs"}
    root.mkdir(parents=True, exist_ok=True)
    (root / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.convert:
        from pufferlib.ocean.drive.drive import process_all_maps
        output = args.workspace / "datasets/drive-binaries/training"
        existing = list(output.glob("map_*.bin"))
        if existing:
            if len(existing) == 10000 and (output / "source-manifest.json").is_file():
                previous = json.loads((output / "source-manifest.json").read_text())
                if any(previous.get(key) != manifest[key] for key in ("repo_id", "revision", "pattern")):
                    raise RuntimeError(f"Existing binary provenance differs in {output}")
                expected = {f"map_{i:06d}.bin" for i in range(10000)}
                if {p.name for p in existing} != expected:
                    raise RuntimeError(f"Existing binary filenames are not contiguous in {output}")
                print(f"Existing complete conversion retained: {output}", flush=True)
                return
            raise RuntimeError(f"Refusing to overwrite existing binaries in {output}")
        process_all_maps(root / "training/group_0", output, max_maps=10000, num_workers=args.workers)
        count = len(list(output.glob("map_*.bin")))
        if count != 10000:
            raise RuntimeError(f"Binary conversion produced {count}/10000 maps")
        manifest["maps"] = [{"binary": f"map_{i:06d}.bin", "source": str(p.relative_to(root))}
                            for i, p in enumerate(files)]
        (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Prepared exactly {count} training maps at {output}")


if __name__ == "__main__":
    main()
