from __future__ import annotations

from pathlib import Path
import argparse
import re

import torch


def infer_latest_checkpoint(folder: Path) -> Path | None:
    checkpoints = list(folder.glob("*.pt"))
    if not checkpoints:
        return None

    def step_key(path: Path) -> int:
        match = re.search(r"(\d+)", path.stem)
        return int(match.group(1)) if match else 0

    return sorted(checkpoints, key=step_key)[-1]


def infer_obs_dim(checkpoint: dict) -> str:
    state_dict = checkpoint.get("policy_state_dict", {})
    weight = next((value for key, value in state_dict.items() if "weight" in key and getattr(value, "ndim", 0) == 2), None)
    return str(weight.shape[1]) if weight is not None else "?"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize checkpoint folders.")
    parser.add_argument("root", nargs="?", default="checkpoints", help="Checkpoint root folder")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Checkpoint root not found: {root}")

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        latest = infer_latest_checkpoint(folder)
        if latest is None:
            continue

        checkpoint = torch.load(latest, map_location="cpu")
        obs_dim = infer_obs_dim(checkpoint)
        policy_type = checkpoint.get("config", {}).get("policy_type", "?")
        step = checkpoint.get("step", "?")
        print(f"{folder.name:40s} obs_dim={obs_dim:>3} type={policy_type} step={step}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
