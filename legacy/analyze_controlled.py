"""
Stage 0.2 — Evaluate controlled A/B/C checkpoints and test whether trust helps.

Loads each stage0 checkpoint, sweeps packet-drop rates under a fixed spoof +
compromised-drone threat (identical for every system), and reports capture rate
with bootstrap 95% CIs plus a Welch t-test on the C - B difference — the actual
trust test. See planning/Controlled Experiment and Metrics in the vault.

Usage:
    python -m experiments.analyze_controlled --smoke
    python -m experiments.analyze_controlled --seeds 0 1 2 3 4 5 6 7 --episodes 200
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy import stats

from border_env import BorderEnv
from mappo_trainer import MAPPOTrainer

CHECKPOINT_ROOT = "checkpoints/stage0"
FULL_DROPS: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SMOKE_DROPS: List[float] = [0.0, 0.4, 0.8]

# Shared adversarial conditions applied to ALL systems during evaluation.
EVAL_SPOOF_RATE = 0.1
EVAL_SPOOF_STD = 2.0
COMPROMISED_DRONE = 1


def find_checkpoint(system: str, seed: int) -> Path:
    run_dir = Path(CHECKPOINT_ROOT) / f"{system}_seed{seed}"
    if not run_dir.exists():
        raise FileNotFoundError(
            f"No checkpoint dir {run_dir}. Run controlled_experiment.py first."
        )
    best = run_dir / "best.pt"
    if best.exists():
        return best
    steps = sorted(run_dir.glob("step_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if steps:
        return steps[-1]
    raise FileNotFoundError(f"No .pt checkpoints in {run_dir}")


def evaluate_checkpoint(system: str, seed: int, drop: float, n_episodes: int) -> float:
    """Return capture rate (fraction) for one (system, seed, drop) condition."""
    ckpt_path = find_checkpoint(system, seed)
    ckpt_cfg = torch.load(ckpt_path, map_location="cpu").get("config", {})

    env = BorderEnv(
        use_pybullet=False,
        domain_rand=False,
        p_drop=drop,
        p_spoof=EVAL_SPOOF_RATE,
        spoof_std=EVAL_SPOOF_STD,
        use_trust=(system == "C"),
        compromised_drone=COMPROMISED_DRONE,
        capture_mode="sustained",   # must match training (ADR-006)
        sustained_steps=1,
        seed=seed,
    )
    trainer = MAPPOTrainer(
        env=env,
        config={
            "use_wandb": False,
            "total_steps": 1,
            "run_name": f"{system}_seed{seed}",
            "seed": seed,
            "policy_type": ckpt_cfg.get("policy_type", "mlp"),
            "hidden_dim": int(ckpt_cfg.get("hidden_dim", 128)),
        },
    )
    trainer.load_checkpoint(str(ckpt_path))

    drone_keys = trainer._drone_keys()
    is_recurrent = trainer.policy.is_recurrent
    captures = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        # Recurrent policies need their hidden state threaded through the episode;
        # calling get_action per step would run the GRU as if memoryless.
        policy_state = trainer._init_policy_state() if is_recurrent else None
        done = False
        info: Dict = {}
        while not done:
            actions = {}
            with torch.no_grad():
                for k in drone_keys:
                    if is_recurrent:
                        action, _, next_hidden = trainer.policy.step(
                            obs[k], deterministic=True, hidden_state=policy_state[k]
                        )
                        if trainer.policy_type == "gru":
                            policy_state[k] = next_hidden.squeeze(0)
                        else:
                            policy_state[k] = (next_hidden[0].squeeze(0), next_hidden[1].squeeze(0))
                    else:
                        action, _ = trainer.policy.get_action(obs[k], deterministic=True)
                    actions[k] = action
            actions["sensor_0"] = 1 if float(obs["sensor_0"][0]) > 0.5 else 0
            obs, _, term, trunc, info = env.step(actions)
            done = any(term[k] or trunc[k] for k in drone_keys)
        captures += int(bool(info.get("drone_0", {}).get("captured", False)))
    env.close()
    return captures / n_episodes


def bootstrap_ci(vals: List[float], n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(0)
    means = np.array([rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the controlled A/B/C experiment")
    parser.add_argument("--systems", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--smoke", action="store_true", help="Few episodes, subset of drops")
    args = parser.parse_args()

    drops = SMOKE_DROPS if args.smoke else FULL_DROPS
    seeds = [0, 1] if args.smoke else list(args.seeds)
    n_ep = 10 if args.smoke else int(args.episodes)

    rows: List[Dict] = []
    for system in args.systems:
        for seed in seeds:
            for drop in drops:
                rate = evaluate_checkpoint(system, seed, drop, n_ep)
                rows.append({"system": system, "seed": seed, "drop_rate": drop, "capture_rate": rate})
                print(f"  {system} seed{seed} drop{drop:.1f} -> capture={rate:.3f}")

    df = pd.DataFrame(rows)
    out_dir = Path("results/stage0")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / ("controlled_results_smoke.csv" if args.smoke else "controlled_results.csv")
    df.to_csv(out_csv, index=False)

    # ---- Verdict: does trust help? (C - B, per drop rate) --------------------
    print("\n" + "=" * 72)
    print("DOES TRUST HELP?  Capture rate mean [95% CI] per system, and C - B test")
    print("=" * 72)
    have_bc = {"B", "C"}.issubset(set(args.systems))
    for drop in drops:
        cell = {}
        for s in args.systems:
            v = df[(df.system == s) & (df.drop_rate == drop)].capture_rate.values
            lo, hi = bootstrap_ci(list(v))
            cell[s] = (float(np.mean(v)) if len(v) else float("nan"), lo, hi)
        line = f"drop={drop:.1f}  " + "  ".join(
            f"{s}={cell[s][0]:.3f}[{cell[s][1]:.2f},{cell[s][2]:.2f}]" for s in args.systems
        )
        if have_bc:
            b = df[(df.system == "B") & (df.drop_rate == drop)].capture_rate.values
            c = df[(df.system == "C") & (df.drop_rate == drop)].capture_rate.values
            delta = float(np.mean(c) - np.mean(b)) if len(b) and len(c) else float("nan")
            if len(b) <= 1 or len(c) <= 1:
                line += f"   C-B={delta:+.3f} (p=n/a, need >1 seed)"
            elif np.std(b) + np.std(c) == 0:
                line += f"   C-B={delta:+.3f} (p=n/a, zero variance)"
            else:
                p = float(stats.ttest_ind(c, b, equal_var=False).pvalue)
                sig = "*" if p < 0.05 else " "
                line += f"   C-B={delta:+.3f} (p={p:.3f}){sig}"
        print(line)

    print("=" * 72)
    print(f"Saved: {out_csv}")
    print("Reminder: with the smoke budget these numbers are meaningless - this only")
    print("proves the pipeline runs end to end. Run the full sweep for real results.")


if __name__ == "__main__":
    main()
