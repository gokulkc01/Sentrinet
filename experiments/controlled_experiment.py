"""
Stage 0.2 — Controlled A/B/C training runner (SentryNet v2).

Enforces the experimental invariant from docs/DESIGN.md §0.2: systems A, B and C
are IDENTICAL in architecture, capture rule, optimizer, budget and seeds. They
differ ONLY in the communication conditions and whether trust is active:

    System A : clean training        (p_drop=0.0, p_spoof=0.0, use_trust=False)
    System B : adversarial training  (p_drop=0.2, p_spoof=0.1, use_trust=False)
    System C : adversarial + trust   (p_drop=0.2, p_spoof=0.1, use_trust=True)

Interpretation: A→B isolates the effect of *adversarial training*; B→C isolates
the effect of the *trust mechanism* (the only difference between B and C).

This deliberately does NOT reuse train.py, whose per-system defaults (GRU +
sustained capture for C, MLP + team for A/B) are the confound that made the old
results uninterpretable. See findings/Known Bugs and Confounds in the vault.

Usage:
    # Fast end-to-end pipeline check (a few minutes)
    python -m experiments.controlled_experiment --smoke

    # Full sweep (heavy — run where compute is cheap, e.g. a GPU box overnight)
    python -m experiments.controlled_experiment --seeds 0 1 2 3 4 5 6 7 --steps 1000000
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import random
from typing import Any, Dict, List

import numpy as np
import torch

from border_env import BorderEnv
from mappo_trainer import MAPPOTrainer

# The ONLY differences between systems.
SYSTEMS: Dict[str, Dict[str, Any]] = {
    "A": {"p_drop": 0.0, "p_spoof": 0.0, "use_trust": False},
    "B": {"p_drop": 0.2, "p_spoof": 0.1, "use_trust": False},
    "C": {"p_drop": 0.2, "p_spoof": 0.1, "use_trust": True},
}

# Held identical across all systems — the controlled part of the experiment.
INVARIANT_ENV: Dict[str, Any] = dict(
    use_pybullet=False,
    domain_rand=True,
    use_curriculum=False,   # curriculum is shelved; it would break the invariant
    capture_mode="team",
    spoof_std=2.0,
)
INVARIANT_TRAIN: Dict[str, Any] = dict(
    lr=3e-4,
    gamma=0.99,
    lam=0.95,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,      # same for all — NOT the old 0.005-for-C special case
    max_grad_norm=10.0,
    n_steps=2048,
    batch_size=256,
    n_epochs=4,
    policy_type="mlp",      # same for all — NOT the old GRU-for-C special case
    hidden_dim=128,
    save_every=50_000,
    eval_every=50_000,
)

CHECKPOINT_ROOT = "checkpoints/stage0"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_env(system: str, seed: int) -> BorderEnv:
    cfg = SYSTEMS[system]
    return BorderEnv(
        seed=seed,
        p_drop=cfg["p_drop"],
        p_spoof=cfg["p_spoof"],
        use_trust=cfg["use_trust"],
        **INVARIANT_ENV,
    )


def train_one(system: str, seed: int, total_steps: int, use_wandb: bool, smoke: bool) -> None:
    set_seed(seed)
    env = build_env(system, seed)

    train_cfg = dict(INVARIANT_TRAIN)
    if smoke:
        # Tiny budget that still exercises collect -> update -> eval -> save.
        train_cfg.update(n_steps=1024, n_epochs=2, save_every=total_steps, eval_every=total_steps)

    config: Dict[str, Any] = dict(train_cfg)
    config.update(
        total_steps=int(total_steps),
        use_wandb=bool(use_wandb),
        run_name=f"{system}_seed{seed}",
        checkpoint_dir=CHECKPOINT_ROOT,
        seed=int(seed),
        metrics_csv=f"logs/stage0_metrics_{system}_seed{seed}.csv",
    )

    print(f"\n=== Stage0 System {system} seed {seed} | {SYSTEMS[system]} | steps={total_steps:,} ===")
    trainer = MAPPOTrainer(env=env, config=config)
    trainer.train()
    env.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 0.2 controlled A/B/C experiment")
    p.add_argument("--systems", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--steps", type=int, default=1_000_000, help="Total env steps per run")
    p.add_argument("--smoke", action="store_true", help="Fast end-to-end pipeline check")
    p.add_argument("--wandb", action="store_true", help="Enable wandb logging (off by default)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    steps = 2048 if args.smoke else int(args.steps)
    seeds: List[int] = [0, 1] if args.smoke else list(args.seeds)

    print(f"Controlled experiment | systems={args.systems} seeds={seeds} "
          f"steps={steps:,} smoke={args.smoke}")
    for system in args.systems:
        for seed in seeds:
            train_one(system, seed, steps, args.wandb, args.smoke)

    print(f"\nAll runs complete. Checkpoints under {CHECKPOINT_ROOT}/<system>_seed<seed>/")
    print("Next: python -m experiments.analyze_controlled "
          + ("--smoke" if args.smoke else "--seeds " + " ".join(str(s) for s in seeds)))


if __name__ == "__main__":
    main()
