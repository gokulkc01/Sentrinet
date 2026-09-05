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
    use_curriculum=False,       # curriculum is shelved; it would break the invariant
    capture_mode="sustained",   # ADR-006: MLP+team never learned; GRU+sustained is
    sustained_steps=1,          #   the known-solvable config (old runs hit ~99%)
    spoof_std=2.0,
)
INVARIANT_TRAIN: Dict[str, Any] = dict(
    lr=3e-4,
    gamma=0.99,
    lam=0.95,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.005,     # ADR-006: was 0.01; the bonus was dominating a dead
                            #   gradient (rising entropy). Applied to ALL systems.
    max_grad_norm=10.0,
    n_steps=2048,
    batch_size=256,
    n_epochs=4,
    policy_type="gru",      # ADR-006: MLP couldn't handle partial observability.
                            #   Same for ALL systems, so the invariant still holds.
    hidden_dim=128,
    save_every=50_000,
    eval_every=50_000,
)

# ADR-010 reward-shaping ablation arms. 'baseline' reproduces the measured 3-seed
# run that did NOT learn (13 captures in 900k steps); every other arm changes ONE
# thing, so a difference is attributable. None of these alter task difficulty:
# capture radius, intruder speed/evasiveness and sustained_steps are untouched.
REWARD_ARMS: Dict[str, Dict[str, Any]] = {
    "baseline": {},
    "prox": {"proximity_weight": 0.5},
    "coll": {"collision_mode": "graded", "collision_weight": 2.0},
    "both": {"proximity_weight": 0.5, "collision_mode": "graded", "collision_weight": 2.0},
}

CHECKPOINT_ROOT = "checkpoints/stage0"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_env(system: str, seed: int, arm: str = "baseline") -> BorderEnv:
    cfg = SYSTEMS[system]
    return BorderEnv(
        seed=seed,
        p_drop=cfg["p_drop"],
        p_spoof=cfg["p_spoof"],
        use_trust=cfg["use_trust"],
        **INVARIANT_ENV,
        **REWARD_ARMS[arm],
    )


def train_one(system: str, seed: int, total_steps: int, use_wandb: bool, smoke: bool,
              tag: str = "", arm: str = "baseline") -> None:
    set_seed(seed)
    env = build_env(system, seed, arm)

    train_cfg = dict(INVARIANT_TRAIN)
    if smoke:
        # Tiny budget that still exercises collect -> update -> eval -> save.
        train_cfg.update(n_steps=1024, n_epochs=2, save_every=total_steps, eval_every=total_steps)

    suffix = f"_{tag}" if tag else ""
    config: Dict[str, Any] = dict(train_cfg)
    config.update(
        total_steps=int(total_steps),
        use_wandb=bool(use_wandb),
        run_name=f"{system}_seed{seed}",
        checkpoint_dir=f"{CHECKPOINT_ROOT}{suffix}",
        seed=int(seed),
        reward_arm=str(arm),          # provenance: recorded into the checkpoint
        metrics_csv=f"logs/stage0{suffix}_metrics_{system}_seed{seed}.csv",
    )

    print(f"\n=== Stage0 System {system} seed {seed} | {SYSTEMS[system]} | "
          f"arm={arm} {REWARD_ARMS[arm]} | steps={total_steps:,} ===")
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
    p.add_argument("--arm", default="baseline", choices=sorted(REWARD_ARMS),
                   help="ADR-010 reward-shaping arm. Does NOT change task difficulty.")
    p.add_argument("--tag", default="", help="Isolate outputs into checkpoints/stage0_<tag>/ and "
                                             "logs/stage0_<tag>_metrics_*.csv (e.g. 'diag') so a "
                                             "diagnostic run does not clobber the full sweep")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    steps = 2048 if args.smoke else int(args.steps)
    seeds: List[int] = [0, 1] if args.smoke else list(args.seeds)

    print(f"Controlled experiment | systems={args.systems} seeds={seeds} "
          f"steps={steps:,} smoke={args.smoke}")
    for system in args.systems:
        for seed in seeds:
            train_one(system, seed, steps, args.wandb, args.smoke, args.tag, args.arm)

    suffix = f"_{args.tag}" if args.tag else ""
    print(f"\nAll runs complete. Checkpoints under {CHECKPOINT_ROOT}{suffix}/<system>_seed<seed>/")
    print("Next: python -m experiments.analyze_controlled "
          + ("--smoke" if args.smoke else "--seeds " + " ".join(str(s) for s in seeds)))


if __name__ == "__main__":
    main()
