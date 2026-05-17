"""
train.py  —  SentryNet Phase 2
===============================
Main training entrypoint for Systems A, B, and C.
"""

from __future__ import annotations

import argparse
import random
from typing import Dict, List

import numpy as np
import torch

from border_env import BorderEnv
from mappo_trainer import MAPPOTrainer


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_env(
    system: str,
    seed: int,
    use_curriculum: bool = False,
    use_trust_shaping: bool = True,
    capture_mode: str = "team",
    sustained_steps: int | None = None,
    capture_k: int = 2,
) -> BorderEnv:
    """Create BorderEnv configured for the selected system."""
    if sustained_steps is None:
        sustained_steps = 3

    if system == "A":
        return BorderEnv(
            use_pybullet=False,
            domain_rand=True,
            p_drop=0.0,
            p_spoof=0.0,
            use_trust=False,
            use_curriculum=use_curriculum,
            capture_mode=capture_mode,
            sustained_steps=sustained_steps,
            capture_k=capture_k,
            seed=seed,
        )
    if system == "B":
        return BorderEnv(
            use_pybullet=False,
            domain_rand=True,
            p_drop=0.2,
            p_spoof=0.0,
            use_trust=False,
            use_curriculum=use_curriculum,
            capture_mode=capture_mode,
            sustained_steps=sustained_steps,
            capture_k=capture_k,
            seed=seed,
        )
    if system == "C":
        return BorderEnv(
            use_pybullet=False,
            domain_rand=True,
            p_drop=0.2,
            p_spoof=0.1,
            use_trust=use_trust_shaping,
            use_curriculum=use_curriculum,
            capture_mode=capture_mode,
            sustained_steps=sustained_steps,
            capture_k=capture_k,
            seed=seed,
        )
    raise ValueError(f"Unknown system '{system}'")


def train_one(
    system: str,
    seed: int,
    total_steps: int,
    use_wandb: bool,
    use_curriculum: bool = False,
    use_trust_shaping: bool = True,
    capture_mode: str = "team",
    sustained_steps: int | None = None,
    capture_k: int = 2,
    policy_type: str = "mlp",
    hidden_dim: int = 128,
    run_name: str | None = None,
) -> None:
    """Train one (system, seed) configuration."""
    set_seed(seed)
    env = build_env(
        system=system,
        seed=seed,
        use_curriculum=use_curriculum,
        use_trust_shaping=use_trust_shaping,
        capture_mode=capture_mode,
        sustained_steps=sustained_steps,
        capture_k=capture_k,
    )
    run_name = run_name or f"system_{system}_seed{seed}"

    config: Dict[str, object] = {
        "lr": 3e-4,
        "gamma": 0.99,
        "lam": 0.95,
        "clip_eps": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 10.0,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 4,
        "total_steps": int(total_steps),
        "save_every": 50_000,
        "eval_every": 50_000,
        "use_wandb": bool(use_wandb),
        "run_name": run_name,
        "checkpoint_dir": "checkpoints",
        "seed": int(seed),
        "policy_type": str(policy_type),
        "hidden_dim": int(hidden_dim),
    }

    print(f"\n=== Training {run_name} for {total_steps:,} steps ===")
    trainer = MAPPOTrainer(env=env, config=config)
    trainer.train()
    env.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train SentryNet MAPPO systems")
    parser.add_argument("--system", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning for the run")
    parser.add_argument("--no-trust-shaping", action="store_true", help="Disable trust-aware reward shaping")
    parser.add_argument("--capture-mode", choices=["team", "sustained"], default="team",
                        help="Capture rule used by the environment")
    parser.add_argument("--sustained-steps", type=int, default=None,
                        help="Required consecutive in-range steps for sustained capture")
    parser.add_argument("--capture-k", type=int, default=2,
                        help="Number of drones required for multi-capture mode")
    parser.add_argument("--policy-type", choices=["mlp", "gru", "lstm"], default="mlp",
                        help="Policy architecture used by MAPPO")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden size for recurrent policy variants")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Override checkpoint run directory name")
    parser.add_argument("--seed", type=int, default=None, help="Single seed; default runs seeds 0/1/2")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Multiple seeds, e.g. --seeds 0 1 2")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total steps per run")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--fast", action="store_true", help="Smoke test mode (10,000 steps)")
    return parser.parse_args()


def main() -> None:
    """Train one or more systems based on CLI options."""
    args = parse_args()

    steps = 10_000 if args.fast else int(args.steps)
    use_wandb = not bool(args.no_wandb)
    use_curriculum = bool(args.curriculum)
    use_trust_shaping = not bool(args.no_trust_shaping)

    systems: List[str] = ["A", "B", "C"] if args.system == "all" else [args.system]
    if args.seeds is not None:
        seeds: List[int] = [int(s) for s in args.seeds]
    elif args.seed is not None:
        seeds = [int(args.seed)]
    else:
        seeds = [0, 1, 2]

    for system in systems:
        for seed in seeds:
            train_one(
                system=system,
                seed=seed,
                total_steps=steps,
                use_wandb=use_wandb,
                use_curriculum=use_curriculum,
                use_trust_shaping=use_trust_shaping,
                capture_mode=args.capture_mode,
                sustained_steps=args.sustained_steps,
                capture_k=args.capture_k,
                policy_type=args.policy_type,
                hidden_dim=args.hidden_dim,
                run_name=args.run_name,
            )


if __name__ == "__main__":
    main()
