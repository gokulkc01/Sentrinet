"""
evaluate.py  —  SentryNet Phase 2
==================================
Evaluate trained checkpoints and export full experiment CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from border_env import BorderEnv
from mappo_trainer import MAPPOTrainer


DROP_RATES: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# All systems evaluated under the SAME adversarial conditions:
#   - Random spoofing (p_spoof=0.1) on all links
#   - Targeted adversary: drone 1 is compromised (always spoofs)
# Only System C has a trust mechanism to detect and isolate the bad drone.
EVAL_SPOOF_RATE: float = 0.1
EVAL_SPOOF_STD: float = 2.0
COMPROMISED_DRONE: int = 1


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    """Sort checkpoints by numeric step, not lexicographic filename."""
    match = re.search(r"step_(\d+)\.pt$", path.name)
    if match:
        return (int(match.group(1)), path.name)
    if path.name == "final.pt":
        return (-1, path.name)
    return (-2, path.name)


def find_best_checkpoint(run_name: str) -> Path:
    """Select checkpoint with highest step number for a run."""
    run_dir = Path("checkpoints") / run_name
    assert run_dir.exists(), f"Checkpoint directory not found: {run_dir}"
    best = run_dir / "best.pt"
    if best.exists():
        return best
    ckpts = sorted(run_dir.glob("step_*.pt"), key=checkpoint_sort_key)
    if ckpts:
        return ckpts[-1]
    final = run_dir / "final.pt"
    assert final.exists(), f"No checkpoints found in {run_dir}"
    return final


def mean_trust_from_info(info: Dict[str, object]) -> float:
    """Compute mean trust scalar from nested trust lists."""
    trust = info.get("trust_scores", [])
    flat: List[float] = []
    if isinstance(trust, list):
        for row in trust:
            if isinstance(row, list):
                flat.extend([float(x) for x in row])
    return float(np.mean(flat)) if flat else 0.0


def evaluate_condition(
    system: str,
    seed: int,
    drop_rate: float,
    n_episodes: int,
    capture_mode: str,
) -> Dict[str, float]:
    """Evaluate one (system, seed, drop_rate) condition."""
    use_trust = system == "C"
    p_spoof = EVAL_SPOOF_RATE  # same adversarial conditions for all systems

    env = BorderEnv(
        use_pybullet=False,
        domain_rand=False,
        p_drop=drop_rate,
        p_spoof=p_spoof,
        spoof_std=EVAL_SPOOF_STD,
        use_trust=use_trust,
        compromised_drone=COMPROMISED_DRONE,
        capture_mode=capture_mode,
        seed=seed,
    )

    ckpt = find_best_checkpoint(f"system_{system}_seed{seed}")
    ckpt_data = torch.load(ckpt, map_location="cpu")
    ckpt_config = ckpt_data.get("config", {}) if isinstance(ckpt_data.get("config", {}), dict) else {}

    trainer = MAPPOTrainer(
        env=env,
        config={
            "use_wandb": False,
            "total_steps": 1,
            "run_name": f"system_{system}_seed{seed}",
            "seed": seed,
            "policy_type": ckpt_config.get("policy_type", "mlp"),
            "hidden_dim": int(ckpt_config.get("hidden_dim", 128)),
        },
    )

    trainer.load_checkpoint(str(ckpt))

    captures = 0
    steps_list: List[int] = []
    rewards_list: List[float] = []
    trust_list: List[float] = []
    n_close_list: List[float] = []
    team_distance_list: List[float] = []
    formation_list: List[float] = []
    coverage_list: List[float] = []
    # per-episode trust trajectories saved to disk per condition
    trust_trajs: List[List[List[float]]] = []  # episodes -> steps -> nested trust lists
    battery_list: List[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        ep_steps = 0
        final_info: Dict[str, object] = {}

        ep_trusts: List[List[List[float]]] = []
        while not done:
            actions = {}
            for i in range(3):
                a, _ = trainer.policy.get_action(obs[f"drone_{i}"], deterministic=True)
                actions[f"drone_{i}"] = a
            actions["sensor_0"] = 1 if float(obs["sensor_0"][0]) > 0.5 else 0

            obs, rew, term, trunc, info = env.step(actions)
            # snapshot trust scores for this step (list per receiver)
            step_trust = [tm.get_trust_scores().tolist() for tm in env.trust_mods]
            ep_trusts.append(step_trust)
            ep_reward += float(np.mean([rew[f"drone_{i}"] for i in range(3)]))
            ep_steps += 1
            info0 = info.get("drone_0", {})
            n_close_list.append(float(info0.get("n_close", 0)))
            team_distance_list.append(float(info0.get("mean_team_distance", 0.0)))
            formation_list.append(float(info0.get("formation_spread", 0.0)))
            coverage_list.append(float(info0.get("angular_coverage_score", 0.0)))
            done = any(term[f"drone_{i}"] or trunc[f"drone_{i}"] for i in range(3))
            final_info = info.get("drone_0", {})

        captures += int(bool(final_info.get("captured", False)))
        steps_list.append(ep_steps)
        rewards_list.append(ep_reward)
        trust_list.append(mean_trust_from_info(final_info))
        trust_trajs.append(ep_trusts)

        drone_pos = final_info.get("drone_pos")
        if isinstance(drone_pos, np.ndarray):
            # Battery is stored in env directly; this keeps requested column present.
            battery_list.append(float(np.mean(env.battery)))
        else:
            battery_list.append(float(np.mean(env.battery)))

    env.close()

    # Save trust trajectories for this condition for later plotting/analysis
    out_dir = Path("results") / "trust_trajs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"trust_{system}_seed{seed}_drop{int(drop_rate*100)}.npz"
    # store as compressed pickle-friendly object
    np.savez_compressed(str(fname), episodes=np.array(trust_trajs, dtype=object))

    return {
        "system": system,
        "seed": seed,
        "drop_rate": drop_rate,
        "capture_rate": captures / n_episodes,
        "mean_steps": float(np.mean(steps_list)) if steps_list else 0.0,
        "mean_reward": float(np.mean(rewards_list)) if rewards_list else 0.0,
        "mean_trust": float(np.mean(trust_list)) if trust_list else 0.0,
        "mean_battery": float(np.mean(battery_list)) if battery_list else 0.0,
        "mean_n_close": float(np.mean(n_close_list)) if n_close_list else 0.0,
        "mean_team_distance": float(np.mean(team_distance_list)) if team_distance_list else 0.0,
        "mean_formation_spread": float(np.mean(formation_list)) if formation_list else 0.0,
        "mean_angular_coverage": float(np.mean(coverage_list)) if coverage_list else 0.0,
        "p_spoof": p_spoof,
        "capture_mode": capture_mode,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI args for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate SentryNet checkpoints")
    parser.add_argument("--system", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--capture-mode", choices=["team", "sustained"], default="team")
    parser.add_argument("--fast", action="store_true", help="Run 20 episodes per condition")
    return parser.parse_args()


def main() -> None:
    """Run full condition sweep and save results CSV."""
    args = parse_args()
    systems = ["A", "B", "C"] if args.system == "all" else [args.system]
    episodes = 20 if args.fast else int(args.episodes)

    rows: List[Dict[str, float]] = []
    for system in systems:
        for seed in args.seeds:
            for drop in DROP_RATES:
                print(f"Evaluating system={system} seed={seed} drop={drop:.1f} ...")
                rows.append(evaluate_condition(system=system, seed=int(seed), drop_rate=float(drop), n_episodes=episodes, capture_mode=args.capture_mode))

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "full_experiment.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")


if __name__ == "__main__":
    main()
