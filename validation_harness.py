#!/usr/bin/env python3
"""
Validation harness: A/B/C comparison of training architectures.
- Baseline: no curriculum, basic reward shaping only
- Curriculum: curriculum learning only
- Full: curriculum + trust-aware reward shaping

Run with 3 seeds, collect metrics, save results.
"""
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from border_env import BorderEnv
from mappo_trainer import MAPPOTrainer

RESULTS_DIR = Path("results/validation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Training config
TOTAL_STEPS = 50_000  # Full training run
EVAL_INTERVAL = 5_000
SEEDS = [0, 1, 2]
N_EVAL_EPISODES = 20

CONFIGS = {
    "baseline": {
        "use_curriculum": False,
        "use_trust": False,
        "description": "No curriculum, no trust shaping",
    },
    "curriculum": {
        "use_curriculum": True,
        "use_trust": False,
        "description": "Curriculum only",
    },
    "full": {
        "use_curriculum": True,
        "use_trust": True,
        "description": "Curriculum + trust-aware shaping",
    },
}


def train_configuration(
    config_name: str,
    config_params: Dict,
    seed: int,
) -> Dict:
    """Train one configuration with one seed."""
    print(f"\n{'=' * 70}")
    print(f"CONFIG: {config_name:12s} | SEED: {seed} | {config_params['description']}")
    print(f"{'=' * 70}")
    
    # Create env
    env = BorderEnv(
        use_pybullet=False,
        domain_rand=False,
        p_drop=0.2,
        p_spoof=0.1,
        use_curriculum=config_params["use_curriculum"],
        curriculum_progress=0.0,
        use_trust=config_params["use_trust"],
        seed=seed,
    )
    
    # Trainer config
    trainer_config = {
        "n_steps": 2048,
        "total_steps": TOTAL_STEPS,
        "seed": seed,
        "n_epochs": 3,
        "batch_size": 256,
    }
    
    trainer = MAPPOTrainer(env, trainer_config)
    
    # Training loop
    results = {
        "config": config_name,
        "seed": seed,
        "steps": [],
        "mean_rewards": [],
        "capture_rates": [],
        "mean_trusts": [],
        "policy_losses": [],
        "value_losses": [],
        "entropies": [],
    }
    
    start_time = time.time()
    epoch = 0
    
    while trainer.total_env_steps < TOTAL_STEPS:
        # Collect rollout
        rollout_metrics = trainer.collect_rollout()
        
        # Train
        train_metrics = trainer.update()
        
        # Log
        trainer.total_env_steps += trainer_config["n_steps"]
        epoch += 1
        
        if epoch % 5 == 0 or trainer.total_env_steps >= TOTAL_STEPS:
            elapsed = time.time() - start_time
            print(f"  Step {trainer.total_env_steps:7d}/{TOTAL_STEPS} | "
                  f"Reward: {rollout_metrics['mean_reward']:7.4f} | "
                  f"Capture: {rollout_metrics['capture_rate']:6.1%} | "
                  f"Trust: {rollout_metrics['mean_trust']:6.4f} | "
                  f"Time: {elapsed:.1f}s")
            
            results["steps"].append(trainer.total_env_steps)
            results["mean_rewards"].append(rollout_metrics["mean_reward"])
            results["capture_rates"].append(rollout_metrics["capture_rate"])
            results["mean_trusts"].append(rollout_metrics["mean_trust"])
            results["policy_losses"].append(train_metrics["policy_loss"])
            results["value_losses"].append(train_metrics["value_loss"])
            results["entropies"].append(train_metrics["entropy"])
        
        if trainer.total_env_steps >= TOTAL_STEPS:
            break
    
    elapsed_total = time.time() - start_time
    print(f"\n  Total time: {elapsed_total:.1f}s")
    
    # Final evaluation
    print(f"  Running final evaluation...")
    eval_metrics = trainer.evaluate(n_episodes=N_EVAL_EPISODES, p_drop_eval=0.2, p_spoof_eval=0.1)
    results["eval_capture_rate"] = eval_metrics["capture_rate"]
    results["eval_mean_reward"] = eval_metrics["mean_reward"]
    results["eval_mean_trust"] = eval_metrics["mean_trust"]
    
    print(f"  Eval capture rate: {eval_metrics['capture_rate']:.1%}")
    print(f"  Eval mean reward: {eval_metrics['mean_reward']:.4f}")
    print(f"  Eval mean trust: {eval_metrics['mean_trust']:.4f}")
    
    return results


def main():
    """Run full validation suite."""
    print("\n" + "=" * 70)
    print("VALIDATION HARNESS: A/B/C Training Architecture Comparison")
    print("=" * 70)
    print(f"\nTotal steps per run: {TOTAL_STEPS:,}")
    print(f"Configurations: {', '.join(CONFIGS.keys())}")
    print(f"Seeds: {SEEDS}")
    print(f"Evaluation episodes: {N_EVAL_EPISODES}")
    
    all_results = []
    summary_stats = {}
    
    # Run all configs
    for config_name, config_params in CONFIGS.items():
        config_results = []
        
        for seed in SEEDS:
            result = train_configuration(config_name, config_params, seed)
            config_results.append(result)
            all_results.append(result)
        
        # Aggregate stats for this config
        final_captures = [r["capture_rates"][-1] if r["capture_rates"] else 0.0 
                         for r in config_results]
        final_rewards = [r["mean_rewards"][-1] if r["mean_rewards"] else 0.0 
                        for r in config_results]
        eval_captures = [r.get("eval_capture_rate", 0.0) for r in config_results]
        eval_rewards = [r.get("eval_mean_reward", 0.0) for r in config_results]
        
        summary_stats[config_name] = {
            "final_capture_mean": float(np.mean(final_captures)),
            "final_capture_std": float(np.std(final_captures)),
            "final_reward_mean": float(np.mean(final_rewards)),
            "final_reward_std": float(np.std(final_rewards)),
            "eval_capture_mean": float(np.mean(eval_captures)),
            "eval_capture_std": float(np.std(eval_captures)),
            "eval_reward_mean": float(np.mean(eval_rewards)),
            "eval_reward_std": float(np.std(eval_rewards)),
        }
    
    # Save results
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    for config_name, stats in summary_stats.items():
        print(f"\n{config_name}:")
        print(f"  Final capture rate: {stats['final_capture_mean']:.2%} +/- {stats['final_capture_std']:.2%}")
        print(f"  Final reward: {stats['final_reward_mean']:.4f} +/- {stats['final_reward_std']:.4f}")
        print(f"  Eval capture rate: {stats['eval_capture_mean']:.2%} +/- {stats['eval_capture_std']:.2%}")
        print(f"  Eval reward: {stats['eval_reward_mean']:.4f} +/- {stats['eval_reward_std']:.4f}")
    
    # Save to JSON
    results_file = RESULTS_DIR / f"validation_results_{int(time.time())}.json"
    with open(results_file, "w") as f:
        json.dump({
            "summary": summary_stats,
            "detailed": all_results,
            "config": {
                "total_steps": TOTAL_STEPS,
                "seeds": SEEDS,
                "eval_episodes": N_EVAL_EPISODES,
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Save CSV for easy spreadsheet analysis
    csv_file = RESULTS_DIR / f"validation_summary.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "config",
            "final_capture_mean", "final_capture_std",
            "final_reward_mean", "final_reward_std",
            "eval_capture_mean", "eval_capture_std",
            "eval_reward_mean", "eval_reward_std",
        ])
        for config_name, stats in summary_stats.items():
            writer.writerow([
                config_name,
                stats["final_capture_mean"], stats["final_capture_std"],
                stats["final_reward_mean"], stats["final_reward_std"],
                stats["eval_capture_mean"], stats["eval_capture_std"],
                stats["eval_reward_mean"], stats["eval_reward_std"],
            ])
    
    print(f"Summary CSV saved to: {csv_file}")


if __name__ == "__main__":
    main()
