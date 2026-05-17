"""
run_trust_grid.py

Quick grid search over trust hyperparameters (fast smoke runs).
Saves aggregated results to `results/trust_grid.csv`.
"""
from pathlib import Path
import itertools
import csv

import numpy as np
from trust_module import TrustModule
from border_env import BorderEnv
from mappo_trainer import MAPPOTrainer

RESULTS = []

alphas = [0.05, 0.1, 0.2]
maxerr_mults = [1.0, 1.5, 2.0]
seeds = [0, 1, 2]

out_dir = Path("results")
out_dir.mkdir(parents=True, exist_ok=True)
out_csv = out_dir / "trust_grid.csv"

for alpha, mm in itertools.product(alphas, maxerr_mults):
    max_err = TrustModule.MAX_ERROR * mm
    for seed in seeds:
        print(f"Running alpha={alpha} max_err={max_err:.2f} seed={seed} ...")
        env = BorderEnv(
            use_pybullet=False,
            domain_rand=True,
            p_drop=0.2,
            p_spoof=0.1,
            use_trust=True,
            compromised_drone=1,
            seed=int(seed),
            trust_alpha=float(alpha),
            trust_max_error=float(max_err),
        )
        cfg = {
            "use_wandb": False,
            "total_steps": 10_000,
            "run_name": f"grid_alpha{alpha}_mm{mm}_seed{seed}",
            "seed": int(seed),
            "save_every": 1000000,
            "eval_every": 1000000,
        }
        trainer = MAPPOTrainer(env=env, config=cfg)
        trainer.train()
        # quick eval
        stats = trainer.evaluate(n_episodes=20, p_drop_eval=0.2, p_spoof_eval=0.1)
        row = {
            "alpha": alpha,
            "maxerr_mult": mm,
            "max_error": max_err,
            "seed": seed,
            "capture_rate": stats["capture_rate"],
            "mean_trust": stats["mean_trust"],
            "mean_reward": stats["mean_reward"],
            "mean_steps": stats["mean_steps"],
        }
        RESULTS.append(row)
        # cleanup env
        try:
            env.close()
        except Exception:
            pass

# save CSV
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(RESULTS[0].keys()))
    writer.writeheader()
    for r in RESULTS:
        writer.writerow(r)

print(f"Grid complete — saved to {out_csv}")
