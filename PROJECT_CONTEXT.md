# SentryNet — Project Context (updated May 18, 2026)

This file gives a concise, current summary of the codebase, experimental systems, recent results, and known limitations. Use it as the canonical reference for collaborators and reviewers.

**High-level summary**
- SentryNet is a modular research prototype that studies trust-aware multi-agent RL for cooperative drone pursuit under adversarial communication. The codebase contains training (MAPPO), evaluation sweeps, visualization (Pygame + PyBullet), and analysis tooling.
- Recent work migrated the pipeline to use per-agent noisy local estimates (obs_dim=42) and added a curriculum and trust-consensus mechanisms to remove earlier ground-truth leakage.

**Key components**
- Environment: `border_env.py` (mock physics + optional PyBullet) with communication channel and trust pipeline.
- Adversary: `adversarial_channel.py` implements distance-dependent drops and probabilistic spoofing.
- Trust: `trust_module.py` (EMA + consensus heuristics) and `trust_aggregator.py`.
- Learning: `networks.py`, `rollout_buffer.py`, `mappo_trainer.py`, and `train.py`.
- Evaluation & analysis: `run_trained.py`, `evaluate.py`, `scripts/inspect_ckpt.py`, `scripts/plot_eval_results.py`, `scripts/plot_full_experiment.py`.
- Visualization: `dashboard.py`, `simulation_trial.py`, PyBullet optional overlay.

**Experimental systems (short)**
- System A — clean baseline (no drops/no spoofing) during training.
- System B — trained with packet loss (p_drop > 0) but no spoofing.
- System C — trust-aware (uses trust aggregation + training with p_spoof > 0).

Each system has multi-seed checkpoints (seeds 0/1/2). Checkpoints are stored in `checkpoints/` with `step_*.pt` artifacts.

Recent results and artifacts
- Aggregated sweep: `results/full_experiment.csv` (per-system, per-seed, multiple drop rates). Rows contain capture_rate, mean_steps, mean_reward, mean_trust, and `p_spoof`.
- Generated plots (saved to `results/plots/`):
  - `capture_vs_drop.png` — capture rate vs drop rate (mean ± std over seeds/conditions)
  - `steps_vs_drop.png` — mean episode length vs drop rate
  - `reward_vs_drop.png` — mean reward vs drop rate
  - `trust_vs_drop.png` — mean trust vs drop rate
  - `capture_rates_systems.png` — deterministic eval capture rates (per-checkpoint files)
- Utility scripts added/updated: `scripts/inspect_ckpt.py`, `scripts/plot_eval_results.py`, `scripts/plot_full_experiment.py` (these produce the above CSV/PNG artifacts).

Representative single-checkpoint deterministic evals (seed=1, 300 eps, headless):
- System A (seed1): ~88.0% capture
- System B (seed1): ~90.7% capture
- System C (seed1): ~77.3% capture

Interpretation note: single-seed deterministic runs can differ substantially from the aggregated CSV (which is multi-seed and uses different evaluation settings). Always prefer aggregated CSV analyses for conclusions.

Known issues & limitations
- Seed variance: performance shows high variance across seeds — run 5+ seeds for robust claims.
- PyBullet stability: long multi-episode visual runs occasionally trigger `pybullet.error: Not connected to physics server` — dashboard needs graceful reconnect/fallback.
- Trust realism migration: Step 1–3 (local estimates, no-GT broadcasts, consensus updates) completed; retraining across seeds with the realistic pipeline (full evaluation) is pending.
- Some plots previously used synthetic traces; real per-step trust trajectories must be recorded for publication-quality figures.

Running and reproducing important outputs
- Regenerate the publication plots (reads `results/full_experiment.csv`):
```powershell
.\sentrinet_env\Scripts\Activate.ps1
python scripts\plot_full_experiment.py
```
- Re-parse single-checkpoint eval files (robust decoding) and regenerate `capture_rates_systems.png`:
```powershell
.\sentrinet_env\Scripts\Activate.ps1
python scripts\plot_eval_results.py
```
- Inspect a checkpoint:
```powershell
.\sentrinet_env\Scripts\Activate.ps1
python scripts\inspect_ckpt.py checkpoints\system_C_gru_seed1\step_1001472.pt
```

Next recommended actions (short)
1. Aggregate results across seeds (run missing seed evaluations or re-run `validation_harness.py` with 5 seeds).
2. Retrain or fine-tune weak seeds (seed 1 for System C showed lower capture in some runs).
3. Instrument evaluation to log per-step trust for real trust-dynamics plots.
4. Harden the dashboard PyBullet loop to recover from disconnects.

If you need a one-line status for a README or PR, use: "Project ready for targeted retraining and statistical aggregation; plotting and analysis scripts produce multi-condition figures in `results/plots/`."
