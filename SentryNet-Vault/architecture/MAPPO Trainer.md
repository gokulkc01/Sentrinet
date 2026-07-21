# MAPPO Trainer

**File:** `mappo_trainer.py` · **Class:** `MAPPOTrainer`. The training loop that drives Layer 4. Algorithm: [[MAPPO]].

## The loop (`train`)
Repeat until `total_steps`:
1. **`collect_rollout()`** — run the shared [[Networks and Rollout Buffer|PolicyNet]] in the env for `n_steps` (2048), storing transitions; reset env on episode end; compute [[PPO and GAE|GAE]].
2. **`update()`** — `n_epochs` (4) of minibatch PPO: clipped policy loss + value regression + entropy bonus; grad-clip; Adam step for policy and value.
3. **Log** metrics (optional wandb), periodically **`evaluate()`** and **`save_checkpoint()`**.

## Key config (`DEFAULT_CONFIG`)
`lr=3e-4`, `gamma=0.99`, `lam=0.95`, `clip_eps=0.2`, `value_coef=0.5`, `entropy_coef=0.01`, `n_steps=2048`, `batch_size=256`, `n_epochs=4`, `total_steps=1e6`, `save/eval_every=50k`.

## Evaluation
`evaluate()` spins up a **fresh** `BorderEnv` (no domain rand) and runs deterministic episodes. ⚠️ Because the env is fresh, the broken observation normalization restarts from zero stats → train/eval mismatch. See [[Known Bugs and Confounds]].

## ⚠️ Design warts
- **Shared value across drones:** `values_dict[k] = v` gives all 3 drones the same joint value, while returns are per-drone → the critic regresses one input against 3 targets (learns the mean). Defensible team-value design, but undocumented.
- **Sensor hard-coded:** `actions_env["sensor_0"] = 1 if obs>0.5` — the "sensor agent" doesn't learn. See [[Observation and Action Spaces]].
- **`use_wandb` defaults True** — noisy; Stage 0 flips this off by default.

## Related
- [[MAPPO]] · [[Networks and Rollout Buffer]] · [[Controlled Experiment]] · [[Known Bugs and Confounds]]
