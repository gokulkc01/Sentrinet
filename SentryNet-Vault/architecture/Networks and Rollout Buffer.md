# Networks and Rollout Buffer

**Files:** `networks.py` (`PolicyNet`, `ValueNet`), `rollout_buffer.py` (`RolloutBuffer`). Part of Layer 4 — the [[MAPPO]] learning machinery.

## PolicyNet (the actor)
- **Structure:** obs encoder (Linear→Tanh) → core (`mlp` = Linear+Tanh, or `gru`/`lstm` recurrent cell) → mean head + a learnable `log_std`.
- **Output:** a Gaussian over actions, **tanh-squashed** into [-1,1] with the correct log-prob correction (`_squashed_log_prob`). Getting this correction right is a common failure point — SentryNet does it correctly.
- **Init:** orthogonal (gain √2 hidden, 0.01 on the mean head) — standard for stable PPO.
- Key methods: `get_action` (execution), `evaluate_actions` (training), `step` (recurrent-aware).

## ValueNet (the critic)
- MLP `obs_dim → 128 → 128 → 1`. Input = concatenation of all drones' observations (**centralized**, CTDE). Default `obs_dim = 3×42 = 126`.

## RolloutBuffer
- Stores per-step `(obs, actions, rewards, values, log_probs, dones)` for all drones (+ hidden states for recurrent policies).
- **`compute_gae(last_values)`** — [[PPO and GAE|GAE]] advantages + returns, then normalizes advantages.
- **`get_batches(batch_size)`** — yields shuffled minibatches for the PPO update.

## Dimensions to remember
- Policy input: **42** (per-drone obs — see [[Observation and Action Spaces]]).
- Critic input: **126** (3×42).
- ⚠️ Docs elsewhere say 20/60 — stale. Code is authoritative: 42/126.

## Related
- [[MAPPO]] · [[PPO and GAE]] · [[MAPPO Trainer]] · [[Observation and Action Spaces]]
