# MAPPO (Multi-Agent PPO)

**Definition:** The multi-agent extension of [[PPO and GAE|PPO]]. Each agent acts with a policy (actor); a **centralized critic** estimates value from joint information during training. A textbook [[Multi-Agent Reinforcement Learning|MARL]] baseline that is simple and surprisingly strong.

## The three ingredients
1. **Shared actor (policy).** All homogeneous agents (the 3 drones) use *one* network. It maps a local observation → an action distribution. Sharing weights makes learning sample-efficient and encourages consistent, coordinated behavior.
2. **Centralized critic (value).** Sees the *concatenation* of all agents' observations and outputs a value estimate. More context → better baselines → lower-variance advantages. Used **only in training** (CTDE).
3. **PPO update.** Clipped policy-gradient step + value regression + entropy bonus. See [[PPO and GAE]].

## Why "shared policy but centralized critic"
- Shared **actor** → efficiency + coordination, and it's what runs on-device at execution (local obs only).
- Centralized **critic** → training-time-only crutch that stabilizes learning by reducing the non-stationarity problem of [[Multi-Agent Reinforcement Learning|MARL]].

## In SentryNet
- Implemented in [[Networks and Rollout Buffer]] (`PolicyNet`, `ValueNet`) and [[MAPPO Trainer]] (`MAPPOTrainer`).
- Policy input = 42-dim per-drone observation; critic input = 3 × 42 = 126-dim joint state. See [[Observation and Action Spaces]].
- ⚠️ **Known design wart:** the critic outputs one joint value assigned to all three drones while returns are per-drone — see [[Known Bugs and Confounds]]. Defensible as "team value," but currently undocumented and noisy.

## Related
- [[PPO and GAE]] · [[Multi-Agent Reinforcement Learning]] · [[MAPPO Trainer]] · [[Networks and Rollout Buffer]]
