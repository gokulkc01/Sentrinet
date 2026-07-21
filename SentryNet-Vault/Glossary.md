# Glossary

Quick definitions. Follow the `[[links]]` for depth.

| Term | Meaning |
|---|---|
| **MARL** | Multi-Agent Reinforcement Learning — many agents learning together. See [[Multi-Agent Reinforcement Learning]]. |
| **MAPPO** | Multi-Agent PPO — the algorithm we use. See [[MAPPO]]. |
| **PPO** | Proximal Policy Optimization — stable policy-gradient RL. See [[PPO and GAE]]. |
| **GAE** | Generalized Advantage Estimation — low-variance advantage targets. See [[PPO and GAE]]. |
| **CTDE** | Centralized Training, Decentralized Execution — global info in training, local-only at run time. |
| **Actor / Policy** | Network mapping observation → action distribution. `PolicyNet`. |
| **Critic / Value** | Network estimating expected return; centralized here. `ValueNet`. |
| **Advantage** | How much better an action was than the policy's average. |
| **Rollout** | A batch of environment steps collected before a learning update. `RolloutBuffer`. |
| **Trust score (τ)** | Per-sender belief ∈ [0,1] in a teammate's messages. See [[Trust and Reputation]]. |
| **EMA** | Exponential Moving Average — the current trust update rule. |
| **Aggregation** | Trust-weighted combination of teammate messages. `TrustAggregator`. |
| **p_drop / p_spoof** | Probabilities a message is dropped / spoofed. See [[Adversarial Communication]]. |
| **Spoofing** | Feeding false coordinates. Here: additive noise or a compromised drone. |
| **GNSS / GPS denial** | Jamming/spoofing satellite navigation. See [[GPS Spoofing and GNSS Denial]]. |
| **Plausibility trust** | Scoring messages by physical consistency, not error-vs-own-estimate. See [[Plausibility-Based Trust]]. |
| **UWB ranging** | Radio time-of-flight distance measurement — the un-spoofable cross-check. |
| **Consensus residual** | Deviation of a message from the swarm's fused estimate. See [[Robust Statistics and Consensus]]. |
| **Breakdown point** | Fraction of corrupt data a robust method tolerates. |
| **Domain randomization** | Randomizing sim params for robust transfer. See [[Sim-to-Real Transfer]]. |
| **CTDE critic** | Training-only value net seeing all agents' observations. |
| **Capture mode** | Rule for "intruder caught": `team` (≥2 drones near) or `sustained` (one drone near for k steps). |
| **Curriculum** | Staged difficulty ramp during training (currently shelved). |
| **ADR** | Architecture Decision Record — a logged, rationale-backed decision. See [[Decision Log]]. |

## Related
- [[00 - START HERE]] · [[System Architecture]] · [[Decision Log]]
