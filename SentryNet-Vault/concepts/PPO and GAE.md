# PPO and GAE

The optimization engine under [[MAPPO]]. Two ideas: **PPO** keeps policy updates small and stable; **GAE** produces low-variance learning targets.

## PPO (Proximal Policy Optimization)
A policy-gradient method that improves the policy without taking destructive steps.

- **Clipped objective:** maximize `min(ratio · A, clip(ratio, 1−ε, 1+ε) · A)`, where `ratio = π_new(a|s) / π_old(a|s)` and `A` is the advantage. The clip (ε≈0.2) stops any single update from moving the policy too far.
- **Value loss:** regress the critic toward observed returns.
- **Entropy bonus:** rewards randomness in the policy to keep it exploring.
- **Total loss:** `policy_loss + c_v · value_loss − c_e · entropy`.

## GAE (Generalized Advantage Estimation)
Estimates the **advantage** `A_t` = "how much better than expected was this action?" while trading off bias vs. variance with a parameter λ.

- `δ_t = r_t + γ·V(s_{t+1}) − V(s_t)` (the TD error)
- `A_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + …`
- γ (≈0.99) = discount; λ (≈0.95) = bias/variance knob. λ=1 → high variance Monte-Carlo; λ=0 → biased 1-step.
- Advantages are then **normalized** (zero mean, unit std) per batch for stable gradients.

## In SentryNet
- GAE computed in [[Networks and Rollout Buffer|RolloutBuffer.compute_gae]].
- PPO update loop in [[MAPPO Trainer|MAPPOTrainer.update]] with `clip_eps=0.2`, `value_coef=0.5`, `entropy_coef=0.01`, `n_epochs=4`.
- Actions are **tanh-squashed Gaussians** with the proper log-prob correction (`networks.py`) — a detail many implementations get wrong.

## Related
- [[MAPPO]] · [[Multi-Agent Reinforcement Learning]] · [[Networks and Rollout Buffer]] · [[MAPPO Trainer]]
