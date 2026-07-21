# ADR-005 — Replace Running Normalization with Fixed World-Scale

**Status:** ✅ Accepted · **First code change of Stage 0**

## Context
`border_env._normalize_obs_features` maintains a running mean/std over the first 20 observation dims. It is broken in three ways (see [[Known Bugs and Confounds]]):
1. The Welford std update is mathematically wrong — variance never converges, drifts upward.
2. Stats live inside the env instance and **re-learn from zero on every fresh eval env** → train/eval distribution mismatch.
3. Stats are **not saved in the checkpoint** → a loaded policy sees different normalization than it trained under.

This is the single most damaging bug — it corrupts every number the project has produced.

## Decision
Delete the running-stats normalizer. Replace with **fixed, deterministic world-scale normalization**: divide positions by world bounds, velocities by `MAX_SPEED`, etc. — constants known from the environment definition.

## Rationale
- Deterministic and stateless → **no train/eval skew, no checkpoint dependence**, fully reproducible.
- Simpler and correct; removes a whole class of silent failure.
- Running normalization (à la VecNormalize) *can* be done right, but it must be frozen at eval and persisted — unnecessary complexity here when world scales are known a priori.

## Consequences
- **Invalidates all existing checkpoints** — accepted, since their numbers were corrupt anyway.
- Requires re-running the [[Controlled Experiment]] from scratch (which we're doing regardless).

## Related
- [[Known Bugs and Confounds]] · [[Environment - BorderEnv]] · [[Controlled Experiment]]
