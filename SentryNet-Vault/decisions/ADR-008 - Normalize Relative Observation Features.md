# ADR-008 — Normalize the Relative Observation Features

**Status:** ✅ Accepted · **Validation of learning-reliability in progress**

## Context
Even after the dense-pursuit reward ([[ADR-007 - Dense Pursuit Reward]]), learning stayed **fragile**: one throwaway subclass run reached 42% capture, but production runs (System A clean, System B adversarial) both stalled at ~0% with rising entropy. Same config — different luck.

Investigation ruled out perception: `rel_target_pos = intruder_pos − drone_pos` (the intruder's **true** relative position) is **always** in the observation (dims 23:26), regardless of the FoV cone. The drone is never actually blind to the intruder.

But that feature — and all the relative features (dims 20:42) — were appended **raw at ~±20 scale**, while the base features (dims 0:20) were normalized to [-1,1]. Empirically `rel_target_pos ≈ [-5.8, 16.4, 1.6]`, the distance feature ≈ 17. Feeding ±17 inputs into a **Tanh** network saturates it, so the single most important feature was the *worst* conditioned → unreliable gradients → fragile learning. Crucially, the earlier "normalization on/off" test ([[Does Trust Actually Help|diagnostics]]) only toggled dims 0:20 and never touched these — so it falsely cleared normalization.

## Decision
Normalize the relative features to ~[-1,1]:
- relative positions ÷ `[WORLD_XY, WORLD_XY, 10]` = `[20, 20, 10]`
- relative velocities ÷ `MAX_SPEED` (5)
- distance ÷ 30 (world diagonal)

Full observation is now in [-1,1] end to end.

## Rationale
Standard input conditioning. A Tanh policy/critic can only reliably learn from a feature that is well-scaled; the intruder-direction signal must not saturate the first layer.

## Consequences
- Invalidates checkpoints trained on the raw features (re-running regardless).
- **Validation in progress:** a 3-seed run must show learning is now *reliable* (multiple seeds climb), not a lucky one-off, before we declare the testbed fixed.
- Complements [[ADR-005 - Fixed Observation Normalization]] (which fixed only the base-feature normalization).

## Related
- [[ADR-005 - Fixed Observation Normalization]] · [[ADR-007 - Dense Pursuit Reward]] · [[Observation and Action Spaces]] · [[Does Trust Actually Help]]
