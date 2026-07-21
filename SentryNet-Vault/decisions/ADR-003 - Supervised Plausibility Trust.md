# ADR-003 — Learn Plausibility Trust, Supervised

**Status:** ✅ Accepted

## Context
The current EMA trust ([[Trust Module and Aggregator]]) fails because its reference (the receiver's own estimate) is unavailable when the receiver is blind — exactly when it matters. We need a trust signal that works when blind, under [[GPS Spoofing and GNSS Denial]].

## Decision
Build a **learned per-sender trust detector** over **physical-plausibility residual features** (kinematic feasibility, RF-ranging cross-check, consensus residual, temporal consistency), trained **supervised** on simulated attacks where the sim provides ground-truth spoof labels. See [[Plausibility-Based Trust]].

## Rationale
- **Supervised, not emergent:** letting trust emerge purely from RL reward is slow, entangled, and hard to verify. Supervised training gives a fast, modular, independently-testable component with clear metrics (detection rate, false-accusation rate).
- **Modularity is a feature for defense:** a separable, verifiable trust module is the architecture reviewers accept — you can test it in isolation and reason about its guarantees.
- The **RF-ranging** feature is the crux: GPS can be spoofed, radio time-of-flight cannot — and it maps to real hardware ([[Sim-to-Real Transfer]]).

## Consequences
- Need a labeled-attack data-generation pipeline in sim.
- The trust module and the RL policy are trained/validated separately, then composed.
- Must still beat classical **trimmed-mean** fusion to justify the complexity ([[Metrics]]).

## Related
- [[Plausibility-Based Trust]] · [[Robust Statistics and Consensus]] · [[Metrics]]
