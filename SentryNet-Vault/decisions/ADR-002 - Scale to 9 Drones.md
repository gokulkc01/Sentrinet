# ADR-002 — Scale the Swarm from N=3 to N=9

**Status:** ✅ Accepted

## Context
The environment is hard-wired to `N_DRONES=3` via module-level globals in [[Environment - BorderEnv|border_env.py]]. With 3 drones, each receiver has only **2 senders**.

## Decision
Parameterize `N_DRONES` and run the core experiments at **N=9** (8 senders per receiver).

## Rationale
- **Consensus and robust statistics need honest samples.** A "median of 2" is meaningless; a single adversary is 50% of the data → above the [[Robust Statistics and Consensus|breakdown point]] of any robust estimator. Trust literally cannot discriminate at N=3.
- At N=9, one or two spoofed nodes are a clear minority → the consensus residual in [[Plausibility-Based Trust]] becomes a real signal.
- Enables the **degradation-vs-k** curve (sweep number of adversaries) — a clean, publishable result ([[Threat Scenarios]] S3).

## Consequences
- Must refactor the hard-coded `N_DRONES` and all shape assumptions (observations, critic input, trust modules).
- Larger joint state → critic input grows; training is heavier (accept it).
- Spawn logic, capture rules, and reward shaping must generalize beyond 3.

## Related
- [[Robust Statistics and Consensus]] · [[Plausibility-Based Trust]] · [[Threat Scenarios]]
