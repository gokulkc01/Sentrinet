# ADR-006 — Revert the Invariant to GRU + Sustained (known-solvable)

**Status:** ✅ Accepted (evidence-based) · **Supersedes the MLP/team choice in [[Controlled Experiment]]**

## Context
The Stage-0 controlled config ([[ADR-001 - GPS-Spoofing Pivot|earlier ADRs]]) fixed the shared setup to **MLP policy + `team` capture + entropy 0.01**. A properly instrumented 400k diagnostic ([[Does Trust Actually Help|diagnostic run]]) showed this config **never learns**:
- eval capture = **0.00** at every checkpoint, all 6 runs;
- train capture flat ~0;
- **entropy rises** (2.16 → ~3.3) — the smoking gun: the policy gradient has no signal, so the entropy bonus dominates and the policy diffuses toward *more* random. This is "never learned", not "learned then collapsed".

Root causes: (1) `team` capture (two drones within 2 m of an evasive target simultaneously) is almost never experienced, so the reward is never seen; (2) a memoryless **MLP** can't track the intruder under the FoV partial observability; (3) the shared-critic value muddies advantages.

Crucially, the old (confounded) runs prove the environment **is** solvable to ~99% with **GRU + sustained** capture.

## Decision
Change the shared invariant config — **identically for A/B/C**, so the "vary only trust" invariant still holds — to the known-solvable setup:
- `policy_type = "gru"` (memory for partial observability)
- `capture_mode = "sustained"`, `sustained_steps = 1`
- `entropy_coef = 0.005` (the 0.01 bonus was dominating the dead gradient)

**Defer** the shared-critic fix to a separate, measured step: the old runs reached ~99% *with* the current critic, so it is not the blocker, and changing it now would confound this change and risk destabilising a known-good configuration.

## Rationale
- Fastest path to a **testbed that actually learns**, so the trust question becomes answerable.
- **Trust stays relevant** even with single-drone `sustained` capture: any drone is frequently blind (target outside its FoV) and must rely on teammates' shared estimates — so [[Trust and Reputation|trust]] still matters.
- One change at a time = interpretable science.

## Consequences
- The eval loop is now **recurrent-aware** (threads GRU hidden state per drone per episode; the old loop ran the GRU as if memoryless — a real trap).
- Invalidates the MLP/team numbers; requires a fresh diagnostic to confirm capture climbs.
- True multi-drone **`team` capture** becomes a possible Stage-1 enhancement once the base learns.
- Shared-critic fix tracked as a future ADR if seed variance stays high.

## Related
- [[Controlled Experiment]] · [[Does Trust Actually Help]] · [[MAPPO]] · [[Decision Log]]
