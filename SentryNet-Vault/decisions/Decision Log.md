# Decision Log (ADR Index)

Every non-obvious decision, recorded with its **rationale** so future-you (and reviewers) understand *why*, not just *what*. Format: Architecture Decision Record (Context → Decision → Rationale → Consequences → Status).

| ADR | Decision | Status |
|---|---|---|
| [[ADR-001 - GPS-Spoofing Pivot]] | Reframe from "trust without crypto for packet loss" → "plausibility trust under GPS spoofing" | ✅ Accepted |
| [[ADR-002 - Scale to 9 Drones]] | Increase swarm from N=3 to N=9 | ✅ Accepted |
| [[ADR-003 - Supervised Plausibility Trust]] | Learn trust from physical residuals, trained supervised (not emergent) | ✅ Accepted |
| [[ADR-004 - Terrain Occlusion Only]] | Add LoS occlusion + RF shadowing; defer photoreal terrain | ✅ Accepted |
| [[ADR-005 - Fixed Observation Normalization]] | Replace broken running normalization with fixed world-scale | ✅ Accepted |
| [[ADR-006 - GRU Sustained Known-Solvable Config]] | Revert invariant to GRU + sustained (MLP+team never learned) | ✅ Accepted |

## The meta-decision behind all of these
**Depth over breadth.** The project's failure mode is not difficulty — it's *scope sprawl with no verified result*. Every ADR above either (a) removes a confound so results become trustworthy, or (b) makes the one core question sharper. Nothing is added for polish.

## How to use this
When you change direction, add a new ADR here rather than silently editing code. A decision without a recorded reason is a decision you'll relitigate in three months.

## Related
- [[Roadmap]] · [[Does Trust Actually Help]] · [[Known Bugs and Confounds]] · [[00 - START HERE]]
