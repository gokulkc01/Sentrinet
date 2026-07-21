# Roadmap

**Canonical detail:** `docs/DESIGN.md`. This note is the vault-side summary. Guiding principle: **depth over breadth — every stage ships something of value.**

## Thesis
Resilient cooperative target tracking under [[GPS Spoofing and GNSS Denial|GPS spoofing]], using [[Plausibility-Based Trust]] — controlled study → reusable benchmark → micro-drone sim-to-real.

## Stage 0 — Foundation: fix, then measure honestly (~2 weeks) ← **WE ARE HERE**
1. Fix correctness bugs ([[Known Bugs and Confounds]]): observation normalization first ([[ADR-005 - Fixed Observation Normalization]]), then `torch` import, `_sender_trust_sum` reset, test syntax error, hygiene.
2. Run **one controlled A/B/C experiment** ([[Controlled Experiment]]): identical net/rules/budget, vary only trust, 8 seeds, CIs + significance.
3. **Decision gate:** if C ≤ B, document the honest negative result — it becomes the motivation for Stage 1. Either way we finally have *trustworthy numbers*.

## Stage 1 — The innovation (~4–6 weeks)
- Pivot to [[GPS Spoofing and GNSS Denial]] with named [[Threat Scenarios]] (S0–S4).
- Scale to **N=9** so consensus works ([[ADR-002 - Scale to 9 Drones]], [[Robust Statistics and Consensus]]).
- Add the load-bearing environmental realism: **LoS occlusion + RF shadowing** ([[ADR-004 - Terrain Occlusion Only]]).
- Build the [[Plausibility-Based Trust]] detector, trained supervised ([[ADR-003 - Supervised Plausibility Trust]]).
- Beat real baselines (uniform, EMA, trimmed-mean, oracle) on [[Metrics]].

## Stage 2 — Benchmark packaging (~2 weeks)
Installable `sentrinet/` package, YAML scenarios, documented threat taxonomy, baseline table, CI, honest README. **This is the "truly useful" deliverable.**

## Stage 3 — Crazyflie sim-to-real (later, ~2 months)
Velocity-setpoint policy → Crazyswarm; real UWB ranging; software-injected GPS spoof. Small, filmed, real. See [[Sim-to-Real Transfer]].

## Explicitly out of scope (shelved)
Curriculum, LSTM, dashboard polish, trust hyperparameter grid, 3-agent claims, photoreal terrain, RF jamming (GPS spoof only), Byzantine-consensus theory.

## Related
- [[Controlled Experiment]] · [[Threat Scenarios]] · [[Metrics]] · [[Decision Log]]
