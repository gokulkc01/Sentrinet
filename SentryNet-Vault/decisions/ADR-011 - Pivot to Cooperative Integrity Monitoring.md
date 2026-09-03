# ADR-011 — Pivot to Cooperative Integrity Monitoring

**Status:** ✅ Accepted · 2026-09-04 · **Supersedes the framing of [[Roadmap]] Stages 0.2–3**

## Context

[[ADR-005 - Fixed Observation Normalization]] through [[ADR-010 - Reward Shaping Is the Blocker]]
made the testbed learn. The validated config — `both` reward arm, per-agent critic, GRU —
reached final-100k training capture of **25.0 / 48.1 / 57.9%** across three seeds at 1M
steps (eval 50 / 54 / 70%), still climbing on every seed, with entropy *rising*
2.15 → 2.70. Against a scripted pursuer that scores 95–100%.

That is the whole picture, and two things follow from it.

**First, the MARL race is unwinnable here.** At ~2 h per 1M steps on 12 CPU cores, the
10–50M steps that serious MARL results require is roughly a month of continuous compute
per configuration. There is no path to convergence on this hardware.

**Second, the dependency graph was inverted.** The novel, defensible contribution —
detecting a GNSS-spoofed node from unforgeable physics — sat *downstream* of the
hardest, least differentiated component. Meanwhile the detector itself trains
**supervised, in minutes, on CPU**. Effort was being spent on the part of the system
with the worst value-per-hour, gating the part with the best.

## Decision

**Reframe the project as cooperative integrity monitoring for GNSS-denied swarms** — the
swarm-level generalisation of aviation **RAIM**. A single receiver uses redundant
satellite measurements to detect a faulty one; the swarm uses redundant *peer*
measurements (GNSS claims cross-checked against UWB ranges) to detect a faulty *node*.

Concretely:

1. **The integrity monitor becomes the product.** Pursuit control becomes scripted;
   MARL is demoted to an optional ablation on top of a working system. This also removes
   the largest remaining confound — with a deterministic controller, degradation under
   attack is attributable to the comms/integrity layer rather than to policy noise.
2. **Capture rate is retired as a primary metric**, replaced by the integrity standard:
   P(missed detection), P(false alert), time-to-detect, false-accusation rate,
   correct-isolation rate, fused position error, and a **cooperative protection level**.
3. **Stage 0.2 — the A/B/C "does trust help?" experiment — is dropped.**
4. **Hardware is deferred**, not cancelled.
5. **v2 code is frozen, not deleted.** New work lives in `sentrinet/` on branch
   `sentrinet-v3-integrity`; `trust_module.py` is ported to `baselines/` as the thing to
   beat. Plan: `docs/DESIGN-v3.md`.

## Rationale — the geometric argument

A node claiming a 3-D position has 3 unknowns and one range constraint per peer:

| Peers | Constraints | Consequence |
|---|---|---|
| 2 (N=3) | 2 | **Under-determined** — a continuum of consistent lies |
| 3 (N=4) | 3 | Generically two solutions: the truth and its mirror |
| 4+ (N≥5) | 4+ | **Over-determined** — generically only the truth |

**...unless the peers are coplanar**, where the mirror solution survives at any swarm
size. Drones at a common altitude are nearly coplanar, making **altitude diversity a
security property**.

This gives [[Does Trust Actually Help]] a structural cause rather than an apology:
**at N=3 the detection problem is unsolvable**, because a consistent lie always exists.
No trust mechanism could have worked. It also turns [[ADR-002 - Scale to 9 Drones]] from
a heuristic into a necessity, and predicts a threshold near **N≈5** — with N=9 as margin
for collusion and occlusion-severed links.

Because occlusion removes links and so lowers effective peer count, **detectability
becomes a function of effective connectivity**. That is the frontier result the project
now exists to produce.

## Consequences

### Validated immediately (phase 0, Test 1 — N=9, 2000 trials)

- **The per-node attribution statistic is exactly χ²**: mean 8.054 against an expected
  8, KS p = **0.691**. The statistic the whole method rests on is sound.
- **The network-level statistic is gauge-degenerate.** Translation and rotation of all
  claims leave every pairwise residual unchanged, so `rank(J) = 3N − 6` — measured at
  exactly 21 for N=9. Using one dof per link inflates the statistic 6× (mean 222.9 vs.
  36). After rank truncation, mean 20.65 vs. 21.
- Remaining tail error is first-order linearisation error, `O(σ_g²/link_length)`,
  confirmed by sweeping minimum separation: mean 679 → 142 → 62 → 42 at 4 → 10 → 20 →
  40 m.
- **This replaces a hand-tuned constant with a calibrated test.** v1 trust was
  `max(0, 1 − error/5.0)`; there was no way to state a false-alarm rate for `5.0`, and
  therefore no way to defend it. `alpha` now *is* the false-alarm probability.

### Measured (phase 0, Test 2)

Residual noise floor `sqrt(2σ_g² + σ_r²)` = 2.12 m. Detection needs **~4σ of
displacement** (52.8% at 8 m, 93.0% at 12 m); false-alarm rate 1.0% matches the
predicted `n_nodes × α`. **Correct isolation lags detection badly** — 80% vs. 99.6% at
20 m — because a spoofed node biases its honest peers' statistics too.

Two consequences for the plan: single-epoch detection is weak, so **sequential detection
(CUSUM/SPRT) is where the power must come from**; and detection-vs-isolation is a real
sub-problem to report separately, not a detail.

### Confirmed (phase 0, Test 3 — the gate)

Largest displacement at which the attacker's best *consistent* lie stays below the
2.12 m noise floor:

| N | 3-D (15 m altitude spread) | peers coplanar, attacker off-plane |
|---|---|---|
| 3 | **≥30 m (unbounded)** | ≥30 m |
| 4 | 20 m | 20 m |
| 5 | 15 m | 20 m |
| 7 | 10 m | 20 m |
| 9 | **5 m** | **20 m** |

- **At N=3 the achievable residual is exactly 0.00 m** for displacements of 5–20 m. The
  under-determination prediction is confirmed, and [[Does Trust Actually Help]] now has a
  structural cause rather than an apology.
- **Attacker capability erodes monotonically with N.** The predicted *sharp threshold at
  N≈5 did not appear* — the decline is smooth, with a qualitative break only at N=3.
  The threshold framing in this ADR's rationale is corrected to **monotone erosion**.
- **Coplanar geometry cancels the benefit of swarm size.** N=9 coplanar is no better
  than N=4 — a 4× worse posture than the 3-D case at the same size. The mirror
  solution's signature appears as a residual dip near twice the attacker's height above
  the peer plane. **Altitude diversity is a hard security requirement**, and this is
  the more operationally actionable finding of the two.
- The first attempt at this sub-experiment made *every* node coplanar, including the
  attacker — in which case the attacker's mirror image is itself and no lie is offered.
  That design tested nothing; it was corrected to peers-in-plane, attacker-off-plane.

**Gate verdict: PASS.** Caveats to tighten in phases C–F: 40 trials/point, medians, one
attacker, fully connected mesh, static snapshot with no temporal accumulation.

### The cost of dropping Stage 0.2 — recorded honestly

The paper loses a measured "the naive mechanism doesn't help" from working code. The
existing `results/full_experiment.csv` **cannot** substitute: it predates
[[ADR-005 - Fixed Observation Normalization]] and its numbers were disowned as corrupt.

**The motivation therefore rests entirely on the geometric argument and phase 0's
Test 3.** If Test 3 shows no threshold, this ADR's rationale is undermined and the
decision must be revisited — including the option of reinstating Stage 0.2 at greater
cost than it would have taken now. That risk was raised, understood, and accepted.

The upside if Test 3 holds: a general structural result ("EMA-style trust at N=3 is
impossible, here is the proof and its numerical demonstration") is **stronger** than one
empirical run, because it is not specific to this codebase.

## What this explicitly does NOT do

- **It does not discard ADR-005 → 010.** Those fixes were correct and the debugging was
  sound. They are what *proved* the RL is not the path — a negative result about the
  approach, not about the work.
- **It does not delete the MARL stack.** `mappo_trainer.py`, `networks.py` and
  `rollout_buffer.py` are frozen for the phase-E ablation.
- **It does not soften the threat model.** Spoofing is still injected at the measurement
  level with no assumption about how the attacker achieved it.

## Related

- [[ADR-002 - Scale to 9 Drones]] · [[ADR-010 - Reward Shaping Is the Blocker]]
- [[Does Trust Actually Help]] · [[Plausibility-Based Trust]] · [[Robust Statistics and Consensus]]
- [[GPS Spoofing and GNSS Denial]] · [[Roadmap]] · [[Decision Log]]
