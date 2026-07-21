# Does Trust Actually Help?

**The honest answer from the project's own data: no — not demonstrably. What helps is *training under packet loss*, not the trust mechanism.** This finding drove the whole pivot ([[ADR-001 - GPS-Spoofing Pivot]]).

Source: `results/full_experiment.csv` — 3 systems × 9 drop rates × 3 seeds.

## The systems
- **A** — trained clean, no trust.
- **B** — trained with drops (adversarial training), no trust.
- **C** — trained with drops + **trust**.

## Result 1: robustness comes from adversarial training, not trust
Capture retained from clean (drop 0.0) to severe (drop 0.8):

| System | drop 0.0 | drop 0.8 | Retained |
|---|---|---|---|
| A | 0.982 | 0.567 | **58%** |
| B | 0.905 | 0.658 | **73%** |
| C | 0.987 | 0.738 | **75%** |

The big jump is **A→B (+15 pts)** — from adversarial *training*. **B→C is only +2 pts** — all the trust mechanism buys. It's inside the noise.

## Result 2: C vs B (the actual trust test) is a wash
Head-to-head `C − B` capture rate by drop rate — no consistent sign:

| drop | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|---|---|---|---|---|
| C−B | +.08 | +.04 | +.02 | −.01 | −.04 | **−.08** | **−.08** | −.00 | +.08 |

At drops 0.5–0.6 (where robustness matters), trust is *worse*. With 3 seeds and std of 0.2–0.4, none of this is significant.

## Result 3: the trust score is a drop-meter, not an adversary detector
System C mean trust falls monotonically with drop rate (0.58 → 0.15) and sits at only **0.58 even at zero drop**. If trust worked it would hold honest senders near 1.0 and crush the compromised drone specifically. Instead it decays *everyone* roughly equally. It's measuring packet loss, not detecting the adversary.

## Why it fails — structurally
1. **Reference only available when not needed** — trust is scored vs the receiver's own estimate, which exists only when it already sees the target. Blind → can't detect spoofing. ([[Trust Module and Aggregator]])
2. **Too few agents** — 2 senders per receiver → consensus is a median of 2, meaningless. ([[Robust Statistics and Consensus]])
3. **Confounded comparison** — C was trained with GRU + sustained capture; A/B with MLP + team. The gap is uninterpretable. ([[Known Bugs and Confounds]])
4. **Ceiling + variance** — everything ~99% below drop 0.4; seeds swing wildly (A@0.8: 0.175 to 0.80).
5. **The normalization bug corrupts all of it anyway.** ([[ADR-005 - Fixed Observation Normalization]])

## What this means
To make trust *demonstrably* help we need: a clean controlled comparison ([[Controlled Experiment]]), a trust signal that discriminates the adversary and works when blind ([[Plausibility-Based Trust]]), and enough agents for consensus ([[ADR-002 - Scale to 9 Drones]]).

## Related
- [[Controlled Experiment]] · [[Plausibility-Based Trust]] · [[Known Bugs and Confounds]] · [[Metrics]]
