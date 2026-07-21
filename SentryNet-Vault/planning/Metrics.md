# Metrics

What we measure, and the baselines any new method must beat. Honest, quantitative evaluation is the whole point — see [[Does Trust Actually Help]] for how *not* to do it.

## Primary metrics
- **Capture rate** — did the swarm catch the intruder? (headline task metric)
- **Intruder tracking error** — mean error of the fused estimate vs truth (the *direct* measure of whether trust improves the shared world-model).
- **Time-to-detect** a spoofed node — how many steps until trust on the adversary drops below a threshold.
- **False-accusation rate** — how often an *honest* node is wrongly distrusted (the cost of being suspicious).
- **Degradation slope** — capture/tracking vs number of spoofed nodes *k* (the [[Robust Statistics and Consensus|breaking-point]] curve).

## Baselines (the bar to clear)
| Baseline | Why it's included |
|---|---|
| Uniform averaging | The "no trust" floor |
| EMA trust | The current design ([[Trust Module and Aggregator]]) |
| **Median / trimmed-mean fusion** | The classical robust answer — the *real* bar |
| **Oracle trust** | Perfect attacker knowledge → the upper bound |

If [[Plausibility-Based Trust]] can't beat trimmed-mean, that's a finding, not a failure.

## Statistical hygiene (non-negotiable)
- ≥8 seeds; report **mean ± 95% CI**, not single runs.
- Significance tests (Welch's t) on head-to-head deltas.
- Avoid **ceiling effects**: if everything captures ~100%, the design is too easy and hides differences (a current problem — see [[Reward Design]]).

## Related
- [[Controlled Experiment]] · [[Threat Scenarios]] · [[Does Trust Actually Help]] · [[Robust Statistics and Consensus]]
