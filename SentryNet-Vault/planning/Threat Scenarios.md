# Threat Scenarios (Stage 1)

Named, reproducible attacks that replace the abstract Gaussian spoof. All are [[GPS Spoofing and GNSS Denial]] variants: a compromised drone **honestly reports a false self-belief** (authentication can't catch this — [[ADR-001 - GPS-Spoofing Pivot]]).

| ID | Scenario | What it tests |
|---|---|---|
| **S0** | No attack | Baseline + **false-accusation floor** (does trust wrongly distrust honest nodes?) |
| **S1** | Constant-offset spoof, 1 node | Easy case — a fixed position error |
| **S2** | **Slow-drift** spoof (ramp), 1 node | Hard case — evades instantaneous checks; needs temporal consistency |
| **S3** | **Coordinated** spoof, *k* nodes (k=1…⌈N/2⌉) | Consensus **breaking point** — how many liars before robust fusion fails |
| **S4** | Intermittent spoof | Tests detection under on/off attacks |

Packet loss is retained throughout as ambient degradation.

## Why these specific scenarios
- **S2 (slow drift)** is the scenario a naive residual check misses — it's the argument for the *temporal-consistency* signal in [[Plausibility-Based Trust]].
- **S3 (coordinated)** is why swarm size matters ([[Robust Statistics and Consensus]]) — it produces a clean "degradation vs number of adversaries" curve, a real publishable result.
- **S0** guards against the failure mode where "trust" just makes everyone paranoid.

## Related
- [[GPS Spoofing and GNSS Denial]] · [[Plausibility-Based Trust]] · [[Robust Statistics and Consensus]] · [[Metrics]]
