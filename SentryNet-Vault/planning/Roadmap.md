# Roadmap

**Canonical detail:** `docs/DESIGN-v3.md` (supersedes `docs/DESIGN.md`). This note is the
vault-side summary. Guiding principle: **depth over breadth — every phase ships something
of value.**

> **Reframed 2026-09-04.** See [[ADR-011 - Pivot to Cooperative Integrity Monitoring]] for
> why. The short version: the novel contribution was sitting *downstream* of a learned
> pursuit policy that cannot converge on the available hardware. v3 inverts that.

## Thesis

**Cooperative integrity monitoring for GNSS-denied swarms** — detect and isolate a
spoofed node from physics that cannot be forged, then package it as a reusable benchmark.

A node's GNSS position can be spoofed. The time-of-flight of the radio link between two
nodes cannot. Everything here is built on that asymmetry. It is the swarm-level
generalisation of aviation **RAIM**: redundant *peer* measurements catching a faulty
*node*, the way redundant satellite measurements catch a faulty satellite.

## The geometric spine

A node claiming a 3-D position has 3 unknowns and one range constraint per peer.
At **N=3** there are only 2 constraints — **under-determined, so a perfectly consistent
lie always exists**. This is the structural reason [[Does Trust Actually Help]] found
nothing: the problem was unsolvable, not badly implemented. It also makes
[[ADR-002 - Scale to 9 Drones]] a necessity rather than a hunch.

Because occlusion severs links and lowers the effective peer count, **detectability is a
function of effective connectivity** — the frontier this project exists to map.

## Phase 0 — The premise test ← **WE ARE HERE**

`experiments/premise_test.py`. No RL, no reward function, no `border_env`.

- **Test 1 — model validity.** ✅ The per-node attribution statistic is exactly χ²
  (mean 8.054 vs 8, KS p = 0.691). Also found: the network-level statistic is
  gauge-degenerate, `rank(J) = 3N − 6`.
- **Test 2 — detectability.** ✅ Needs ~4σ of displacement per epoch; isolation lags
  detection badly. Motivates sequential detection (CUSUM/SPRT).
- **Test 3 — the crux.** ✅ **GATE PASSED.** At N=3 a consistent lie costs *exactly zero*
  residual, confirming the under-determination argument. Attacker reach then erodes
  monotonically with N (≥30 → 20 → 15 → 10 → 5 m) — smoothly, not at the sharp N≈5
  threshold predicted. And **coplanar peers cancel that benefit entirely**: N=9 coplanar
  is no better than N=4, so **altitude diversity is a hard security requirement**.

## Phases A–H

| Phase | Work |
|---|---|
| **A** | Scripted pursuer; detector decoupled from RL; capture rate retired as primary metric |
| **B** | N=9, mesh topology, LoS raycast, DS-TWR ranging + NLoS bias |
| **C** | [[Threat Scenarios]] S0–S4 **+ the adaptive range-consistent attacker** |
| **D** | Classical baselines: χ² gating, trimmed mean, W-MSR, the v1 EMA trust module |
| **E** | Learned detector — supervised sequence model over residual features |
| **F** | **The frontier**: detectability vs. connectivity × attacker fraction × sophistication |
| **G** | *(deferred)* UWB bench calibration against real hardware |
| **H** | `pip install sentrinet`, YAML scenarios, CI, repointed demo video, paper |

≈10–11 weeks part-time. Gated at phase 0 and again after D.

## Metrics

Capture rate is **retired** as a headline number — downstream, high-variance,
policy-dependent. Replaced by the integrity standard: P(missed detection), P(false
alert), time-to-detect, false-accusation rate, correct-isolation rate, fused position
error, and a **cooperative protection level**. See [[Metrics]].

## Explicitly out of scope

GNSS signal processing / SDR (we work at the measurement level) · ROS 2 and Crazyswarm
until hardware is committed · dashboard polish until phase H · geodetic frames · MARL as
a load-bearing component · curriculum · the trust hyperparameter grid.

## What v2 leaves behind

ADR-005 → 010 were correct work and are **not** discarded — they are what proved the RL
is not the path. `mappo_trainer.py`, `networks.py` and `rollout_buffer.py` are frozen for
the phase-E ablation; `trust_module.py` becomes a baseline to beat.

## Related

- [[ADR-011 - Pivot to Cooperative Integrity Monitoring]] · [[Decision Log]]
- [[Does Trust Actually Help]] · [[Threat Scenarios]] · [[Metrics]] · [[Learning Roadmap]]
