# Roadmap

**Canonical detail:** `docs/DESIGN-v3.md` (supersedes `docs/DESIGN.md`). This note is the
vault-side summary. Guiding principle: **depth over breadth — every phase ships something
of value.**

> **Reframed twice.** [[ADR-011 - Pivot to Cooperative Integrity Monitoring]] (2026-09-04)
> inverted the dependency graph: the integrity monitor became the product.
> [[ADR-012 - Utility First, Novelty Is Not a Goal]] (2026-09-05) then made **novelty an
> explicit non-goal** — optimise for realism, utility and applicability instead, adopt
> prior art rather than route around it, and design for *faults* as well as attacks.

## Thesis

**A position-integrity monitor for drone swarms that a real operator can run** — catch
any drone whose reported position cannot be trusted, whether the cause is spoofing, a lost
RTK correction, multipath, or a sensor fault. A fault and an attack produce the *same*
residual, and faults are far more common — which is what widens the user base from defence
to any swarm operator. Target user: commercial swarms, starting with light shows.

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

## Phase 0 — The premise test ✅ complete

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

## Phases A–H ← **WE ARE HERE (A)**

| Phase | Work |
|---|---|
| **A** | **Formation-flight scenarios** (show-like waypoints, velocity/acceleration limited); detector decoupled from RL; pursuit task and capture rate retired outright |
| **B** | **Realism**: `gym-pybullet-drones` dynamics, N=9, mesh, LoS occlusion, DS-TWR ranging with NLoS bias, temporally correlated GNSS error |
| **C** | [[Threat Scenarios]] S0–S4 **plus faults**: RTK dropout, multipath, sensor failure. Includes the rigid common-mode case |
| **D** | Established baselines: χ² FDE, trimmed mean, W-MSR, anchor-rooted MDS+RANSAC, the v1 EMA module |
| **E** | **Anchor integration** — surveyed ground references, the light-show deployment case *(replaces the cut learned detector)* |
| **F** | **Operating envelope** — where the monitor works and where it is blind |
| **G** | **Hardware calibration** — bench UWB modules; measured noise replaces assumed *(promoted from deferred)* |
| **H** | **Real-data validation** (flight logs, ADS-B cross-check), packaging, live demo, write-up |

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
is not the path. All of it now lives frozen under `legacy/` (moved with `git mv`, history
intact): the MAPPO stack, the dashboard, the v1 analysis scripts and the superseded root
docs. `legacy/trust_module.py` stays importable as the baseline to beat.

## Related

- [[ADR-011 - Pivot to Cooperative Integrity Monitoring]] · [[ADR-012 - Utility First, Novelty Is Not a Goal]] · [[Decision Log]]
- [[Does Trust Actually Help]] · [[Threat Scenarios]] · [[Metrics]] · [[Learning Roadmap]]
