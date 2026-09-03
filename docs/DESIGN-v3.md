# SentryNet v3 — Design Document

**Status:** Approved plan. Phase 0 in progress.
**Supersedes:** `docs/DESIGN.md` (v2). That document remains the record of Stages 0–3
as planned; this one replaces its framing and its stage list from Stage 0.2 onward.
**Last updated:** 2026-09-04

---

## Thesis

> **Cooperative integrity monitoring for GNSS-denied drone swarms** — detecting and
> isolating a spoofed node from physical constraints that cannot be forged, validated
> as a controlled study, packaged as a reusable benchmark.

A node's GNSS-derived self-position can be spoofed. The time-of-flight of the radio
link between two nodes cannot. Every detector in this project is built on that
asymmetry: a node's *claimed* position must stay consistent with the ranges its peers
physically measure to it.

This is the swarm-level generalisation of **RAIM** (Receiver Autonomous Integrity
Monitoring). A single receiver uses redundant satellite measurements to detect and
exclude a faulty one, and reports a protection level bounding its position error. Here
the swarm uses redundant *peer* measurements — GNSS claims cross-checked against UWB
ranges — to detect and exclude a faulty *node*.

---

## Why v3 exists

### The inverted dependency

In v2, the novel and defensible contribution — catching a spoofed node from physics —
sat **downstream** of the least differentiated component: a learned multi-agent pursuit
policy. Six ADRs of correct, careful debugging (ADR-005 → ADR-010) took System A from
0% to ~44% capture against a scripted pursuer that scores 95–100%, still unconverged at
1M steps, with entropy *rising*.

That race is not winnable on the available hardware. At ~2 h per 1M steps on 12 CPU
cores, the 10–50M steps that serious MARL results require is roughly a month of
continuous compute for a single configuration. And the prize for winning is a drone
pursuit controller, a problem classical methods already handle.

The detection problem — where no clean classical answer exists in the mobile,
sparse-connectivity, adaptive-adversary regime — trains **supervised, in minutes, on
CPU**. v3 inverts the dependency: the integrity monitor becomes the product, the
pursuit controller becomes scripted, and RL is demoted to an optional ablation on top
of a working system.

### The geometric argument

A node claiming a 3-D position has **3 unknowns** and must satisfy **one range
constraint per peer**:

| Peers | Constraints | Consequence |
|---|---|---|
| 2 (N=3) | 2 | **Under-determined** — a continuum of consistent lies |
| 3 (N=4) | 3 | Generically two solutions: the truth and its mirror |
| 4+ (N≥5) | 4+ | **Over-determined** — generically only the truth |

**...unless the peers are coplanar**, in which case the mirror solution survives at any
swarm size. Drones holding a common altitude are nearly coplanar, so **altitude
diversity is a security property**, not an aesthetic one.

If this holds, the v2 negative result — "EMA trust degenerates into a global drop-rate
meter" ([[Does Trust Actually Help]]) — was not a bad implementation. **At N=3 the
detection problem is structurally unsolvable**, because a consistent lie always exists.
No trust mechanism, learned or heuristic, could have worked. This turns ADR-002 (scale
to 9 drones) from a heuristic into a necessity, and predicts a critical threshold near
**N≈5**, with N=9 providing margin for colluding attackers and occlusion-severed links.

**Occlusion removes links, which lowers effective peer count, which pushes a swarm back
below the threshold.** Detectability is therefore a function of *effective
connectivity* — and that is the headline result this project exists to produce.

---

## Phase 0 — The premise test (the gate)

Standalone: no RL, no reward function, no `border_env`. Driver:
`experiments/premise_test.py`. Modules: `sentrinet/{world,sensing,integrity,attacks}`.

| Test | Question | Status |
|---|---|---|
| **1** | With no attacker, is the statistic actually χ² at the claimed dof? | ✅ **PASS** (per-node) |
| **2** | Detection and false-alarm rate vs. a naive constant-offset spoofer | ✅ Measured |
| **3** | Against an *adaptive* spoofer, does the N≈5 threshold appear? | ⏳ Running |

### Test 1 results (N=9, 2000 trials)

| Model | dof | mean | expected | KS p |
|---|---|---|---|---|
| **per-node (full covariance)** | 8 | **8.054** | 8 | **0.691** ✅ |
| global, rank-truncated | 21 | 20.65 | 21 | 0.001 |
| global, naive dof *(control)* | 36 | 222.9 | 36 | 0 ❌ |
| independence *(control)* | 36 | 36.18 | 36 | 5e-31 ❌ |

Two findings, both load-bearing:

1. **The per-node attribution statistic is exactly χ².** This is the statistic the
   method depends on, and it is well conditioned regardless of geometry.
2. **The network-level statistic is gauge-degenerate.** Translating or rotating every
   claim together leaves all pairwise residuals unchanged, so the Jacobian has a
   six-dimensional null space and the residual lives in a subspace of dimension
   `rank(J) = 3N − 6` — confirmed empirically at exactly 21 for N=9. Using one degree
   of freedom per link inflates the statistic by 6×. After rank truncation the mean is
   correct; the remaining tail error is first-order linearisation error, which scales
   as `O(σ_g² / link_length)` and was confirmed by sweeping minimum separation
   (mean 679 → 142 → 62 → 42 as separation goes 4 → 10 → 20 → 40 m).

The independence control fails exactly as designed: residuals sharing a node share that
node's GNSS error, so treating them as independent gives the right *mean* but the wrong
*distribution*.

### Test 2 results (N=9, 500 trials/point, α=10⁻³)

Honest residual noise floor `sqrt(2σ_g² + σ_r²)` = **2.12 m**.

| offset (m) | detection | correct isolation |
|---|---|---|
| 0 | *(false alarm 1.0%)* | — |
| 3 | 0.046 | 0.008 |
| 5 | 0.144 | 0.066 |
| 8 | 0.528 | 0.324 |
| 12 | 0.930 | 0.648 |
| 20 | 0.996 | 0.800 |

- The false-alarm rate of 1.0% matches the predicted `n_nodes × α` — the model is
  self-consistent.
- **Single-epoch detection needs roughly 4σ of displacement.** Anything subtler is
  invisible to a per-epoch test, which is the direct motivation for sequential
  detection (CUSUM/SPRT) in phase E.
- **Isolation lags detection badly** (80% vs. 99.6% at 20 m). A spoofed node biases its
  honest peers' statistics too. Detection-vs-isolation is a real sub-problem, not a
  detail.

**Gate:** Test 1 passing for the per-node statistic clears the model. Test 3 decides
whether the geometric thesis holds; if no threshold appears, the plan is re-opened
before further work.

---

## Phases

| Phase | Work | Deliverable | Gate |
|---|---|---|---|
| **−1** | Push branches, freeze v2, scaffold `sentrinet/` | Work off one disk | — |
| **0** | Premise test (above) | Threshold figure | **HARD** |
| **A** | Scripted pursuer; detector decoupled from RL; retire capture rate as primary metric | Deterministic testbed | — |
| **B** | N=9, mesh topology, LoS raycast, DS-TWR + NLoS bias | `world/`, `sensing/` | — |
| **C** | Threat scenarios S0–S4 + the adaptive range-consistent attacker | `attacks/`, YAML | — |
| **D** | Classical baselines: χ² gating, trimmed mean, W-MSR, v1 EMA trust | `fusion/`, `baselines/` | **SOFT** |
| **E** | Learned detector: supervised sequence model over residual features | `integrity/learned.py` | — |
| **F** | Frontier sweeps: detectability vs. connectivity × attacker fraction × sophistication | **Headline figure** | — |
| **G** | *(deferred)* UWB bench calibration against real hardware | measured noise model | — |
| **H** | `pip install sentrinet`, scenarios, CI, repointed demo, paper | End products | — |

≈10–11 weeks part-time. Phase G is deferred by decision (2026-09-04); nothing in phases
0–F depends on it.

---

## Metrics

Capture rate is **retired as a primary metric** — it is downstream, high-variance and
policy-dependent. The replacements are the integrity-monitoring standard:

- **P(missed detection)** and **P(false alert)** — ROC, with a stated operating point
- **Time-to-detect** — especially for slow-drift attacks (S2)
- **False-accusation rate** on honest nodes
- **Correct-isolation rate**, reported separately from detection
- **Fused position error** under attack (the thing an operator actually cares about)
- **Cooperative protection level** — a bound on swarm position error given observed
  residuals and an assumed attacker fraction. Novel, and immediately legible to the
  navigation community.

---

## Decisions taken

- **Stage 0.2 (the v2 A/B/C "does trust help?" experiment) is dropped**, with its cost
  recorded. See ADR-011. Phase 0's Test 3 now carries the paper's motivation.
- **Hardware is deferred**, not cancelled. Sim-only until after the phase-0 gate.
- **v2 code is frozen, not deleted.** Root-level scripts stay where they are;
  `trust_module.py` is ported to `baselines/` as the thing to beat.

## Out of scope

GNSS signal processing and SDR (we work at the measurement level) · ROS 2 / Crazyswarm
until hardware is committed · dashboard polish until phase H · geodetic frames (local
ENU is sufficient) · Byzantine-consensus theory beyond r-robustness · MARL as a
load-bearing component.

---

## Related

- `docs/DESIGN.md` — the v2 plan this supersedes
- Vault: `decisions/ADR-011 - Pivot to Cooperative Integrity Monitoring`,
  `planning/Roadmap`, `findings/Does Trust Actually Help`
