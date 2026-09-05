# SentryNet v3 — Design Document

**Status:** Approved plan, rewritten 2026-09-05 after a literature survey and a change of goal.
**Supersedes:** `docs/DESIGN.md` (v2), and the 2026-09-04 revision of this file.
**Rationale:** ADR-011 (the pivot), ADR-012 (utility-first reframe).

---

## Goal

> **A position-integrity monitor for drone swarms that a real operator can run** —
> catching any drone whose reported position cannot be trusted, whether the cause is an
> attack, an RTK dropout, multipath, or a sensor fault.

### Explicit non-goal: novelty

This is stated up front because it changes every downstream decision. **We are not
trying to invent a new method.** The core principle — cross-check a drone's claimed
GNSS position against inter-drone ranges the radio physically measures — is established
prior art (see the survey below). We are engineering it into something usable, honest
about its limits, and validated against reality.

That inverts what was previously the project's largest risk. Prior art is no longer a
threat to a novelty claim; it is a foundation to build on and cite.

### What we optimise for instead

1. **Realism** — measured noise models, real flight dynamics, real failure modes.
2. **Utility** — it solves a problem someone actually has, today.
3. **Applicability** — it can be run where that problem exists, by the people who have it.

---

## The problem, and whose problem it is

GNSS spoofing is real and large-scale: it now affects thousands of commercial flights
daily, and cannot be fixed by cryptography, because a spoofed receiver reports — honestly
— a position it genuinely believes.

But the crucial insight for utility is this:

> **A drone reporting a wrong position because of multipath, an RTK dropout, an
> ionospheric event or a sensor fault produces exactly the same residual signature as a
> spoofed one.** The detector cannot tell them apart, and for a real operator that is a
> feature, not a limitation.

Natural integrity faults are orders of magnitude more common than attacks. Designing for
both at once turns a defence-sector tool into something ordinary swarm operators need.

### Target user: commercial drone swarms, starting with light shows

Drone light shows are the accessible instance of this problem, and the failure record is
public: Hong Kong 2018 · Taichung 2020 · SeaTac 2024 · Orlando December 2024 (drones
collided, one left the show airspace, a bystander was injured) · Ho Chi Minh City 2025 ·
**Sydney Vivid Festival, May 2026 — roughly 89 drones dropped out of formation into the
harbour.**

Two things make this the right target:

- **The named failure mode is a common-mode fault.** When an RTK ground reference station
  drops mid-show, every drone loses its correction *simultaneously*. That is precisely the
  hardest case for a relative-only detector (see the honest limit below) — and precisely
  the case a well-placed absolute anchor solves.
- **The industry already describes our product.** Operators say they want multiple
  independent sensor sets feeding simultaneous position estimators so that a bad reading
  from one sensor is never trusted blindly. That is an integrity monitor.

Secondary targets, same machinery: survey and agricultural swarms (RTK dropout, multipath
near structures), urban delivery (canyon multipath). The defence and counter-UAS case is
real but institutionally inaccessible to a solo builder; success in the commercial case is
what makes it reachable later.

---

## Prior-art survey (2026-09-05)

Conducted to find what to **reuse**, not to find a gap.

### The core idea is established

| Work | What it establishes |
|---|---|
| **Čapkun & Hubaux**, verifiable multilateration / secure positioning in wireless networks (2005–06) | The foundational secure-positioning result: distance-bounding from multiple verifiers lets you verify a claimed position. The ancestor of everything here. |
| **arXiv:2301.12766** — *GPS-Spoofing Attack Detection Mechanism for UAV Swarms* | Our exact primitive, published: compare GPS-derived inter-drone distances against IR-UWB ranging, flag when the discrepancy exceeds a threshold. |
| **arXiv:2312.03787** — *Detection and Mitigation of Position Spoofing Attacks on Cooperative UAV Swarm Formations* | Detection *and* mitigation for cooperative swarm formations. |
| **Cooperative Integrity Monitoring (CIM)** literature — multi-sensor cooperative positioning, VANET "local integrity" | The concept already has a name and a body of work in the GNSS/vehicular community, with residual decomposition into common and specific parts and greedy exclusion. Outperforms standalone RAIM. |

**Consequence:** adopt the established terminology (*cooperative integrity monitoring*,
*fault detection and exclusion*), cite this lineage, and stop describing the primitive as
new.

### The honest limit — and it is recent, formal, and load-bearing

**arXiv:2608.06885** — *Rigid-Covert GNSS Spoofing of UAV Swarms: A Structural Blind
Spot, Its Detection Limit, and Absolute-Anchor Defenses* (August 2026) is the single most
important paper for this project.

Its Proposition 1: **any detector built on relative quantities alone is invariant under a
common translation.** If every drone is shifted by the same vector, all inter-drone
distances and all residuals are unchanged, and the attack is undetectable. They derive a
drift-dependent detection floor and validate it (measured slope 2.66 against a predicted
2.67, R² = 0.99).

**Our phase-0 Test 1 independently rediscovered this.** We measured `rank(J) = 3N − 6` —
exactly the six gauge freedoms, three translation and three rotation. That is a strong
validation of our implementation, and simultaneously tells us the ceiling of a pure
relative approach.

Their proposed defence is what we adopt: **anchor-rooted recovery** — reconstruct swarm
geometry from ranges via classical MDS, align it to trusted absolute anchors with RANSAC
for Byzantine robustness, propagate corrected positions. Their code, configs and swarm
harness are released.

### What nobody has done — and it is exactly what "useful" requires

Every result in this space is **simulation-only**. The rigid-covert authors state
plainly that no RF transmission hardware was used in any experiment; their vision and
ArduPilot SITL work is Gazebo-rendered. Their stated limitations include no physical
swarm, ≥3 non-collinear honest anchors required, no tolerance above 50% compromised
anchors, and RANSAC that does not sustain 10 Hz at N ≥ 64.

The remaining gaps, all of which serve utility rather than novelty:

1. **No grounding in measured hardware.** Ranging noise and NLoS bias are assumed, never
   measured, for swarm integrity work.
2. **No validation against real flight data.** Nothing runs these detectors over logs of
   flights that actually happened.
3. **Nobody targets an operator's real failure mode.** The literature models attacks;
   operators mostly lose drones to faults.
4. **Nothing is deployable.** No clean API, no flight-stack integration path.

**That is where SentryNet goes.**

### The insight that makes the blind spot tractable

A pure relative detector is blind to a rigid common-mode shift. In the abstract that is
fatal. **In our target deployment it is not**, because a drone light show already has
surveyed absolute references: the RTK base station, the surveyed launch grid, ground
cameras with known positions.

So the deployment context supplies exactly the absolute anchor the theory says is
required. The limitation that dooms this approach as pure research is **solvable in
practice** — which is an argument for building it for real users rather than for a
benchmark.

---

## What SentryNet is

A monitor that consumes what a swarm already produces and answers one question per drone,
every epoch: **can I trust this position?**

**Inputs** — each drone's reported position; inter-drone range measurements; optionally
one or more surveyed absolute anchors.

**Method** — the established one, engineered properly:

1. Per-link residual: claim-implied distance minus measured range.
2. Normalise by the full residual covariance (**not** treating residuals as independent —
   they share per-drone GNSS error; our Test 1 controls show that mistake gives the right
   mean and the wrong distribution).
3. Per-drone χ² statistic with a calibrated threshold — `alpha` **is** the false-alarm
   rate, replacing v1's undefendable `max(0, 1 − error/5.0)`.
4. Isolate the worst offender; exclude it from fusion.
5. **Anchor-rooted recovery** for the common-mode case, per arXiv:2608.06885.

**Outputs** — a per-drone trust verdict, an excluded set, a fused position estimate, and
an operating-envelope statement: under these conditions, this is what the monitor can and
cannot see.

### What we deliberately do not build

**No learned detector.** Under a Gaussian model the χ² test is already statistically
near-optimal, so a neural network has nothing to learn; it would only help where the model
is misspecified. More decisively, for a safety-conscious operator the χ² test is
*explainable*, needs no training data, and has no model to maintain. Learning returns only
if a **measured** real-world failure mode demands it. This removes a phase and makes the
product more deployable, not less.

---

## Phases

| Phase | Work | Status |
|---|---|---|
| **0** | Premise test — validate the statistic and the geometry | ✅ Passed |
| **A** | Scripted controller; detector decoupled from RL; capture rate retired | Next |
| **B** | **Realism**: `gym-pybullet-drones` dynamics, N=9, mesh, LoS occlusion, DS-TWR ranging with NLoS bias, **temporally correlated GNSS error** | ~2 wk |
| **C** | Threat + fault model: spoofing scenarios *and* RTK dropout, multipath, sensor fault. Includes the rigid common-mode case | ~1 wk |
| **D** | Established baselines: χ² FDE, trimmed mean, W-MSR, anchor-rooted MDS+RANSAC, the v1 EMA module | ~1 wk |
| **E** | **Anchor integration** — surveyed ground references, the light-show deployment case *(replaces the learned detector)* | ~1 wk |
| **F** | **Operating envelope** — where the monitor works and where it is blind, as connectivity, fault fraction and anchor availability vary | ~1.5 wk |
| **G** | **Hardware calibration** — bench UWB modules, measured noise and NLoS bias feeding back into B *(promoted from deferred)* | ~1 wk |
| **H** | **Real-data validation**, packaging, live demo, write-up | ~3 wk |

Two changes of emphasis from the previous plan: **the learned detector is gone**, and
**hardware calibration is promoted** because realism is now a stated goal rather than a
nice-to-have.

### Phase H — validation against reality, ranked by evidence per hour

1. **Real flight logs.** Multi-drone logs from a light-show operator, a university lab, or
   a public dataset, run through the monitor. If it flags integrity events that actually
   occurred, that is proof against reality rather than simulation.
2. **ADS-B cross-check.** Aircraft broadcast GNSS-derived positions; ground and space-based
   networks independently determine position. Same residual structure, real data, real
   documented spoofing. Prior work exists (Stanford GPS Lab, ION GNSS+ 2024; Aireon), so
   this is a validation path rather than a novel claim. Investigate feasibility early.
3. **Bench-measured ranging**, feeding phase G back into the simulator.
4. **A live, parameterised demo** — see below.

---

## Metrics

Capture rate is retired. The monitor is judged as an integrity monitor:

- **P(missed detection)** and **P(false alert)**, with a stated operating point
- **Time-to-detect**, especially for slow drift
- **False-accusation rate** on honest drones
- **Correct-isolation rate**, reported separately from detection — our Test 2 showed
  isolation lags detection badly (80% vs 99.6% at 20 m)
- **Fused position error** under fault or attack — what an operator actually feels
- **Protection level** — a bound on swarm position error given observed residuals

---

## The demo

A committed deliverable, not an afterthought — it is the only artifact that reaches people
who will never read a ROC curve.

**Two modes from one renderer.** `--live` runs and renders interactively, so a parameter
can be changed in front of an audience (drop to 3 drones, watch the monitor go blind —
the operating envelope demonstrated in ten seconds). `--record` writes frames from a
saved state log for the case where a live run is not possible.

**3-D is not the same decision as PyBullet.** The live view needs positions drawn in
perspective, which the kinematic path plus a renderer provides without a physics server to
lose. A 3-D main view carries the swarm and the altitude story; a flat inset — top-down
plus per-drone χ² bars — carries the detection mechanism, which reads poorly in
perspective.

**What the frames must show:** each drone's true position with a ghost marker for its
claim and a tether between them; the measured range ring the claim falls outside; the χ²
bars crossing threshold on the right drone; and the fused estimate staying locked
throughout.

---

## Scope

**In:** realistic dynamics and error models · fault cases alongside attacks · absolute
anchors · measured ranging · real-data validation · a deployable API · honest operating
limits.

**Out:** novelty claims · GNSS signal processing and SDR (we work at the measurement
level) · a learned detector until a measured failure mode demands one · Byzantine
consensus theory beyond what W-MSR needs · dashboard polish before phase H.

## Positioning, stated honestly

SentryNet does not introduce a new detection principle. It takes an established one —
cooperative integrity monitoring via range/claim residuals — implements it with a
correctly derived covariance and a calibrated threshold, extends it with anchor-rooted
recovery for the common-mode blind spot that the literature has formally shown to be
otherwise undetectable, grounds it in measured hardware and real flight data rather than
assumed noise, and packages it so an operator can run it.

That is an engineering contribution. It is the useful one.

---

## Related

- ADR-011 (pivot to integrity monitoring) · ADR-012 (utility-first reframe)
- `SentryNet-Vault/planning/Roadmap`, `findings/Does Trust Actually Help`
- Phase-0 results: `results/premise/`
