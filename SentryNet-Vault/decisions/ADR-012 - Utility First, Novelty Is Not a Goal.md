# ADR-012 — Utility First; Novelty Is Not a Goal

**Status:** ✅ Accepted · 2026-09-05 · **Refines [[ADR-011 - Pivot to Cooperative Integrity Monitoring]]**

## Context

ADR-011 reframed the project as cooperative integrity monitoring but kept a research
framing: a *detectability frontier* as the headline contribution, aimed at a navigation
venue, with "someone has already published this" recorded as the project's largest risk.

The user then set a different target explicitly: **the simulation must be realistic, the
project must solve a real problem effectively, it must be useful, and the solution must be
applied where that problem exists. It does not have to be novel.**

That is a different optimisation target, and it changes what counts as success.

## Decision

**Novelty is an explicit non-goal.** Optimise instead for realism, utility, and
applicability.

1. **Adopt prior art rather than route around it.** Use the established terminology
   (*cooperative integrity monitoring*, *fault detection and exclusion*), cite the lineage,
   and stop describing the primitive as new.
2. **Design for faults as well as attacks.** A drone reporting a wrong position from
   multipath, an RTK dropout or a sensor fault produces the same residual as a spoofed one.
   The detector cannot separate them, and for an operator that is a feature.
3. **Target commercial drone swarms, starting with light shows.**
4. **Cut the learned detector.** Under a Gaussian model the χ² test is already
   near-optimal, so a network has nothing to learn; and an explainable test with no
   training data and no model to maintain is what a safety-conscious operator will accept.
   Phase E becomes **anchor integration** instead.
5. **Promote hardware calibration** from deferred, because realism is now a stated goal.
6. **Validate against real data** — flight logs first, then an ADS-B cross-check.
7. **Use realistic flight dynamics** (`gym-pybullet-drones`) as the main trajectory source,
   keeping the fast kinematic path for large sweeps and verifying the two agree.

## The survey that informed this (2026-09-05)

Conducted to find what to **reuse**, not to find a gap.

| Work | Establishes | We take |
|---|---|---|
| Čapkun & Hubaux, verifiable multilateration (2005–06) | Distance measurements from multiple verifiers can verify a claimed position | The founding principle |
| arXiv:2301.12766 — *GPS-Spoofing Attack Detection Mechanism for UAV Swarms* | Our exact primitive, already published: GPS-derived distances vs IR-UWB ranging with a threshold | Confirmation, not novelty |
| arXiv:2312.03787 — *Detection and Mitigation of Position Spoofing on Cooperative UAV Swarm Formations* | Detection and mitigation for swarm formations | Prior art to cite |
| Cooperative Integrity Monitoring literature (multi-sensor cooperative positioning; VANET "local integrity") | The concept already has a name and outperforms standalone RAIM | The vocabulary |
| **arXiv:2608.06885** — *Rigid-Covert GNSS Spoofing of UAV Swarms* (Aug 2026) | **Proposition 1: any detector on relative quantities alone is invariant under a common translation.** Plus a detection-limit law and anchor-rooted recovery. Code released | The limit **and** the defence |

**Our phase-0 Test 1 independently rediscovered that last result** as `rank(J) = 3N − 6`
— the six gauge freedoms. This validates our implementation and simultaneously caps what a
pure relative approach can ever do.

### The gap that remains is the utility gap

Every result in this space is **simulation-only** — the rigid-covert authors state that no
RF hardware was used in any experiment. Nothing is validated on real flight logs, nothing
targets an operator's actual failure mode, and nothing is deployable. That is where
SentryNet goes.

## Why light shows

Documented incidents: Hong Kong 2018 · Taichung 2020 · SeaTac 2024 · Orlando Dec 2024
(collision, a drone left the show airspace, a bystander injured) · Ho Chi Minh City 2025 ·
**Sydney Vivid, May 2026 — roughly 89 drones dropped into the harbour.**

Two properties make it the right target:

- **Their named failure mode is our hardest case.** A lost RTK correction hits every drone
  *simultaneously and identically* — a common-mode shift, i.e. exactly the blind spot
  above.
- **The deployment supplies the fix for free.** A show already has surveyed absolute
  references: the RTK base station, the surveyed launch grid, ground cameras. The
  limitation that would kill this as abstract research is solved in practice.

## Consequences

- The project's previously largest risk is **inverted**: prior art is now foundation, not
  threat.
- A phase is removed (learned detector) and the product becomes more deployable, not less.
- Metrics stay the integrity-monitoring set; the "frontier" is reframed as an
  **operating envelope** — where the monitor works and where it is blind.
- Positioning must be stated honestly in every artifact: this is an **engineering**
  contribution, not a new detection principle.
- **Phase A is corrected.** It previously said "scripted pursuer, capture rate retired" —
  a leftover from the pursuit task. Under this framing there is no intruder to pursue; the
  swarm flies a *formation*. Phase A becomes a formation-flight scenario generator.

## Risks this does not remove

- **Occlusion may erase the swarm-size benefit** measured in phase 0's fully connected mesh.
- **NLoS range bias mimics spoofing** — it is a bias, not noise, so it does not average out.
- **Colluding attackers can frame honest drones**; isolation is already the weak link.
- **Temporal correlation in GNSS error** may gut the gains sequential detection promises.
- Reaching a real operator for flight logs is a relationship problem, not a technical one,
  and nothing in the plan de-risks it.

## Related

- [[ADR-011 - Pivot to Cooperative Integrity Monitoring]] · [[Decision Log]] · [[Roadmap]]
- `docs/DESIGN-v3.md` · [[Does Trust Actually Help]] · [[GPS Spoofing and GNSS Denial]]
