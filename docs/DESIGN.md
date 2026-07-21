# SentryNet v2 — Design Document

**Status:** Approved plan. Stage 0 execution begins on explicit "go".
**Last updated:** 2026-07-21

---

## Thesis

> **Resilient cooperative target tracking under GPS spoofing, using physics-based
> plausibility trust — validated in a controlled sim study, packaged as a reusable
> benchmark, with a micro-drone sim-to-real path.**

Why this framing (and not the original "lightweight trust without cryptography for
packet loss"):

- **Realistic:** GPS spoofing/jamming is *the* dominant real-world drone threat and a
  live concern on contested borders. Cryptography cannot fix a spoofed GNSS signal —
  it is authenticated by no one — so a plausibility/trust layer has a legitimate,
  non-redundant reason to exist. This removes the "no crypto is a weakness" objection.
- **Innovative:** Learning per-node *physical-plausibility* trust for GPS-denied
  cooperative tracking is under-explored, unlike the crowded packet-drop-robustness space.
- **Useful:** Maps directly onto counter-UAS / border surveillance under electronic
  warfare, *and* the environment itself becomes a reusable open benchmark — the most
  achievable form of "truly useful" for a solo builder.

## Guiding principles

1. **Depth over breadth.** One sharp, verified result beats ten half-features.
2. **Every stage ships something of value.** If work stops after any stage, what exists
   is still complete and useful — never a half-built cathedral.
3. **The protocol freezes once Stage 0 starts** so the A/B/C comparison stays clean.
4. **Realism that changes the result is in; realism that only changes the screenshot is
   deferred.**

---

## Stage 0 — Foundation: fix, then measure honestly (~2 weeks)

### 0.1 Correctness fixes (in order)

1. **Observation normalization** — delete the broken running-stats code
   (`border_env.py:340-357`; Welford update is mathematically wrong, mutates during
   eval, and is never checkpointed). Replace with *fixed* world-scale normalization
   (positions ÷ world bounds, velocities ÷ MAX_SPEED, etc.). Deterministic,
   checkpoint-free, no train/eval skew.
   - *Consequence:* invalidates existing checkpoints. Accepted — their numbers were
     corrupt anyway.
2. **`evaluate.py`** — add missing `import torch`; reconcile the CSV schema with the
   plotting scripts.
3. **`_sender_trust_sum`** — reset per-episode in `reset()` (currently only init'd in
   `__init__`, so `tanh(sum)` saturates permanently).
4. **`test_curriculum_integration.py:70`** — fix the f-string syntax error.
5. **Hygiene that blocks the science** — delete throwaway scripts (`abcd.py`,
   `diagnose_*`, `verify_step1.py`, `test_obs_dims.py`), pin `requirements.txt`,
   stop tracking new binaries.

### 0.2 The controlled experiment (the invariant, done right)

| Held identical across A/B/C | Varies |
|---|---|
| MLP policy (128×128), entropy 0.01, `team` capture, 1M steps, **no curriculum**, identical eval protocol, identical seeds | **A:** clean training, no trust · **B:** adversarial training (drop 0.2, spoof 0.1), no trust · **C:** adversarial training + EMA trust |

- **8 seeds** per system.
- **Evaluation:** 200 episodes × drop ∈ {0.0, 0.1, …, 0.8} × spoof 0.1 with a
  compromised drone — identical for all systems.
- **Statistics:** bootstrap 95% CIs on capture rate; Welch's t-test on C−B per drop
  rate; report effect sizes. Log per-step trust traces (real, not synthetic).

### 0.3 Decision gate

- If **C > B** with significance → EMA trust helps even at small scale; carry it forward
  as a baseline into Stage 1.
- If **C ≤ B** (as current data suggests) → document the honest negative result
  ("EMA trust adds nothing over adversarial training at N=3, because trust degenerates
  into a global drop-rate meter"). This becomes the **motivation** for Stage 1.

Either outcome is publishable progress.

---

## Stage 1 — The innovation: GPS-spoofing + plausibility trust (~4–6 weeks)

### 1.1 Threat model

The adversary attacks the **physical layer**, not the crypto layer. Each drone estimates
its own position from "GPS" and broadcasts `(own_position, intruder_estimate)`. A spoofed
drone *honestly* reports a *false self-belief* — authentication cannot catch this, which
is precisely why trust must exist.

Named, reproducible attack scenarios:

| ID | Scenario | Purpose |
|---|---|---|
| S0 | No attack | Baseline / false-accusation floor |
| S1 | Constant-offset spoof, 1 node | Easy case |
| S2 | Slow-drift spoof (ramp), 1 node | Hard case — evades naive residual checks |
| S3 | Coordinated spoof, *k* nodes (k = 1…⌈N/2⌉) | Find the consensus breaking point |
| S4 | Intermittent spoof | Tests temporal detection |

Packet loss retained throughout as ambient degradation.

### 1.2 Scale

**N = 9 drones** (parameterize `N_DRONES`, currently hard-wired to 3). Each receiver now
has 8 senders, so consensus and robust statistics finally have power. **Mesh topology**
with distance- and LoS-dependent connectivity, not all-to-all.

### 1.3 Environmental realism — the load-bearing slice (terrain amendment)

Add the parts of terrain that **change the result**, modeled cheaply:

- **Heightmap + raycast line-of-sight.** Both *sensing* (can this drone see the intruder?)
  and *communication* (can these two drones reach each other?) are LoS-gated. This
  creates spatially-correlated blind spots — the physical reason a drone must depend on
  (possibly spoofed) peer data. **This is why occlusion strengthens the core experiment,
  not just decorates it.**
- **Terrain RF shadowing** — additional path loss when the LoS ray is obstructed,
  layered on the existing distance model. Makes "realistic comms" a defensible claim.
- **No-fly volumes / altitude floor** from terrain — navigation realism, minor.

Explicitly **deferred:** photorealistic terrain meshes / rendering. Slow (taxes training
throughput), weeks of effort, and for credibility a real flight video beats a simulated
canyon. Revisit only at Stage 3, if at all.

### 1.4 Trust redesign — the core novelty

Decouple trust from the RL loop: a **learned plausibility detector** feeds a
trust-weighted fusion that MAPPO consumes. Per received message, compute physics-residual
features:

1. **Kinematic feasibility** — claimed Δposition vs. platform max speed/acceleration.
2. **RF-ranging cross-check** — distance implied by claimed positions vs. distance the
   radio *measures* (UWB time-of-flight model). *The killer signal: GPS can be spoofed;
   the physics of the radio link cannot.* Real hardware — Crazyflie's Loco/UWB deck does
   exactly this — which makes the Stage-3 sim-to-real story coherent, not decorative.
3. **Consensus residual** — deviation from the trust-weighted fused estimate (meaningful
   now at N=9).
4. **Temporal self-consistency** — innovation sequence of that sender over time (catches
   slow drift, S2).

A small GRU over these features → per-sender trust score, trained **supervised** on
simulated attacks (sim provides ground-truth spoof labels). Modular, independently
testable, verifiable — the architecture defense evaluators accept — and it works when the
receiver is blind, fixing the structural flaw in the current design.

### 1.5 Baselines it must beat (honestly)

- Uniform averaging
- EMA trust (Stage-0 System C)
- Median / trimmed-mean robust fusion (the classical answer — the real bar to clear)
- **Oracle trust** (perfect attacker knowledge — upper bound)

If learned plausibility cannot beat trimmed-mean, that is itself a finding.

### 1.6 Metrics

Intruder tracking error · capture rate · **time-to-detect** spoofed node ·
**false-accusation rate** on honest nodes · degradation slope vs. number of spoofed
nodes. Same 8-seed statistical protocol as Stage 0.

---

## Stage 2 — Benchmark packaging (~2 weeks)

- Restructure into an installable `sentrinet/` package (`env/`, `comms/`, `trust/`,
  `learning/`, `cli/`); move/kill root-level one-off scripts.
- Scenario configs as YAML; documented threat taxonomy; baseline results table.
- CI running the test suite; pinned deps; README with the *real* results and repro steps.

This is the "truly useful" deliverable — infrastructure others can build on.

---

## Stage 3 — Crazyflie sim-to-real (later, ~2 months)

- Policy outputs **velocity setpoints** (not raw thrust) → Crazyswarm / PX4.
- UWB ranging is real; GPS spoof injected in software.
- Small, filmed, real — proof the robust policy survives the reality gap.
- (Only here does visual/terrain realism get reconsidered, and even then real footage
  wins.)

Scoped out until Stages 0–2 ship.

---

## Explicitly out of scope (shelved, not argued with)

Curriculum learning · LSTM (pick GRU or none) · dashboard polish · the trust
hyperparameter grid · 3-agent claims · photorealistic terrain · outdoor/EW-jamming
(GPS spoofing only) · Byzantine-consensus theory. Each is an hour taken from the critical
path.

---

## Approval & gates

- **This document = approved plan.** ✅
- **Stage 0 code changes begin on explicit "go".**
- After Stage 0 starts, the experimental protocol (§0.2) freezes to keep the comparison
  clean.
- Stage 1 begins only after Stage 0's decision gate (§0.3) reports the honest baseline.
