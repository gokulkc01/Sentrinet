# Learning Roadmap

A staged curriculum for the concepts SentryNet v3 is built on, ordered by what the
project needs next.

> **Rewritten 2026-09-04 for [[ADR-011 - Pivot to Cooperative Integrity Monitoring]].**
> The previous version spent weeks 1–7 on RL foundations → PPO/GAE → MARL. Under the
> reframe that is the wrong 60% of the document: you already know PPO/GAE cold — ten ADRs
> deep into a reward decomposition and a per-agent critic fix is a level of understanding
> no tutorial produces — and MARL stops being load-bearing the moment the pursuit
> controller becomes scripted. Those phases are **demoted, not deleted**; they return for
> the phase-E ablation.

## How to use this

- **Make every checkpoint a project artifact.** On a 10–12 week timeline you cannot
  afford to learn a thing twice. Don't "implement a toy Kalman filter" — implement *the*
  EKF this project needs, and let the exercise ship.
- **Don't front-load.** Eight weeks of reading before any code is the same scope-sprawl
  failure in a new costume. Learn topic 1 properly, then interleave; topics 2, 4 and 6
  are reading you do while sweeps run.
- **Depth is asymmetric.** Go genuinely deep on estimation theory. Stay deliberately
  shallow on GNSS signal processing, RF propagation and embedded work — you need correct
  vocabulary and a defensible measurement model, nothing more. Getting that ratio wrong
  in either direction costs the project.

## What you can stop learning

- **GNSS signal processing / SDR** (correlation peaks, C/N₀ monitoring, GPS-SDR-SIM). We
  work at the **measurement level** — a node's *reported position* is wrong. You never
  touch a signal.
- **Becoming a GNSS expert.** Two weeks of reading, not a semester.
- **ROS 2 / Crazyswarm** until hardware is actually committed (deferred, 2026-09-04).

---

## 1 — Estimation theory ⭐ THE critical gap

**Why here:** it is the spine. Without it, trust stays a heuristic with hand-tuned
constants — v1 computed `accuracy = max(0, 1 − error/5.0)`, and there is no way to state
a false-alarm rate for `5.0`, therefore no way to defend it. With estimation theory the
same idea becomes a hypothesis test whose `alpha` **is** the false-alarm probability.

**Core:** Kalman filter → EKF; state and covariance; the **innovation (residual)
sequence** and its covariance; **NIS / Mahalanobis distance** → **χ² gating**; fault
detection, identification and exclusion (FDI); covariance consistency.

**Resources:** **Bar-Shalom, Li & Kirubarajan, *Estimation with Applications to Tracking
and Navigation*** (primary — has the NIS/χ²/gating machinery explicitly) ·
**Thrun, Burgard & Fox, *Probabilistic Robotics*** (gentler entry) · Simon, *Optimal
State Estimation* (fault detection).

**Ties to:** `sentrinet/integrity/{residuals,chi2}.py`.

**Checkpoint:** the EKF fusing peer reports; plot the NIS sequence; verify it is χ²
under no attack; watch it spike under spoof. *(Phase 0 already did the static version of
this — mean 8.054 against an expected 8, KS p = 0.691.)*

---

## 2 — GNSS fundamentals & integrity monitoring ⭐

**Why here:** the framing. Without the vocabulary, "swarm RAIM" doesn't land with the
audience that actually cares about GNSS spoofing.

**Core:** how a fix is computed (pseudoranges, trilateration, least squares, DOP); error
sources; spoofing vs jamming vs meaconing; **RAIM** — residual-based fault detection,
**protection levels**, integrity risk, P(missed detection) / P(false alert), ARAIM; the
language of HMI, alert limits, time-to-alert.

**Resources:** **Kaplan & Hegarty, *Understanding GPS/GNSS*** (read selectively) ·
**Todd Humphreys** (UT Austin Radionavigation Lab) on civilian spoofing · the **Stanford
GPS Lab** RAIM/ARAIM publications (Blanch, Walter) · skim **ION GNSS+** proceedings as
much for the venue's *style* as its content.

**Checkpoint:** express your own Test 2 results in Pmd/Pfa terms and state an operating
point you would defend.

---

## 3 — UWB ranging & enough radio physics

**Core:** two-way ranging, single- vs **double-sided TWR** (and why clock drift forces
DS); TDoA; path loss and link budget; **NLoS positive range bias**.

**The insight worth internalising:** blocked line-of-sight makes the signal travel
*further*, so NLoS bias is strictly positive — and a spoofed position claim also produces
a range/claim disagreement. **"Behind a hill" and "lying" are genuinely confusable**, and
separating them is part of the research problem, not a nuisance. It couples
[[ADR-004 - Terrain Occlusion Only]] directly to detection.

**Resources:** DW1000/DW3000 user manuals and Qorvo app notes (unusually practical) ·
UWB localisation surveys · Bitcraze Loco documentation.

**Ties to:** `sentrinet/sensing/uwb.py`.

---

## 4 — Detection theory

**Core:** ROC/AUC, operating points, Neyman–Pearson, likelihood-ratio tests;
**CUSUM / SPRT** for sequential detection; calibration.

**Why it matters here:** Test 2 showed single-epoch detection needs ~4σ of displacement.
Anything subtler has to be caught by **accumulating evidence over time** — which is
exactly what sequential detection is for, and what makes time-to-detect a meaningful
metric. This is also the principled answer to the slow-drift attack (S2).

**Resources:** **Kay, *Fundamentals of Statistical Signal Processing, Vol. II: Detection
Theory*** (as a lookup, not cover-to-cover) · scikit-learn docs for the practical metrics.

---

## 5 — Resilient consensus & robust statistics

**Core:** Byzantine fault tolerance; **r-robustness and (r,s)-robustness** of graphs;
**W-MSR**; breakdown point; M-estimators; trimmed means; algebraic connectivity.

**Why it matters here:** this is what makes the frontier a *theoretical* result rather
than an empirical sweep — it tells you whether a sparse mobile mesh can tolerate *f*
attackers at all.

**Resources:** **LeBlanc, Zhang, Koutsoukos & Sundaram, *Resilient Asymptotic Consensus
in Robust Networks* (2013)** — the r-robustness paper · Lamport, Shostak & Pease, *The
Byzantine Generals Problem* (1982) · a robust-statistics survey.

**Tooling:** `networkx`.

**Checkpoint:** compute r-robustness of your actual mesh as swarm size and comms range
vary — that is a figure in the paper.

---

## 6 — Stealthy attacks in cyber-physical systems

**Why here:** the adaptive attacker. Almost all of this literature assumes a naive
adversary; the question an evaluator asks within ninety seconds is *what if the attacker
knows the detector exists?* There is an entire CPS-security literature on attacks that
evade residual-based detectors — you don't have to invent it.

**Resources:** **Mo & Sinopoli** on false-data injection in control systems ·
**Pasqualetti, Dörfler & Bullo** on attack detection and identification in CPS ·
Teixeira et al. on CPS attack models.

**Ties to:** `sentrinet/attacks/adaptive.py`.

---

## 7 — Research packaging & craft

**Core:** `pyproject.toml` + src layout; GitHub Actions CI; `pytest` discipline;
**Hydra/OmegaConf** for YAML scenarios; DVC or git-LFS (the repo carries 431 MB of
checkpoints and a 62 MB `.git`); IEEE LaTeX; reproducibility packaging.

**Retained from the old roadmap:** **Henderson et al., *Deep Reinforcement Learning that
Matters* (2018)** — still the best statement of the seed-variance and reproducibility
problem this project has already lived through.

---

## Demoted (return for the phase-E ablation)

RL foundations · deep RL / PPO+GAE · MARL and CTDE. You already have these. The old
phase notes are preserved in git history if you want them back.

---

## Suggested sequence

| Week | Learn | Build |
|---|---|---|
| 1–2 | **Estimation ⭐** | EKF over peer reports + NIS validation |
| 2–3 | GNSS + RAIM vocabulary *(parallel)* | Phase A: scripted pursuer |
| 3–4 | UWB ranging, NLoS bias | Phase B: N=9, mesh, LoS, TWR |
| 4–5 | Stealthy CPS attacks | Phase C: S0–S4 + adaptive attacker |
| 5–6 | RAIM exclusion, robust stats | Phase D: classical baselines |
| 6–7 | Detection theory, CUSUM/SPRT | Phase E: learned detector + ROC |
| 7–8 | r-robustness | Phase F: frontier sweeps |
| 10–12 | Packaging, IEEE LaTeX | Phase H: package + paper |

## The meta-skill

The fastest deep learning available is still in this repo: for every bug in
[[Known Bugs and Confounds]] and every decision in [[Decision Log]], make sure you
understand it well enough to have caught or made it yourself. That now includes
[[ADR-011 - Pivot to Cooperative Integrity Monitoring]] — a project you can explain
*including why it changed direction* is worth more than ten tutorials.

## Related

- [[00 - START HERE]] · [[Glossary]] · [[Roadmap]] · [[Decision Log]]
