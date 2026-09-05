# SentryNet

**Position-integrity monitoring for drone swarms.** Detect and isolate a drone whose
reported position cannot be trusted — whether the cause is GNSS spoofing, a lost RTK
correction, multipath, or a plain sensor fault.

A drone's GNSS position can be faked or simply wrong. The time-of-flight of the radio
link between two drones cannot. SentryNet holds every claimed position against the
distances peers physically measure, and reports which drones to stop believing.

> **Status:** early development. Phase 0 (premise validation) is complete and its gate
> passed. See [`docs/DESIGN-v3.md`](docs/DESIGN-v3.md) for the plan.

---

## What this is, and what it is not

This is **not a new detection principle**. Cross-checking GNSS claims against inter-drone
ranges is established prior art, with a lineage running from Čapkun & Hubaux's verifiable
multilateration (2005–06) through the cooperative-integrity-monitoring literature to
recent UAV-swarm work. Novelty is an explicit non-goal.

What this *is*: that established method, engineered properly — a correctly derived
residual covariance, a calibrated threshold instead of a magic constant, honest operating
limits, and validation against measured hardware and real flight data rather than
convenient assumptions.

It is the swarm-level analogue of aviation **RAIM**: where a receiver uses redundant
satellite measurements to exclude a faulty satellite, SentryNet uses redundant peer
measurements to exclude a faulty *drone*.

## Install

```bash
git clone https://github.com/gokulkc01/Sentrinet.git
cd Sentrinet
pip install -e ".[dev]"
```

Python 3.10+. The runtime is numpy, scipy and pandas — the whole pipeline runs on a
laptop CPU.

## Run the premise test

Phase 0 validates the statistics and the geometry before anything is built on them:

```bash
python -m experiments.premise_test              # all three tests
python -m experiments.premise_test --test 1     # model validity only
```

Results are written to `results/premise/`.

| Test | Question | Result |
|---|---|---|
| 1 | With no attacker, is the statistic actually χ² at the claimed dof? | **Pass** — mean 8.054 vs 8 expected, KS p = 0.69 |
| 2 | How large an error is detectable per epoch? | ~4σ; 93% detection at 12 m, false alarms match `N·alpha` |
| 3 | Does it survive an attacker that lies *consistently*? | Holds at 9 drones, collapses at 3 |

Two findings from Test 1 shaped the design. The per-node statistic — the one used for
attribution — is exactly χ² and well conditioned. The **network-level** statistic is
gauge-degenerate: translating or rotating every claim together leaves all residuals
unchanged, so `rank(J) = 3N − 6` and using one degree of freedom per link inflates the
statistic sixfold.

## How it works

```
claimed positions ─┐
                   ├─→ residual = claim-implied distance − measured range
measured ranges ───┘        │
                            ├─→ normalise by covariance   (residuals sharing a
                            │                              drone are correlated)
                            ├─→ χ² test, calibrated: alpha IS the false-alarm rate
                            ├─→ isolate the worst offender (not everything over
                            │   threshold — a bad drone inflates its neighbours)
                            └─→ exclude it; fuse the rest
```

The one thing worth understanding is the normalisation. Dividing by how much disagreement
ordinary sensor noise would produce is what turns a raw number into a statistical test
with a stateable error rate.

## Known limits

Stated up front, because a monitor whose limits are undocumented is not trustworthy.

- **Common-mode shift is invisible.** If *every* drone's position is displaced by the same
  vector, all inter-drone geometry is preserved and every residual stays zero. This is a
  proved limit of any purely relative detector, not an implementation gap. The defence is
  an **absolute anchor** at a surveyed position (phase E).
- **Coplanar formations lose the benefit of swarm size.** With peers in one plane a mirror
  solution survives at any N — nine drones perform no better than four. Altitude diversity
  is a security property.
- **Small errors need time.** A single epoch needs roughly 4σ of displacement; subtler
  errors require evidence accumulated across epochs.
- **Isolation lags detection.** At 20 m we detect 99.6% of the time but attribute
  correctly only 80% — a spoofed drone inflates its honest neighbours' statistics too.
- **Honest residuals carry a small positive bias** of `2σ_g²/d` from the linearisation.
  It has the same sign as NLoS range bias, so the two are confounded.

## Repository layout

```
sentrinet/          the package — the deliverable
  world/            formations, links, geometry
  sensing/          GNSS claims; UWB two-way ranging with NLoS bias
  attacks/          threat and fault models, including the consistent liar
  integrity/        residuals, covariance, χ² tests, isolation
  fusion/           robust combination of surviving claims
  baselines/        prior-art methods this work must beat
experiments/        research drivers (premise_test.py …)
tests/              statistical regression tests — see below
docs/               DESIGN-v3.md, the current plan
SentryNet-Vault/    Obsidian knowledge vault: concepts, decisions (ADRs), findings
legacy/             frozen v2 reinforcement-learning code, kept for reference
```

## Development

```bash
pytest                  # full suite (~14 s)
pytest -m "not slow"    # skip the high-trial statistical tests
ruff check . && ruff format --check .
```

### On the tests

The largest technical risk in a project like this is not a crash — it is a **silent
modelling error** that returns plausible but wrong numbers. Linting cannot catch that.

So every statistic is tested against the distribution it should follow when nothing is
wrong, and each such test is paired with a **control that must fail**. The independence
control, for example, produces the right mean and the wrong distribution; if it ever
starts passing, the covariance has stopped doing its job.

Tests are seeded and deterministic. A failure is a real regression, not flakiness.

## Project history

SentryNet began as a multi-agent reinforcement-learning project and pivoted twice. The
reasoning for both changes, including the negative results that drove them, is recorded as
ADRs in `SentryNet-Vault/decisions/` — ADR-011 (pivot to integrity monitoring) and ADR-012
(utility-first reframe) are the relevant ones. The v2 code is frozen in `legacy/`.

## License

MIT
