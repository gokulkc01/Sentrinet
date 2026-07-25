# ADR-010 — Reward Shaping, Not Task Difficulty, Was the Blocker

**Status:** ✅ Accepted · **Both knobs validated on 3 seeds; absolute performance still low**

## Context
[[ADR-009 - Per-Agent Critic]] repaired credit assignment and, on its own, produced
**nothing**: 3 seeds × 300k steps gave **13 captures in 900k env-steps**, flat entropy
(2.16 → 2.31), no trend on any seed. That was the outcome that ruled credit assignment out.

A reward decomposition against a **scripted pursuer that captures 100%** (mean episode
min-distance 1.98 m) explained why. The decomposition reconstructs the env's reward exactly
(residual 0.0000), so it is trustworthy:

| term | scripted (100%) | trained (0%) |
|---|---|---|
| approach | +418.5 | +384.9 |
| time | −188.9 | −600.0 |
| collision | −1540.0 | −1760.0 |
| capture | +6000.0 | +0.0 |
| **total** | **+4689.5** | **−1975.1** |

Decomposing the +6664.6 gap between always-capturing and never-capturing policies:
**capture 90.0% · time 6.2% · collision 3.3% · approach 0.5%.**

**The dense approach reward accounted for 0.5% of the difference between total success and
total failure.** The cause is structural and was present in [[ADR-007 - Dense Pursuit Reward]]
from the start: `approach = prev_dist − dist` **telescopes** to `d_initial − d_final` over an
episode, so it is path-independent. It rewards net displacement, not pursuit. Per drone, net
closure was 7.0 m (scripted) vs 6.4 m (trained) — a 9% difference between 100% and 0% capture.
So "dense_pursuit" was **not dense in the way that matters**; 99.5% of the signal remained the
sparse terminal bonus, which had fired 13 times in 900k steps.

Second effect: the trained policy parked at **4.27 m**. Closing the last 2 m is worth ~+2 per
drone, while converging pushes drones inside the 1.5 m collision cliff at **−5.0/step — 20× the
maximum per-step approach reward** (`MAX_SPEED × DT` = 0.25). The penalty formed a **moat
around the goal**, and the policy learned to sit outside it (1.8% collision steps vs the
scripted policy's 4.9%).

## Decision
Two independent knobs on `_dense_pursuit_rewards`, **both defaulting OFF** so the measured
baseline stays reproducible:

- `proximity_weight` (arm `prox`, 0.5) — adds `w / (1 + dist)` per step. Non-telescoping:
  rewards *being* close, and its gradient strengthens as `d → 0`.
- `collision_mode="graded"` + `collision_weight` (arm `coll`, 2.0) — replaces the flat −5.0
  cliff with a ramp from 0 at 1.5 m to −2.0 at contact. Collision avoidance is preserved; the
  step discontinuity is not.

Arms live in `REWARD_ARMS` in `experiments/controlled_experiment.py`, selected via `--arm`, and
the arm name is written into each checkpoint config for provenance.

**Pre-flight check before training:** a proximity term can create a *new* failure mode where
parking outside the capture radius farms reward forever. Scoring the scripted pursuer against a
standoff-hover policy under all four arms confirmed pursuit wins everywhere, discounted and
undiscounted. No arm was broken.

## Consequences
3 seeds × 300k steps per arm, System A, full difficulty:

| arm | total captures | final-100k per seed | mean | trend (pp/100k) | p vs baseline |
|---|---|---|---|---|---|
| baseline | 13 | 0.00, 2.50, 0.00 | 0.83% | −0.15, +1.06, −0.24 | — |
| prox | 60 | 9.80, 6.17, 1.00 | 5.66% | +4.97, +2.59, +0.02 | 0.100 |
| **coll** | **115** | **8.33, 3.50, 8.80** | **6.88%** | **+1.74, +1.53, +3.45** | **0.050** |

- `coll` achieves **complete separation** — every coll seed beats every baseline seed. All three
  climb. This is the largest single effect (8.8× more captures).
- `prox` helps but is inconsistent: seed2's slope is +0.02, i.e. flat.
- **p = 0.050 is the arithmetic floor at n=3.** Perfect separation is the most significant
  result possible with 3 vs 3 seeds. This is "as clean as 3 seeds can show", **not** strong
  evidence, and must not be reported as such without more seeds.
- **Entropy still did not fall** (2.16 → 2.31, same as baseline). The policy remains
  near-maximally stochastic — these captures are not a converged pursuit strategy.
- **6–9% vs the scripted pursuer's 95–100%.** Movement off zero, not a solved task.
- **No arm had plateaued at 300k** — every coll and prox seed still had positive slope at the
  end of training. The runs may simply be too short.
- ADR-009's contribution cannot be separated from these runs (the arms sit on top of it), but on
  its own it produced nothing.

## What this explicitly does NOT do
**Task difficulty is unchanged.** Capture radius 2.0 m, evasive intruder at full speed,
`sustained_steps=1`, `domain_rand=True` — all identical to baseline. Only *our own shaping
instrumentation* changed. Easing capture was proposed earlier and **rejected**: a scripted
pursuer scores 95–100% at these settings, which proves the task is fair and locates the failure
in the learning stack. That rejection was correct, and this ADR is the vindication of it — the
blocker was our reward function, and the environment never needed to be softened.

## Related
- [[ADR-007 - Dense Pursuit Reward]] · [[ADR-009 - Per-Agent Critic]] · [[Reward Design]]
- [[Known Bugs and Confounds]] · [[Does Trust Actually Help]] · [[Controlled Experiment]]
