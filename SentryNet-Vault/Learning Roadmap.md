# Learning Roadmap

A staged, in-depth curriculum for the concepts SentryNet is built on. Ordered by dependency: earlier phases unlock later ones.

## How to use this

- **Learn grounded, not abstract.** After each topic, open the matching SentryNet file/note and explain it line by line. If you can't, you haven't learned it yet.
- **Two directions, use both.** *Top-down:* start from a file that confuses you and pull the threads. *Bottom-up:* build the fundamentals in order below. A hybrid sticks best.
- **You learn RL by implementing it.** Every phase has a **hands-on checkpoint** — a small thing you build from scratch. Reading alone will fool you into thinking you understand.
- **Priority if you want to be effective on the project *now*:** Phase 2 (PPO/GAE) → Phase 6 (experimental rigor) → Phase 5 (trust/robust stats). Those three map to what you're doing in Stage 0–1 and to the bugs that bit you. Fill in Phases 1/3/4 around them.

```
Phase 0 (math, as-needed)
      │
Phase 1  RL foundations ──► Phase 2  Deep RL / PPO+GAE ──► Phase 3  Multi-agent RL
                                        │                        │
                                        ▼                        ▼
                              Phase 6  Experimental rigor   Phase 4  Drones / control / estimation
                                        │                        │
                                        └────────► Phase 5  Adversarial / trust / robust stats ◄──┘
```

---

## Phase 0 — Mathematical foundations (as-needed, don't front-load)

**Why here:** RL is applied probability + optimization. You need *just enough*, pulled in when a topic demands it — not a semester up front.

**Core concepts:** probability (expectation, variance, conditional prob, Bayes, Gaussians), linear algebra (vectors, matrices, norms, dot products), calculus/optimization (gradients, chain rule, SGD, convexity intuition), a little information theory (entropy, KL divergence).

**Resources:** *Mathematics for Machine Learning* (Deisenroth, Faisal, Ong — free PDF); 3Blue1Brown's *Essence of Linear Algebra* and *Essence of Calculus* (YouTube).

**SentryNet tie-in:** Gaussians → the policy's action distribution ([[Networks and Rollout Buffer]]); KL divergence → PPO's `approx_kl` ([[MAPPO Trainer]]); norms → every distance in [[Environment - BorderEnv]].

**Checkpoint:** derive the gradient of a scalar loss w.r.t. a weight matrix by hand; explain why entropy measures "randomness" of the policy.

---

## Phase 1 — Reinforcement Learning foundations

**Why here:** Everything else is built on the MDP framework. Skip this and PPO is cargo-cult.

**Core concepts:** Markov Decision Processes (states, actions, transitions, rewards, γ); return; value functions V(s) and Q(s,a); the Bellman equations; policy vs value methods; exploration/exploitation; temporal-difference learning; on-policy vs off-policy.

**Resources:** **Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed, free PDF)** — chapters 3 (MDPs), 4 (DP), 5 (Monte Carlo), 6 (TD), 9 (approximation), 13 (policy gradients). **David Silver's RL Course** (DeepMind/UCL lectures on YouTube) — the canonical video companion.

**SentryNet tie-in:** [[Multi-Agent Reinforcement Learning]], [[Glossary]]. `gamma=0.99` in [[MAPPO Trainer]] is the discount γ; the critic is V(s).

**Checkpoint:** implement **value iteration** and **tabular Q-learning** on a small gridworld from scratch. Watch the value function converge.

---

## Phase 2 — Deep RL & policy gradients (the core of what you use) ⭐

**Why here:** SentryNet *is* a policy-gradient method. This is the phase to go deepest.

**Core concepts:** function approximation with neural nets; the policy-gradient theorem; REINFORCE; baselines & variance reduction; actor-critic; advantage estimation → **GAE**; trust-region idea → **PPO**'s clipped objective; entropy regularization; the tanh-squashed Gaussian for bounded continuous actions and its log-prob correction.

**Resources:** **OpenAI *Spinning Up in Deep RL*** — the single best practical path from vanilla PG to PPO, with clean code. **Schulman et al., *Proximal Policy Optimization Algorithms* (2017)** — the PPO paper. **Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (2015)** — the GAE paper. Costa Huang's *The 37 Implementation Details of PPO* — invaluable for the gotchas (this is the level at which your normalization bug lived).

**SentryNet tie-in:** [[PPO and GAE]], [[Networks and Rollout Buffer]], [[MAPPO Trainer]]. Map every term: `compute_gae` ↔ the GAE equations; `surr1/surr2` ↔ the clip; `_squashed_log_prob` ↔ the tanh correction; `entropy_coef` ↔ entropy bonus.

**Checkpoint:** implement **REINFORCE**, then **vanilla PPO**, from scratch on CartPole and Pendulum. *Then* read `mappo_trainer.py` and `networks.py` and explain every line against your own implementation. This alone will make you genuinely dangerous.

---

## Phase 3 — Multi-Agent RL

**Why here:** SentryNet is 3 (soon 9) cooperating agents. MARL adds non-stationarity, credit assignment, and partial observability.

**Core concepts:** Dec-POMDPs; the non-stationarity problem; **CTDE** (Centralized Training, Decentralized Execution); shared vs independent policies; centralized critics; value factorization (QMIX) vs policy-gradient MARL (MAPPO, MADDPG); the credit-assignment problem.

**Resources:** **Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games* (2021)** — the MAPPO paper, read it closely. **Lowe et al., *MADDPG* (2017)** and **Rashid et al., *QMIX* (2018)** for contrast. **Oliehoek & Amato, *A Concise Introduction to Decentralized POMDPs*** (book) for the theory.

**SentryNet tie-in:** [[MAPPO]], [[Observation and Action Spaces]]. The centralized critic (126-dim joint state) vs local policy (42-dim) is CTDE in the flesh. The "shared value across drones" wart in [[Known Bugs and Confounds]] is a credit-assignment subtlety — you'll understand it fully after this phase.

**Checkpoint:** take your Phase-2 PPO and extend it to 2 cooperating agents with a centralized critic. Feel the non-stationarity.

---

## Phase 4 — Drones: dynamics, control & state estimation

**Why here:** to make it *realistic* and eventually *hardware-ready*, you must understand what the RL is actually commanding — and why raw thrust is the wrong abstraction.

**Core concepts:** quadrotor rigid-body dynamics (6-DOF, thrust/torque, motor model); the **cascaded control stack** (position → velocity → attitude → motor); PID control; **state estimation** — Kalman filter → **Extended Kalman Filter (EKF)** — and **sensor fusion**; coordinate frames; domain randomization for **[[Sim-to-Real Transfer|sim-to-real]]**.

**Resources:** **Thrun, Burgard, Fox, *Probabilistic Robotics*** — the canonical book for Kalman/EKF and sensor fusion (this underpins your whole trust-as-estimation angle). **Panerati et al., *Learning to Fly* (gym-pybullet-drones, 2021)** — the paper for the drone sim you literally vendored. Quadrotor dynamics + PID: any standard aerial-robotics course (e.g. the classic Coursera *Aerial Robotics* by Vijay Kumar).

**SentryNet tie-in:** [[Environment - BorderEnv]] (`_MockPhysics`, `_to_thrust`, domain randomization), [[Sim-to-Real Transfer]], [[ADR-004 - Terrain Occlusion Only]]. The "output velocity setpoints, not thrust" fix is a direct consequence of understanding the cascaded stack.

**Checkpoint:** derive a quadrotor's equations of motion; implement a **1-D Kalman filter** fusing two noisy sensors; implement a **PID hover** controller. The Kalman filter is the conceptual seed of plausibility trust.

---

## Phase 5 — Adversarial ML, security & robust estimation (your innovation) ⭐

**Why here:** this is where the *novel contribution* lives — [[Plausibility-Based Trust]] under [[GPS Spoofing and GNSS Denial]].

**Core concepts:** GNSS fundamentals and **GPS spoofing/jamming**; **trust & reputation** systems in networks; the **Byzantine Generals Problem** and Byzantine-fault-tolerant aggregation; **robust statistics** (median, trimmed mean, M-estimators, breakdown point); anomaly/outlier detection; the distinction between crypto (authentication) and physical-plausibility (integrity); multi-sensor fusion under attack.

**Resources:** **Lamport, Shostak, Pease, *The Byzantine Generals Problem* (1982)** — foundational. **Huber, *Robust Statistics*** (or a robust-stats survey for a gentler start). Todd Humphreys' body of work on **civilian GPS spoofing** (UT Austin) — the canonical academic source; search his talks/papers. For trust/reputation, read surveys on *trust management in multi-agent / sensor networks* (the field is broad — read 2–3 surveys rather than one paper).

**SentryNet tie-in:** [[Trust and Reputation]], [[Plausibility-Based Trust]], [[Robust Statistics and Consensus]], [[Adversarial Channel]], the ADRs [[ADR-001 - GPS-Spoofing Pivot]] / [[ADR-003 - Supervised Plausibility Trust]]. [[Robust Statistics and Consensus]] is *why* you're scaling to N=9.

**Checkpoint:** implement trust-weighted mean, **median**, and **trimmed-mean** aggregation; inject a spoofed sensor and measure each estimator's **breakdown point** as you increase the number of liars. Then build a toy **kinematic-plausibility** detector. This is a direct dry-run of Stage 1.

---

## Phase 6 — Experimental rigor & research craft ⭐

**Why here:** this is the phase that would have *prevented most of your bugs*. Rigor is a skill, not an afterthought.

**Core concepts:** reproducibility (seeds, determinism, environment pinning); the confound problem (why A/B/C must be identical except one variable); statistical testing (bootstrap CIs, Welch's t-test, effect sizes, multiple-comparison pitfalls); ceiling effects; ablations; reading and writing papers; honest reporting of negative results.

**Resources:** **Henderson et al., *Deep Reinforcement Learning that Matters* (2018)** — read this early; it is *exactly* your seed-variance / reproducibility problem, documented. Andrew Ng's *Machine Learning Yearning* for experimental discipline. For paper-reading: Andrew Ng's "how to read papers" method (the three-pass approach).

**SentryNet tie-in:** [[Controlled Experiment]], [[Metrics]], [[Does Trust Actually Help]], [[Known Bugs and Confounds]]. Your Stage-0 pipeline (`experiments/`) is applied Phase 6.

**Checkpoint:** re-run a small experiment with 8 seeds; compute bootstrap CIs and a Welch t-test by hand (then check against `analyze_controlled.py`); design one ablation and predict its result before running it.

---

## A suggested 12-week sequence (adjust to your pace)

| Weeks | Focus | Outcome |
|---|---|---|
| 1–2 | Phase 1 (+ Phase 0 as needed) | Value iteration & Q-learning working |
| 3–5 | **Phase 2** ⭐ | REINFORCE + PPO from scratch; can read all of `networks.py`/`mappo_trainer.py` |
| 6–7 | Phase 3 | 2-agent CTDE PPO; understand the critic |
| 8 | **Phase 6** ⭐ | Rigorous re-run with CIs; understand every Stage-0 bug |
| 9–10 | Phase 5 ⭐ | Robust aggregators + a plausibility detector (previews Stage 1) |
| 11–12 | Phase 4 | Kalman filter + PID + cascaded-control understanding for sim-to-real |

## The meta-skill: learn by interrogating this project

The fastest deep learning available to you is already in this repo: for every **bug** in [[Known Bugs and Confounds]] and every **decision** in [[Decision Log]], make sure you understand it well enough to have caught/made it yourself. A project you can fully explain — including its mistakes — is worth more than ten tutorials.

## Related
- [[00 - START HERE]] · [[Glossary]] · [[Roadmap]] (the *project* roadmap, distinct from this *learning* roadmap)
