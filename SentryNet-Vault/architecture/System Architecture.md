# System Architecture

SentryNet is organized in **five layers**. Data flows: environment → communication → trust → learning → experiment.

```
┌─────────────────────────────────────────────────────────────┐
│ 5. EXPERIMENT   train.py · evaluate.py · run_trained.py       │
│                 dashboard.py · plot_results.py                │
├─────────────────────────────────────────────────────────────┤
│ 4. LEARNING     networks.py · rollout_buffer.py               │
│                 mappo_trainer.py                              │
├─────────────────────────────────────────────────────────────┤
│ 3. TRUST        trust_module.py · trust_aggregator.py         │
├─────────────────────────────────────────────────────────────┤
│ 2. COMMUNICATION  adversarial_channel.py                      │
├─────────────────────────────────────────────────────────────┤
│ 1. ENVIRONMENT   border_env.py  (physics, obs, rewards)       │
└─────────────────────────────────────────────────────────────┘
```

## The layers
1. **Environment** — [[Environment - BorderEnv]] (`border_env.py`). The 20×20×10 m world, drone/intruder physics, observations, rewards, and the comms pipeline that ties layers 2–3 together.
2. **Communication** — [[Adversarial Channel]] (`adversarial_channel.py`). Drops and spoofs messages.
3. **Trust** — [[Trust Module and Aggregator]] (`trust_module.py`, `trust_aggregator.py`). Per-sender EMA scoring + trust-weighted fusion.
4. **Learning** — [[Networks and Rollout Buffer]] + [[MAPPO Trainer]]. The [[MAPPO]] actor-critic, rollout storage + [[PPO and GAE|GAE]], and the training loop.
5. **Experiment** — training/eval/plotting/visualization entry points.

## The core per-step data flow
Every environment step (`border_env.step`) runs:
1. Convert actions → thrust, advance **physics** (mock or PyBullet).
2. Move the **intruder** (passive / evasive / reactive profile).
3. Each drone updates its **local FoV estimate** of the intruder.
4. **Comms pipeline:** each drone broadcasts its estimate → [[Adversarial Channel|channel]] drops/spoofs → [[Trust Module and Aggregator|trust updated & messages fused]] → fused estimate injected into observations.
5. Compute **rewards** (capture + heavy shaping).
6. Build **observations**, return to the trainer.

The comms detail (steps 3–4) is the research core — see [[Adversarial Channel]] and [[Trust Module and Aggregator]]. Reward detail (step 5) is in [[Reward Design]].

## Health & caveats
- 25/25 tests in `tests/test_phase1_3d.py` pass — the core is functional.
- But several **correctness bugs** corrupt results and several **design choices** confound the science. See [[Known Bugs and Confounds]].
- The codebase has **scope sprawl** (24 root scripts, throwaways) that Stage 2 cleans up. See [[Roadmap]].

## Related
- [[Environment - BorderEnv]] · [[MAPPO Trainer]] · [[Observation and Action Spaces]] · [[Known Bugs and Confounds]]
