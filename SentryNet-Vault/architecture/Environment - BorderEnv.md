# Environment — BorderEnv

**File:** `border_env.py` · **Class:** `BorderEnv(ParallelEnv)` (PettingZoo)

The heart of the project: a 20×20×10 m airspace where 3 hunter drones pursue 1 intruder, with an optional ground sensor.

## Responsibilities
- **Physics:** `_MockPhysics` (fast point-mass with drag/gravity/wind/clamping) or PyBullet (`_init_pybullet`, `_pb_step`).
- **Intruder motion:** `_step_intruder` — profiles `passive` / `evasive` / `reactive`.
- **Local sensing:** `_update_local_estimates` — each drone gets a noisy intruder estimate only inside a FoV cone (`DETECT_RANGE=8m`, `DETECT_ANGLE=60°`).
- **Comms pipeline:** `_comms_pipeline` — the research core; broadcasts, channel, trust, fusion. See [[Adversarial Channel]], [[Trust Module and Aggregator]].
- **Rewards:** `_compute_rewards` — capture reward + a large stack of shaping terms.
- **Observations:** `_drone_obs` / `_obs_all`. See [[Observation and Action Spaces]].
- **Domain randomization:** `_domain_randomise` — mass, wind, sensor noise, intruder speed.

## Key constants (module-level)
`N_DRONES=3`, `MAX_STEPS=500`, `CAPTURE_R=2.0m`, `DT=0.05s`, plus ~15 reward weights (`W_CAPTURE=100`, `W_TEAM`, `W_CLOSE`, …). ⚠️ Being module-level globals, they hard-wire the env to 3 drones — [[ADR-002 - Scale to 9 Drones]] must parameterize `N_DRONES`.

## Capture modes
- `team`: caught when ≥2 drones are within `CAPTURE_R`.
- `sustained`: caught when one drone stays within range for `sustained_steps`.

## ⚠️ Bugs living here
- **Observation normalization** (`_normalize_obs_features`, lines ~340-357): broken Welford math, mutates during eval, not checkpointed. The single most damaging bug. See [[Known Bugs and Confounds]] and [[ADR-005 - Fixed Observation Normalization]].
- **`_sender_trust_sum`** never reset per episode → trust reward shaping saturates. See [[Known Bugs and Confounds]].
- **Curriculum override:** when enabled, `_update_curriculum_params` overrides p_drop/p_spoof for all systems, breaking the A/B/C invariant. Curriculum is shelved for Stage 0.

## Related
- [[Adversarial Channel]] · [[Trust Module and Aggregator]] · [[Observation and Action Spaces]] · [[Reward Design]]
