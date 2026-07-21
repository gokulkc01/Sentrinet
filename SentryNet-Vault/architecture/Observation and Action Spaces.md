# Observation and Action Spaces

Exact interfaces between the [[Environment - BorderEnv|environment]] and the [[MAPPO|policy]]. Built in `border_env._drone_obs` / `_obs_all`.

## Drone observation — 42 dims
| Slice | Dims | Contents |
|---|---|---|
| 0:3 | 3 | own position |
| 3:6 | 3 | own velocity |
| 6:12 | 6 | **trust-aggregated** intruder pos+vel (the fused message) |
| 12 | 1 | sensor alert flag |
| 13:16 | 3 | relative intruder pos *if in FoV*, else zeros |
| 16 | 1 | battery |
| 17:20 | 3 | wind vector |
| 20:23 | 3 | one-hot drone ID |
| 23:42 | 19 | relative target pos/vel + teammate rel pos/vel + distance |

First 20 dims are passed through the (buggy) normalizer; the rest are raw — a scale mismatch worth noting.

## Sensor observation — 4 dims
`[detected_flag, noisy_x, noisy_y, noisy_z]`.

## Actions
- **Drones:** `Box(3)` in [-1,1] → interpreted as thrust (`_to_thrust`). ⚠️ For [[Sim-to-Real Transfer]] this should become velocity/waypoint setpoints, not raw thrust.
- **Sensor:** `Discrete(2)` (idle / alert) — but **hard-coded, not learned** (echoes the detection flag). The "QMIX sensor agent" in old docs doesn't exist.

## Why dims [6:12] are the crux
That's where the **fused teammate estimate** enters the policy's input. Everything about [[Trust and Reputation|trust]], [[Adversarial Communication|attacks]], and [[Plausibility-Based Trust|the redesign]] is ultimately about making those 6 numbers trustworthy when the drone is blind.

## Related
- [[Environment - BorderEnv]] · [[Networks and Rollout Buffer]] · [[Trust Module and Aggregator]]
