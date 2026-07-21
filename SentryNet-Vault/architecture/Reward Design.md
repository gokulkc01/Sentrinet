# Reward Design

**Where:** `border_env._compute_rewards`. How the drones are told what "good" means.

## The components (per drone, per step)
- **Terminal capture:** `+W_CAPTURE (100)` when the intruder is caught — dominates everything.
- **Time penalty:** `−0.1`/step (finish faster).
- **Energy penalty:** proportional to battery drain.
- **Security penalty:** proportional to empirical spoof rate.
- **Team pursuit:** reward for reducing *mean* team→intruder distance (not a single hero drone).
- **Formation / coverage:** shaping for useful spacing and angular surround.
- **Capture-imminence gradient:** smooth `exp(−d/cap_r)` pull toward the intruder.
- **Milestone bonuses:** step rewards at d < 6/4/3/2.5 m to stop "hovering just outside".
- **Collision penalty:** `−5` per too-close drone pair.
- **Trust shaping:** small bounded term rewarding high trust on honest senders.

## The problem: this is a 90-line pile of magic numbers
~15 hand-tuned weights. This is almost certainly why results **saturate near 100%** at low drop rates — the shaping is so strong that capture is easy, leaving no headroom to show a trust benefit. It also makes behavior hard to reason about or reproduce.

## ⚠️ Bug living here
`_sender_trust_sum` (used by the trust-shaping term) is never reset per episode → `tanh(sum)` saturates to ~1 permanently, so the term is a constant, not a signal. See [[Known Bugs and Confounds]].

## Stage-1 direction
Simplify aggressively: a clean capture reward + minimal shaping, so the *task* (not the shaping) drives behavior and the trust effect is observable. Fewer knobs = more reproducible science.

## Related
- [[Environment - BorderEnv]] · [[Controlled Experiment]] · [[Known Bugs and Confounds]]
