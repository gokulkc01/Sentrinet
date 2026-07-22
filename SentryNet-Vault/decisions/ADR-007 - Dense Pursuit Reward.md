# ADR-007 — Dense Pursuit Reward (fix the no-learning local optimum)

**Status:** ✅ Accepted (evidence-based) · **The fix that finally made the testbed learn**

## Context
After [[ADR-006 - GRU Sustained Known-Solvable Config|ADR-006]] the controlled experiment *still* never learned (capture ~0, entropy rising). A systematic elimination hunt found the cause:

| Suspect | Verdict | Evidence |
|---|---|---|
| Task / reward feasibility | ✅ fine | scripted "fly at intruder" captures **95%**, +34.9 reward |
| Architecture (MLP/GRU) | ❌ not it | both flat-zero |
| Capture mode (team/sustained) | ❌ not it | both flat-zero |
| Observation normalization | ❌ not it | ON and OFF both flat-zero |
| Critic quality | ✅ works | explained variance **0.5–0.7** |
| **Reward shaping** | 🎯 **the bug** | clean reward learns; shaped never does |

The legacy 90-line, ~15-weight [[Reward Design|shaped reward]] has a **"spread out and hover" local optimum**: the policy banks the formation/coverage/energy terms while never pursuing, so the +100 capture is essentially never experienced and can't pull the policy toward pursuit.

## Decision
Add a `reward_mode` parameter to `BorderEnv`, defaulting to **`"dense_pursuit"`** (legacy kept as `"shaped"`). Per drone, per step:

```
r = (prev_dist − curr_dist)   # dense distance-reduction (dominant, smooth gradient)
    − 0.02                     # small time cost
    + 100 · captured           # sparse terminal capture bonus (W_CAPTURE)
    − 5 · collisions           # the one shaping term kept, for safety
```

## Rationale
- The **dense distance-reduction** term gives a smooth gradient toward pursuit at every step, which the sparse capture bonus alone could not once the policy left the pursuit basin.
- Removing the competing shaping terms removes the local optimum.
- **Validated:** capture **0% → ~60% and climbing in 120k steps** (shaped stayed flat 0%); mean team distance closed from ~20 m to ~8.5 m; reward went positive.

## Consequences
- Default reward behaviour changes; applies **identically to A/B/C**, so the "vary only trust" invariant still holds.
- The legacy shaped reward is preserved (`reward_mode="shaped"`) for reference, nothing deleted.
- The **trust question is finally answerable** on a testbed that learns.
- Reinforces the general lesson: **run the cheap decisive check (scripted pursuer) first**.

## Related
- [[Reward Design]] · [[Does Trust Actually Help]] · [[ADR-006 - GRU Sustained Known-Solvable Config]] · [[Controlled Experiment]]
