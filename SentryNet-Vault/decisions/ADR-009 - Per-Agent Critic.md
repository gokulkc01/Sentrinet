# ADR-009 — Per-Agent (Agent-Conditioned) Critic

**Status:** ✅ Accepted · **Mechanism verified; learning validation pending**

## Context
After [[ADR-007 - Dense Pursuit Reward]] made rewards **per-drone**, the centralised critic
still emitted **one value for all three drones**:

```python
v = float(self.value(obs_all)...)
for k in self._drone_keys():
    values_dict[k] = v          # same number for every drone
```

and in `update()` every drone's critic input was byte-identical:

```python
obs_all_flat = np.repeat(obs_all_t, repeats=n_drones, axis=0)   # (3T, 126) identical rows
```

So the critic was asked to predict **three different returns from one identical input**.
The MSE-optimal fit is their mean, making the residual `return_i − mean_return`
**mathematically irreducible**. That residual is not a nuisance — it *is* the advantage,
because `advantage = return − value`.

Measured on the pre-fix build (dense_pursuit, zero captures):

| rollout | non-capture \|reward\| | RAW \|adv\| | RAW adv std |
|---|---|---|---|
| 0 | 0.130 | 1.97 | 2.85 |
| 1 | 0.177 | 2.50 | 6.40 |
| 2 | 0.191 | 3.04 | 9.25 |

Advantages were **15–20× reward scale and growing**. The policy was being trained almost
entirely on critic error. Worse, mean normalised advantage was **declining** (0.49 → 0.33 →
0.30): as the raw std exploded, normalisation progressively squashed the real reward signal
toward zero. The learning signal was *decaying over training*.

This is the "critic wart" deferred earlier when explained variance (0.23/0.56/0.71) looked
healthy. That reading was wrong — EV was measured when rewards were still team-shaped, so a
single shared value was still an adequate predictor. ADR-007 is what turned the latent flaw
into an active blocker.

## Decision
Make the critic **agent-conditioned**: input = joint observation **+ one-hot agent ID**.

- `critic_obs_dim = obs_dim * n_drones + n_drones` = 42×3 + 3 = **129** (was 126 shared)
- `_obs_all_tensor` returns `(n_drones, 129)` — one row per drone — instead of `(1, 126)`
- `collect_rollout` and `last_values` store a **distinct** value per drone
- `update()` appends `np.tile(np.eye(n_d), (T,1))` to the repeated joint state; this matches
  the `np.repeat` row ordering `[t0a0, t0a1, t0a2, t1a0, …]`, which is also the ordering of
  `buffer.returns.reshape(total)` — so each row's ID names the drone whose return it is
  trained against

This is the standard MAPPO "agent-specific global state" formulation (Yu et al., 2022).

## Rationale
CTDE lets the critic see everything, but a value function is defined **per agent**. Once
agents receive different rewards they have different returns, and one shared scalar cannot
represent them. The agent ID is the minimum information needed to make the value head
agent-specific while keeping the critic centralised.

## Consequences
Verified at full, unmodified difficulty (evasive intruder, 2 m capture radius):

| rollout | value spread | EV | RAW \|adv\| | RAW adv std | \|adv\|/\|reward\| |
|---|---|---|---|---|---|
| 0 | 0.034 | 0.162 | 0.417 | 0.423 | 4.6× |
| 1 | 0.035 | 0.566 | 0.390 | 0.393 | 4.9× |
| 2 | 0.072 | 0.518 | 0.385 | 0.474 | 3.6× |

- **Value spread across drones > 0** (was *exactly* 0.0) — binary proof the fix is wired in
- **Raw advantage std no longer diverges**: 2.85→6.40→9.25 became 0.42→0.39→0.47 (stable)
- **Advantage magnitude fell ~5–8×** and the reward ratio dropped from 15–20× to 3.6–4.9×
- **Normalised advantage stopped decaying**: 0.49→0.33→0.30 became a stable ~0.65–0.69, so
  the policy now receives a consistent learning signal instead of a shrinking one
- Explained variance climbing within three rollouts (0.16 → 0.57) — the critic is learning
- **Invalidates all earlier checkpoints** (129-dim critic ≠ 126). `load_checkpoint` now
  raises a clear error instead of a cryptic shape mismatch

**Still zero captures at 6k steps** — expected for an untrained policy and *not* what this
ADR claims to fix. This ADR fixes the credit-assignment mechanism; whether that yields
learning is the open question, and the bar is **multiple seeds climbing**, not one lucky run.

## What this explicitly does NOT do
Task difficulty was **not** reduced. Easing capture (bigger radius, slower intruder) was
proposed and **rejected** — a scripted "fly at the intruder" pursuer already scores **95%**
at the current settings, which proves the task is fair and that the failure is in the
learning stack, not the environment. Reported results must never come from a softened task.

## Related
- [[ADR-007 - Dense Pursuit Reward]] · [[ADR-008 - Normalize Relative Observation Features]]
- [[Known Bugs and Confounds]] · [[Does Trust Actually Help]] · [[MAPPO and CTDE]]
