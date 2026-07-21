# Known Bugs and Confounds

The catalog of what's broken and why it matters. Stage 0 fixes the correctness bugs; the confounds are removed by the [[Controlled Experiment]]. Until these are addressed, **no existing number is trustworthy.**

## 🔴 Correctness bugs (Stage 0 fixes, in order)

### 1. Observation normalization — CRITICAL
`border_env._normalize_obs_features` (~lines 340-357). Three failures:
- Welford std update is mathematically wrong → variance never converges, drifts up.
- Stats live in the env instance and **re-learn from zero on every fresh eval env** → train/eval mismatch.
- Stats **not saved in checkpoints** → loaded policy sees different normalization than it trained under.
- **Impact:** corrupts every result. Fix = fixed world-scale normalization ([[ADR-005 - Fixed Observation Normalization]]).

### 2. `evaluate.py` crashes — CRITICAL
Calls `torch.load()` at line 91 but **never imports `torch`**. The committed `full_experiment.csv` also has a *different schema* than this code emits → the results were produced by code no longer in the repo. Fix: `import torch` + reconcile schema.

### 3. `_sender_trust_sum` never reset — HIGH
Initialized in `__init__`, never in `reset()`. Accumulates forever → `tanh(sum)` saturates to ~1, so the trust reward-shaping term is a constant, not a signal. ([[Reward Design]])

### 4. `test_curriculum_integration.py:70` — HIGH
`SyntaxError` (escaped quotes in an f-string). The file can't even be parsed.

## 🟠 Design warts (document or fix)
- **Shared critic value across drones** — one joint value assigned to 3 drones while returns are per-drone. Defensible team-value design, but undocumented. ([[MAPPO Trainer]])
- **Sensor not learned** — hard-coded echo of the detection flag; old "QMIX agent" docs are fiction. ([[Observation and Action Spaces]])
- **Reward function** — 90 lines, ~15 magic weights → ceiling effects. ([[Reward Design]])

## 🔴 Confounds (invalidate the science, removed by Stage 0)
- **Architecture mismatch:** System C = GRU + sustained capture + entropy 0.005; A/B = MLP + team + 0.01. C-vs-A/B is uninterpretable. This is the biggest one.
- **Curriculum override:** when enabled, overrides p_drop/p_spoof for all systems → breaks the A/B/C invariant. (Curriculum shelved.)
- **Only 3 seeds** with huge variance → nothing is significant.
- **Ceiling effect** below drop 0.4.

## 🧹 Repo hygiene (blocks reproducibility)
- 74 checkpoint `.pt` files + 636 `tmp/` files committed → 61 MB `.git`. Use LFS/DVC or don't commit.
- No README; `requirements.txt` unpinned.
- ~10 throwaway root scripts (`abcd.py`, `diagnose_*`, `verify_step1.py`, `test_obs_dims.py`).
- Docs contradict code (obs dim 20 vs actual 42) and each other.

## Related
- [[Does Trust Actually Help]] · [[Controlled Experiment]] · [[ADR-005 - Fixed Observation Normalization]] · [[Roadmap]]
