# SentryNet — Current Status & Open Questions

## Current Status

### What Works

| Component | Status | Notes |
|-----------|--------|-------|
| Environment (BorderEnv) | ✅ Functional | 3D world, mock physics, PyBullet optional |
| Domain randomization | ✅ Functional | Mass, wind, sensor noise, intruder speed |
| Mock physics | ✅ Functional | Drag, gravity, wind, clamping |
| Adversarial channel | ✅ Functional | Drop, spoof, distance-dependent drop rates |
| Trust module (EMA) | ✅ Functional | Converges correctly — uses local-reference updates and consensus-based evaluation (weighted-median + temporal consistency), no GT access |
| Trust aggregator | ✅ Functional | Weighted averaging, handles drops |
| PolicyNet | ✅ Functional | Tanh-squashed Gaussian, orthogonal init |
| ValueNet | ✅ Functional | Centralized 60-dim critic |
| MAPPO trainer | ✅ Functional | Collect, GAE, PPO update, checkpoint |
| Training pipeline | ✅ Functional | System C: 5 seeds completed at 100k; broader multi-system retraining pending |
| Evaluation pipeline | ✅ Functional | 81 conditions × 200 episodes completed |
| Results CSV | ✅ Exists | `results/full_experiment.csv` |
| Publication plots | ✅ Generated | 5 plots in `results/plots/` |
| Dashboard | ✅ Functional | Pygame + optional PyBullet overlay |
| 3D visualization | ✅ Functional | PyBullet with URDF drones |
| Local estimate infra | ✅ Added | `_update_local_estimates()` works and is wired into `_comms_pipeline()` |
| Consensus-based trust | ✅ Implemented | Weighted-median consensus + temporal consistency added to `TrustModule.update()` |
| Step 1 verification | ✅ Passed | `verify_step1.py` confirms no behavior change |
| Curriculum learning | ✅ Implemented | 5 progressive difficulty stages, obs_dim=23, drone identity embedding |
| Observation normalization | ✅ Implemented | Running mean/std on shifting feature distributions |
| Trust-aware reward shaping | ✅ Implemented | Lightweight rewards (max ±0.02) for honest communication learning |
| Trainer curriculum wiring | ✅ Implemented | Progress tracking and env update each rollout |
| Network dimension updates | ✅ Implemented | PolicyNet(23), ValueNet(69), all tests passing |

### What's Broken or Misleading

| Issue | Severity | Impact |
|-------|----------|--------|
| Trust evaluates against ground truth | ~~**Critical**~~ Resolved | Trust updates now compare received messages to the receiver's local estimate (see `trust_module.py:update()` and `border_env.py:_comms_pipeline()`) |
| Comms broadcast ground-truth intruder state | ~~**Critical**~~ Resolved | Senders now broadcast their noisy local estimates; stale estimates self-drop (see `border_env.py:_comms_pipeline()`) |
| Trust dynamics plot is synthetic | **High** | Misleading for publication |
| Single-drone capture (2.0m threshold) | **Medium** | Coordination not required |
| Intruder is non-reactive | **Medium** | Pursuit is trivially easy |
| Seed count remains limited across the full project | **Medium** | System C has 5 recent seeds; broader multi-system retraining still needed for publication-grade confidence |
| Sensor agent never learns | **Low** | Functional but not interesting |
| No curriculum training | **Low** | ✅ **RESOLVED** — **5-stage curriculum now implemented** |

---

## Recent Work Summary (May 17, 2026)

### Curriculum Learning Implementation — COMPLETE ✅

**Status**: Fully implemented and smoke-tested

**What was implemented**:

1. **border_env.py**
   - Added 5 progressive difficulty stages (clean → full adversarial)
   - Interpolate difficulty by training progress: `progress = total_steps / max_steps`
   - Stage 0 (prog=0.0): p_drop=0.0, p_spoof=0.0
   - Stage 1 (prog=0.2): p_drop=0.1, p_spoof=0.0
   - Stage 2 (prog=0.4): p_drop=0.2, p_spoof=0.05
   - Stage 3 (prog=0.6): p_drop=0.2, p_spoof=0.1, 1 compromised drone
   - Stage 4 (prog=1.0): p_drop=0.3, p_spoof=0.2, 1 compromised drone
   - Added observation normalization (running mean/std)
   - Added one-hot drone identity embedding: **obs_dim = 23** (was 20)
   - Added trust-aware reward shaping (max ±0.02 bounded)

2. **networks.py**
   - PolicyNet: obs_dim=23 (individual drone obs with identity)
   - ValueNet: obs_dim=69 (3 drones × 23-dim concatenated)

3. **mappo_trainer.py**
   - Curriculum progress tracking and env update each rollout
   - RolloutBuffer initialized with obs_dim=23
   - All dimension-dependent code updated

4. **test_obs_dims.py** (NEW)
   - Smoke test passed: PolicyNet(23), ValueNet(69), training updates work

5. **validation_harness.py** (NEW)
   - A/B/C comparison: Baseline vs. Curriculum vs. Full (curriculum + shaping)
   - 3 seeds each, 50k steps, produces JSON + CSV results
   - Fixed step accounting bug: now advances by 1 env step per rollout, not `n_steps * 3`
   - Config variants now differ correctly: baseline/curriculum use `use_trust=False`, full uses `use_trust=True`

### System C Capture Diagnosis — UPDATED ✅

**Status**: Inspectable and validated against current checkpoints

**A) Exact capture logic found**
- `_captured()` uses `CAPTURE_R = 2.0 m`
- `capture_mode="sustained"` requires one drone to remain inside radius for `sustained_steps` consecutive steps
- The curriculum now makes `sustained_steps=1` in phases A/B and restores `sustained_steps=3` in phase C
- There is no hidden multi-drone requirement unless `capture_mode="multi"` is chosen explicitly

**B) Modifications made**
- Replaced the old interpolated 5-stage schedule with explicit curriculum phases A/B/C
- Added configurable capture mode and sustained-capture duration in `train.py`
- Increased dense pursuit reward strength in `border_env.py`
- Added checkpoint loader compatibility for old 20-dim policies and new 23-dim observations

**C) Before/after reward structure**
- Before: `r += 0.5 * (prev_dist - current_dist)` plus a weak proximity term
- After: `r += 2.0 * (prev_dist - current_dist)` plus tiered distance bonuses at 8m / 5m / 3m / 2m, while keeping battery cost, security penalty, and collision penalties

**D) Curriculum schedule**
- Phase A: 0–20% progress, `p_drop=0`, `p_spoof=0`, `domain_rand=False`, `sustained_steps=1`
- Phase B: 20–50% progress, `p_drop=0.1`, `p_spoof=0`, `domain_rand=True`, `sustained_steps=1`
- Phase C: 50–100% progress, `p_drop=0.2`, `p_spoof=0.1`, compromised drone enabled, `domain_rand=True`, `sustained_steps=3`

**E) Capture metrics**
- Training rollout capture stayed near 0% in the 100k run, but that metric is rollout-local and very sparse
- Smoke training with the new code completed successfully for System C seed 0 at 10,240 steps

**F) Clean vs adversarial comparison**
- `run_trained.py --checkpoint checkpoints/system_C_seed0 --episodes 100 --p_drop 0 --p_spoof 0 --use_trust --no-domain-rand` gave **15% capture**
- `run_trained.py --checkpoint checkpoints/system_C_seed0 --episodes 100 --p_drop 0.2 --p_spoof 0.1 --use_trust --no-domain-rand` gave **11% capture**
- `diagnose_trust.py` on the same checkpoint reported **25% clean** and **30% adversarial** capture over 20 episodes

**G) Whether System C improved**
- Yes, it can capture in clean conditions and under adversarial conditions, but capture is still sparse and not yet reliable enough for 100% capture claims
- The 0% rollout logs do not mean the policy never learned pursuit; they mainly reflect the strict sustained-capture rule and the local rollout metric

**H) Recommended next actions**
- Retrain System C with the new staged curriculum for at least 1M steps
- Keep `sustained_steps=1` during early curriculum and restore `3` only after pursuit stabilizes
- Re-run evaluation after retraining and generate trust trajectory plots from recorded episodes rather than synthetic traces

**Interpretation of the first run**:
- Zero capture at 50k steps is still possible, because capture is sparse and the environment uses sustained capture by default.
- The key improvement signal in the short validation run is reward/trust trend, not capture yet.
- Re-run validation after the step-accounting fix before drawing conclusions.

### What's Next

From `full_experiment.csv` (200 episodes per condition, all systems face p_spoof=0.1 + compromised drone 1):

**System C (seed 0)** maintains **100% capture rate across all drop rates (0.0–0.8)** — the strongest single-seed result.

**System C aggregate** shows degradation at high drops due to seed 1 weakness:

| Drop Rate | System A (mean) | System B (mean) | System C (mean) |
|-----------|-----------------|-----------------|-----------------|
| 0.0 | 98.2% | 90.5% | 98.7% |
| 0.2 | 99.2% | 97.8% | 99.5% |
| 0.4 | 91.2% | 99.5% | 96.0% |
| 0.6 | 73.5% | 93.0% | 85.5% |
| 0.8 | 56.7% | 65.8% | 73.8% |

**Key observation**: System C's advantage is clearest at extreme drop rates — but the variance across seeds undermines confidence. System B sometimes outperforms C on specific seeds, which would concern a reviewer.

**Trust scores**: System C mean trust decreases monotonically with drop rate (0.58 → 0.15), confirming the EMA mechanism responds to drops. But this is expected behavior of a system with ground-truth access — the real question is whether it would work without it.

---

## Migration Roadmap (In Progress)

The previous conversation established a multi-step plan to remove ground-truth leakage. Step 1 is complete:

| Step | Description | Status |
|------|-------------|--------|
| 1 | Add local estimate infrastructure (`_update_local_estimates()`, `_estimate_age`) | ✅ Complete |
| 2 | Replace broadcast ground truth with local noisy estimates in `_comms_pipeline()` | ✅ Complete |
| 3 | Replace ground-truth comparison in `TrustModule.update()` with consensus-based evaluation | ✅ Complete |
| 3a | Advanced consensus (weighted median, temporal consistency) | ✅ Complete |
| 4 | Retrain all systems with realistic communication | ⬜ Not started — smoke-test passed for System C (10k steps, seed 0) |
| 4a | Smoke-test (fast) | ✅ Passed for System C seed 0 (10k steps) |
| 5 | Re-evaluate and regenerate results | ⬜ Not started |

---

## Open Questions

### Architecture Questions

**Q1**: When switching to local estimates in `_comms_pipeline()`, what should a drone broadcast when its estimate is stale (age > `STALE_THRESH` = 25 steps)?
- Option A: Broadcast the stale estimate anyway (with age metadata)
- Option B: Broadcast nothing (treated as a self-imposed drop)
- Option C: Broadcast last known estimate but mark confidence as low
- **This affects how trust evaluation interprets silence vs. old information.**

**Q2**: For consensus-based trust (Step 3), how should trust be computed without ground truth?
- Option A: Median-based outlier detection (compare each message against the consensus of other messages)
- Option B: Temporal consistency (does this sender's reports change plausibly over time?)
- Option C: Cross-correlation with own local estimate (when available)
- Option D: Combination of A + B + C with confidence weighting
- **This is the hardest design decision in the project. The mechanism must be robust enough to still differentiate honest from compromised drones, but without access to truth.**

**Q3**: Should the centralized critic (ValueNet) also lose access to ground-truth intruder position?
- Currently, the critic sees the trust-aggregated message through all three drone observations. If drones no longer have perfect information in their observations, the critic automatically loses it too.
- But during training, should the critic have privileged access? CTDE (centralized training, decentralized execution) traditionally allows this. The question is whether it undermines the research claim.

**Q4**: How should the capture mechanism change?
- Multi-drone containment (require 2+ drones within threshold)?
- Sustained containment (require proximity for N consecutive steps)?
- Either change makes the task significantly harder and may require retuning reward weights.

### Evaluation Questions

**Q5**: How many seeds are needed for publication-quality confidence intervals?
- Current: 3 seeds. Standard practice for MARL venues: 5–10 seeds.
- Cost: each full training run is ~1M steps. At 3 systems × 10 seeds, that's 30 training runs.

**Q6**: Should the trust dynamics plot be replaced with real recorded trajectories?
- This requires instrumenting the evaluation loop to log per-step trust scores per sender.
- It would provide genuine evidence of trust convergence behavior.

**Q7**: What ablation experiments are needed?
- Trust vs. no-trust under identical realistic communication (the current core comparison)
- Effect of trust alpha parameter (0.05 vs 0.1 vs 0.2)
- Effect of FoV parameters (range, angle)
- Effect of stale threshold
- Effect of number of compromised drones (0, 1, 2)
- Importance of local estimates vs. global broadcasts

### Research Direction Questions

**Q8**: Should the intruder become an RL adversary?
- This turns the problem into a two-player game and adds a curriculum-like effect.
- But it significantly increases complexity and training time.
- **Recommendation**: Implement reactive intruder profiles first (fast, evasive, stealth). RL adversary is a separate future contribution.

**Q9**: Should role emergence be pursued in this phase?
- The shared policy means all drones behave identically. Adding agent IDs or role tokens could enable specialization.
- But role emergence is difficult to claim without extensive analysis.
- **Recommendation**: Defer. Focus on trust realism first.

**Q10**: What is the minimum viable set of changes needed for a defensible research contribution?
- Complete Steps 2–5 of the migration (replace ground truth in comms and trust)
- Run 5+ seeds
- Record real trust trajectories for the trust dynamics plot
- Show that System C still outperforms A and B under realistic conditions
- If it doesn't, that's also a valid (and arguably more interesting) finding

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| System C fails to outperform after removing ground truth | **High** | This is a genuine research outcome. Publish the negative result or redesign the trust mechanism. |
| Consensus-based trust is too slow to converge | **Medium** | Tune alpha, add temporal consistency, use confidence weighting |
| Stale estimates cause policy collapse | **Medium** | Gradual integration — start with fresh estimates only, fall back to old pipeline for stale ones |
| Retraining takes too long | **Low** | Use `--fast` mode for prototyping, full runs only for final results |
| Local sensing too noisy for trust | **Medium** | The noise model (0.05 + 0.02×dist) is relatively mild. May need to tune for realistic ranges. |

---

## ⚡ Next Actions (Priority Order — Critical Path)

### Phase 1: Validate Curriculum Impact — START HERE

**Action 1.1** — Run A/B/C validation harness (2–3 hours wall-clock)
```bash
cd d:\Sentrinet
python validation_harness.py
```
Trains 9 configurations (3 configs × 3 seeds) at 50k steps each.
Produces: `results/validation/validation_results_<timestamp>.json` + CSV summary.
**Critical decision**: Confirms whether curriculum + shaping improves convergence.

**Action 1.2** — Analyze results and decide
- Compare capture rates, rewards, trust metrics across A/B/C
- Identify best configuration
- Decision: Scale to 100k + 5 seeds vs. redesign approach

---

### Phase 2: Full Training (12–18 GPU hours)

Train with best configuration from Phase 1:
- 5 seeds minimum (improves statistical rigor)
- Systems A, B, C (apply curriculum to all for fair comparison)
- 100k steps per run
- Full eval sweep: p_drop ∈ {0.0–0.8}
- **Important**: Record real trust trajectories for authentic trust dynamics plot

---

### Phase 3: Trust Realism Migration (parallel OK — 8–12 GPU hours)

Validate Steps 1–3 of ground-truth leakage removal:
- Retrain A/B/C with realistic trust pipeline
- Compare old (ground-truth) vs. new (realistic) results
- Quantify impact of ground-truth removal

---

### Phase 4: Publication Output (4–6 hours)

- Replace synthetic plot3 with real trust trajectories
- Add confidence intervals (5+ seeds)
- Update documentation with curriculum architecture

---

## Checkpoint: Phase 1 is next

**Command**:
```
python validation_harness.py
```

**Expected timeline**:
- Phase 1: 3–4 hours
- Phase 2: 12–18 hours GPU
- Phase 3: 8–12 hours GPU (parallel)
- Phase 4: 4–6 hours
- **Total**: ~30–40 hours to publication-ready results
