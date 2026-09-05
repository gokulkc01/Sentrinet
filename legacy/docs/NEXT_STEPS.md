# SentryNet: Next Important Steps (May 17, 2026)

## Summary of Current State

**Curriculum Learning**: ✅ COMPLETE — 5 progressive difficulty stages implemented
- obs_dim updated 20 → 23 (added one-hot drone ID)
- Observation normalization added
- Trust-aware reward shaping implemented (max ±0.02 bounded)
- All networks updated (PolicyNet: 23-dim, ValueNet: 69-dim)
- Smoke test passed

**Ground-truth Leakage Migration**: ✅ STEPS 1-3 COMPLETE
- Local estimate infrastructure working
- Consensus-based trust (weighted-median + temporal consistency) implemented
- Ready for retraining validation

**Previous Results**: Baseline A/B/C with 3 seeds × 3 systems (ground-truth version)
- System C best single-seed (100% capture across all drop rates)
- But variance across seeds undermines confidence

---

## ⚡ PRIORITY 1: Validate Curriculum Learning Impact (2–3 hours)

**Command**:
```bash
cd d:\Sentrinet
python validation_harness.py
```

**What it does**:
- Trains 9 configurations: 3 approaches × 3 seeds
  - Config A: Baseline (no curriculum, basic reward shaping)
  - Config B: Curriculum only (progressive difficulty)
  - Config C: Full (curriculum + trust-aware reward shaping)
- 50k steps per run (fast smoke test)
- Outputs: `results/validation/validation_results_<timestamp>.json` + CSV summary

**Critical decision**:
- If Config C wins decisively → Proceed to Phase 2 (full 100k training)
- If Config B wins → Use curriculum without shaping
- If Config A wins → Curriculum doesn't help; investigate why

**Timeline**: ~2–3 hours wall-clock time (9 sequential training runs)

---

## PRIORITY 2: Full Training with Best Config (12–18 GPU hours)

**After Phase 1**, run full training:

```bash
# Modify train.py or create new script
# Use best configuration from validation harness
# Run with 5 seeds (improves statistical rigor)
```

**For each of Systems A, B, C**:
- 5 seeds × 100k steps each
- Same adversarial conditions as old baseline:
  - Evaluation: p_drop ∈ {0.0, 0.1, ..., 0.8}
  - p_spoof=0.1 (fixed)
  - compromised_drone=1

**Important**: Record real trust trajectories during evaluation for authentic plot3 (not synthetic)

**Timeline**: ~15 hours GPU time total (3 systems × 5 seeds)

---

## PRIORITY 3: Ground-truth Leakage Validation (8–12 GPU hours, can run parallel)

**In parallel with Priority 2**, verify ground-truth removal:

1. **Smoke-test** (already done): System C trains with realistic trust
2. **Full retraining**: Retrain with Steps 2–3 of migration applied
3. **Compare results**: Old (ground-truth) vs. New (realistic)
4. **Quantify impact**: How much does ground-truth leakage inflate performance?

**Critical question**: Does System C still outperform under realistic trust?
- If yes: Publish results with ground-truth migration complete
- If no: Either redesign trust mechanism or accept negative result (still valid)

**Timeline**: ~10 hours GPU time (3 systems × 3 seeds for validation comparison)

---

## PRIORITY 4: Publication-Ready Output (4–6 hours)

**After Phases 2 & 3**:

1. **Replace plot3** (trust dynamics)
   - Was: Synthetic exponential decay formula
   - Now: Real trust trajectories recorded from evaluation
   - Add confidence intervals (5+ seeds)

2. **Update documentation**
   - PROJECT_CONTEXT.md: Add curriculum architecture section
   - CURRENT_STATUS.md: Document ground-truth migration
   - CURRICULUM_LEARNING.md: Reference implementation guide
   - Create supplementary: Trust algorithm details

3. **Generate final plots**
   - Compare old (ground-truth) vs. new (realistic) results
   - Show improvement from curriculum learning
   - Confidence intervals on all metrics

**Timeline**: ~5 hours (data aggregation + plotting + writing)

---

## Immediate Next Step (RIGHT NOW)

**Execute Phase 1**:
```bash
cd d:\Sentrinet
python validation_harness.py
```

**Expected completion**: 2–3 hours

**Decision**: Use results to determine Phase 2 configuration

---

## Timeline Estimate to Publication-Ready

| Phase | Task | Time | GPU Cost |
|-------|------|------|----------|
| **1** | Validation harness (A/B/C × 3 seeds, 50k steps) | 2–3h | Low |
| **2** | Full training (3 systems × 5 seeds, 100k steps) | 12–18h | High ✓✓✓ |
| **3** | Ground-truth migration validation | 8–12h | High ✓✓✓ |
| **4** | Publication plots & documentation | 4–6h | None |
| **Total** | **To publication-ready** | **~30–40 hours** | **~20h GPU** |

---

## Key Files to Know

**Modified for curriculum learning**:
- `d:\Sentrinet\border_env.py` — Curriculum stages, normalization, reward shaping
- `d:\Sentrinet\networks.py` — PolicyNet(23), ValueNet(69)
- `d:\Sentrinet\mappo_trainer.py` — Curriculum wiring
- `d:\Sentrinet\rollout_buffer.py` — obs_dim=23

**New files**:
- `d:\Sentrinet\test_obs_dims.py` — Smoke test (already passed ✓)
- `d:\Sentrinet\validation_harness.py` — A/B/C comparison (READY TO RUN)
- `d:\Sentrinet\CURRICULUM_LEARNING.md` — Implementation details

**Documentation**:
- `d:\Sentrinet\CURRENT_STATUS.md` — Updated with curriculum + next phases
- `d:\Sentrinet\PROJECT_CONTEXT.md` — Architecture overview (needs curriculum section reference)

---

## Deferred (After Phase 4)

These can wait until after publication-ready results:
- Reactive intruder profiles (fast, evasive, stealth)
- Multi-drone containment capture mechanism
- Role emergence analysis
- Sensor agent learning (currently rule-based)
- RL adversary (two-player game)
- Ablation studies (alpha parameter, FoV, etc.)

---

## Checkpoints for Success

✅ **Checkpoint 1 (NOW)**: Phase 1 validation harness runs without errors (2–3h)
- Best config identified
- Confidence: Is curriculum helpful?

✅ **Checkpoint 2 (After Phase 2)**: Full training completes with 5 seeds (12–18h GPU)
- Final System A/B/C models trained
- Full evaluation sweep done
- Real trust trajectories recorded

✅ **Checkpoint 3 (After Phase 3)**: Ground-truth migration validated (8–12h GPU)
- Know impact of realistic trust
- Decision: Include in paper or acknowledge limitation

✅ **Checkpoint 4 (After Phase 4)**: Publication-ready (4–6h work)
- All plots updated
- Documentation complete
- Ready for peer review

---

## Notes for Next Session

**If resuming**: Start with this command:
```bash
cd d:\Sentrinet
python validation_harness.py
```

**Expected outputs**: 
- `results/validation/validation_results_<timestamp>.json` (detailed metrics)
- `results/validation/validation_summary.csv` (aggregated by config)

**Decision point**: Based on validation results, decide whether to proceed with Phase 2 or iterate curriculum design.

