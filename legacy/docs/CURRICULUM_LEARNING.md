# Curriculum Learning Architecture (May 17, 2026)

## Overview

Curriculum learning improves System C convergence by gradually increasing difficulty during training:
- Starts easy (clean channel, basic pursuit)  
- Progressively introduces communication attacks  
- Ends challenging (high drop + spoof + compromised drone)

## Five Curriculum Stages

Interpolated linearly by training progress: `stage = min(int(progress × 5), 4)` where progress ∈ [0, 1]

| Stage | Progress | p_drop | p_spoof | Compromised | Objective |
|-------|----------|--------|---------|-------------|-----------|
| 0 | 0.0-0.2 | 0.0 | 0.0 | None | Learn pursuit mechanics |
| 1 | 0.2-0.4 | 0.1 | 0.0 | None | Handle packet drops |
| 2 | 0.4-0.6 | 0.2 | 0.05 | None | Adapt to mixed corruption |
| 3 | 0.6-0.8 | 0.2 | 0.1 | Drone 1 | Trust critical (adversarial) |
| 4 | 0.8-1.0 | 0.3 | 0.2 | Drone 1 | Full adversarial challenge |

## Implementation Summary

### Observation Space (Updated to 23-dim)

**Before (20-dim)**:
- [0:3]: position
- [3:6]: velocity  
- [6:12]: trust-aggregated intruder messages
- [12]: sensor alert
- [13:16]: relative intruder position
- [16]: battery
- [17:20]: wind vector

**After (23-dim)**:
- [0:20]: same as before, with normalization applied
- [20:23]: NEW — one-hot drone ID [1,0,0] | [0,1,0] | [0,0,1]

**Rationale**:
- One-hot ID enables role specialization in shared policy
- Normalization (running mean/std) handles shifting distributions (trust decays 1.0→0.2)

### Reward Shaping (Trust-Aware)

Added lightweight signals to help drone learn honest vs compromised senders:
- Max bounded ±0.02 to avoid drowning main objective
- Rewards high trust on non-compromised senders
- Credit assignment for communication learning

### Networks Updated

- **PolicyNet**: obs_dim=23 (individual drone observation with identity)
- **ValueNet**: obs_dim=69 (3 drones × 23-dim concatenated)
- Both ready for augmentation with trust scores + channel statistics

### Trainer Integration

**Curriculum Progress Tracking**:
```python
curriculum_progress = total_env_steps / total_training_steps
```

**Environment Update** (each rollout):
```python
env.update_curriculum_progress(curriculum_progress)
```

**Methods Added**:
- `border_env._update_curriculum_params()`: interpolate difficulty
- `border_env.update_curriculum_progress(progress)`: set current stage
- `border_env._normalize_obs_features()`: running mean/std normalization

## Testing & Validation

**Smoke Test** (test_obs_dims.py): PASSED ✓
- PolicyNet obs_dim=23 ✓
- ValueNet obs_dim=69 ✓  
- One-hot drone ID confirmed ✓
- Rollout collection successful ✓
- Training update successful ✓

**Validation Harness** (validation_harness.py): READY
- Compares 3 configurations: baseline, curriculum-only, curriculum+shaping
- 3 seeds each, 50k steps
- Produces JSON detailed results + CSV summary
- Determines best approach before full training

## Next Steps

1. **Phase 1**: Run validation harness (2–3 hours)
   - `python validation_harness.py`
   - Analyze results to identify best configuration

2. **Phase 2**: Full training (12–18 GPU hours)
   - Use best config from Phase 1
   - 5 seeds × 3 systems × 100k steps
   - Full evaluation sweep

3. **Phase 3**: Trust realism migration (8–12 GPU hours, parallel)
   - Verify ground-truth leakage removal didn't break results
   - Retrain with realistic trust pipeline

4. **Phase 4**: Publication output (4–6 hours)
   - Replace synthetic plot3 with real trust trajectories
   - Add confidence intervals
   - Final documentation

## Files Modified/Created

**Modified**:
- `border_env.py`: Curriculum infrastructure, normalization, reward shaping
- `networks.py`: obs_dim parameters (23, 69)
- `mappo_trainer.py`: Curriculum wiring, progress tracking
- `rollout_buffer.py`: obs_dim=23

**Created**:
- `test_obs_dims.py`: Smoke test
- `validation_harness.py`: A/B/C comparison framework
- `CURRICULUM_LEARNING.md`: This document
