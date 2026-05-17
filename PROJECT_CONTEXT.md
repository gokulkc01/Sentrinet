# SentryNet — Project Context

## Summary of Understanding

SentryNet is a trust-aware multi-agent reinforcement learning system for autonomous drone border surveillance. It studies whether lightweight EMA-based trust scoring can improve cooperative drone swarm robustness under adversarial communication attacks — without requiring cryptographic mechanisms.

The project is positioned between a research prototype and a publishable experimental system. It has functional training, evaluation, and visualization pipelines, but carries several known scientific validity issues that must be resolved before results can withstand peer review.

---

## Architecture — What Exists Today

### Agents

| Agent | Type | Observation | Action |
|-------|------|-------------|--------|
| `drone_0`, `drone_1`, `drone_2` | Hunter drones (MAPPO) | 20-dim float32 | Box(3) ∈ [-1,1] (thrust) |
| `sensor_0` | Ground sensor (rule-based) | 4-dim float32 | Discrete(2) (idle/trigger) |
| Intruder | Autonomous (biased random walk) | N/A | N/A |

### World

- 20×20×10m airspace
- Mock physics with Ornstein-Uhlenbeck wind turbulence
- Optional PyBullet physics and 3D rendering
- Domain randomization: mass ±18%, wind up to 15 km/h, sensor noise 0–0.30, intruder speed 1.5–4.0 m/s

### Communication Pipeline (Per Step)

```
1. Each drone broadcasts honest = [intruder_pos, intruder_vel]  ← GROUND TRUTH
2. Compromised drone (if set) replaces message with adversarial mirror-image position
3. AdversarialChannel applies:
   a. Distance-dependent drop (base p_drop + 0.025 × distance, capped at 0.95)
   b. Bernoulli packet drop
   c. Additive Gaussian spoofing (p_spoof probability)
4. TrustModule updates EMA trust per sender:
   - On receive: tau = 0.1 × accuracy + 0.9 × tau
   - accuracy = max(0, 1 - ||recv_pos - TRUE_pos|| / 5.0)  ← GROUND TRUTH
   - On drop: tau = 0.95 × tau
5. TrustAggregator: weighted_avg = Σ(tau_j × msg_j) / Σ(tau_j)
6. Aggregated message → observation dims [6:12]
```

### Observation Layout (20-dim per drone)

| Dims | Content |
|------|---------|
| [0:3] | Own position |
| [3:6] | Own velocity |
| [6:9] | Trust-aggregated intruder position |
| [9:12] | Trust-aggregated intruder velocity |
| [12] | Sensor alert flag |
| [13:16] | Relative intruder position (if in FoV, else zeros) |
| [16] | Battery |
| [17:20] | Wind vector |

### Field of View Sensing

Each drone has independent local sensing:
- Detection range: 8.0m
- Detection cone: 60° half-angle (relative to velocity direction)
- When detected: stores noisy local estimate (noise_std = 0.05 + 0.02 × distance)
- When not detected: previous estimate ages but persists

**Key finding**: Local estimates (`_local_estimates`, `_estimate_age`) exist in the environment but are **not yet used** in the communication pipeline. The `_comms_pipeline()` still broadcasts ground-truth intruder state.

### Neural Networks

| Network | Input | Hidden | Output | Init |
|---------|-------|--------|--------|------|
| PolicyNet (shared actor) | 20-dim | 128→Tanh→128→Tanh | 3-dim Gaussian (tanh-squashed) | Orthogonal |
| ValueNet (centralized critic) | 60-dim (3×20 concatenated) | 128→Tanh→128→Tanh | 1-dim scalar | Orthogonal |

### Training (MAPPO)

- PPO clip ratio: 0.2
- GAE: γ=0.99, λ=0.95
- Rollout: 2048 steps
- Mini-batch: 256
- Epochs per update: 4
- Learning rate: 3e-4 (Adam)
- Entropy coefficient: 0.01
- Gradient clipping: 10.0
- Total steps: 1,000,000 per run

### Reward Structure

| Component | Weight | Notes |
|-----------|--------|-------|
| Capture bonus | +W1 = +10.0 | On successful capture |
| Time penalty | -W2 = -0.1 | Per step |
| Energy cost | -W3 × (1 - battery) | Per step |
| Security penalty | -W4 × min(1, emp_spoof_rate) | Proportional, W4=5.0 |
| Approach shaping | +0.5 × Δdist | Closing distance reward |
| Proximity bonus | +1.0 × (1 - dist/5.0) | When within 5.0m |
| Team coordination | +0.3 × (1 - min_dist/5.0) | When team closest drone < 5.0m |
| Collision penalty | -5.0 | When drones < 1.5m apart |
| Sensor | +1.0 correct trigger, +0.05 correct idle, -0.5 false alarm | |

### Capture Logic

Single-drone proximity: any drone within CAPTURE_R = 2.0m triggers capture.

### Three Experimental Systems

| System | p_drop (train) | p_spoof (train) | use_trust | Purpose |
|--------|----------------|-----------------|-----------|---------|
| A | 0.0 | 0.0 | False | Clean baseline |
| B | 0.2 | 0.0 | False | Packet loss only |
| C | 0.2 | 0.1 | True | Trust-aware (full) |

All evaluated under identical adversarial conditions: p_spoof=0.1, compromised_drone=1, sweep p_drop ∈ {0.0, 0.1, …, 0.8}.

---

## Existing Infrastructure

### Files

| File | Purpose |
|------|---------|
| `border_env.py` | Environment (588 lines) — world, physics, comms, rewards |
| `adversarial_channel.py` | Packet drop + spoofing simulation |
| `trust_module.py` | EMA trust scoring per sender |
| `trust_aggregator.py` | Trust-weighted message averaging |
| `networks.py` | PolicyNet + ValueNet |
| `rollout_buffer.py` | Rollout storage + GAE computation |
| `mappo_trainer.py` | Full MAPPO training loop |
| `train.py` | CLI for training systems A/B/C |
| `evaluate.py` | Full condition sweep → CSV |
| `plot_results.py` | Publication plots from CSV |
| `run_trained.py` | Inference + PyBullet visualization |
| `simulation_trial.py` | Cleaner PyBullet visualizer |
| `dashboard.py` | Real-time Pygame + optional PyBullet dashboard |
| `diagnose_trust.py` | Trust mechanism + checkpoint health tests |
| `verify_step1.py` | Verification of local estimate infrastructure |
| `diagnostic_3d.py` | Environment sanity checks |
| `tests/test_phase1_3d.py` | Pytest suite for Phase 1 |

### Checkpoints

Trained models exist for all 3 systems × 3 seeds (0, 1, 2), plus an extra `system_A_seed42`. Each run directory contains `best.pt` and/or `step_*.pt` files.

### Results

`results/full_experiment.csv`: 81 rows (3 systems × 3 seeds × 9 drop rates), 200 episodes per condition.

5 publication plots in `results/plots/`.

---

## Scientific Validity Assessment

### Critical Issues (in priority order)

#### Issue 1: Ground-Truth Leakage in Trust

**Where**: `border_env.py:404`, `trust_module.py:94`

The trust update compares received messages against the **true intruder position** passed directly from the environment. This means trust evaluation has access to information no real drone would possess.

**Impact**: Trust scores are artificially accurate. The entire System C claim — that trust-aware aggregation improves robustness — is built on privileged information. A reviewer would reject this immediately.

**What exists already**: Local FoV-based estimates (`_local_estimates`, `_estimate_age`) are computed per step in `_update_local_estimates()`, but the communication pipeline (`_comms_pipeline()`) ignores them and broadcasts ground truth.

#### Issue 2: Global Broadcasts Use Privileged Information

**Where**: `border_env.py:359-362`

The "honest" messages broadcast by each drone are `[intruder_pos, intruder_vel]` — the true state from the environment. In reality, each drone would only know its own noisy local estimate.

**Impact**: The entire communication subsystem is operating on perfect information. This makes the adversarial communication challenge far easier than it would be in practice.

#### Issue 3: Single-Drone Capture is Trivial

**Where**: `border_env.py:463-465`

Only one drone needs to reach within 2.0m to capture. This is too easy and doesn't require real coordination.

#### Issue 4: Intruder is Non-Reactive

**Where**: `border_env.py:305-315`

The intruder follows a biased random walk toward the center with random noise. It doesn't evade, doesn't react to hunters, and doesn't adapt. This makes the pursuit problem significantly easier than any realistic scenario.

#### Issue 5: No Curriculum Training

All three systems train under fixed adversarial settings. System C sees the same p_drop=0.2 and p_spoof=0.1 for the entire training run. There is no progressive difficulty escalation.

#### Issue 6: Trust Dynamics Plot is Synthetic

**Where**: `plot_results.py:213-243`

The trust dynamics plot (`plot3_trust_dynamics.png`) is generated from a hardcoded exponential decay formula, not from actual trust trajectories recorded during evaluation. This is misleading for publication.

#### Issue 7: High Variance Across Seeds

The CSV data shows significant performance variance. For example, System C seed 1 degrades to 59.5% capture at p_drop=0.8, while seed 0 maintains 100% across all drop rates. With only 3 seeds, confidence intervals are wide.

#### Issue 8: Sensor Agent is Rule-Based

`sensor_0` is controlled by a simple threshold rule (`action = 1 if alert else 0`) everywhere — in training, evaluation, and inference. It never learns. Its reward signal exists but is not connected to any learning loop.

---

## What Has Already Been Done (Prior Conversations)

Based on conversation history:

1. **Local estimate infrastructure** has been added (`_update_local_estimates()`, `_local_estimates`, `_estimate_age`, FoV detection) — Step 1 of the trust realism migration is complete.
2. `verify_step1.py` confirms the infrastructure works without breaking existing behavior.
3. Experiments have been run, results collected, plots generated.
4. Dashboard and visualization tooling is mature.
5. The system has been through debugging cycles (trust convergence issues, reward structure adjustments, spoofing strategy changes).

---

## What the Codebase Actually Does Well

1. **Clean separation of concerns**: env / channel / trust / learning / evaluation are modular.
2. **Reproducibility**: Seeded RNG, checkpoint save/load, deterministic evaluation.
3. **Visualization**: PyBullet 3D rendering, Pygame dashboard with live controls.
4. **Domain randomization**: Non-trivial physics realism (mass variation, wind turbulence, distance-dependent drops).
5. **Adversarial modeling**: Compromised drone sends mirror-image positions — a reasonably sophisticated targeted attack.
6. **Evaluation methodology**: Systematic sweep across conditions with CSV export.

---

## Relationship Between Components

```
train.py
  └─ MAPPOTrainer
       ├─ PolicyNet (shared)
       ├─ ValueNet (centralized)
       ├─ RolloutBuffer (GAE)
       └─ BorderEnv
            ├─ _MockPhysics / PyBullet
            ├─ _update_local_estimates()  ← exists but unused in pipeline
            ├─ _comms_pipeline()
            │    ├─ AdversarialChannel.transmit()
            │    ├─ TrustModule.update()  ← uses TRUE intruder pos
            │    └─ TrustAggregator.aggregate()
            ├─ _compute_rewards()
            └─ _drone_obs()
```
