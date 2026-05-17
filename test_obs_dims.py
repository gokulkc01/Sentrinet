#!/usr/bin/env python3
"""Smoke test for obs dimension changes."""
import numpy as np
from border_env import BorderEnv
from mappo_trainer import MAPPOTrainer

print("=" * 60)
print("OBS DIMENSION SMOKE TEST")
print("=" * 60)

# Create environment with curriculum
env = BorderEnv(
    use_pybullet=False,
    domain_rand=False,
    p_drop=0.2,
    p_spoof=0.1,
    use_curriculum=True,
    curriculum_progress=0.0,
    use_trust=True,
    seed=42,
)

print("\n[1] Environment initialization OK")

# Create trainer
config = {
    "n_steps": 128,
    "total_steps": 100_000,
    "seed": 42,
}
trainer = MAPPOTrainer(env, config)
print("[2] Trainer initialization OK")

# Reset and check obs dimensions
obs_dict, _ = env.reset()
print(f"[3] Policy obs_dim: {trainer.policy.obs_dim} (expected 23)")
print(f"[4] Value obs_dim: {trainer.value.obs_dim} (expected 69)")

sample_obs = obs_dict['drone_0']
print(f"[5] Drone obs shape: {sample_obs.shape} (expected (23,))")
assert sample_obs.shape == (23,), f"FAIL: Expected (23,) got {sample_obs.shape}"

print(f"[6] One full drone obs: {sample_obs}")

# Try one rollout
print("\n[7] Collecting rollout...")
metrics = trainer.collect_rollout()
print(f"    Mean reward: {metrics['mean_reward']:.4f}")
print(f"    Capture rate: {metrics['capture_rate']:.1%}")

# Try one update
print("\n[8] Running one training epoch...")
train_metrics = trainer.update()
print(f"    Policy loss: {train_metrics['policy_loss']:.6f}")
print(f"    Value loss: {train_metrics['value_loss']:.6f}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
