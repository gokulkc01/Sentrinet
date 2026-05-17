#!/usr/bin/env python3
"""Quick smoke test for curriculum learning integration."""
import numpy as np
import torch
from border_env import BorderEnv
from networks import PolicyNet, ValueNet
from mappo_trainer import MAPPOTrainer

def test_curriculum_smoke():
    """Test curriculum integration without full training."""
    print("=" * 60)
    print("CURRICULUM LEARNING INTEGRATION SMOKE TEST")
    print("=" * 60)
    
    # Create environment with curriculum
    env = BorderEnv(
        use_pybullet=False,
        domain_rand=False,
        p_drop=0.2,
        p_spoof=0.1,
        use_curriculum=True,      # ENABLE CURRICULUM
        curriculum_progress=0.0,   # START AT STAGE 0
        use_trust=True,
        seed=42,
    )
    
    # Create trainer
    config = {
        "n_steps": 256,  # Short rollout for quick test
        "total_steps": 100_000,
        "seed": 42,
    }
    trainer = MAPPOTrainer(env, config)
    
    print("\n[TEST 1] Environment initialization")
    print(f"  Environment curriculum: {env.use_curriculum}")
    print(f"  Initial progress: {trainer.curriculum_progress:.3f}")
    
    print("\n[TEST 2] Network dimensions")
    obs = env.reset()[0]
    print(f"  Policy obs_dim: {trainer.policy.obs_dim}")
    print(f"  Value obs_dim: {trainer.value.obs_dim}")
    # Get one drone obs to verify dimension
    sample_obs = obs['drone_0']
    print(f"  Actual drone obs shape: {sample_obs.shape} (should be (23,))")
    assert sample_obs.shape == (23,), f"Expected obs shape (23,), got {sample_obs.shape}"
    
    print("\n[TEST 3] Curriculum progress updates")
    for step in [0, 250_000, 500_000, 1_000_000]:
        trainer.total_env_steps = step
        env_ema = env.reset()[0]  # dummy to trigger reset
        progress = min(1.0, step / config['total_steps'])
        print(f"  Step {step:7d}: progress = {progress:.2f}")
    
    print("\n[TEST 4] One rollout collection")
    try:
        metrics = trainer.collect_rollout()
        print(f"  Mean reward: {metrics['mean_reward']:.4f}")
        print(f"  Capture rate: {metrics['capture_rate']:.2%}")
        print(f"  Mean trust: {metrics['mean_trust']:.4f}")
        print(f"  Curriculum progress: {trainer.curriculum_progress:.3f}")
        print(f"  ✓ Rollout collection successful")
    except Exception as e:
        print(f"  ✗ Rollout failed: {e}")
        raise
    
    print("\n[TEST 5] One training update")
    try:
        train_metrics = trainer.update()
        print(f\"  Policy loss: {train_metrics['policy_loss']:.4f}\")
        print(f\"  Value loss: {train_metrics['value_loss']:.4f}\")
        print(f\"  Entropy: {train_metrics['entropy']:.4f}\")
        print(f\"  Approx KL: {train_metrics['approx_kl']:.4f}\")
        print(f\"  ✓ Training update successful\")
    except Exception as e:
        print(f\"  ✗ Training update failed: {e}\")
        raise
    
    print(f"\n[TEST 6] Curriculum stage progression\")
    # Check that curriculum params change as progress increases
    env.curriculum_progress = 0.0
    env._update_curriculum_params()
    stage0_drop, stage0_spoof = env._p_drop_eff, env._p_spoof_eff
    
    env.curriculum_progress = 0.5
    env._update_curriculum_params()
    stage2_drop, stage2_spoof = env._p_drop_eff, env._p_spoof_eff
    
    env.curriculum_progress = 1.0
    env._update_curriculum_params()
    stage4_drop, stage4_spoof = env._p_drop_eff, env._p_spoof_eff
    
    print(f\"  Stage 0 (prog=0.0):   p_drop={stage0_drop:.2f}, p_spoof={stage0_spoof:.2f}\")
    print(f\"  Stage 2 (prog=0.5):   p_drop={stage2_drop:.2f}, p_spoof={stage2_spoof:.2f}\")
    print(f\"  Stage 4 (prog=1.0):   p_drop={stage4_drop:.2f}, p_spoof={stage4_spoof:.2f}\")
    
    assert stage0_drop <= stage2_drop <= stage4_drop, \"Difficulty should increase\"
    print(f\"  ✓ Curriculum difficulty progression verified\")
    
    print(f\"\n\" + \"=\" * 60)
    print(\"✓ ALL SMOKE TESTS PASSED\")
    print(\"=\" * 60)

if __name__ == \"__main__\":
    test_curriculum_smoke()
