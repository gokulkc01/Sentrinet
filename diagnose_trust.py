"""
diagnose_trust.py — Verify trust mechanism + checkpoint health
================================================================
Run: python diagnose_trust.py
"""
import numpy as np
import torch
from border_env import BorderEnv
from networks import PolicyNet
from pathlib import Path
import re


def resolve_ckpt(path_str):
    p = Path(path_str)
    if p.is_file():
        return str(p)
    if p.is_dir():
        for name in ["best.pt", "final.pt"]:
            if (p / name).exists():
                return str(p / name)
        steps = sorted(p.glob("step_*.pt"),
                       key=lambda x: int(re.search(r"(\d+)", x.stem).group(1)))
        if steps:
            return str(steps[-1])
    raise FileNotFoundError(f"No checkpoint found at {path_str}")


def load_policy(ckpt_path):
    LEGACY_KEY_MAP = {
        "fc1.weight": "net.0.weight", "fc1.bias": "net.0.bias",
        "fc2.weight": "net.2.weight", "fc2.bias": "net.2.bias",
        "fc_mean.weight": "mean_head.weight", "fc_mean.bias": "mean_head.bias",
    }
    ckpt_path = resolve_ckpt(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["policy_state_dict"]
    if not any(k in sd for k in LEGACY_KEY_MAP.values()):
        sd = {LEGACY_KEY_MAP.get(k, k): v for k, v in sd.items()}
    obs_dim = int(sd["net.0.weight"].shape[1] if "net.0.weight" in sd else sd["fc1.weight"].shape[1])
    policy = PolicyNet(obs_dim=obs_dim)
    policy.load_state_dict(sd)
    policy.eval()
    step = ckpt.get("step", "?")
    print(f"[Loaded] {ckpt_path} ({step:,} steps)" if isinstance(step, int)
          else f"[Loaded] {ckpt_path}")
    return policy


def slice_obs_for_policy(obs, policy):
    obs_dim = int(getattr(policy, "obs_dim", len(obs)))
    if len(obs) > obs_dim:
        return obs[:obs_dim]
    return obs


def run_episodes(policy, env, n_episodes=20, label=""):
    captures = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=42 + ep)
        done = False
        while not done:
            actions = {}
            for i in range(3):
                a, _ = policy.get_action(slice_obs_for_policy(obs[f"drone_{i}"], policy), deterministic=True)
                actions[f"drone_{i}"] = a
            actions["sensor_0"] = 1 if obs["sensor_0"][0] > 0.5 else 0
            obs, rew, term, trunc, info = env.step(actions)
            done = any(term[f"drone_{i}"] or trunc[f"drone_{i}"] for i in range(3))
        captured = info.get("drone_0", {}).get("captured", False)
        if captured:
            captures += 1
    rate = captures / n_episodes * 100
    print(f"  {label}: {captures}/{n_episodes} = {rate:.0f}% capture rate")
    return rate


def test_trust_mechanism():
    """Verify trust scores converge correctly for honest vs compromised senders."""
    from trust_module import TrustModule

    print("\n" + "="*60)
    print("TEST 1: Trust Module Convergence")
    print("="*60)

    tm = TrustModule(n_senders=2)
    true_pos = np.array([10.0, 10.0, 3.0])

    # Simulate: sender 0 = honest, sender 1 = compromised (always spoofed)
    for step in range(200):
        honest_msg = true_pos + np.random.normal(0, 0.1, 3)  # tiny noise
        spoofed_msg = true_pos + np.random.normal(0, 2.0, 3)  # big noise

        received = np.array([honest_msg, spoofed_msg])
        dropped = np.array([False, False])

        tm.update(received, true_pos, dropped)

        if step in [0, 10, 50, 100, 199]:
            scores = tm.get_trust_scores()
            print(f"  Step {step:3d}: honest_trust={scores[0]:.3f}  "
                  f"compromised_trust={scores[1]:.3f}  "
                  f"gap={scores[0]-scores[1]:.3f}")

    scores = tm.get_trust_scores()
    print(f"\n  Final: honest={scores[0]:.3f}, compromised={scores[1]:.3f}")
    if scores[0] > scores[1] + 0.2:
        print("  ✓ Trust correctly differentiates honest vs compromised")
    else:
        print("  ✗ Trust FAILS to differentiate — gap too small!")

    # Now test with drops mixed in
    print("\n  --- With 30% drops ---")
    tm.reset()
    for step in range(200):
        honest_msg = true_pos + np.random.normal(0, 0.1, 3)
        spoofed_msg = true_pos + np.random.normal(0, 2.0, 3)
        received = np.array([honest_msg, spoofed_msg])
        dropped = np.array([np.random.random() < 0.3, np.random.random() < 0.3])
        tm.update(received, true_pos, dropped)

    scores = tm.get_trust_scores()
    print(f"  Final: honest={scores[0]:.3f}, compromised={scores[1]:.3f}, "
          f"gap={scores[0]-scores[1]:.3f}")


def test_checkpoint_health():
    """Test each system's checkpoint WITHOUT any adversary."""
    print("\n" + "="*60)
    print("TEST 2: Checkpoint Health (no adversary, no spoofing)")
    print("="*60)

    for system in ["A", "B", "C"]:
        ckpt_dir = f"checkpoints/system_{system}_seed0"
        if not Path(ckpt_dir).exists():
            print(f"  System {system}: checkpoint not found, skipping")
            continue
        policy = load_policy(ckpt_dir)
        use_trust = (system == "C")
        env = BorderEnv(
            use_pybullet=False, domain_rand=False,
            p_drop=0.0, p_spoof=0.0,
            use_trust=use_trust, seed=42,
        )
        run_episodes(policy, env, n_episodes=20,
                     label=f"System {system} (clean channel)")
        env.close()


def test_with_compromised_drone():
    """Test each system WITH compromised drone."""
    print("\n" + "="*60)
    print("TEST 3: With Compromised Drone 1 (targeted adversary)")
    print("="*60)

    for system in ["A", "B", "C"]:
        ckpt_dir = f"checkpoints/system_{system}_seed0"
        if not Path(ckpt_dir).exists():
            continue
        policy = load_policy(ckpt_dir)
        use_trust = (system == "C")
        env = BorderEnv(
            use_pybullet=False, domain_rand=False,
            p_drop=0.2, p_spoof=0.1,
            use_trust=use_trust,
            compromised_drone=1,
            seed=42,
        )
        run_episodes(policy, env, n_episodes=20,
                     label=f"System {system} (compromised drone, drop=0.2)")
        env.close()


if __name__ == "__main__":
    test_trust_mechanism()
    test_checkpoint_health()
    test_with_compromised_drone()
    print("\n" + "="*60)
    print("DONE")
    print("="*60)
