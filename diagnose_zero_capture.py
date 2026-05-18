"""
diagnose_zero_capture.py
========================
Paste this into d:/Sentrinet/ and run:
    python diagnose_zero_capture.py

It tests 5 hypotheses in order and stops at the first one that explains 0% capture.
"""

import numpy as np
import torch
from pathlib import Path
import re

# ── helpers ────────────────────────────────────────────────────────────────

CKPT = "checkpoints/system_C_seed0"

def resolve_ckpt(path_str):
    p = Path(path_str)
    if p.is_file(): return str(p)
    for name in ["best.pt", "final.pt"]:
        if (p / name).exists(): return str(p / name)
    steps = sorted(p.glob("step_*.pt"),
                   key=lambda x: int(re.search(r"(\d+)", x.stem).group(1)))
    if steps: return str(steps[-1])
    raise FileNotFoundError(path_str)

LEGACY_KEY_MAP = {
    "fc1.weight": "obs_encoder.0.weight", "fc1.bias": "obs_encoder.0.bias",
    "fc2.weight": "core.0.weight", "fc2.bias": "core.0.bias",
    "fc_mean.weight": "mean_head.weight", "fc_mean.bias": "mean_head.bias",
    "net.0.weight": "obs_encoder.0.weight", "net.0.bias": "obs_encoder.0.bias",
    "net.2.weight": "core.0.weight", "net.2.bias": "core.0.bias",
}


def remap_policy_state_dict(state_dict):
    """Translate legacy checkpoint keys to the current PolicyNet layout."""
    if any(k.startswith("obs_encoder.") or k.startswith("core.") for k in state_dict):
        return state_dict
    return {LEGACY_KEY_MAP.get(k, k): v for k, v in state_dict.items()}

def load_policy(path):
    from networks import PolicyNet
    ckpt_path = resolve_ckpt(path)
    ck = torch.load(ckpt_path, map_location="cpu")
    config = ck.get("config", {}) if isinstance(ck.get("config", {}), dict) else {}
    sd = remap_policy_state_dict(ck["policy_state_dict"])

    if "obs_encoder.0.weight" in sd:
        obs_dim = int(sd["obs_encoder.0.weight"].shape[1])
    elif "net.0.weight" in sd:
        obs_dim = int(sd["net.0.weight"].shape[1])
    else:
        raise KeyError("Could not infer obs_dim from checkpoint state_dict")

    policy = PolicyNet(
        obs_dim=int(config.get("obs_dim", obs_dim)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        policy_type=str(config.get("policy_type", "mlp")),
    )
    policy.load_state_dict(sd)
    policy.eval()
    print(f"  Loaded: {ckpt_path}")
    print(f"  Policy obs_dim={policy.obs_dim}, step={ck.get('step','?')}")
    return policy, int(policy.obs_dim)

def make_env(p_drop=0.0, p_spoof=0.0, use_trust=True, capture_mode="sustained",
             sustained_steps=1, domain_rand=False, seed=0):
    from border_env import BorderEnv
    return BorderEnv(
        use_pybullet=False, domain_rand=domain_rand,
        p_drop=p_drop, p_spoof=p_spoof,
        use_trust=use_trust,
        capture_mode=capture_mode,
        sustained_steps=sustained_steps,
        seed=seed,
    )

def slice_obs(obs, obs_dim):
    if len(obs) >= obs_dim:
        return obs[:obs_dim]
    raise ValueError(f"obs len {len(obs)} < policy obs_dim {obs_dim}")

def run_episodes(policy, env, obs_dim, n=50, label=""):
    captures = 0
    min_dists = []
    for ep in range(n):
        obs, _ = env.reset(seed=42 + ep)
        done = False
        ep_min_dist = 999.0
        while not done:
            actions = {}
            for i in range(3):
                o = slice_obs(obs[f"drone_{i}"], obs_dim)
                a, _ = policy.get_action(o, deterministic=True)
                actions[f"drone_{i}"] = a
            actions["sensor_0"] = 1 if obs["sensor_0"][0] > 0.5 else 0
            obs, rew, term, trunc, info = env.step(actions)
            done = any(term[f"drone_{i}"] or trunc[f"drone_{i}"] for i in range(3))
            dists = np.linalg.norm(
                env.drone_pos - env.intruder_pos, axis=1)
            ep_min_dist = min(ep_min_dist, float(np.min(dists)))
        if info["drone_0"].get("captured", False):
            captures += 1
        min_dists.append(ep_min_dist)
    rate = captures / n * 100
    print(f"  [{label}] captures={captures}/{n} ({rate:.0f}%)  "
          f"min_dist_mean={np.mean(min_dists):.2f}m  "
          f"min_dist_best={np.min(min_dists):.2f}m")
    return rate, np.mean(min_dists), np.min(min_dists)

# ── HYPOTHESIS 1: Policy produces non-random actions ───────────────────────

print("\n" + "="*60)
print("H1: Policy action entropy (random vs learned)")
print("="*60)

policy, obs_dim = load_policy(CKPT)
env = make_env()
obs, _ = env.reset(seed=0)

actions_det = []
actions_sto = []
for _ in range(200):
    o = slice_obs(obs[f"drone_0"], obs_dim)
    a_det, _ = policy.get_action(o, deterministic=True)
    a_sto, _ = policy.get_action(o, deterministic=False)
    actions_det.append(a_det)
    actions_sto.append(a_sto)

det_arr = np.array(actions_det)
sto_arr = np.array(actions_sto)
print(f"  Deterministic action mean: {det_arr.mean(axis=0).round(3)}")
print(f"  Deterministic action std:  {det_arr.std(axis=0).round(3)}")
print(f"  Stochastic action std:     {sto_arr.std(axis=0).round(3)}")

if det_arr.std() < 0.01:
    print("  *** DEGENERATE: Policy outputs near-constant actions → undertrained or collapsed")
else:
    print("  OK: Policy produces varied actions")

env.close()

# ── HYPOTHESIS 2: Drone actually moves toward intruder ─────────────────────

print("\n" + "="*60)
print("H2: Drone approaches intruder (min distance over episode)")
print("="*60)

env = make_env(capture_mode="sustained", sustained_steps=1)
obs, _ = env.reset(seed=0)

print(f"  Initial intruder pos: {env.intruder_pos.round(2)}")
print(f"  Initial drone 0 pos:  {env.drone_pos[0].round(2)}")
print(f"  Initial dist:         {np.linalg.norm(env.drone_pos[0]-env.intruder_pos):.2f}m")

dists_over_time = []
for step in range(500):
    if not env.agents: break
    actions = {}
    for i in range(3):
        o = slice_obs(obs[f"drone_{i}"], obs_dim)
        a, _ = policy.get_action(o, deterministic=True)
        actions[f"drone_{i}"] = a
    actions["sensor_0"] = 0
    obs, rew, term, trunc, info = env.step(actions)
    min_d = float(np.min(np.linalg.norm(env.drone_pos - env.intruder_pos, axis=1)))
    dists_over_time.append(min_d)

print(f"  Min dist achieved: {np.min(dists_over_time):.2f}m  (capture needs <2.0m)")
print(f"  Final dist:        {dists_over_time[-1]:.2f}m")
print(f"  Dist at step 50:   {dists_over_time[49]:.2f}m")
print(f"  Dist at step 100:  {dists_over_time[99]:.2f}m")
print(f"  Dist at step 200:  {dists_over_time[199]:.2f}m")

if np.min(dists_over_time) < 2.0:
    print("  OK: Drone reached capture range — capture_mode/condition is the bug")
elif np.min(dists_over_time) < 5.0:
    print("  PARTIAL: Drone gets close but not inside 2.0m — reward shaping issue")
else:
    print("  *** FAIL: Drone never approaches — policy doesn't pursue")

env.close()

# ── HYPOTHESIS 3: Capture condition fires correctly ────────────────────────

print("\n" + "="*60)
print("H3: Capture condition unit test (force drone inside 2.0m)")
print("="*60)

from border_env import BorderEnv, CAPTURE_R
env2 = BorderEnv(use_pybullet=False, capture_mode="sustained", sustained_steps=1, seed=0)
env2.reset()

# Manually place drone 0 on top of intruder
env2.drone_pos[0] = env2.intruder_pos.copy()
captured, n_close, dists = env2._capture_status()
print(f"  CAPTURE_R = {CAPTURE_R}m")
print(f"  Drone placed at intruder pos, dist={dists[0]:.4f}m")
print(f"  captured={captured}, n_close={n_close}")

if not captured:
    print("  *** BUG: Capture condition failed even with drone AT intruder position!")
    print(f"  capture_mode={env2.capture_mode}, sustained_steps={env2.sustained_steps}")
    print(f"  _capture_counters={env2._capture_counters}")
else:
    print("  OK: Capture condition fires correctly")

env2.close()

# ── HYPOTHESIS 4: Reward actually incentivises approach ────────────────────

print("\n" + "="*60)
print("H4: Reward signal for approach (2 steps, moving toward intruder)")
print("="*60)

env3 = BorderEnv(use_pybullet=False, domain_rand=False, capture_mode="sustained",
                 sustained_steps=1, seed=0)
obs3, _ = env3.reset()

# Step 1: random baseline
actions_rand = {a: env3.action_space(a).sample() for a in env3.agents}
obs3, rew3, _, _, info3 = env3.step(actions_rand)
mean_rew_rand = np.mean([rew3[f"drone_{i}"] for i in range(3)])

# Get position of intruder relative to drone 0
rel = env3.intruder_pos - env3.drone_pos[0]
direction = rel / (np.linalg.norm(rel) + 1e-8)

# Step 2: move directly toward intruder
actions_pursuit = {
    "drone_0": (direction * 0.8).astype(np.float32),
    "drone_1": (direction * 0.8).astype(np.float32),
    "drone_2": (direction * 0.8).astype(np.float32),
    "sensor_0": 0,
}
obs3, rew3_pursuit, _, _, info3 = env3.step(actions_pursuit)
mean_rew_pursuit = np.mean([rew3_pursuit[f"drone_{i}"] for i in range(3)])

print(f"  Mean reward (random action):  {mean_rew_rand:.4f}")
print(f"  Mean reward (toward intruder):{mean_rew_pursuit:.4f}")

if mean_rew_pursuit <= mean_rew_rand:
    print("  *** BUG: Approaching intruder gives no better reward than random!")
    print("  Reward shaping is not driving approach behavior.")
else:
    print(f"  OK: Approach rewarded better by {mean_rew_pursuit - mean_rew_rand:.4f}")

env3.close()

# ── HYPOTHESIS 5: Capture rate sweep across conditions ─────────────────────

print("\n" + "="*60)
print("H5: Capture rate across conditions (20 eps each)")
print("="*60)

conditions = [
    dict(capture_mode="sustained", sustained_steps=1, p_drop=0.0, p_spoof=0.0,
         use_trust=True,  domain_rand=False, label="sustained-1  clean    trust"),
    dict(capture_mode="sustained", sustained_steps=3, p_drop=0.0, p_spoof=0.0,
         use_trust=True,  domain_rand=False, label="sustained-3  clean    trust"),
    dict(capture_mode="team",      sustained_steps=3, p_drop=0.0, p_spoof=0.0,
         use_trust=True,  domain_rand=False, label="team         clean    trust"),
    dict(capture_mode="sustained", sustained_steps=1, p_drop=0.2, p_spoof=0.1,
         use_trust=True,  domain_rand=False, label="sustained-1  adversarial trust"),
    dict(capture_mode="sustained", sustained_steps=1, p_drop=0.0, p_spoof=0.0,
         use_trust=False, domain_rand=False, label="sustained-1  clean    no-trust"),
]

for cond in conditions:
    label = cond.pop("label")
    env_c = make_env(**cond, seed=0)
    policy_c, obs_dim_c = load_policy(CKPT)
    rate, mean_d, best_d = run_episodes(policy_c, env_c, obs_dim_c, n=20, label=label)
    env_c.close()

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("="*60)
print("Interpretation guide:")
print("  H1 degenerate  → policy collapsed during training, must retrain")
print("  H2 >5m always  → policy doesn't pursue, check reward sign / obs mismatch")
print("  H2 <2m but H3  → capture condition bug in border_env.py")
print("  H4 bad reward  → reward shaping pushing away not toward")
print("  H5 all 0%      → policy fundamentally broken")
print("  H5 some >0%    → specific condition is the blocker, fix that config")