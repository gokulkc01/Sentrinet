"""
run_trained.py — Run a trained SentryNet policy
=================================================
Loads a MAPPO checkpoint and runs episodes with the trained policy.

Modes:
  --mode stats   : Run N episodes headless, print capture statistics (default)
  --mode visual  : Run with PyBullet 3D visualization

Usage:
  python run_trained.py --checkpoint checkpoints/system_C/final.pt
  python run_trained.py --checkpoint checkpoints/system_C/final.pt --mode visual
  python run_trained.py --checkpoint checkpoints/system_C/final.pt --episodes 100 --p_drop 0.3
"""

import argparse
from pathlib import Path
import re
import numpy as np
import torch
import time

from networks import PolicyNet
from border_env import BorderEnv

N_DRONES = 3


LEGACY_POLICY_KEY_MAP = {
    "fc1.weight": "net.0.weight",
    "fc1.bias": "net.0.bias",
    "fc2.weight": "net.2.weight",
    "fc2.bias": "net.2.bias",
    "fc_mean.weight": "mean_head.weight",
    "fc_mean.bias": "mean_head.bias",
}


def infer_policy_obs_dim(state_dict) -> int:
    """Infer the policy observation dimension from the first linear layer."""
    if "net.0.weight" in state_dict:
        return int(state_dict["net.0.weight"].shape[1])
    if "fc1.weight" in state_dict:
        return int(state_dict["fc1.weight"].shape[1])
    # fallback: find the first 2D weight tensor and use its input dim
    for k, v in state_dict.items():
        try:
            if hasattr(v, "ndim") and v.ndim == 2:
                return int(v.shape[1])
        except Exception:
            continue
    raise KeyError("Could not infer PolicyNet input dimension from checkpoint")


def slice_obs_for_policy(obs, policy: PolicyNet):
    """Trim observations to the policy's expected width for legacy checkpoints."""
    obs_dim = int(getattr(policy, "obs_dim", len(obs)))
    if len(obs) > obs_dim:
        return obs[:obs_dim]
    return obs


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    """Sort checkpoints by numeric step, not filename string order."""
    match = re.search(r"step_(\d+)\.pt$", path.name)
    if match:
        return (int(match.group(1)), path.name)
    if path.name == "final.pt":
        return (-1, path.name)
    return (-2, path.name)


def resolve_checkpoint_path(checkpoint: str) -> str:
    """Resolve checkpoint input to a concrete .pt file path.

    Supports:
      - direct file path to .pt
      - directory path containing final.pt or step_*.pt
    """
    p = Path(checkpoint)
    if p.is_file():
        return str(p)

    if p.is_dir():
        best_ckpt = p / "best.pt"
        if best_ckpt.exists():
            return str(best_ckpt)
        final_ckpt = p / "final.pt"
        if final_ckpt.exists():
            return str(final_ckpt)
        step_ckpts = sorted(p.glob("step_*.pt"), key=checkpoint_sort_key)
        if step_ckpts:
            return str(step_ckpts[-1])

    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint}. "
        "Pass a .pt file or a run directory containing final.pt / step_*.pt"
    )


def load_policy(checkpoint_path: str, device: str = "cpu") -> PolicyNet:
    """Load a trained PolicyNet from checkpoint."""
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {}) if isinstance(ckpt.get("config", {}), dict) else {}

    state_dict = ckpt["policy_state_dict"]
    if not any(key in state_dict for key in LEGACY_POLICY_KEY_MAP.values()):
        remapped_state_dict = {
            LEGACY_POLICY_KEY_MAP.get(key, key): value for key, value in state_dict.items()
        }
        state_dict = remapped_state_dict

    policy_type = str(config.get("policy_type", "mlp")).lower()
    obs_dim = int(config.get("obs_dim", infer_policy_obs_dim(state_dict)))
    hidden_dim = int(config.get("hidden_dim", 128))
    policy = PolicyNet(obs_dim=obs_dim, hidden_dim=hidden_dim, policy_type=policy_type).to(device)

    policy.load_state_dict(state_dict)
    policy.eval()
    step = ckpt.get("total_steps", ckpt.get("step", "?"))
    step_str = f"{step:,}" if isinstance(step, int) else str(step)
    print(f"[Loaded] {checkpoint_path} (trained for {step_str} steps)")
    return policy


def get_drone_actions(policy, obs_dict, policy_state=None, device="cpu", deterministic=True):
    """Get actions for all drones from the trained policy."""
    action_dict = {}
    next_state = None
    if policy.is_recurrent:
        next_state = {}
    for i in range(N_DRONES):
        obs_i = slice_obs_for_policy(obs_dict[f"drone_{i}"], policy)
        if policy.is_recurrent:
            hidden_in = policy_state[f"drone_{i}"]
            action, _, hidden_out = policy.step(obs_i, deterministic=deterministic, hidden_state=hidden_in)
            action_dict[f"drone_{i}"] = action
            if policy.policy_type == "lstm":
                next_state[f"drone_{i}"] = (hidden_out[0].squeeze(0), hidden_out[1].squeeze(0))
            else:
                next_state[f"drone_{i}"] = hidden_out.squeeze(0)
        else:
            action, _ = policy.get_action(obs_i, deterministic=deterministic)
            action_dict[f"drone_{i}"] = action

    # Sensor: reactive rule (trigger when alert detected)
    action_dict["sensor_0"] = 1 if obs_dict["sensor_0"][0] > 0.5 else 0
    return action_dict, next_state


# ─────────────────────────────────────────────────────────────────────────────
#  Stats mode — headless evaluation
# ─────────────────────────────────────────────────────────────────────────────
def run_stats(policy, args):
    """Run N episodes and print capture statistics."""
    env = BorderEnv(
        use_pybullet=False,
        domain_rand=bool(args.domain_rand),
        p_drop=args.p_drop,
        p_spoof=args.p_spoof,
        use_trust=args.use_trust,
        capture_mode=args.capture_mode,
        sustained_steps=args.sustained_steps,
        capture_k=args.capture_k,
        seed=args.seed,
    )

    captures = 0
    total_steps = 0
    rewards_all = []
    n_close_all = []
    team_distance_all = []
    formation_all = []
    coverage_all = []
    policy_state = None
    if policy.is_recurrent:
        policy_state = {f"drone_{i}": policy.init_hidden(1) for i in range(N_DRONES)}

    print(f"\nRunning {args.episodes} episodes  |  p_drop={args.p_drop}  "
          f"p_spoof={args.p_spoof}  use_trust={args.use_trust}  capture_mode={args.capture_mode}\n")

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        if policy.is_recurrent:
            policy_state = {f"drone_{i}": policy.init_hidden(1) for i in range(N_DRONES)}
        ep_reward = 0.0
        step = 0

        while env.agents:
            action_dict, policy_state = get_drone_actions(policy, obs, policy_state=policy_state, deterministic=True)
            obs, rew, term, trunc, info = env.step(action_dict)
            ep_reward += sum(rew[f"drone_{i}"] for i in range(N_DRONES)) / N_DRONES
            step += 1
            info0 = info.get("drone_0", {})
            n_close_all.append(float(info0.get("n_close", 0)))
            team_distance_all.append(float(info0.get("mean_team_distance", 0.0)))
            formation_all.append(float(info0.get("formation_spread", 0.0)))
            coverage_all.append(float(info0.get("angular_coverage_score", 0.0)))

        captured = info.get("drone_0", {}).get("captured", False)
        if captured:
            captures += 1
        total_steps += step
        rewards_all.append(ep_reward)

        if ep % 20 == 0 or ep == args.episodes:
            rate = captures / ep * 100
            print(f"  Episode {ep:>4}/{args.episodes}  |  "
                  f"Captures: {captures:>3}  |  Rate: {rate:5.1f}%  |  "
                  f"Avg reward: {np.mean(rewards_all):>7.2f}  |  "
                                    f"Avg length: {total_steps/ep:>5.0f}  |  "
                                    f"NClose: {np.mean(n_close_all):>4.2f}  |  "
                                    f"TeamDist: {np.mean(team_distance_all):>5.2f}  |  "
                                    f"Form: {np.mean(formation_all):>5.2f}  |  "
                                    f"Cov: {np.mean(coverage_all):>5.2f}")

    rate = captures / args.episodes * 100
    print(f"\n{'='*55}")
    print(f"  RESULTS  ({args.episodes} episodes)")
    print(f"{'='*55}")
    print(f"  Capture rate : {rate:.1f}%  ({captures}/{args.episodes})")
    print(f"  Avg reward   : {np.mean(rewards_all):.2f}")
    print(f"  Avg length   : {total_steps / args.episodes:.0f} steps")
    print(f"  Mean n_close : {np.mean(n_close_all):.2f}")
    print(f"  Mean team dist: {np.mean(team_distance_all):.2f}")
    print(f"  Mean formation: {np.mean(formation_all):.2f}")
    print(f"  Mean coverage : {np.mean(coverage_all):.2f}")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Visual mode — PyBullet 3D rendering
# ─────────────────────────────────────────────────────────────────────────────
def run_visual(policy, args):
    """Run episodes with PyBullet visualization."""
    import pybullet as p
    import os
    import gym_pybullet_drones as g

    ASSETS_DIR = os.path.join(os.path.dirname(g.__file__), 'assets')
    CF2X_URDF  = os.path.join(ASSETS_DIR, 'cf2x.urdf')
    RACER_URDF = os.path.join(ASSETS_DIR, 'racer.urdf')

    HUNTER_SCALE = 17.5
    TARGET_SCALE = 8.0
    DRONE_COLORS = [
        [0.93, 0.32, 0.28, 1.0],
        [0.14, 0.72, 0.48, 1.0],
        [0.20, 0.50, 0.93, 1.0],
    ]
    TARGET_COLOR = [0.99, 0.80, 0.18, 1.0]

    env = BorderEnv(
        use_pybullet=True,
        render_mode='human',
        domain_rand=bool(args.domain_rand),
        p_drop=args.p_drop,
        p_spoof=args.p_spoof,
        use_trust=args.use_trust,
        capture_mode=args.capture_mode,
        sustained_steps=args.sustained_steps,
        capture_k=args.capture_k,
    )

    episode = 0
    print(f"\nVisualizing trained policy  |  p_drop={args.p_drop}  "
        f"p_spoof={args.p_spoof}  use_trust={args.use_trust}  capture_mode={args.capture_mode}")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            episode += 1
            obs, _ = env.reset()
            policy_state = None
            if policy.is_recurrent:
                policy_state = {f"drone_{i}": policy.init_hidden(1) for i in range(3)}
            pb = env._pb

            # Camera & world
            p.resetDebugVisualizerCamera(29, 38, -28, [10, 10, 3], physicsClientId=pb)
            corners = [([0,0,0],[20,0,0]),([20,0,0],[20,20,0]),
                       ([20,20,0],[0,20,0]),([0,20,0],[0,0,0])]
            for a, b in corners:
                p.addUserDebugLine(a, b, [0.98,0.60,0.16], 3, physicsClientId=pb)

            # Load drone models
            drone_ids = []
            for i in range(3):
                did = p.loadURDF(CF2X_URDF, env.drone_pos[i].tolist(),
                                p.getQuaternionFromEuler([0,0,0]),
                                physicsClientId=pb, globalScaling=HUNTER_SCALE)
                if DRONE_COLORS[i]:
                    for link in range(-1, p.getNumJoints(did, physicsClientId=pb)):
                        p.changeVisualShape(did, link, rgbaColor=DRONE_COLORS[i],
                                           physicsClientId=pb)
                        p.setCollisionFilterGroupMask(did, link, 0, 0, physicsClientId=pb)
                p.changeDynamics(did, -1, mass=0.0, physicsClientId=pb)
                drone_ids.append(did)

            intruder_id = p.loadURDF(RACER_URDF, env.intruder_pos.tolist(),
                                    p.getQuaternionFromEuler([0,0,0]),
                                    physicsClientId=pb, globalScaling=TARGET_SCALE)
            for link in range(-1, p.getNumJoints(intruder_id, physicsClientId=pb)):
                p.changeVisualShape(intruder_id, link, rgbaColor=TARGET_COLOR,
                                   physicsClientId=pb)
                p.setCollisionFilterGroupMask(intruder_id, link, 0, 0, physicsClientId=pb)
            p.changeDynamics(intruder_id, -1, mass=0.0, physicsClientId=pb)

            # Labels
            for i, did in enumerate(drone_ids):
                p.addUserDebugText(f'H{i}', [0,0,1.15],
                                  textColorRGB=DRONE_COLORS[i][:3],
                                  textSize=1.1, parentObjectUniqueId=did,
                                  physicsClientId=pb)
            p.addUserDebugText('TGT', [0,0,1.35], textColorRGB=TARGET_COLOR[:3],
                              textSize=1.1, parentObjectUniqueId=intruder_id,
                              physicsClientId=pb)

            print(f"\n── Episode {episode} ──")
            print(f"   Intruder at: {env.intruder_pos.round(2)}  "
                  f"Speed: {env.intruder_speed:.1f} m/s  "
                  f"Wind: {env.wind_vec.round(1)} m/s")

            step = 0
            while env.agents:
                action_dict, policy_state = get_drone_actions(policy, obs, policy_state=policy_state, deterministic=True)
                obs, rew, term, trunc, info = env.step(action_dict)
                step += 1

                # Update visual positions
                for i in range(3):
                    vx, vy = float(env.drone_vel[i][0]), float(env.drone_vel[i][1])
                    orn = p.getQuaternionFromEuler([
                        float(np.clip(-vy*0.12, -0.4, 0.4)),
                        float(np.clip( vx*0.12, -0.4, 0.4)), 0])
                    p.resetBasePositionAndOrientation(
                        drone_ids[i], env.drone_pos[i].tolist(), orn,
                        physicsClientId=pb)

                p.resetBasePositionAndOrientation(
                    intruder_id, env.intruder_pos.tolist(),
                    p.getQuaternionFromEuler([0,0,0]), physicsClientId=pb)

                time.sleep(0.02)

            captured = info.get("drone_0", {}).get("captured", False)
            result = "CAPTURED!" if captured else f"Timeout ({step} steps)"
            print(f"   Result: {result}")

            # Cleanup models
            for bid in drone_ids + [intruder_id]:
                try: p.removeBody(bid, physicsClientId=pb)
                except: pass

    except KeyboardInterrupt:
        print("\nStopped.")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trained SentryNet policy")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--mode", type=str, default="stats", choices=["stats", "visual"],
                        help="stats = headless evaluation, visual = PyBullet 3D")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes (stats mode, default: 100)")
    parser.add_argument("--p_drop", type=float, default=0.0,
                        help="Packet drop rate during evaluation (default: 0.0)")
    parser.add_argument("--p_spoof", type=float, default=0.0,
                        help="Spoof rate during evaluation (default: 0.0)")
    parser.add_argument("--use_trust", action="store_true",
                        help="Enable trust-weighted aggregation")
    parser.add_argument("--capture-mode", choices=["team", "sustained"], default="team",
                        help="Capture rule used during evaluation")
    parser.add_argument("--sustained-steps", type=int, default=3,
                        help="Required consecutive in-range steps for sustained mode")
    parser.add_argument("--capture-k", type=int, default=2,
                        help="Number of drones required for team mode fallback or experiments")
    parser.add_argument("--domain-rand", dest="domain_rand", action="store_true", default=True,
                        help="Enable domain randomization during evaluation")
    parser.add_argument("--no-domain-rand", dest="domain_rand", action="store_false",
                        help="Disable domain randomization during evaluation")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")

    args = parser.parse_args()

    policy = load_policy(args.checkpoint)

    if args.mode == "stats":
        run_stats(policy, args)
    else:
        run_visual(policy, args)
