"""
simulation_trial.py
===================
Cleaner PyBullet visualizer for SentryNet.

By default this script can either:
  - run a trained checkpoint if one is provided
  - fall back to random actions for debugging/demo use

Examples:
  python simulation_trial.py --checkpoint checkpoints/system_A_seed0
  python simulation_trial.py --checkpoint checkpoints/system_C_seed0 --p_drop 0.2 --use_trust
  python simulation_trial.py --random
"""

from __future__ import annotations

import argparse
import os
import time

import gym_pybullet_drones as g
import numpy as np
import pybullet as p

from border_env import BorderEnv
from run_trained import get_drone_actions, load_policy


HUNTER_SCALE = 10.5
TARGET_SCALE = 8.0
DEFAULT_SPEED = 0.02
DRONE_COLORS = [
    [0.93, 0.32, 0.28, 1.0],
    [0.14, 0.72, 0.48, 1.0],
    [0.20, 0.50, 0.93, 1.0],
]
TARGET_COLOR = [0.99, 0.80, 0.18, 1.0]

ASSETS_DIR = os.path.join(os.path.dirname(g.__file__), "assets")
CF2X_URDF = os.path.join(ASSETS_DIR, "cf2x.urdf")
RACER_URDF = os.path.join(ASSETS_DIR, "racer.urdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SentryNet episodes with cleaner visuals")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint file or run directory. If omitted, uses random actions.")
    parser.add_argument("--random", action="store_true",
                        help="Force random actions even if a checkpoint is provided.")
    parser.add_argument("--episodes", type=int, default=0,
                        help="Number of episodes to run. 0 means loop forever.")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                        help="Delay between frames in seconds.")
    parser.add_argument("--p_drop", type=float, default=0.0,
                        help="Packet drop rate for the communication channel.")
    parser.add_argument("--p_spoof", type=float, default=0.0,
                        help="Spoofing rate for the communication channel.")
    parser.add_argument("--use_trust", action="store_true",
                        help="Enable trust-weighted aggregation.")
    parser.add_argument("--domain-rand", action="store_true",
                        help="Enable domain randomization per episode.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed.")
    return parser.parse_args()


def load_drone(pb: int, urdf_path: str, position, color, scale: float) -> int:
    body_id = p.loadURDF(
        urdf_path,
        basePosition=position,
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        physicsClientId=pb,
        globalScaling=scale,
    )
    for link_idx in range(-1, p.getNumJoints(body_id, physicsClientId=pb)):
        p.changeVisualShape(body_id, link_idx, rgbaColor=color, physicsClientId=pb)
        p.setCollisionFilterGroupMask(body_id, link_idx, 0, 0, physicsClientId=pb)
    p.changeDynamics(body_id, -1, mass=0.0, physicsClientId=pb)
    return body_id


def set_camera(pb: int) -> None:
    p.resetDebugVisualizerCamera(
        cameraDistance=29,
        cameraYaw=38,
        cameraPitch=-28,
        cameraTargetPosition=[10, 10, 3.0],
        physicsClientId=pb,
    )


def draw_world(pb: int) -> None:
    corners = [
        ([0, 0, 0], [20, 0, 0]),
        ([20, 0, 0], [20, 20, 0]),
        ([20, 20, 0], [0, 20, 0]),
        ([0, 20, 0], [0, 0, 0]),
    ]
    for a, b in corners:
        p.addUserDebugLine(a, b, lineColorRGB=[0.98, 0.60, 0.16], lineWidth=3, physicsClientId=pb)

    for x, y in ([0, 0], [20, 0], [20, 20], [0, 20]):
        p.addUserDebugLine(
            [x, y, 0], [x, y, 10],
            lineColorRGB=[0.40, 0.40, 0.40],
            lineWidth=1,
            physicsClientId=pb,
        )


def add_labels(pb: int, drone_ids: list[int], intruder_id: int) -> None:
    for i, did in enumerate(drone_ids):
        p.addUserDebugText(
            f"H{i}",
            [0, 0, 1.15],
            textColorRGB=DRONE_COLORS[i][:3],
            textSize=1.1,
            parentObjectUniqueId=did,
            physicsClientId=pb,
        )
    p.addUserDebugText(
        "TGT",
        [0, 0, 1.35],
        textColorRGB=TARGET_COLOR[:3],
        textSize=1.1,
        parentObjectUniqueId=intruder_id,
        physicsClientId=pb,
    )


def remove_bodies(pb: int, ids: list[int]) -> None:
    for bid in ids:
        try:
            p.removeBody(bid, physicsClientId=pb)
        except Exception:
            pass


def build_actions(env: BorderEnv, obs, policy):
    if policy is None:
        return {a: env.action_space(a).sample() for a in env.agents}
    return get_drone_actions(policy, obs, deterministic=True)


def main() -> None:
    args = parse_args()
    policy = None if args.random or not args.checkpoint else load_policy(args.checkpoint)

    env = BorderEnv(
        use_pybullet=True,
        render_mode="human",
        domain_rand=bool(args.domain_rand),
        p_drop=float(args.p_drop),
        p_spoof=float(args.p_spoof),
        use_trust=bool(args.use_trust),
        seed=args.seed,
    )

    mode = "random policy" if policy is None else "trained policy"
    print("=" * 58)
    print("  SentryNet Visualizer")
    print("=" * 58)
    print(f"  Control mode  : {mode}")
    print(f"  Attacks       : drop={args.p_drop}, spoof={args.p_spoof}")
    print(f"  Trust         : {args.use_trust}")
    print(f"  Domain rand   : {args.domain_rand}")
    print(f"  Frame delay   : {args.speed}s")
    print("=" * 58)
    print("  Press Ctrl+C to stop\n")

    episode = 0
    try:
        while args.episodes == 0 or episode < args.episodes:
            episode += 1
            obs, _ = env.reset()
            pb = env._pb

            set_camera(pb)
            draw_world(pb)

            drone_ids = [
                load_drone(pb, CF2X_URDF, env.drone_pos[i].tolist(), DRONE_COLORS[i], HUNTER_SCALE)
                for i in range(3)
            ]
            intruder_id = load_drone(pb, RACER_URDF, env.intruder_pos.tolist(), TARGET_COLOR, TARGET_SCALE)
            add_labels(pb, drone_ids, intruder_id)

            print(
                f"Episode {episode:>3} | intruder={env.intruder_pos.round(2)} "
                f"| wind={env.wind_vec.round(2)} | speed={env.intruder_speed:.2f}"
            )

            step = 0
            while env.agents:
                actions = build_actions(env, obs, policy)
                obs, rew, term, trunc, info = env.step(actions)
                step += 1

                for i in range(3):
                    vx = float(env.drone_vel[i][0])
                    vy = float(env.drone_vel[i][1])
                    roll = float(np.clip(-vy * 0.12, -0.4, 0.4))
                    pitch = float(np.clip(vx * 0.12, -0.4, 0.4))
                    orn = p.getQuaternionFromEuler([roll, pitch, 0])
                    p.resetBasePositionAndOrientation(
                        drone_ids[i], env.drone_pos[i].tolist(), orn, physicsClientId=pb
                    )

                p.resetBasePositionAndOrientation(
                    intruder_id,
                    env.intruder_pos.tolist(),
                    [0, 0, 0, 1],
                    physicsClientId=pb,
                )

                if args.speed > 0:
                    time.sleep(args.speed)

            captured = bool(info["drone_0"]["captured"])
            result = "CAPTURED" if captured else f"TIMEOUT ({step} steps)"
            print(f"  Result: {result}\n")

            remove_bodies(pb, drone_ids + [intruder_id])
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
