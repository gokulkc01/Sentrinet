"""
dashboard.py — SentryNet Real-Time Dashboard
=============================================
Run: python dashboard.py --checkpoint checkpoints/system_C_seed0
      python dashboard.py --checkpoint checkpoints/system_C_seed0 --use_trust --pybullet
Controls:
  A         — toggle attack (p_drop=0.4, p_spoof=0.15)
  UP/DOWN   — adjust drop rate ±0.05
  R         — reset episode
  P         — toggle PyBullet 3D window (if launched with --pybullet)
  ESC/Q     — quit
"""
import argparse, threading, time, re, sys, os
from pathlib import Path
import numpy as np
import torch
import pygame

from border_env import BorderEnv
from networks import PolicyNet

# ── Constants ──────────────────────────────────────────────────────────────
W, H = 1280, 720
FPS = 30
N_DRONES = 3
WORLD_XY = 20.0

# Colors
BG        = (15, 17, 23)
PANEL_BG  = (22, 25, 35)
BORDER    = (45, 50, 65)
TEXT      = (210, 215, 225)
TEXT_DIM  = (120, 128, 145)
ACCENT    = (88, 166, 255)
GREEN     = (46, 204, 113)
RED       = (231, 76, 60)
AMBER     = (241, 196, 15)
ORANGE    = (230, 126, 34)
WHITE     = (255, 255, 255)
DRONE_C   = [(216, 90, 48), (29, 158, 117), (55, 138, 221)]
INTRUDER_C= (241, 196, 15)

LEGACY_KEY_MAP = {
    "fc1.weight": "net.0.weight", "fc1.bias": "net.0.bias",
    "fc2.weight": "net.2.weight", "fc2.bias": "net.2.bias",
    "fc_mean.weight": "mean_head.weight", "fc_mean.bias": "mean_head.bias",
}


def infer_policy_obs_dim(sd):
    # Try common encoder weight keys first (newer checkpoints)
    for k in ("obs_encoder.0.weight", "obs_encoder.weight", "net.0.weight", "fc1.weight"):
        if k in sd:
            w = sd[k]
            if hasattr(w, "shape") and len(w.shape) >= 2:
                return int(w.shape[1])

    # For recurrent cores we sometimes have weight_ih with shape (4*hidden, input)
    for k in ("core.weight_ih", "rnn.weight_ih_l0", "core.weight_hh", "rnn.weight_hh_l0"):
        if k in sd:
            w = sd[k]
            if hasattr(w, "shape") and len(w.shape) >= 2:
                return int(w.shape[1])

    # Fallback: pick the first 2D weight whose input dim looks plausible
    for k, v in sd.items():
        if hasattr(v, "shape") and len(v.shape) == 2:
            in_dim = int(v.shape[1])
            if 4 <= in_dim <= 2048:
                return in_dim

    raise KeyError("Could not infer policy input dimension from checkpoint")

# ── Shared State ───────────────────────────────────────────────────────────
state = {
    "drone_pos": np.zeros((3, 3)), "drone_vel": np.zeros((3, 3)),
    "intruder_pos": np.zeros(3), "trust": np.ones((3, 2)),
    "battery": np.ones(3), "wind": np.zeros(3),
    "reward": 0.0, "step": 0, "episode": 1, "captured": False,
    "p_drop": 0.0, "p_spoof": 0.0, "sensor_alert": False,
    "capture_mode": "team",
    "ch_stats": {}, "reward_hist": [], "capture_hist": [],
    "metrics": {},
    "attack_on": False, "use_trust": True, "running": True,
    "reset_req": False, "speed": 0.02,
}
lock = threading.Lock()


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
    ckpt_path = resolve_ckpt(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt.get("config", {}) if isinstance(ckpt.get("config", {}), dict) else {}
    sd = ckpt["policy_state_dict"]
    if not any(k in sd for k in LEGACY_KEY_MAP.values()):
        sd = {LEGACY_KEY_MAP.get(k, k): v for k, v in sd.items()}
    policy = PolicyNet(
        obs_dim=int(config.get("obs_dim", infer_policy_obs_dim(sd))),
        hidden_dim=int(config.get("hidden_dim", 128)),
        policy_type=str(config.get("policy_type", "mlp")).lower(),
    )
    policy.load_state_dict(sd)
    policy.eval()
    step = ckpt.get("step", "?")
    print(f"[Loaded] {ckpt_path} ({step:,} steps)" if isinstance(step, int)
          else f"[Loaded] {ckpt_path}")
    return policy


# ── PyBullet 3D overlay helpers ────────────────────────────────────────────
def _setup_pybullet_episode(env):
    """Load URDF drone/intruder models into the PyBullet GUI for one episode."""
    import pybullet as p
    try:
        import gym_pybullet_drones as g
        ASSETS_DIR = os.path.join(os.path.dirname(g.__file__), 'assets')
        CF2X_URDF  = os.path.join(ASSETS_DIR, 'cf2x.urdf')
        RACER_URDF = os.path.join(ASSETS_DIR, 'racer.urdf')
    except ImportError:
        CF2X_URDF = RACER_URDF = "sphere_small.urdf"

    HUNTER_SCALE = 17.5
    TARGET_SCALE = 8.0
    PB_DRONE_COLORS = [
        [0.93, 0.32, 0.28, 1.0],
        [0.14, 0.72, 0.48, 1.0],
        [0.20, 0.50, 0.93, 1.0],
    ]
    TARGET_COLOR = [0.99, 0.80, 0.18, 1.0]

    pb = env._pb
    p.resetDebugVisualizerCamera(29, 38, -28, [10, 10, 3], physicsClientId=pb)
    corners = [([0,0,0],[20,0,0]),([20,0,0],[20,20,0]),
               ([20,20,0],[0,20,0]),([0,20,0],[0,0,0])]
    for a, b in corners:
        p.addUserDebugLine(a, b, [0.98,0.60,0.16], 3, physicsClientId=pb)

    drone_ids = []
    for i in range(N_DRONES):
        did = p.loadURDF(CF2X_URDF, env.drone_pos[i].tolist(),
                         p.getQuaternionFromEuler([0,0,0]),
                         physicsClientId=pb, globalScaling=HUNTER_SCALE)
        for link in range(-1, p.getNumJoints(did, physicsClientId=pb)):
            p.changeVisualShape(did, link, rgbaColor=PB_DRONE_COLORS[i],
                               physicsClientId=pb)
            p.setCollisionFilterGroupMask(did, link, 0, 0, physicsClientId=pb)
        p.changeDynamics(did, -1, mass=0.0, physicsClientId=pb)
        p.addUserDebugText(f'H{i}', [0,0,1.15],
                           textColorRGB=PB_DRONE_COLORS[i][:3],
                           textSize=1.1, parentObjectUniqueId=did,
                           physicsClientId=pb)
        drone_ids.append(did)

    intruder_id = p.loadURDF(RACER_URDF, env.intruder_pos.tolist(),
                             p.getQuaternionFromEuler([0,0,0]),
                             physicsClientId=pb, globalScaling=TARGET_SCALE)
    for link in range(-1, p.getNumJoints(intruder_id, physicsClientId=pb)):
        p.changeVisualShape(intruder_id, link, rgbaColor=TARGET_COLOR,
                           physicsClientId=pb)
        p.setCollisionFilterGroupMask(intruder_id, link, 0, 0, physicsClientId=pb)
    p.changeDynamics(intruder_id, -1, mass=0.0, physicsClientId=pb)
    p.addUserDebugText('TGT', [0,0,1.35], textColorRGB=TARGET_COLOR[:3],
                       textSize=1.1, parentObjectUniqueId=intruder_id,
                       physicsClientId=pb)

    return drone_ids, intruder_id


def _update_pybullet_positions(env, drone_ids, intruder_id):
    """Sync PyBullet visual model positions with env state."""
    import pybullet as p
    pb = env._pb
    for i in range(N_DRONES):
        vx = float(env.drone_vel[i][0])
        vy = float(env.drone_vel[i][1])
        orn = p.getQuaternionFromEuler([
            float(np.clip(-vy * 0.12, -0.4, 0.4)),
            float(np.clip( vx * 0.12, -0.4, 0.4)), 0])
        p.resetBasePositionAndOrientation(
            drone_ids[i], env.drone_pos[i].tolist(), orn,
            physicsClientId=pb)
    p.resetBasePositionAndOrientation(
        intruder_id, env.intruder_pos.tolist(),
        p.getQuaternionFromEuler([0,0,0]), physicsClientId=pb)


def _cleanup_pybullet_models(env, drone_ids, intruder_id):
    """Remove URDF bodies between episodes."""
    import pybullet as p
    pb = env._pb
    for bid in drone_ids + [intruder_id]:
        try:
            p.removeBody(bid, physicsClientId=pb)
        except Exception:
            pass


# ── Simulation Thread ──────────────────────────────────────────────────────
def sim_loop(policy, args):
    use_pb = getattr(args, 'pybullet', False)
    env = BorderEnv(
        use_pybullet=use_pb, domain_rand=True,
        render_mode='human' if use_pb else None,
        p_drop=args.p_drop, p_spoof=args.p_spoof,
        use_trust=args.use_trust, capture_mode=args.capture_mode,
        sustained_steps=args.sustained_steps, capture_k=args.capture_k,
        intruder_profile=getattr(args, 'intruder_profile', 'evasive'),
        seed=args.seed,
    )
    obs, _ = env.reset()
    episode = 1
    ep_rewards = []
    ep_captures = 0
    ep_total = 0
    policy_state = None
    if policy.is_recurrent:
        policy_state = {f"drone_{i}": policy.init_hidden(1) for i in range(N_DRONES)}

    # Set up PyBullet models for the first episode
    pb_drone_ids, pb_intruder_id = None, None
    if use_pb and env._pb is not None:
        pb_drone_ids, pb_intruder_id = _setup_pybullet_episode(env)

    while True:
        with lock:
            if not state["running"]:
                break
            speed = state["speed"]
            if state["reset_req"]:
                state["reset_req"] = False
                if use_pb and pb_drone_ids is not None:
                    _cleanup_pybullet_models(env, pb_drone_ids, pb_intruder_id)
                obs, _ = env.reset()
                if policy.is_recurrent:
                    policy_state = {f"drone_{i}": policy.init_hidden(1) for i in range(N_DRONES)}
                if use_pb and env._pb is not None:
                    pb_drone_ids, pb_intruder_id = _setup_pybullet_episode(env)
                episode += 1
                state["episode"] = episode
                ep_rewards = []
                continue
            # apply live attack controls
            drop = state["p_drop"]
            spoof = state["p_spoof"]
            env.channel.p_drop = drop
            env.channel.p_spoof = spoof
            env.use_trust = state["use_trust"]

        actions = {}
        for i in range(N_DRONES):
            if policy.is_recurrent:
                hidden_in = policy_state[f"drone_{i}"]
                a, _, hidden_out = policy.step(obs[f"drone_{i}"], deterministic=True, hidden_state=hidden_in)
                actions[f"drone_{i}"] = a
                if policy.policy_type == "lstm":
                    policy_state[f"drone_{i}"] = (hidden_out[0].squeeze(0), hidden_out[1].squeeze(0))
                else:
                    policy_state[f"drone_{i}"] = hidden_out.squeeze(0)
            else:
                a, _ = policy.get_action(obs[f"drone_{i}"], deterministic=True)
                actions[f"drone_{i}"] = a
        actions["sensor_0"] = 1 if obs["sensor_0"][0] > 0.5 else 0

        obs, rew, term, trunc, info = env.step(actions)
        mean_rew = float(np.mean([rew[f"drone_{i}"] for i in range(N_DRONES)]))
        ep_rewards.append(mean_rew)

        # Update PyBullet 3D positions
        if use_pb and pb_drone_ids is not None:
            _update_pybullet_positions(env, pb_drone_ids, pb_intruder_id)

        trust_scores = []
        for tm in env.trust_mods:
            trust_scores.append(tm.get_trust_scores())

        with lock:
            state["drone_pos"] = env.drone_pos.copy()
            state["drone_vel"] = env.drone_vel.copy()
            state["intruder_pos"] = env.intruder_pos.copy()
            state["trust"] = np.array(trust_scores)
            state["battery"] = env.battery.copy()
            state["wind"] = env.wind_vec.copy()
            state["reward"] = mean_rew
            state["step"] = env.step_count
            state["sensor_alert"] = bool(env.sensor_alert)
            state["ch_stats"] = env.channel.get_stats()
            state["capture_mode"] = env.capture_mode
            state["metrics"] = {
                "n_close": float(info.get("drone_0", {}).get("n_close", 0)),
                "mean_team_distance": float(info.get("drone_0", {}).get("mean_team_distance", 0.0)),
                "formation_spread": float(info.get("drone_0", {}).get("formation_spread", 0.0)),
                "angular_coverage_score": float(info.get("drone_0", {}).get("angular_coverage_score", 0.0)),
                "capture_count": int(bool(info.get("drone_0", {}).get("captured", False))),
            }
            # keep last 200 reward values for chart
            rh = state["reward_hist"]
            rh.append(mean_rew)
            if len(rh) > 200:
                state["reward_hist"] = rh[-200:]

        if not env.agents:
            captured = info.get("drone_0", {}).get("captured", False)
            ep_total += 1
            if captured:
                ep_captures += 1
            with lock:
                state["captured"] = captured
                ch = state["capture_hist"]
                ch.append(1 if captured else 0)
                if len(ch) > 50:
                    state["capture_hist"] = ch[-50:]
            time.sleep(0.5)
            # Clean up and reset
            if use_pb and pb_drone_ids is not None:
                _cleanup_pybullet_models(env, pb_drone_ids, pb_intruder_id)
            obs, _ = env.reset()
            if use_pb and env._pb is not None:
                pb_drone_ids, pb_intruder_id = _setup_pybullet_episode(env)
            episode += 1
            ep_rewards = []
            with lock:
                state["episode"] = episode

        time.sleep(speed)

    env.close()


# ── Drawing Helpers ────────────────────────────────────────────────────────
def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def draw_rounded_rect(surf, color, rect, radius=8):
    pygame.draw.rect(surf, color, rect, border_radius=radius)


def draw_panel(surf, x, y, w, h, title=""):
    draw_rounded_rect(surf, PANEL_BG, (x, y, w, h), 10)
    pygame.draw.rect(surf, BORDER, (x, y, w, h), 1, border_radius=10)
    if title:
        font_sm = pygame.font.SysFont("Segoe UI", 13, bold=True)
        ts = font_sm.render(title, True, TEXT_DIM)
        surf.blit(ts, (x + 12, y + 8))


def draw_bar(surf, x, y, w, h, val, color, bg=(40, 44, 55)):
    pygame.draw.rect(surf, bg, (x, y, w, h), border_radius=3)
    bw = max(0, min(int(val * w), w))
    if bw > 0:
        pygame.draw.rect(surf, color, (x, y, bw, h), border_radius=3)


# ── Main Panels ───────────────────────────────────────────────────────────
def draw_world(surf, s, font, x0, y0, pw, ph):
    draw_panel(surf, x0, y0, pw, ph, "WORLD VIEW (Top-Down)")
    mx, my = x0 + 10, y0 + 28
    mw, mh = pw - 20, ph - 38
    pygame.draw.rect(surf, (30, 34, 48), (mx, my, mw, mh), border_radius=6)

    sx, sy = mw / WORLD_XY, mh / WORLD_XY

    # Grid
    for g in range(0, 21, 5):
        gx = mx + int(g * sx)
        gy = my + int(g * sy)
        pygame.draw.line(surf, (38, 42, 56), (gx, my), (gx, my + mh), 1)
        pygame.draw.line(surf, (38, 42, 56), (mx, gy), (mx + mw, gy), 1)

    # Border lines
    pygame.draw.rect(surf, ORANGE, (mx, my, mw, mh), 2, border_radius=4)

    # Intruder
    ip = s["intruder_pos"]
    ix, iy = mx + int(ip[0] * sx), my + int(ip[1] * sy)
    pts = [(ix, iy - 9), (ix - 7, iy + 6), (ix + 7, iy + 6)]
    pygame.draw.polygon(surf, INTRUDER_C, pts)
    pygame.draw.polygon(surf, (180, 140, 0), pts, 2)
    label = font.render("TGT", True, INTRUDER_C)
    surf.blit(label, (ix + 10, iy - 8))

    # Drones
    for i in range(N_DRONES):
        dp = s["drone_pos"][i]
        dx, dy = mx + int(dp[0] * sx), my + int(dp[1] * sy)
        avg_t = float(s["trust"][i].mean()) if s["use_trust"] else 1.0
        ring_c = lerp_color(RED, GREEN, avg_t)
        pygame.draw.circle(surf, ring_c, (dx, dy), 14, 2)
        pygame.draw.circle(surf, DRONE_C[i], (dx, dy), 10)
        lbl = font.render(f"D{i}", True, DRONE_C[i])
        surf.blit(lbl, (dx + 12, dy - 8))

        # Distance line to intruder
        dist = float(np.linalg.norm(dp[:2] - ip[:2]))
        if dist < 8.0:
            alpha_line = max(40, 150 - int(dist * 15))
            lc = (*DRONE_C[i][:3],)
            pygame.draw.line(surf, lerp_color(lc, BG, 0.5), (dx, dy), (ix, iy), 1)

    # Capture radius indicator around intruder
    cr_px = int(2.0 * sx)
    pygame.draw.circle(surf, (80, 60, 20), (ix, iy), cr_px, 1)


def draw_trust_panel(surf, s, font, font_sm, x0, y0, pw, ph):
    draw_panel(surf, x0, y0, pw, ph, "TRUST SCORES")
    labels = ["D0", "D1", "D2"]
    yy = y0 + 30
    for i in range(N_DRONES):
        t_lbl = font.render(f"{labels[i]}", True, DRONE_C[i])
        surf.blit(t_lbl, (x0 + 12, yy))
        senders = [j for j in range(N_DRONES) if j != i]
        for k, j in enumerate(senders):
            sy = yy + k * 22
            sl = font_sm.render(f"→D{j}", True, TEXT_DIM)
            surf.blit(sl, (x0 + 38, sy + 1))
            val = float(s["trust"][i][k]) if s["use_trust"] else 1.0
            color = GREEN if val > 0.6 else AMBER if val > 0.3 else RED
            draw_bar(surf, x0 + 80, sy + 3, 120, 12, val, color)
            vt = font_sm.render(f"{val:.2f}", True, TEXT)
            surf.blit(vt, (x0 + 206, sy + 1))
        yy += len(senders) * 22 + 14


def draw_battery_panel(surf, s, font, font_sm, x0, y0, pw, ph):
    draw_panel(surf, x0, y0, pw, ph, "BATTERY")
    for i in range(N_DRONES):
        yy = y0 + 28 + i * 26
        lbl = font_sm.render(f"D{i}", True, DRONE_C[i])
        surf.blit(lbl, (x0 + 12, yy))
        bv = float(s["battery"][i])
        color = GREEN if bv > 0.5 else AMBER if bv > 0.2 else RED
        draw_bar(surf, x0 + 40, yy + 3, 130, 12, bv, color)
        vt = font_sm.render(f"{bv*100:.0f}%", True, TEXT)
        surf.blit(vt, (x0 + 178, yy))


def draw_channel_panel(surf, s, font, font_sm, x0, y0, pw, ph):
    draw_panel(surf, x0, y0, pw, ph, "CHANNEL STATUS")
    ch = s["ch_stats"]
    yy = y0 + 28
    items = [
        ("Drop Rate", f"{s['p_drop']*100:.0f}%",
         RED if s['p_drop'] > 0.3 else AMBER if s['p_drop'] > 0 else GREEN),
        ("Spoof Rate", f"{s['p_spoof']*100:.0f}%",
         RED if s['p_spoof'] > 0.05 else GREEN),
        ("Msgs Sent", str(ch.get("total_messages", 0)), TEXT),
        ("Drops", str(ch.get("total_drops", 0)),
         RED if ch.get("total_drops", 0) > 0 else TEXT),
        ("Spoofed", str(ch.get("total_spoofs", 0)),
         RED if ch.get("total_spoofs", 0) > 0 else TEXT),
    ]
    for label, val, color in items:
        lt = font_sm.render(label, True, TEXT_DIM)
        vt = font_sm.render(val, True, color)
        surf.blit(lt, (x0 + 12, yy))
        surf.blit(vt, (x0 + pw - 60, yy))
        yy += 20

    # Sensor alert
    yy += 5
    alert = s["sensor_alert"]
    at = font.render("SENSOR", True, TEXT)
    surf.blit(at, (x0 + 12, yy))
    if alert:
        pygame.draw.circle(surf, RED, (x0 + pw - 30, yy + 8), 6)
        st = font_sm.render("ALERT", True, RED)
    else:
        pygame.draw.circle(surf, GREEN, (x0 + pw - 30, yy + 8), 6)
        st = font_sm.render("CLEAR", True, GREEN)
    surf.blit(st, (x0 + pw - 68, yy + 2))


def draw_reward_chart(surf, s, font_sm, x0, y0, pw, ph):
    draw_panel(surf, x0, y0, pw, ph, "REWARD HISTORY")
    rh = s["reward_hist"]
    if len(rh) < 2:
        return
    cx, cy = x0 + 15, y0 + 28
    cw, ch_ = pw - 30, ph - 42
    pygame.draw.rect(surf, (30, 34, 48), (cx, cy, cw, ch_), border_radius=4)

    vals = rh[-min(len(rh), cw):]
    if not vals:
        return
    mn, mx = min(vals), max(vals)
    rng = mx - mn if mx != mn else 1.0

    pts = []
    for idx, v in enumerate(vals):
        px = cx + int(idx * cw / max(len(vals) - 1, 1))
        py = cy + ch_ - int((v - mn) / rng * (ch_ - 10)) - 5
        pts.append((px, py))

    if len(pts) >= 2:
        pygame.draw.lines(surf, ACCENT, False, pts, 2)

    # Zero line
    if mn < 0 < mx:
        zy = cy + ch_ - int((0 - mn) / rng * (ch_ - 10)) - 5
        pygame.draw.line(surf, (60, 65, 80), (cx, zy), (cx + cw, zy), 1)


def draw_info_panel(surf, s, font, font_sm, x0, y0, pw, ph):
    draw_panel(surf, x0, y0, pw, ph, "ENVIRONMENT")
    yy = y0 + 28
    wind = s["wind"]
    wind_spd = float(np.linalg.norm(wind)) * 3.6
    metrics = s.get("metrics", {})
    items = [
        ("Wind", f"{wind_spd:.1f} km/h"),
        ("Wind Vec", f"[{wind[0]:.1f}, {wind[1]:.1f}, {wind[2]:.1f}]"),
        ("Trust Mode", "ON" if s["use_trust"] else "OFF"),
        ("Capture Mode", s.get("capture_mode", "team")),
        ("n_close", str(int(metrics.get("n_close", 0)))),
        ("Team Dist", f"{float(metrics.get('mean_team_distance', 0.0)):.2f}"),
        ("Formation", f"{float(metrics.get('formation_spread', 0.0)):.2f}"),
        ("Coverage", f"{float(metrics.get('angular_coverage_score', 0.0)):.2f}"),
    ]
    for label, val in items:
        lt = font_sm.render(label, True, TEXT_DIM)
        vt = font_sm.render(val, True, TEXT)
        surf.blit(lt, (x0 + 12, yy))
        surf.blit(vt, (x0 + 100, yy))
        yy += 20


def draw_controls(surf, s, font_sm, x0, y0, pw, ph):
    draw_panel(surf, x0, y0, pw, ph, "CONTROLS")
    yy = y0 + 28
    controls = [
        ("[A]", f"Attack: {'ON' if s['attack_on'] else 'OFF'}",
         RED if s["attack_on"] else GREEN),
        ("[↑/↓]", f"Drop: {s['p_drop']*100:.0f}%", AMBER),
        ("[R]", "Reset Episode", TEXT),
        ("[Q]", "Quit", TEXT),
    ]
    for key, desc, color in controls:
        kt = font_sm.render(key, True, ACCENT)
        dt = font_sm.render(desc, True, color)
        surf.blit(kt, (x0 + 12, yy))
        surf.blit(dt, (x0 + 58, yy))
        yy += 20


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SentryNet Live Dashboard")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--p_drop", type=float, default=0.0)
    parser.add_argument("--p_spoof", type=float, default=0.0)
    parser.add_argument("--use_trust", action="store_true")
    parser.add_argument("--capture-mode", choices=["team", "sustained"], default="team")
    parser.add_argument("--sustained-steps", type=int, default=3)
    parser.add_argument("--capture-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pybullet", action="store_true",
                        help="Also open PyBullet 3D window (synced with dashboard)")
    parser.add_argument("--speed", type=float, default=0.02,
                        help="Sim sleep per step in seconds (0.02=50Hz)")
    parser.add_argument("--intruder-profile", choices=["passive", "evasive", "reactive"], default="evasive",
                        help="Set the intelligence profile of the intruder drone")
    args = parser.parse_args()

    policy = load_policy(args.checkpoint)

    with lock:
        state["p_drop"] = args.p_drop
        state["p_spoof"] = args.p_spoof
        state["use_trust"] = args.use_trust
        state["capture_mode"] = args.capture_mode
        state["speed"] = args.speed

    sim = threading.Thread(target=sim_loop, args=(policy, args), daemon=True)
    sim.start()

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("SentryNet Live Dashboard")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Segoe UI", 14)
    font_sm = pygame.font.SysFont("Segoe UI", 12)
    font_lg = pygame.font.SysFont("Segoe UI", 18, bold=True)
    font_title = pygame.font.SysFont("Segoe UI", 22, bold=True)

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key == pygame.K_a:
                    with lock:
                        state["attack_on"] = not state["attack_on"]
                        if state["attack_on"]:
                            state["p_drop"] = 0.4
                            state["p_spoof"] = 0.15
                        else:
                            state["p_drop"] = 0.0
                            state["p_spoof"] = 0.0
                elif ev.key == pygame.K_UP:
                    with lock:
                        state["p_drop"] = min(0.95, state["p_drop"] + 0.05)
                elif ev.key == pygame.K_DOWN:
                    with lock:
                        state["p_drop"] = max(0.0, state["p_drop"] - 0.05)
                elif ev.key == pygame.K_r:
                    with lock:
                        state["reset_req"] = True

        with lock:
            s = {k: (v.copy() if isinstance(v, np.ndarray) else
                      v.copy() if isinstance(v, dict) else
                      list(v) if isinstance(v, list) else v)
                 for k, v in state.items()}

        screen.fill(BG)

        # ── Header ──
        title = font_title.render("SentryNet Live Dashboard", True, WHITE)
        screen.blit(title, (15, 10))

        # Live indicator
        pulse = abs(int(time.time() * 3) % 2)
        pygame.draw.circle(screen, RED if pulse else (80, 20, 20), (280, 22), 5)
        lt = font.render("LIVE", True, RED)
        screen.blit(lt, (290, 14))

        # Step / Episode / Reward
        info_txt = (f"Step: {s['step']}   Episode: {s['episode']}   "
                    f"Reward: {s['reward']:.2f}")
        it = font.render(info_txt, True, TEXT)
        screen.blit(it, (340, 16))

        # Capture rate
        ch = s["capture_hist"]
        if ch:
            cr = sum(ch) / len(ch) * 100
            crt = font.render(f"Capture Rate: {cr:.0f}%", True,
                              GREEN if cr > 60 else AMBER if cr > 30 else RED)
            screen.blit(crt, (W - 200, 16))

        # Attack status
        if s["attack_on"]:
            at = font_lg.render("⚠ ATTACK ACTIVE", True, RED)
            screen.blit(at, (W - 420, 12))

        # ── Layout ──
        top = 45
        # Left column: world view + reward chart
        draw_world(screen, s, font, x0=10, y0=top, pw=640, ph=420)
        draw_reward_chart(screen, s, font_sm, x0=10, y0=top+430, pw=640, ph=240)

        # Right column: trust, battery, channel, env, controls
        rx = 660
        rw = W - rx - 10
        draw_trust_panel(screen, s, font, font_sm, rx, top, rw, 200)
        draw_battery_panel(screen, s, font, font_sm, rx, top+210, rw, 110)
        draw_channel_panel(screen, s, font, font_sm, rx, top+330, rw, 170)
        draw_info_panel(screen, s, font, font_sm, rx, top+510, rw, 90)
        draw_controls(screen, s, font_sm, rx, top+610, rw, 105)

        pygame.display.flip()
        clock.tick(FPS)

    with lock:
        state["running"] = False
    pygame.quit()


if __name__ == "__main__":
    main()
