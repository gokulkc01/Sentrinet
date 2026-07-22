"""
border_env.py  ─  SentryNet Phase 1 (3D)
==========================================
3D Border Surveillance Environment  |  gym-pybullet-drones + PettingZoo

World: 20×20×10 m airspace
Agents: 3 hunter drones (MAPPO) + 1 ground sensor (QMIX)
Target: 1 intruder drone (autonomous random-walk)

Observation per drone (20-dim):
  [own_x,y,z (3), own_vx,vy,vz (3), agg_x,y,z (3), agg_vx,vy,vz (3),
   sensor_alert (1), rel_x,y,z (3), battery (1), wind_x,y,z (3)]

Observation sensor (4-dim): [detected, noisy_x, noisy_y, noisy_z]

Action drones : Box(3) in [-1,1]     ← [Δx, Δy, Δz] thrust
Action sensor : Discrete(2)          ← 0=Idle, 1=Trigger

Install:
  pip install pybullet pettingzoo gymnasium numpy
  pip install git+https://github.com/utiasDSL/gym-pybullet-drones.git
"""

from __future__ import annotations
import os
import numpy as np
from typing import Optional, Dict, Any, Tuple

from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec

from adversarial_channel import AdversarialChannel
from trust_module        import TrustModule
from trust_aggregator    import TrustAggregator

# ── Optional deps ─────────────────────────────────────────────────────────────
try:
    import pybullet as p
    import pybullet_data
    _PYBULLET = True
except ImportError:
    _PYBULLET = False
    print("[SentryNet] pybullet not found → mock physics mode.")

try:
    import gym_pybullet_drones as _gpd_pkg
    _GPD = True
except ImportError:
    _GPD = False

# ── World constants ───────────────────────────────────────────────────────────
N_DRONES      = 3
MAX_STEPS     = 500
CAPTURE_R     = 2.0      # metres
CLOSE_R       = 5.0      # metres (legacy shaping radius)
WORLD_XY      = 20.0     # [0, WORLD_XY]
MAX_ALT       = 10.0
DT            = 0.05     # seconds per step
MASS_NOM      = 0.027    # kg (Crazyflie 2.x)
G             = 9.81
MAX_SPEED     = 5.0      # m/s
W1,W2,W3,W4   = 10.0, 0.1, 0.05, 5.0  # legacy capture/time/energy/security weights
W_CAPTURE     = 100.0    # terminal team-capture reward
W_TEAM        = 2.0      # mean-distance team pursuit shaping
W_FORMATION   = 0.8      # pairwise spacing shaping
W_COVERAGE    = 0.8      # angular surround shaping
W_CLOSE       = 1.5      # capture imminence shaping
W_PARTICIPATION = 1.0    # keep all hunters engaged near the target
TEAM_CAPTURE_MIN_DRONES = 2
COLLISION_THRESHOLD = 1.5
MAX_COORDINATION_DISTANCE = 8.0
IDEAL_SPACING = 4.5
BASE_DRONE_OBS_DIM = 23  # 20 base + 3 one-hot ID
EXTRA_REL_OBS_DIM = 19   # target rel pos/vel + teammate rel pos/vel + distance
DRONE_OBS_DIM = BASE_DRONE_OBS_DIM + EXTRA_REL_OBS_DIM
SENSOR_OBS_DIM= 4

# Curriculum learning phases (staged difficulty progression)
CURRICULUM_PHASES = [
    {
        "name": "stage1",
        "progress_end": 0.20,
        "p_drop": 0.0,
        "p_spoof": 0.0,
        "domain_rand": False,
        "wind_max": 0.0,
        "noise_max": 0.0,
    },
    {
        "name": "stage2",
        "progress_end": 0.40,
        "p_drop": 0.0,
        "p_spoof": 0.0,
        "domain_rand": True,
        "wind_max": 0.35,
        "noise_max": 0.05,
    },
    {
        "name": "stage3",
        "progress_end": 0.60,
        "p_drop": 0.1,
        "p_spoof": 0.0,
        "domain_rand": True,
        "wind_max": 0.70,
        "noise_max": 0.08,
    },
    {
        "name": "stage4",
        "progress_end": 0.80,
        "p_drop": 0.1,
        "p_spoof": 0.05,
        "domain_rand": True,
        "wind_max": 1.00,
        "noise_max": 0.12,
    },
    {
        "name": "stage5",
        "progress_end": 1.00,
        "p_drop": 0.2,
        "p_spoof": 0.1,
        "domain_rand": True,
        "wind_max": 1.50,
        "noise_max": 0.20,
    },
]
DRONE_ACT_DIM = 3
STALE_THRESH  = 25       # steps before local estimate is too stale for trust
DETECT_RANGE  = 8.0      # metres — FoV detection range
DETECT_ANGLE  = 60.0     # degrees — FoV half-cone


# ── Mock physics (no pybullet needed for testing) ─────────────────────────────
class _MockPhysics:
    DRAG = 0.08
    def __init__(self, n, mass, wind):
        self.n, self.mass, self.wind = n, mass.copy(), wind.copy()
        self.pos = np.zeros((n, 3))
        self.vel = np.zeros((n, 3))

    def reset(self, pos):
        self.pos = pos.copy()
        self.vel = np.zeros_like(pos)

    def step(self, thrust):
        for i in range(self.n):
            grav  = np.array([0., 0., -G * self.mass[i]])
            wind_f= self.wind * self.mass[i] * 0.08
            drag  = -self.DRAG * self.vel[i]
            acc   = (thrust[i] + grav + wind_f + drag) / self.mass[i]
            self.vel[i] = np.clip(self.vel[i] + acc*DT, -MAX_SPEED, MAX_SPEED)
            self.pos[i] = np.clip(
                self.pos[i] + self.vel[i]*DT,
                [0,0,0], [WORLD_XY, WORLD_XY, MAX_ALT]
            )

    @property
    def states(self):
        return self.pos.copy(), self.vel.copy()


# ── BorderEnv ─────────────────────────────────────────────────────────────────
class BorderEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv for SentryNet 3D border surveillance.

    Quick start (mock physics, no pybullet needed):
        env = BorderEnv(use_pybullet=False)
        obs, info = env.reset()
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rew, term, trunc, info = env.step(actions)

    Full pybullet mode:
        env = BorderEnv(use_pybullet=True, render_mode="human")
    """

    metadata = {"render_modes": ["human","rgb_array"], "name": "sentrinet_v1"}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        use_pybullet: bool = True,
        domain_rand: bool = True,
        p_drop: float = 0.0,
        p_spoof: float = 0.0,
        spoof_std: float = 2.0,
        use_trust: bool = True,
        compromised_drone: Optional[int] = None,
        # Trust hyperparameters (exposed for tuning)
        trust_alpha: float = None,
        trust_max_error: float = None,
        # Capture / intruder realism
        capture_mode: str = "team",       # one of: 'team', 'sustained'
        capture_k: int = 2,                 # for 'multi' mode: required drones
        sustained_steps: int = 3,           # for 'sustained' mode: steps required
        intruder_profile: str = "evasive", # one of: 'passive', 'evasive', 'reactive'
        reward_mode: str = "dense_pursuit", # 'dense_pursuit' (default) or 'shaped' (legacy)
        # Curriculum learning
        use_curriculum: bool = False,
        curriculum_progress: float = 0.0,  # 0.0 (easiest) to 1.0 (hardest)
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.render_mode  = render_mode
        self.use_pybullet = use_pybullet and _PYBULLET
        self.use_trust    = use_trust
        self.rng          = np.random.default_rng(seed)
        
        # Curriculum learning
        self.use_curriculum = use_curriculum
        self.curriculum_progress = float(curriculum_progress)
        self.curriculum_stage = "stage1"
        # Initialize curriculum params (will be updated if curriculum enabled)
        self._p_drop_eff = p_drop
        self._p_spoof_eff = p_spoof
        self.domain_rand = domain_rand
        self.compromised_drone = compromised_drone
        self.capture_mode = str(capture_mode)
        self.capture_k = int(capture_k)
        self.sustained_steps = max(1, int(sustained_steps))
        self.reward_mode = str(reward_mode)
        self.curriculum_intruder_speed = 1.5
        self.curriculum_wind_max = 0.0
        self.curriculum_noise_max = 0.0
        self.curriculum_shaping_weight = 1.0
        self.curriculum_capture_weight = 0.5
        self.curriculum_capture_r = float(CAPTURE_R)
        self._update_curriculum_params()

        self.possible_agents = [f"drone_{i}" for i in range(N_DRONES)] + ["sensor_0"]
        self.agents: list[str] = []
        self.drone_obs_dim = DRONE_OBS_DIM
        self.sensor_obs_dim = SENSOR_OBS_DIM
        
# Observation normalization: fixed, deterministic world-scale (ADR-005).
        # Stateless — no running statistics, so there is no train/eval skew and
        # nothing to persist in checkpoints.  Layout matches the 20-dim base obs
        # built in _drone_obs: pos(3), vel(3), agg_pos(3), agg_vel(3),
        # sensor_alert(1), rel_intruder_pos(3), battery(1), wind(3).
        self._normalize_obs = True
        self._obs_center = np.array([
            10., 10., 5.,   0., 0., 0.,   10., 10., 5.,   0., 0., 0.,
            0.,   0., 0., 0.,   0.,   0., 0., 0.,
        ], dtype=np.float32)
        self._obs_scale = np.array([
            10., 10., 5.,   5., 5., 5.,   10., 10., 5.,   5., 5., 5.,
            1.,   10., 10., 10.,   1.,   6., 6., 6.,
        ], dtype=np.float32)

          # Spaces (DRONE_OBS_DIM=42: 20 base + 3 one-hot ID + 19 relative features)
        inf = np.inf
        self._obs_sp = {
            **{f"drone_{i}": spaces.Box(-inf, inf, (DRONE_OBS_DIM,), np.float32)
               for i in range(N_DRONES)},
            "sensor_0": spaces.Box(-inf, inf, (SENSOR_OBS_DIM,), np.float32),
        }
        self._act_sp = {
            **{f"drone_{i}": spaces.Box(-1., 1., (DRONE_ACT_DIM,), np.float32)
               for i in range(N_DRONES)},
            "sensor_0": spaces.Discrete(2),
        }

        # Comms layer (will be updated by curriculum if enabled)
        self.channel    = AdversarialChannel(p_drop=self._p_drop_eff, p_spoof=self._p_spoof_eff,
                              spoof_std=spoof_std, seed=seed)
        # Pass explicit trust params if provided
        t_alpha = trust_alpha if trust_alpha is not None else TrustModule.EMA_ALPHA
        t_maxerr = trust_max_error if trust_max_error is not None else TrustModule.MAX_ERROR
        self.trust_mods = [TrustModule(n_senders=N_DRONES-1, max_error=t_maxerr, alpha=t_alpha)
                   for _ in range(N_DRONES)]
        self.aggregator = TrustAggregator(n_senders=N_DRONES-1, msg_dim=6)

        # State (init'd in reset)
        self.drone_pos   = np.zeros((N_DRONES, 3))
        self.drone_vel   = np.zeros((N_DRONES, 3))
        self.intruder_pos= np.zeros(3)
        self.intruder_vel= np.zeros(3)
        self.battery     = np.ones(N_DRONES)
        self.sensor_alert= 0
        self._noisy_int_pos  = np.zeros(3)
        self._agg_msgs       = np.zeros((N_DRONES, 6))
        self.drone_mass  = np.full(N_DRONES, MASS_NOM)
        self.wind_vec    = np.zeros(3)
        self.sensor_noise_std = 0.0
        self.intruder_speed   = 2.5
        self.step_count  = 0
        self._prev_dists = np.full(N_DRONES, WORLD_XY)  # for reward shaping
        self._prev_mean_team_dist = float(WORLD_XY)
        self._local_estimates = np.zeros((N_DRONES, 6), dtype=np.float32)
        self._estimate_age    = np.full(N_DRONES, MAX_STEPS, dtype=int)

        # Legacy sustained-capture state (only used when capture_mode == 'sustained').
        self._capture_counters = np.zeros(N_DRONES, dtype=int) if self.capture_mode == "sustained" else None

        # intruder realism profile
        self.intruder_profile = intruder_profile

        # PyBullet handles
        self._pb   = None
        self._dids = []
        self._iid  = None
        self._mock: Optional[_MockPhysics] = None
        
        # Reward shaping state (for tracking sender trust/reliability)
        self._sender_trust_sum = np.zeros(N_DRONES, dtype=np.float32)  # for averaging
        self._sender_reliable_count = np.zeros(N_DRONES, dtype=int)

    def observation_space(self, agent): return self._obs_sp[agent]
    def action_space(self, agent):      return self._act_sp[agent]
    
    def _update_curriculum_params(self):
        """Select curriculum parameters based on training progress."""
        if not self.use_curriculum:
            return
        prog = np.clip(self.curriculum_progress, 0.0, 1.0)
        stage = CURRICULUM_PHASES[-1]
        for candidate in CURRICULUM_PHASES:
            if prog <= float(candidate["progress_end"]):
                stage = candidate
                break
        prev_end = 0.0
        for candidate in CURRICULUM_PHASES:
            if candidate is stage:
                break
            prev_end = float(candidate["progress_end"])
        stage_span = max(1e-6, float(stage["progress_end"]) - prev_end)
        stage_progress = np.clip((prog - prev_end) / stage_span, 0.0, 1.0)

        self.curriculum_stage = str(stage["name"])
        self._p_drop_eff = float(stage["p_drop"])
        self._p_spoof_eff = float(stage["p_spoof"])
        self.domain_rand = bool(stage["domain_rand"])
        self.compromised_drone = 1 if prog >= 0.60 else None
        self.curriculum_wind_max = float(stage.get("wind_max", 0.0))
        self.curriculum_noise_max = float(stage.get("noise_max", 0.0))
        # Gradually increase intruder speed from easy-to-catch to full difficulty.
        self.curriculum_intruder_speed = float(0.4 + prog * (1.5 - 0.4))
        self.curriculum_shaping_weight = float(1.0 - prog)
        self.curriculum_capture_weight = float(0.5 + prog)
        self.curriculum_capture_r = float(4.0 - 2.0 * prog) if self.use_curriculum else float(CAPTURE_R)
    
    def update_curriculum_progress(self, progress: float):
        """Update curriculum stage based on training progress (0.0 to 1.0)."""
        if self.use_curriculum:
            self.curriculum_progress = np.clip(progress, 0.0, 1.0)
            self._update_curriculum_params()
            # Update channel drop rates
            if hasattr(self, 'channel'):
                self.channel.set_drop_rate(self._p_drop_eff)
    
    def _normalize_obs_features(self, obs_raw: np.ndarray, drone_idx: int) -> np.ndarray:
        """Fixed world-scale normalization of the 20-dim base obs (ADR-005).

        Deterministic and stateless: (obs - center) / scale using constants
        derived from the world bounds.  drone_idx is unused (kept for a stable
        call signature).
        """
        if not self._normalize_obs or obs_raw.shape[0] < 20:
            return obs_raw
        obs = obs_raw.copy()
        obs[:20] = (obs[:20] - self._obs_center) / self._obs_scale
        return obs

    # ── reset ──────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agents      = self.possible_agents[:]
        self.step_count  = 0
        self.battery     = np.ones(N_DRONES)
        self.sensor_alert= 0

        self._domain_randomise()

        # Spawn positions
        self.drone_pos = np.array([
            [self.rng.uniform(2, 8),   self.rng.uniform(2, 8),   self.rng.uniform(3,6)],
            [self.rng.uniform(12, 18), self.rng.uniform(2, 8),   self.rng.uniform(3,6)],
            [self.rng.uniform(2, 8),   self.rng.uniform(12, 18), self.rng.uniform(3,6)],
        ], dtype=np.float64)
        self.drone_vel     = np.zeros((N_DRONES, 3))
        self.intruder_pos  = self._border_spawn()
        self.intruder_vel  = self._inward_vel(self.intruder_pos)
        self._noisy_int_pos= self.intruder_pos.copy()
        self._prev_dists   = np.linalg.norm(self.drone_pos - self.intruder_pos, axis=1)
        self._local_estimates = np.zeros((N_DRONES, 6), dtype=np.float32)
        self._estimate_age    = np.full(N_DRONES, MAX_STEPS, dtype=int)
        self._agg_msgs     = np.zeros((N_DRONES, 6), dtype=np.float32)
        if self._capture_counters is not None:
            self._capture_counters.fill(0)
        self._prev_mean_team_dist = float(np.mean(self._prev_dists))
        # Reset per-episode trust reward-shaping accumulators.  These previously
        # persisted across the whole run, saturating tanh() to ~1 and turning the
        # trust-shaping term into a constant bias instead of a signal.
        self._sender_trust_sum[:] = 0.0
        self._sender_reliable_count[:] = 0

        # capture/intruder state already initialized in __init__

        if self.use_pybullet:
            self._init_pybullet()
        else:
            self._mock = _MockPhysics(N_DRONES, self.drone_mass, self.wind_vec)
            self._mock.reset(self.drone_pos)

        for tm in self.trust_mods:
            tm.reset()
        self.channel.reset_stats()
        return self._obs_all(), {}

    def _domain_randomise(self):
        if not self.domain_rand:
            self.drone_mass       = np.full(N_DRONES, MASS_NOM)
            self.wind_vec         = np.zeros(3)
            self.sensor_noise_std = 0.0
            self.intruder_speed   = float(self.curriculum_intruder_speed if self.use_curriculum else 2.5)
            return
        # Stage-aware disturbances: early curriculum uses light randomization,
        # later stages gradually widen the same family of disturbances.
        mass_span = 0.18 if not self.use_curriculum else np.clip(0.05 + 0.13 * self.curriculum_progress, 0.05, 0.18)
        self.drone_mass = self.rng.uniform(MASS_NOM * (1.0 - mass_span), MASS_NOM * (1.0 + mass_span), N_DRONES)
        wmax = self.curriculum_wind_max if self.use_curriculum else 15.0 / 3.6
        self.wind_vec = self.rng.uniform(-wmax, wmax, 3) if wmax > 0 else np.zeros(3)
        self.wind_vec[2] *= 0.25
        nmax = self.curriculum_noise_max if self.use_curriculum else 0.30
        self.sensor_noise_std = float(self.rng.uniform(0.0, nmax)) if nmax > 0 else 0.0
        if self.use_curriculum:
            self.intruder_speed = float(self.curriculum_intruder_speed)
        else:
            self.intruder_speed = float(self.rng.uniform(1.5, 4.0))

    # ── step ───────────────────────────────────────────────────────────────
    def step(self, actions: Dict[str, Any]):
        assert self.agents, "Episode done — call reset()."
        self.step_count += 1

        # Reward for sensor action should match the alert state visible at decision time.
        prev_sensor_alert = int(self.sensor_alert)

        thrust = self._to_thrust(actions)
        self._step_physics(thrust)
        self._step_intruder()
        self._step_sensor(actions.get("sensor_0", 0))
        self._update_local_estimates()
        self._comms_pipeline()

        effort = np.linalg.norm(thrust, axis=1) / (MASS_NOM * G)
        self.battery = np.clip(self.battery - 0.0005*effort, 0., 1.)

        captured, n_close, dists = self._capture_status()
        coord_metrics = self._coordination_metrics(dists)
        rewards  = self._compute_rewards(actions, prev_sensor_alert, captured, n_close, coord_metrics)
        trunc    = self.step_count >= MAX_STEPS
        done     = captured or trunc
        if done:
            self.agents = []

        term  = {a: captured for a in self.possible_agents}
        trunc_ = {a: trunc   for a in self.possible_agents}
        info  = self._info(captured, n_close, coord_metrics)
        return self._obs_all(), rewards, term, trunc_, info

    # ── physics ────────────────────────────────────────────────────────────
    def _to_thrust(self, actions) -> np.ndarray:
        T = np.zeros((N_DRONES, 3))
        for i in range(N_DRONES):
            act  = np.clip(np.asarray(
                actions.get(f"drone_{i}", np.zeros(3)), np.float64), -1., 1.)
            hov  = self.drone_mass[i] * G
            T[i, 2]  = hov * (1.0 + act[2]*0.50)
            T[i, :2] = hov * act[:2] * 0.35
            # Battery voltage sag under high thrust.
            scale = 0.6 + 0.4 * self.battery[i]
            T[i] *= scale
        return T

    def _step_physics(self, thrust):
        if self.use_pybullet and self._pb is not None:
            self._pb_step(thrust)
        else:
            self._mock.mass = self.drone_mass
            self._mock.wind = self.wind_vec
            self._mock.step(thrust)
            self.drone_pos, self.drone_vel = self._mock.states

        # Ornstein-Uhlenbeck wind turbulence process.
        theta, sigma = 0.1, 0.5
        self.wind_vec += (
            -theta * self.wind_vec * DT
            + sigma * np.sqrt(DT) * self.rng.normal(0.0, 1.0, 3)
        )
        self.wind_vec[2] *= 0.3
        self.wind_vec = np.clip(self.wind_vec, -6.0, 6.0)

    def _step_intruder(self):
        # Intruder motion profiles: passive (toward center), evasive (move away
        # from nearest drone when too close), reactive (evade and speed up).
        centre  = np.array([WORLD_XY/2, WORLD_XY/2, self.intruder_pos[2]])
        to_c    = centre - self.intruder_pos
        base_bias = to_c / (np.linalg.norm(to_c)+1e-8) * 0.6

        # default noise
        noise   = np.array([*self.rng.uniform(-1,1,2), self.rng.uniform(-0.2,0.2)]) * 0.4

        if self.intruder_profile == "passive":
            d = base_bias + noise

        else:
            # find nearest drone
            rels = self.drone_pos - self.intruder_pos
            dists = np.linalg.norm(rels, axis=1)
            nearest = int(np.argmin(dists))
            nearest_dist = float(dists[nearest])
            nearest_vec = rels[nearest]

            if self.intruder_profile == "evasive":
                # if a drone is within 6m, steer away strongly
                if nearest_dist < 6.0:
                    away = -nearest_vec / (np.linalg.norm(nearest_vec)+1e-8)
                    d = 0.9 * away + 0.1 * base_bias + noise
                else:
                    d = base_bias + noise

            elif self.intruder_profile == "reactive":
                # reactive: accelerate away if any drone is within 8m
                if nearest_dist < 8.0:
                    away = -nearest_vec / (np.linalg.norm(nearest_vec)+1e-8)
                    d = 0.7 * away + 0.3 * base_bias + noise
                    self.intruder_speed = min(self.intruder_speed * 1.05, 6.0)
                else:
                    d = base_bias + noise
            else:
                d = base_bias + noise

        self.intruder_vel = d / (np.linalg.norm(d)+1e-8) * self.intruder_speed
        self.intruder_pos = np.clip(
            self.intruder_pos + self.intruder_vel*DT,
            [0,0,0.5], [WORLD_XY, WORLD_XY, MAX_ALT]
        )

    def _step_sensor(self, action):
        noise = (self.rng.normal(0, self.sensor_noise_std, 3)
                 if self.sensor_noise_std > 0 else np.zeros(3))
        self._noisy_int_pos = self.intruder_pos + noise
        sensor_loc = np.array([WORLD_XY/2, WORLD_XY/2, 0.])
        self.sensor_alert = int(
            np.linalg.norm(self._noisy_int_pos[:2] - sensor_loc[:2]) < 8.0)

    # ── local estimates ────────────────────────────────────────────────────
    def _update_local_estimates(self):
        """Update each drone's local intruder estimate from FoV sensing.

        Each drone independently checks whether the intruder is within its
        field of view.  When detected, it stores a noisy absolute-position
        and velocity estimate.  When not detected, it keeps the previous
        (increasingly stale) estimate and increments the age counter.
        """
        for i in range(N_DRONES):
            rel  = self.intruder_pos - self.drone_pos[i]
            dist = float(np.linalg.norm(rel))
            fwd  = self.drone_vel[i] / (np.linalg.norm(self.drone_vel[i]) + 1e-8)
            cos_a = float(np.clip(np.dot(fwd, rel / (dist + 1e-8)), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cos_a)))
            detected = (dist < DETECT_RANGE) and (angle < DETECT_ANGLE)

            if detected:
                # Range-dependent sensor noise (closer = more accurate)
                noise_std = 0.05 + 0.02 * dist
                pos_noise = self.rng.normal(0, noise_std, 3)
                vel_noise = self.rng.normal(0, noise_std * 0.5, 3)
                self._local_estimates[i, :3] = (
                    self.intruder_pos + pos_noise
                ).astype(np.float32)
                self._local_estimates[i, 3:] = (
                    self.intruder_vel + vel_noise
                ).astype(np.float32)
                self._estimate_age[i] = 0
            else:
                self._estimate_age[i] += 1

    # ── comms ──────────────────────────────────────────────────────────────
    def _comms_pipeline(self):
        """Communication pipeline using LOCAL estimates (no ground truth).

        Each drone broadcasts its own noisy local estimate of the intruder.
        Drones with stale estimates (age >= STALE_THRESH) self-drop — they
        have nothing useful to contribute.

        Trust is evaluated against the RECEIVER's own local estimate when
        available.  When the receiver has no fresh estimate, only drop-decay
        is applied (no accuracy signal).

        The compromised drone is an exception: the adversary is assumed to
        have intelligence about the true intruder position and deliberately
        sends misleading coordinates.
        """
        base_drop = float(self.channel.p_drop)

        recv_msgs  = np.zeros((N_DRONES, N_DRONES, 6), np.float32)
        drop_masks = np.zeros((N_DRONES, N_DRONES), bool)

        for sender in range(N_DRONES):
            for receiver in range(N_DRONES):
                if receiver == sender:
                    continue

                # Sender with stale estimate → self-imposed drop
                if self._estimate_age[sender] >= STALE_THRESH:
                    recv_msgs[receiver, sender]  = 0.0
                    drop_masks[receiver, sender] = True
                    continue

                # Sender broadcasts its LOCAL noisy estimate (not GT)
                msg = self._local_estimates[sender][np.newaxis, :]  # (1, 6)

                dist = float(np.linalg.norm(
                    self.drone_pos[sender] - self.drone_pos[receiver]))
                effective_drop = min(0.95, base_drop + 0.025 * dist)
                self.channel.set_drop_rate(effective_drop)
                recv, drops = self.channel.transmit(msg)

                # Targeted adversary: compromised drone reports ADVERSARIAL
                # position.  The adversary has intelligence about the true
                # intruder position (stronger threat model).
                if (self.compromised_drone is not None
                        and sender == self.compromised_drone
                        and not drops[0]):
                    fake_pos = 2.0 * self.drone_pos[sender][:3] - self.intruder_pos
                    fake_pos = np.clip(fake_pos, [0,0,0],
                                       [WORLD_XY, WORLD_XY, MAX_ALT])
                    fake_vel = -self.intruder_vel
                    recv[0] = np.concatenate([fake_pos, fake_vel]).astype(np.float32)

                recv_msgs[receiver, sender]  = recv[0]
                drop_masks[receiver, sender] = drops[0]

        # Restore original configured drop rate.
        self.channel.set_drop_rate(base_drop)

        agg = np.zeros((N_DRONES, 6), np.float32)
        for i in range(N_DRONES):
            snd    = [j for j in range(N_DRONES) if j != i]
            msgs_i = recv_msgs[i][snd]
            drp_i  = drop_masks[i][snd]
            if self.use_trust:
                scr_i = self.trust_mods[i].get_trust_scores()
                if self._estimate_age[i] < STALE_THRESH:
                    # Receiver has a fresh local estimate → use as reference
                    reference = self._local_estimates[i, :3].copy()
                    self.trust_mods[i].update(msgs_i[:, :3], reference, drp_i)
                else:
                    # No local estimate → can only apply drop-decay
                    self.trust_mods[i].decay_on_drops(drp_i)
            else:
                # Systems A/B: uniform weights (no trust mechanism)
                scr_i = np.ones(N_DRONES - 1, dtype=np.float64)
            agg[i] = self.aggregator.aggregate(msgs_i, scr_i, drp_i)
        self._agg_msgs = agg

    # ── rewards ────────────────────────────────────────────────────────────
    def _compute_rewards(
        self,
        actions,
        sensor_alert_for_reward: int,
        captured: bool,
        n_close: int,
        coord_metrics: Dict[str, Any],
    ) -> Dict[str, float]:
        dists = coord_metrics.get("dists", np.linalg.norm(self.drone_pos - self.intruder_pos, axis=1))
        if self.reward_mode == "dense_pursuit":
            return self._dense_pursuit_rewards(actions, sensor_alert_for_reward, captured, dists)
        cap_r = float(getattr(self, "curriculum_capture_r", CAPTURE_R))
        empirical_spoof = self.channel.get_stats()["empirical_spoof_rate"]
        # Proportional security penalty instead of binary cliff.
        sec_penalty = min(1.0, empirical_spoof)
        mean_team_distance = float(coord_metrics.get("mean_team_distance", float(np.mean(dists))))
        formation_spread = float(coord_metrics.get("formation_spread", 0.0))
        angular_coverage_score = float(coord_metrics.get("angular_coverage_score", 0.0))
        participation_count = int(coord_metrics.get("participation_count", int(np.sum(dists < 4.0))))
        rew: Dict[str, float] = {}
        
        # Reward shaping: trust-aware communication learning
        # Goal: reward policy for learning to identify and trust honest senders
        if self.use_trust and not captured:  # don't over-reward once task is complete
            for i in range(N_DRONES):
                trust_scores = self.trust_mods[i].get_trust_scores()  # (N_DRONES-1,)
                if len(trust_scores) > 0:
                    honest_idx = np.where(np.arange(N_DRONES) != i)
                    # Reward: high trust on non-compromised drones
                    if self.compromised_drone is not None:
                        honest_scores = trust_scores[honest_idx[0] != self.compromised_drone] if self.compromised_drone != i else trust_scores[:self.compromised_drone] if self.compromised_drone > i else trust_scores[self.compromised_drone:]
                        if len(honest_scores) > 0:
                            honest_avg = float(np.mean(honest_scores))
                            if honest_avg > 0.6:  # threshold for "good trust"
                                # Small reward for learning to trust honest senders
                                rew_shaping_trust = 0.1 * min(1.0, honest_avg - 0.6)
                            else:
                                rew_shaping_trust = -0.05  # penalty for low trust on honest senders
                        else:
                            rew_shaping_trust = 0.0
                    else:
                        # No compromised drone: reward high trust across board
                        avg_trust = float(np.mean(trust_scores))
                        rew_shaping_trust = 0.05 * min(1.0, avg_trust)  # small reward
                    self._sender_trust_sum[i] += rew_shaping_trust
        
        team_approach = self._prev_mean_team_dist - mean_team_distance
        shaping_weight = float(self.curriculum_shaping_weight if self.use_curriculum else 1.0)
        capture_weight = float(self.curriculum_capture_weight if self.use_curriculum else 1.0)

        for i in range(N_DRONES):
            r  = -W2                                 # time penalty (-0.1/step)
            r -= W3 * (1.0 - self.battery[i])        # energy cost
            r -= W4 * sec_penalty                    # proportional security cost

            # Team pursuit: reward the whole formation for reducing average
            # intruder distance, rather than incentivizing a single hero drone.
            r += shaping_weight * W_TEAM * team_approach

            # Formation geometry: keep the team in a useful spacing band.
            r += shaping_weight * W_FORMATION * formation_spread

            # Angular coverage: reward surround-like approach patterns.
            r += shaping_weight * W_COVERAGE * angular_coverage_score

            # Capture imminence: smooth gradient toward the intruder.
            close_grad = 0.0
            for d in dists:
                if d < 8.0:
                    close_grad += float(np.exp(-d / max(cap_r, 1e-6)))
            close_grad /= float(N_DRONES)
            r += shaping_weight * W_CLOSE * close_grad

            # Participation reward: keep all hunters engaged within a useful radius.
            r += shaping_weight * W_PARTICIPATION * (participation_count / float(N_DRONES))

            # Capture (dominates all shaping)
            if captured:
                r += capture_weight * W_CAPTURE
            
            # Add lightweight trust shaping (max 0.1 to avoid drowning main signal)
            if self.use_trust and hasattr(self, '_sender_trust_sum'):
                r += shaping_weight * 0.02 * np.tanh(self._sender_trust_sum[i])  # bounded by 0.02

            # Milestone bonuses to prevent hovering just outside capture range.
            d_self = float(dists[i])
            if d_self < 6.0:
                r += shaping_weight * 0.5
            if d_self < 4.0:
                r += shaping_weight * 1.0
            if d_self < 3.0:
                r += shaping_weight * 2.0
            if d_self < 2.5:
                r += shaping_weight * 3.0

            rew[f"drone_{i}"] = float(r)

        self._prev_dists = dists.copy()  # update for next step
        self._prev_mean_team_dist = mean_team_distance

        # Collision penalty between hunter drones.
        for i in range(N_DRONES):
            for j in range(i + 1, N_DRONES):
                if np.linalg.norm(self.drone_pos[i] - self.drone_pos[j]) < 1.5:
                    rew[f"drone_{i}"] -= 5.0
                    rew[f"drone_{j}"] -= 5.0

        alert  = int(sensor_alert_for_reward)
        action = int(actions.get("sensor_0", 0))
        rew["sensor_0"] = 1.0 if (alert and action==1) else \
                          0.05 if (not alert and action==0) else -0.5
        return rew

    def _dense_pursuit_rewards(self, actions, sensor_alert_for_reward, captured, dists):
        """Clean dense pursuit reward (default; see ADR-007).

        Each drone is rewarded for reducing its own distance to the intruder,
        pays a small time cost, receives a large sparse bonus on capture, and a
        collision-safety penalty.  This avoids the 'spread out and hover' local
        optimum of the legacy 'shaped' reward, under which the policy never
        learned to pursue (capture stayed at 0%; dense reward reaches ~60% and
        climbing in 120k steps).
        """
        rew: Dict[str, float] = {}
        approach = self._prev_dists - dists  # per-drone one-step distance reduction
        for i in range(N_DRONES):
            r = float(approach[i]) - 0.02      # close the gap; small time cost
            if captured:
                r += W_CAPTURE                 # large sparse capture bonus
            rew[f"drone_{i}"] = r
        # Collision-safety penalty between hunters (the one shaping term we keep).
        for i in range(N_DRONES):
            for j in range(i + 1, N_DRONES):
                if np.linalg.norm(self.drone_pos[i] - self.drone_pos[j]) < 1.5:
                    rew[f"drone_{i}"] -= 5.0
                    rew[f"drone_{j}"] -= 5.0
        # Bookkeeping for the next step (mirrors the shaped path).
        self._prev_dists = dists.copy()
        self._prev_mean_team_dist = float(np.mean(dists))
        # Sensor-agent reward (unchanged from the shaped path).
        alert = int(sensor_alert_for_reward)
        action = int(actions.get("sensor_0", 0))
        rew["sensor_0"] = 1.0 if (alert and action == 1) else \
                          0.05 if (not alert and action == 0) else -0.5
        return rew

    def _capture_status(self):
        cap_r = float(getattr(self, "curriculum_capture_r", CAPTURE_R))
        dists = np.linalg.norm(self.drone_pos - self.intruder_pos, axis=1)
        within = dists < cap_r
        n_close = int(np.sum(within))

        if self.capture_mode == "team":
            return n_close >= TEAM_CAPTURE_MIN_DRONES, n_close, dists

        if self.capture_mode == "sustained":
            if self._capture_counters is None:
                self._capture_counters = np.zeros(N_DRONES, dtype=int)
            for i in range(N_DRONES):
                if within[i]:
                    self._capture_counters[i] += 1
                else:
                    self._capture_counters[i] = 0
            return bool(np.any(self._capture_counters >= self.sustained_steps)), n_close, dists

        raise ValueError(f"Unknown capture_mode '{self.capture_mode}'")

    def _captured(self) -> bool:
        captured, _, _ = self._capture_status()
        return captured

    def _coordination_metrics(self, dists: np.ndarray) -> Dict[str, Any]:
        mean_team_distance = float(np.mean(dists))
        participation_count = int(np.sum(dists < 4.0))

        pairwise = []
        pair_scores = []
        for i in range(N_DRONES):
            for j in range(i + 1, N_DRONES):
                dist_ij = float(np.linalg.norm(self.drone_pos[i] - self.drone_pos[j]))
                pairwise.append(dist_ij)
                if dist_ij < COLLISION_THRESHOLD:
                    pair_scores.append(-1.0)
                elif dist_ij > MAX_COORDINATION_DISTANCE:
                    pair_scores.append(-0.5)
                else:
                    pair_scores.append(max(0.0, 1.0 - abs(dist_ij - IDEAL_SPACING) / 3.0))

        formation_spread = float(np.mean(pair_scores)) if pair_scores else 0.0

        rel = self.drone_pos[:, :2] - self.intruder_pos[:2]
        angles = np.sort(np.arctan2(rel[:, 1], rel[:, 0]))
        if len(angles) >= 2:
            gaps = np.diff(np.r_[angles, angles[0] + 2.0 * np.pi])
            ideal_gap = 2.0 * np.pi / float(len(angles))
            angular_coverage = float(np.clip(1.0 - np.std(gaps) / max(ideal_gap, 1e-6), 0.0, 1.0))
        else:
            angular_coverage = 0.0

        return {
            "dists": dists,
            "mean_team_distance": mean_team_distance,
            "formation_spread": formation_spread,
            "angular_coverage_score": angular_coverage,
            "mean_pairwise_distance": float(np.mean(pairwise)) if pairwise else 0.0,
            "participation_count": participation_count,
        }

    # ── observations ───────────────────────────────────────────────────────
    def _obs_all(self):
        obs = {f"drone_{i}": self._drone_obs(i) for i in range(N_DRONES)}
        obs["sensor_0"] = np.array(
            [float(self.sensor_alert), *self._noisy_int_pos], np.float32)
        return obs

    def _drone_obs(self, i) -> np.ndarray:
        rel = self.intruder_pos - self.drone_pos[i]
        dist = float(np.linalg.norm(rel))
        fwd = self.drone_vel[i] / (np.linalg.norm(self.drone_vel[i]) + 1e-8)
        cos_a = float(np.clip(np.dot(fwd, rel / (dist + 1e-8)), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cos_a)))
        detected = (dist < DETECT_RANGE) and (angle < DETECT_ANGLE)
        intruder_rel = rel if detected else np.zeros(3, dtype=np.float64)

        # Base observation (20 dims)
        obs_base = np.concatenate([
            self.drone_pos[i],                      # 3
            self.drone_vel[i],                      # 3
            self._agg_msgs[i],                      # 6
            [float(self.sensor_alert)],             # 1
            intruder_rel,                           # 3
            [self.battery[i]],                      # 1
            self.wind_vec,                          # 3
        ]).astype(np.float32)                       # = 20
        
        # Apply observation normalization
        obs_base = self._normalize_obs_features(obs_base, i)
        
        # Append one-hot drone identity (3 dims) → total 23 dims
        one_hot_id = np.zeros(N_DRONES, dtype=np.float32)
        one_hot_id[i] = 1.0

        # Relative features normalized to ~[-1,1] (ADR-008). These were previously
        # raw at ±20 scale — including rel_target_pos, the single most important
        # feature — which saturated the Tanh network and made learning fragile even
        # though the intruder position is always present here.
        rel_pos_scale = np.array([WORLD_XY, WORLD_XY, 10.0], dtype=np.float32)
        rel_target_pos = ((self.intruder_pos - self.drone_pos[i]) / rel_pos_scale).astype(np.float32)
        rel_target_vel = ((self.intruder_vel - self.drone_vel[i]) / MAX_SPEED).astype(np.float32)

        teammate_rel_pos = []
        teammate_rel_vel = []
        for j in range(N_DRONES):
            if j == i:
                continue
            teammate_rel_pos.append(((self.drone_pos[j] - self.drone_pos[i]) / rel_pos_scale).astype(np.float32))
            teammate_rel_vel.append(((self.drone_vel[j] - self.drone_vel[i]) / MAX_SPEED).astype(np.float32))

        teammate_rel_pos_arr = np.concatenate(teammate_rel_pos).astype(np.float32)
        teammate_rel_vel_arr = np.concatenate(teammate_rel_vel).astype(np.float32)
        dist_feature = np.array([dist / 30.0], dtype=np.float32)  # world diagonal ~30 m

        rel_features = np.concatenate([
            rel_target_pos,
            rel_target_vel,
            teammate_rel_pos_arr,
            teammate_rel_vel_arr,
            dist_feature,
        ]).astype(np.float32)

        return np.concatenate([obs_base, one_hot_id, rel_features]).astype(np.float32)  # = 42

    def _info(self, captured, n_close: int, coord_metrics: Dict[str, Any]):
        base = dict(
            captured=captured, step=self.step_count,
            capture_count=int(captured),
            capture_mode=self.capture_mode,
            curriculum_stage=self.curriculum_stage,
            intruder_speed=float(self.intruder_speed),
            shaping_weight=float(self.curriculum_shaping_weight if self.use_curriculum else 1.0),
            capture_weight=float(self.curriculum_capture_weight if self.use_curriculum else 1.0),
            n_close=n_close,
            participation_count=int(coord_metrics.get("participation_count", 0)),
            mean_team_distance=float(coord_metrics.get("mean_team_distance", 0.0)),
            formation_spread=float(coord_metrics.get("formation_spread", 0.0)),
            angular_coverage_score=float(coord_metrics.get("angular_coverage_score", 0.0)),
            mean_pairwise_distance=float(coord_metrics.get("mean_pairwise_distance", 0.0)),
            intruder_pos=self.intruder_pos.copy(),
            drone_pos=self.drone_pos.copy(),
            wind=self.wind_vec.copy(),
            drone_mass=self.drone_mass.copy(),
            sensor_noise_std=self.sensor_noise_std,
            trust_scores=[tm.get_trust_scores().tolist() for tm in self.trust_mods],
        )
        return {a: base for a in self.possible_agents}

    # ── helpers ────────────────────────────────────────────────────────────
    def _border_spawn(self):
        alt  = float(self.rng.uniform(1.5, 5.0))
        edge = int(self.rng.integers(0, 4))
        r    = float(self.rng.uniform(0, WORLD_XY))
        edges = [
            [r,        0.,       alt],
            [r,        WORLD_XY, alt],
            [0.,       r,        alt],
            [WORLD_XY, r,        alt],
        ]
        return np.array(edges[edge])

    def _inward_vel(self, pos):
        c = np.array([WORLD_XY/2, WORLD_XY/2, pos[2]])
        d = c - pos
        return d / (np.linalg.norm(d)+1e-8) * self.intruder_speed

    # ── pybullet ───────────────────────────────────────────────────────────
    def _init_pybullet(self):
        if not _PYBULLET: return
        if self._pb is not None:
            try: p.disconnect(self._pb)
            except: pass
        self._pb = p.connect(p.GUI if self.render_mode=="human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._pb)
        p.setGravity(self.wind_vec[0]*0.08, self.wind_vec[1]*0.08, -G,
                     physicsClientId=self._pb)
        p.setTimeStep(DT, physicsClientId=self._pb)
        p.loadURDF("plane.urdf", physicsClientId=self._pb)

        urdf = self._get_urdf()
        self._dids = []
        for i in range(N_DRONES):
            did = p.loadURDF(urdf,
                basePosition=self.drone_pos[i].tolist(),
                baseOrientation=p.getQuaternionFromEuler([0,0,0]),
                physicsClientId=self._pb)
            p.changeDynamics(did, -1, mass=float(self.drone_mass[i]),
                             physicsClientId=self._pb)
            self._dids.append(did)
        self._iid = p.loadURDF("sphere_small.urdf",
            basePosition=self.intruder_pos.tolist(),
            physicsClientId=self._pb)

    def _pb_step(self, thrust):
        for i, did in enumerate(self._dids):
            p.applyExternalForce(did, -1, thrust[i].tolist(), [0,0,0],
                                 p.WORLD_FRAME, physicsClientId=self._pb)
        p.stepSimulation(physicsClientId=self._pb)
        for i, did in enumerate(self._dids):
            pos, _ = p.getBasePositionAndOrientation(did, physicsClientId=self._pb)
            vel, _ = p.getBaseVelocity(did, physicsClientId=self._pb)
            self.drone_pos[i] = np.array(pos)
            self.drone_vel[i] = np.array(vel)
        p.resetBasePositionAndOrientation(self._iid,
            self.intruder_pos.tolist(), [0,0,0,1], physicsClientId=self._pb)

    def _get_urdf(self) -> str:
        if _GPD:
            import gym_pybullet_drones as g
            c = os.path.join(os.path.dirname(g.__file__), "assets", "cf2x.urdf")
            if os.path.exists(c): return c
        return "sphere_small.urdf"

    def render(self):
        if self.render_mode == "rgb_array" and _PYBULLET and self._pb:
            w, h = 640, 480
            vm = p.computeViewMatrix([WORLD_XY/2,-5,15],
                [WORLD_XY/2,WORLD_XY/2,0],[0,0,1],physicsClientId=self._pb)
            pm = p.computeProjectionMatrixFOV(60,w/h,0.1,100,physicsClientId=self._pb)
            _, _, rgb, _, _ = p.getCameraImage(w,h,vm,pm,physicsClientId=self._pb)
            return np.array(rgb,np.uint8)[:,:,:3]

    def close(self):
        if _PYBULLET and self._pb:
            try: p.disconnect(self._pb)
            except: pass
            self._pb = None


def make_aec_env(**kw): return parallel_to_aec(BorderEnv(**kw))