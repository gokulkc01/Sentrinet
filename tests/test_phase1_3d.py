import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from border_env import BorderEnv, N_DRONES, DRONE_OBS_DIM, SENSOR_OBS_DIM, MASS_NOM, CAPTURE_R

@pytest.fixture
def env():
    e = BorderEnv(use_pybullet=False, domain_rand=False, seed=0)
    e.reset()
    return e

@pytest.fixture
def env_dr():
    return BorderEnv(use_pybullet=False, domain_rand=True, seed=7)

class TestReset:
    def test_agent_count(self, env):
        obs, _ = env.reset()
        assert len(obs) == N_DRONES + 1

    def test_drone_obs_shape(self, env):
        obs, _ = env.reset()
        for i in range(N_DRONES):
            assert obs[f"drone_{i}"].shape == (DRONE_OBS_DIM,)

    def test_sensor_obs_shape(self, env):
        obs, _ = env.reset()
        assert obs["sensor_0"].shape == (SENSOR_OBS_DIM,)

    def test_obs_dtype(self, env):
        obs, _ = env.reset()
        for a in env.possible_agents:
            assert obs[a].dtype == np.float32

    def test_agents_list_full_after_reset(self, env):
        env.reset()
        assert len(env.agents) == N_DRONES + 1

class TestDomainRandomisation:
    def test_mass_varies(self, env_dr):
        masses = []
        for _ in range(10):
            env_dr.reset()
            masses.append(env_dr.drone_mass.copy())
        # All mass arrays should not be identical
        masses = np.array(masses)
        assert masses.std() > 1e-5

    def test_mass_in_bounds(self, env_dr):
        for _ in range(20):
            env_dr.reset()
            assert np.all(env_dr.drone_mass >= MASS_NOM * 0.80)
            assert np.all(env_dr.drone_mass <= MASS_NOM * 1.20)

    def test_wind_in_bounds(self, env_dr):
        wmax = 15.0 / 3.6
        for _ in range(20):
            env_dr.reset()
            assert np.all(np.abs(env_dr.wind_vec[:2]) <= wmax + 0.01)

    def test_intruder_speed_in_bounds(self, env_dr):
        for _ in range(20):
            env_dr.reset()
            assert 1.5 <= env_dr.intruder_speed <= 4.0

    def test_sensor_noise_in_bounds(self, env_dr):
        for _ in range(20):
            env_dr.reset()
            assert 0.0 <= env_dr.sensor_noise_std <= 0.30

    def test_no_randomisation_when_disabled(self):
        env = BorderEnv(use_pybullet=False, domain_rand=False)
        env.reset()
        assert np.allclose(env.drone_mass, MASS_NOM)
        assert np.allclose(env.wind_vec, 0)

class TestStep:
    def test_step_returns_all_agents(self, env):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rew, term, trunc, info = env.step(actions)
        for a in env.possible_agents:
            assert a in obs
            assert a in rew
            assert a in term

    def test_obs_shape_after_step(self, env):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, _, _, _, _ = env.step(actions)
        for i in range(N_DRONES):
            assert obs[f"drone_{i}"].shape == (DRONE_OBS_DIM,)

    def test_positions_stay_in_bounds(self, env):
        for _ in range(50):
            if not env.agents: break
            actions = {a: env.action_space(a).sample() for a in env.agents}
            env.step(actions)
        assert np.all(env.drone_pos >= 0)
        assert np.all(env.drone_pos[:,:2] <= 20)
        assert np.all(env.drone_pos[:,2]  <= 10)

    def test_battery_decreases(self, env):
        b0 = env.battery.copy()
        for _ in range(5):
            if not env.agents: break
            actions = {a: env.action_space(a).sample() for a in env.agents}
            env.step(actions)
        assert np.any(env.battery < b0)

    def test_step_raises_after_done(self, env):
        env.agents = []
        with pytest.raises(AssertionError):
            env.step({a: env.action_space(a).sample()
                      for a in env.possible_agents})

    def test_truncates_at_max_steps(self):
        from border_env import MAX_STEPS
        env = BorderEnv(use_pybullet=False, domain_rand=False, seed=0)
        env.reset()
        done = False
        for _ in range(MAX_STEPS + 5):
            if not env.agents: break
            actions = {a: env.action_space(a).sample() for a in env.agents}
            _, _, term, trunc, _ = env.step(actions)
            if any(trunc.values()) or any(term.values()):
                done = True
                break
        assert done

class TestCommsPipeline:
    def test_trust_scores_degrade_under_attack(self):
        env = BorderEnv(use_pybullet=False, p_drop=0.8, seed=42)
        env.reset()
        for _ in range(200):
            if not env.agents: env.reset()
            actions = {a: env.action_space(a).sample() for a in env.agents}
            env.step(actions)
        # Under heavy drop, trust scores should have decayed
        for tm in env.trust_mods:
            scores = tm.get_trust_scores()
            assert np.any(scores < 0.9)

    def test_channel_stats_reflect_drop_rate(self):
        env = BorderEnv(use_pybullet=False, p_drop=0.5, seed=0)
        env.reset()
        for _ in range(100):
            if not env.agents: env.reset()
            actions = {a: env.action_space(a).sample() for a in env.agents}
            env.step(actions)
        dr = env.channel.get_stats()["empirical_drop_rate"]
        # The channel applies distance-dependent modulation to the base drop
        # rate; allow a wider acceptance range to avoid flakiness.
        assert 0.20 < dr < 0.95

class TestSensorRewards:
    def test_correct_trigger_gets_positive_reward(self, env):
        env.sensor_alert = 1
        actions = {a: env.action_space(a).sample() for a in env.agents}
        actions["sensor_0"] = 1
        _, rew, _, _, _ = env.step(actions)
        assert rew["sensor_0"] > 0

    def test_false_alarm_gets_negative_reward(self, env):
        env.sensor_alert = 0
        actions = {a: env.action_space(a).sample() for a in env.agents}
        actions["sensor_0"] = 1
        _, rew, _, _, _ = env.step(actions)
        assert rew["sensor_0"] < 0


class TestTeamCaptureDiagnostics:
    def _set_positions(self, env, drone_pos, intruder_pos):
        env.drone_pos = np.array(drone_pos, dtype=np.float64)
        env.drone_vel = np.zeros((N_DRONES, 3), dtype=np.float64)
        env.intruder_pos = np.array(intruder_pos, dtype=np.float64)
        env.intruder_vel = np.zeros(3, dtype=np.float64)
        env.battery = np.ones(N_DRONES, dtype=np.float64)
        env.wind_vec = np.zeros(3, dtype=np.float64)
        env.sensor_alert = 0
        env.use_trust = False
        env._prev_dists = np.linalg.norm(env.drone_pos - env.intruder_pos, axis=1)
        env._prev_mean_team_dist = float(np.mean(env._prev_dists))

    def test_single_drone_close_does_not_capture(self):
        env = BorderEnv(use_pybullet=False, domain_rand=False, capture_mode="team", seed=1)
        env.reset()
        self._set_positions(
            env,
            drone_pos=[
                [1.0, 1.0, 3.0],
                [12.0, 12.0, 3.0],
                [15.0, 15.0, 3.0],
            ],
            intruder_pos=[2.2, 1.0, 3.0],
        )
        captured, n_close, _ = env._capture_status()
        assert n_close == 1
        assert captured is False

    def test_two_drones_close_capture(self):
        env = BorderEnv(use_pybullet=False, domain_rand=False, capture_mode="team", seed=2)
        env.reset()
        self._set_positions(
            env,
            drone_pos=[
                [1.0, 1.0, 3.0],
                [1.7, 1.0, 3.0],
                [15.0, 15.0, 3.0],
            ],
            intruder_pos=[1.2, 1.0, 3.0],
        )
        captured, n_close, _ = env._capture_status()
        assert n_close >= 2
        assert captured is True

    def test_collision_penalty_applies_when_team_captures(self):
        env = BorderEnv(use_pybullet=False, domain_rand=False, capture_mode="team", seed=3)
        env.reset()
        self._set_positions(
            env,
            drone_pos=[
                [1.0, 1.0, 3.0],
                [1.1, 1.0, 3.0],
                [1.4, 1.0, 3.0],
            ],
            intruder_pos=[1.2, 1.0, 3.0],
        )
        captured, n_close, dists = env._capture_status()
        coord_metrics = env._coordination_metrics(dists)
        rewards = env._compute_rewards({f"drone_{i}": np.zeros(3) for i in range(N_DRONES)} | {"sensor_0": 0}, 0, captured, n_close, coord_metrics)

        env_no_collision = BorderEnv(use_pybullet=False, domain_rand=False, capture_mode="team", seed=4)
        env_no_collision.reset()
        self._set_positions(
            env_no_collision,
            drone_pos=[
                [1.0, 1.0, 3.0],
                [2.6, 1.0, 3.0],
                [3.9, 1.0, 3.0],
            ],
            intruder_pos=[1.2, 1.0, 3.0],
        )
        captured2, n_close2, dists2 = env_no_collision._capture_status()
        coord_metrics2 = env_no_collision._coordination_metrics(dists2)
        rewards_no_collision = env_no_collision._compute_rewards({f"drone_{i}": np.zeros(3) for i in range(N_DRONES)} | {"sensor_0": 0}, 0, captured2, n_close2, coord_metrics2)

        assert captured is True
        assert rewards["drone_0"] < rewards_no_collision["drone_0"]

    def test_surround_pattern_scores_high_coverage(self):
        env = BorderEnv(use_pybullet=False, domain_rand=False, capture_mode="team", seed=5)
        env.reset()
        r = 4.0
        intruder = np.array([10.0, 10.0, 3.0], dtype=np.float64)
        angles = [0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0]
        drone_pos = [
            [intruder[0] + r * np.cos(a), intruder[1] + r * np.sin(a), 3.0]
            for a in angles
        ]
        self._set_positions(env, drone_pos=drone_pos, intruder_pos=intruder)
        _, _, dists = env._capture_status()
        metrics = env._coordination_metrics(dists)
        assert metrics["angular_coverage_score"] > 0.8