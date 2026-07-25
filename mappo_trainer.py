"""
mappo_trainer.py  —  SentryNet Phase 2
========================================
Full MAPPO trainer for shared-policy multi-drone learning.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import torch.nn as nn

from border_env import BorderEnv
from networks import PolicyNet, ValueNet
from rollout_buffer import RolloutBuffer


class MAPPOTrainer:
    """Full MAPPO training loop with shared actor and centralized critic."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "lr": 3e-4,
        "gamma": 0.99,
        "lam": 0.95,
        "clip_eps": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 10.0,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 4,
        "total_steps": 1_000_000,
        "save_every": 50_000,
        "eval_every": 50_000,
        "use_wandb": True,
        "run_name": "system_A",
        "checkpoint_dir": "checkpoints",
        "seed": 0,
        "policy_type": "mlp",
        "hidden_dim": 128,
    }

    def __init__(self, env: BorderEnv, config: Optional[Dict[str, Any]] = None) -> None:
        self.env = env
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        if config is not None:
            self.config.update(config)

        self.policy_type = str(self.config.get("policy_type", "mlp")).lower()
        self.hidden_dim = int(self.config.get("hidden_dim", 128))
        self.obs_dim = int(getattr(env, "drone_obs_dim", 23))
        self.config["policy_type"] = self.policy_type
        self.config["hidden_dim"] = self.hidden_dim
        self.config["obs_dim"] = self.obs_dim

        self.seed = int(self.config["seed"])
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Policy receives the expanded per-drone observation.
        # Value receives the joint observation plus a one-hot agent ID, so it can
        # predict each drone's own return instead of the team mean (ADR-009).
        self.n_drones = len(self._drone_keys())
        self.critic_obs_dim = self.obs_dim * self.n_drones + self.n_drones
        self.policy = PolicyNet(obs_dim=self.obs_dim, act_dim=3, hidden_dim=self.hidden_dim, policy_type=self.policy_type).to(self.device)
        self.value = ValueNet(obs_dim=self.critic_obs_dim).to(self.device)

        lr = float(self.config["lr"])
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=lr)

        self.buffer = RolloutBuffer(
            n_steps=int(self.config["n_steps"]),
            n_drones=3,
            obs_dim=self.obs_dim,
            act_dim=3,
            policy_type=self.policy_type,
            hidden_dim=self.hidden_dim,
            gamma=float(self.config["gamma"]),
            lam=float(self.config["lam"]),
            device=self.device,
        )

        self.total_env_steps = 0
        self.best_eval_capture_rate = float("-inf")
        
        # Curriculum learning tracking
        self.curriculum_progress = 0.0
        self.total_training_steps = int(self.config.get("total_steps", 1_000_000))

    @staticmethod
    def _drone_keys() -> List[str]:
        return ["drone_0", "drone_1", "drone_2"]

    def _init_policy_state(self):
        if not self.policy.is_recurrent:
            return None
        device = self.device
        if self.policy_type == "gru":
            return {k: torch.zeros(self.hidden_dim, device=device) for k in self._drone_keys()}
        return {k: (torch.zeros(self.hidden_dim, device=device), torch.zeros(self.hidden_dim, device=device)) for k in self._drone_keys()}

    def _obs_all_tensor(self, obs_dict: Dict[str, np.ndarray], info_dict: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Build the agent-conditioned critic input: one row per drone.

        Each row is the joint observation (42*3 = 126 dims today) followed by a
        one-hot agent ID. Without the ID every drone hands the critic an identical input
        while the dense reward gives them different returns, so the best the critic
        can do is predict their mean and the leftover error lands in the advantage.

        Returns: (n_drones, obs_dim*3 + n_drones)
        """
        n = self.n_drones
        joint = np.concatenate([obs_dict[k] for k in self._drone_keys()], axis=0).astype(np.float32)
        critic_in = np.concatenate([np.tile(joint, (n, 1)), np.eye(n, dtype=np.float32)], axis=1)
        obs_all = torch.as_tensor(critic_in, dtype=torch.float32, device=self.device)
        assert obs_all.shape == (n, self.critic_obs_dim), f"Expected ({n}, {self.critic_obs_dim}), got {tuple(obs_all.shape)}"
        return obs_all

    @staticmethod
    def _mean_trust_from_info(info: Dict[str, Any]) -> float:
        trust = info.get("trust_scores", [])
        if not trust:
            return 0.0
        flat: List[float] = []
        for row in trust:
            if isinstance(row, (list, tuple)):
                flat.extend([float(x) for x in row])
        return float(np.mean(flat)) if flat else 0.0

    def collect_rollout(self) -> Dict[str, float]:
        """Collect one rollout and compute GAE-ready buffer targets.
        
        Also updates curriculum progress based on training steps.
        """
        self.buffer.reset()
        n_steps = int(self.config["n_steps"])
        
        # Update curriculum progress (0.0 to 1.0 based on total training steps)
        progress = self.total_env_steps / max(1, self.total_training_steps)
        if hasattr(self.env, 'update_curriculum_progress'):
            self.env.update_curriculum_progress(progress)
        self.curriculum_progress = progress

        obs, _ = self.env.reset(seed=self.seed + self.total_env_steps)
        policy_state = self._init_policy_state()

        rollout_rewards: List[float] = []
        rollout_captures = 0
        rollout_episode_count = 0
        steps_to_capture: List[int] = []
        trust_values: List[float] = []
        stage_values: List[str] = []
        speed_values: List[float] = []
        n_close_values: List[float] = []
        participation_values: List[float] = []
        team_distance_values: List[float] = []
        formation_values: List[float] = []
        coverage_values: List[float] = []

        for step in range(n_steps):
            actions_dict: Dict[str, np.ndarray] = {}
            log_probs_dict: Dict[str, float] = {}
            values_dict: Dict[str, float] = {}
            hidden_state_dict: Optional[Dict[str, np.ndarray]] = None
            cell_state_dict: Optional[Dict[str, np.ndarray]] = None

            with torch.no_grad():
                if self.policy.is_recurrent:
                    hidden_state_dict = {}
                    if self.policy_type == "gru":
                        for k in self._drone_keys():
                            hidden_in = policy_state[k]
                            hidden_state_dict[k] = hidden_in.detach().cpu().numpy().astype(np.float32)
                            action, log_prob, next_hidden = self.policy.step(obs[k], deterministic=False, hidden_state=hidden_in)
                            actions_dict[k] = action
                            log_probs_dict[k] = float(log_prob)
                            policy_state[k] = next_hidden.squeeze(0)
                    else:
                        cell_state_dict = {}
                        for k in self._drone_keys():
                            hidden_in = policy_state[k]
                            hidden_state_dict[k] = hidden_in[0].detach().cpu().numpy().astype(np.float32)
                            cell_state_dict[k] = hidden_in[1].detach().cpu().numpy().astype(np.float32)
                            action, log_prob, next_hidden = self.policy.step(obs[k], deterministic=False, hidden_state=hidden_in)
                            actions_dict[k] = action
                            log_probs_dict[k] = float(log_prob)
                            policy_state[k] = (next_hidden[0].squeeze(0), next_hidden[1].squeeze(0))
                else:
                    for k in self._drone_keys():
                        action, log_prob = self.policy.get_action(obs[k], deterministic=False)
                        actions_dict[k] = action
                        log_probs_dict[k] = float(log_prob)

                v_per_agent = self.value(self._obs_all_tensor(obs)).squeeze(-1).cpu().numpy()
                for i, k in enumerate(self._drone_keys()):
                    values_dict[k] = float(v_per_agent[i])

            actions_env: Dict[str, Any] = {k: actions_dict[k] for k in self._drone_keys()}
            actions_env["sensor_0"] = 1 if float(obs["sensor_0"][0]) > 0.5 else 0

            next_obs, rewards, term, trunc, info_all = self.env.step(actions_env)
            dones_dict = {k: bool(term[k] or trunc[k]) for k in self._drone_keys()}

            self.buffer.add(
                step=step,
                obs_dict=obs,
                actions_dict=actions_dict,
                rewards_dict={k: float(rewards[k]) for k in self._drone_keys()},
                values_dict=values_dict,
                log_probs_dict=log_probs_dict,
                dones_dict=dones_dict,
                hidden_state_dict=hidden_state_dict,
                cell_state_dict=cell_state_dict,
            )

            rollout_rewards.append(float(np.mean([rewards[k] for k in self._drone_keys()])))
            info0 = info_all.get("drone_0", {})
            stage_values.append(str(info0.get("curriculum_stage", "")))
            speed_values.append(float(info0.get("intruder_speed", 0.0)))
            n_close_values.append(float(info0.get("n_close", 0)))
            participation_values.append(float(info0.get("participation_count", 0)))
            team_distance_values.append(float(info0.get("mean_team_distance", 0.0)))
            formation_values.append(float(info0.get("formation_spread", 0.0)))
            coverage_values.append(float(info0.get("angular_coverage_score", 0.0)))

            if any(dones_dict.values()):
                rollout_episode_count += 1
                trust_values.append(self._mean_trust_from_info(info0))
                if bool(info0.get("captured", False)):
                    rollout_captures += 1
                    steps_to_capture.append(int(info0.get("step", 0)))
                next_obs, _ = self.env.reset()
                policy_state = self._init_policy_state()

            obs = next_obs

        with torch.no_grad():
            last_values = self.value(self._obs_all_tensor(obs)).squeeze(-1).cpu().numpy().astype(np.float32)

        self.buffer.compute_gae(last_values=last_values)

        capture_rate = float(rollout_captures / rollout_episode_count) if rollout_episode_count > 0 else 0.0
        mean_steps_to_capture = float(np.mean(steps_to_capture)) if steps_to_capture else float(500)
        mean_trust = float(np.mean(trust_values)) if trust_values else 0.0

        return {
            "mean_reward": float(np.mean(rollout_rewards)) if rollout_rewards else 0.0,
            "capture_rate": capture_rate,
            "capture_count": float(rollout_captures),
            "mean_trust": mean_trust,
            "curriculum_stage": stage_values[-1] if stage_values else "",
            "mean_intruder_speed": float(np.mean(speed_values)) if speed_values else 0.0,
            "mean_n_close": float(np.mean(n_close_values)) if n_close_values else 0.0,
            "mean_participation_count": float(np.mean(participation_values)) if participation_values else 0.0,
            "mean_team_distance": float(np.mean(team_distance_values)) if team_distance_values else 0.0,
            "mean_formation_spread": float(np.mean(formation_values)) if formation_values else 0.0,
            "mean_angular_coverage": float(np.mean(coverage_values)) if coverage_values else 0.0,
            "steps_to_capture": mean_steps_to_capture,
        }

    def update(self) -> Dict[str, float]:
        """Run PPO updates for actor and critic using collected rollout."""
        n_epochs = int(self.config["n_epochs"])
        batch_size = int(self.config["batch_size"])
        clip_eps = float(self.config["clip_eps"])
        entropy_coef = float(self.config["entropy_coef"])
        value_coef = float(self.config["value_coef"])
        max_grad_norm = float(self.config["max_grad_norm"])

        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropy_values: List[float] = []
        approx_kls: List[float] = []

        total = self.buffer.ptr * self.buffer.n_drones
        n_d = self.buffer.n_drones
        # np.repeat orders rows [t0a0, t0a1, t0a2, t1a0, ...]; np.tile(eye) and
        # buffer.returns.reshape(total) share that ordering, so the one-hot ID on
        # each row names the drone whose return that row is trained against.
        obs_all_t = self.buffer.obs[: self.buffer.ptr].reshape(self.buffer.ptr, -1)
        obs_all_flat = np.concatenate(
            [
                np.repeat(obs_all_t, repeats=n_d, axis=0),
                np.tile(np.eye(n_d, dtype=np.float32), (self.buffer.ptr, 1)),
            ],
            axis=1,
        )
        flat_returns = self.buffer.returns[: self.buffer.ptr].reshape(total)
        flat_hidden = None
        flat_cell = None
        if self.buffer.hidden_states is not None:
            flat_hidden = self.buffer.hidden_states[: self.buffer.ptr].reshape(total, self.hidden_dim)
        if self.buffer.cell_states is not None:
            flat_cell = self.buffer.cell_states[: self.buffer.ptr].reshape(total, self.hidden_dim)

        for _ in range(n_epochs):
            idx = np.random.permutation(total)

            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                b = idx[start:end]

                obs_b = torch.as_tensor(
                    self.buffer.obs[: self.buffer.ptr].reshape(total, self.buffer.obs_dim)[b],
                    dtype=torch.float32,
                    device=self.device,
                )
                act_b = torch.as_tensor(
                    self.buffer.actions[: self.buffer.ptr].reshape(total, self.buffer.act_dim)[b],
                    dtype=torch.float32,
                    device=self.device,
                )
                old_log_b = torch.as_tensor(
                    self.buffer.log_probs[: self.buffer.ptr].reshape(total)[b],
                    dtype=torch.float32,
                    device=self.device,
                )
                adv_b = torch.as_tensor(
                    self.buffer.advantages[: self.buffer.ptr].reshape(total)[b],
                    dtype=torch.float32,
                    device=self.device,
                )
                ret_b = torch.as_tensor(flat_returns[b], dtype=torch.float32, device=self.device)
                obs_all_b = torch.as_tensor(obs_all_flat[b], dtype=torch.float32, device=self.device)

                if self.policy.is_recurrent and flat_hidden is not None:
                    hidden_b = torch.as_tensor(flat_hidden[b], dtype=torch.float32, device=self.device)
                    if self.policy_type == "lstm" and flat_cell is not None:
                        cell_b = torch.as_tensor(flat_cell[b], dtype=torch.float32, device=self.device)
                        hidden_input = (hidden_b, cell_b)
                    else:
                        hidden_input = hidden_b
                    new_log_b, entropy_b = self.policy.evaluate_actions(obs_b, act_b, hidden_state=hidden_input)
                else:
                    new_log_b, entropy_b = self.policy.evaluate_actions(obs_b, act_b)
                ratio = torch.exp(new_log_b - old_log_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred = self.value(obs_all_b).squeeze(-1)
                value_loss = 0.5 * torch.mean((value_pred - ret_b) ** 2)

                entropy_term = entropy_b.mean()
                total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_term

                self.policy_opt.zero_grad()
                self.value_opt.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                nn.utils.clip_grad_norm_(self.value.parameters(), max_grad_norm)
                self.policy_opt.step()
                self.value_opt.step()

                with torch.no_grad():
                    approx_kl = torch.mean(old_log_b - new_log_b).item()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropy_values.append(float(entropy_term.item()))
                approx_kls.append(float(approx_kl))

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
        }

    def evaluate(
        self,
        n_episodes: int = 50,
        p_drop_eval: float = 0.0,
        p_spoof_eval: float = 0.0,
    ) -> Dict[str, float]:
        """Evaluate policy deterministically on a separate environment."""
        eval_env = BorderEnv(
            use_pybullet=False,
            domain_rand=False,
            p_drop=p_drop_eval,
            p_spoof=p_spoof_eval,
            use_trust=getattr(self.env, "use_trust", True),
            capture_mode=getattr(self.env, "capture_mode", "team"),
            sustained_steps=getattr(self.env, "sustained_steps", 3),
            capture_k=getattr(self.env, "capture_k", 2),
            seed=self.seed,
        )

        captures = 0
        ep_rewards: List[float] = []
        ep_steps: List[int] = []
        trust_values: List[float] = []
        stage_values: List[str] = []
        speed_values: List[float] = []
        n_close_values: List[float] = []
        participation_values: List[float] = []
        team_distance_values: List[float] = []
        formation_values: List[float] = []
        coverage_values: List[float] = []

        for ep in range(n_episodes):
            obs, _ = eval_env.reset(seed=self.seed + ep)
            done = False
            ep_reward = 0.0
            steps = 0
            policy_state = self._init_policy_state()

            while not done:
                actions_env: Dict[str, Any] = {}
                if self.policy.is_recurrent:
                    if self.policy_type == "gru":
                        for k in self._drone_keys():
                            hidden_in = policy_state[k]
                            a, _, next_hidden = self.policy.step(obs[k], deterministic=True, hidden_state=hidden_in)
                            actions_env[k] = a
                            policy_state[k] = next_hidden.squeeze(0)
                    else:
                        for k in self._drone_keys():
                            hidden_in = policy_state[k]
                            a, _, next_hidden = self.policy.step(obs[k], deterministic=True, hidden_state=hidden_in)
                            actions_env[k] = a
                            policy_state[k] = (next_hidden[0].squeeze(0), next_hidden[1].squeeze(0))
                else:
                    for k in self._drone_keys():
                        a, _ = self.policy.get_action(obs[k], deterministic=True)
                        actions_env[k] = a
                actions_env["sensor_0"] = 1 if float(obs["sensor_0"][0]) > 0.5 else 0

                obs, rewards, term, trunc, info = eval_env.step(actions_env)
                ep_reward += float(np.mean([rewards[k] for k in self._drone_keys()]))
                steps += 1
                done = any(term[k] or trunc[k] for k in self._drone_keys())
                info0 = info.get("drone_0", {})
                stage_values.append(str(info0.get("curriculum_stage", "")))
                speed_values.append(float(info0.get("intruder_speed", 0.0)))
                n_close_values.append(float(info0.get("n_close", 0)))
                participation_values.append(float(info0.get("participation_count", 0)))
                team_distance_values.append(float(info0.get("mean_team_distance", 0.0)))
                formation_values.append(float(info0.get("formation_spread", 0.0)))
                coverage_values.append(float(info0.get("angular_coverage_score", 0.0)))

            info0 = info.get("drone_0", {})
            captures += int(bool(info0.get("captured", False)))
            trust_values.append(self._mean_trust_from_info(info0))
            ep_rewards.append(ep_reward)
            ep_steps.append(steps)

        eval_env.close()

        return {
            "capture_rate": float(captures / n_episodes),
            "capture_count": float(captures),
            "mean_steps": float(np.mean(ep_steps)) if ep_steps else 0.0,
            "mean_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
            "mean_trust": float(np.mean(trust_values)) if trust_values else 0.0,
            "curriculum_stage": stage_values[-1] if stage_values else "",
            "mean_intruder_speed": float(np.mean(speed_values)) if speed_values else 0.0,
            "mean_n_close": float(np.mean(n_close_values)) if n_close_values else 0.0,
            "mean_participation_count": float(np.mean(participation_values)) if participation_values else 0.0,
            "mean_team_distance": float(np.mean(team_distance_values)) if team_distance_values else 0.0,
            "mean_formation_spread": float(np.mean(formation_values)) if formation_values else 0.0,
            "mean_angular_coverage": float(np.mean(coverage_values)) if coverage_values else 0.0,
        }

    def save_checkpoint(self, step: int) -> None:
        """Persist training state to checkpoints/{run_name}/step_{step}.pt."""
        root = Path(str(self.config["checkpoint_dir"])) / str(self.config["run_name"])
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"step_{step}.pt"
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "value_state_dict": self.value.state_dict(),
                "policy_opt_state_dict": self.policy_opt.state_dict(),
                "value_opt_state_dict": self.value_opt.state_dict(),
                "step": step,
                "config": self.config,
                "seed": self.seed,
                "best_eval_capture_rate": self.best_eval_capture_rate,
            },
            path,
        )

    def save_best_checkpoint(self, step: int) -> None:
        """Persist the strongest evaluated model for easier downstream usage."""
        root = Path(str(self.config["checkpoint_dir"])) / str(self.config["run_name"])
        root.mkdir(parents=True, exist_ok=True)
        path = root / "best.pt"
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "value_state_dict": self.value.state_dict(),
                "policy_opt_state_dict": self.policy_opt.state_dict(),
                "value_opt_state_dict": self.value_opt.state_dict(),
                "step": step,
                "config": self.config,
                "seed": self.seed,
                "best_eval_capture_rate": self.best_eval_capture_rate,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model and optimizer state from checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        ckpt_critic_dim = int(ckpt["value_state_dict"]["net.0.weight"].shape[1])
        if ckpt_critic_dim != self.critic_obs_dim:
            raise ValueError(
                f"Checkpoint '{path}' has a {ckpt_critic_dim}-dim critic but this build "
                f"expects {self.critic_obs_dim} (joint state + one-hot agent ID, ADR-009). "
                "Checkpoints from before the per-agent critic fix cannot be resumed; retrain."
            )
        self.value.load_state_dict(ckpt["value_state_dict"])
        self.policy_opt.load_state_dict(ckpt["policy_opt_state_dict"])
        self.value_opt.load_state_dict(ckpt["value_opt_state_dict"])
        self.total_env_steps = int(ckpt.get("step", 0))
        self.best_eval_capture_rate = float(ckpt.get("best_eval_capture_rate", float("-inf")))

    def train(self) -> None:
        """Run MAPPO training loop until configured total_steps."""
        use_wandb = bool(self.config["use_wandb"])
        wandb = None
        if use_wandb:
            try:
                import wandb as _wandb  # type: ignore[import-not-found]

                wandb = _wandb
                wandb.init(
                    project="sentrinet",
                    name=str(self.config["run_name"]),
                    config=self.config,
                    reinit=True,
                )
            except Exception:
                wandb = None

        n_steps = int(self.config["n_steps"])
        total_steps = int(self.config["total_steps"])
        save_every = int(self.config["save_every"])
        eval_every = int(self.config["eval_every"])

        # Optional per-run metrics CSV (the training curve) — robust to console
        # redirection issues; captures the learning curve regardless of logging.
        metrics_csv_path = self.config.get("metrics_csv")
        csv_file = None
        csv_writer = None
        if metrics_csv_path:
            import os as _os
            import csv as _csv

            _os.makedirs(_os.path.dirname(metrics_csv_path) or ".", exist_ok=True)
            csv_file = open(metrics_csv_path, "w", newline="")
            _fieldnames = [
                "step", "train/reward", "train/capture_rate", "train/capture_count",
                "train/entropy", "train/approx_kl", "train/policy_loss", "train/value_loss",
                "env/trust_mean", "env/drop_rate",
                "eval/capture_rate", "eval/mean_reward", "eval/mean_steps", "eval/mean_trust",
            ]
            csv_writer = _csv.DictWriter(
                csv_file, fieldnames=_fieldnames, extrasaction="ignore", restval=""
            )
            csv_writer.writeheader()

        while self.total_env_steps < total_steps:
            rollout_stats = self.collect_rollout()
            loss_stats = self.update()
            self.total_env_steps += n_steps
            # Fire eval/save when we CROSS a multiple of the interval. The old
            # `% interval == 0` check silently never fired when the interval was
            # not an exact multiple of n_steps, so best.pt was never saved.
            _prev_steps = self.total_env_steps - n_steps
            do_eval = (self.total_env_steps // eval_every) > (_prev_steps // eval_every)
            do_save = (self.total_env_steps // save_every) > (_prev_steps // save_every)

            metrics = {
                "train/reward": rollout_stats["mean_reward"],
                "train/capture_rate": rollout_stats["capture_rate"],
                "train/capture_count": rollout_stats["capture_count"],
                "train/curriculum_stage": rollout_stats["curriculum_stage"],
                "train/intruder_speed": rollout_stats["mean_intruder_speed"],
                "train/n_close_mean": rollout_stats["mean_n_close"],
                "train/participation_mean": rollout_stats["mean_participation_count"],
                "train/team_distance_mean": rollout_stats["mean_team_distance"],
                "train/formation_spread_mean": rollout_stats["mean_formation_spread"],
                "train/angular_coverage_mean": rollout_stats["mean_angular_coverage"],
                "train/policy_loss": loss_stats["policy_loss"],
                "train/value_loss": loss_stats["value_loss"],
                "train/entropy": loss_stats["entropy"],
                "train/approx_kl": loss_stats["approx_kl"],
                "env/trust_mean": rollout_stats["mean_trust"],
                "env/drop_rate": float(getattr(self.env.channel, "p_drop", 0.0)),
                "env/battery_mean": float(np.mean(getattr(self.env, "battery", np.zeros((3,), dtype=np.float32)))),
            }

            if do_eval:
                eval_stats = self.evaluate(
                    n_episodes=50,
                    p_drop_eval=float(getattr(self.env.channel, "p_drop", 0.0)),
                    p_spoof_eval=float(getattr(self.env.channel, "p_spoof", 0.0)),
                )
                metrics.update(
                    {
                        "eval/capture_rate": eval_stats["capture_rate"],
                        "eval/capture_count": eval_stats["capture_count"],
                        "eval/curriculum_stage": eval_stats["curriculum_stage"],
                        "eval/intruder_speed": eval_stats["mean_intruder_speed"],
                        "eval/mean_steps": eval_stats["mean_steps"],
                        "eval/mean_reward": eval_stats["mean_reward"],
                        "eval/mean_trust": eval_stats["mean_trust"],
                        "eval/n_close_mean": eval_stats["mean_n_close"],
                        "eval/participation_mean": eval_stats["mean_participation_count"],
                        "eval/team_distance_mean": eval_stats["mean_team_distance"],
                        "eval/formation_spread_mean": eval_stats["mean_formation_spread"],
                        "eval/angular_coverage_mean": eval_stats["mean_angular_coverage"],
                    }
                )
                if eval_stats["capture_rate"] >= self.best_eval_capture_rate:
                    self.best_eval_capture_rate = float(eval_stats["capture_rate"])
                    self.save_best_checkpoint(step=self.total_env_steps)

            if wandb is not None:
                wandb.log(metrics, step=self.total_env_steps)

            print(
                f"[Step {self.total_env_steps:>8,}] "
                f"Stage={rollout_stats['curriculum_stage']:<7} | "
                f"Speed={rollout_stats['mean_intruder_speed']:4.2f} | "
                f"Capture={rollout_stats['capture_rate']*100:5.1f}% | "
                f"Reward={rollout_stats['mean_reward']:7.2f} | "
                f"Close={rollout_stats['mean_n_close']:4.2f} | "
                f"Part={rollout_stats['mean_participation_count']:4.2f} | "
                f"TeamDist={rollout_stats['mean_team_distance']:5.2f} | "
                f"Form={rollout_stats['mean_formation_spread']:5.2f} | "
                f"Cov={rollout_stats['mean_angular_coverage']:5.2f} | "
                f"PolicyLoss={loss_stats['policy_loss']:8.4f} | "
                f"ValueLoss={loss_stats['value_loss']:8.4f}"
            )

            if csv_writer is not None:
                _row = {"step": self.total_env_steps}
                _row.update(metrics)
                csv_writer.writerow(_row)
                csv_file.flush()

            if do_save:
                self.save_checkpoint(step=self.total_env_steps)

        self.save_checkpoint(step=self.total_env_steps)
        if csv_file is not None:
            csv_file.close()

        if wandb is not None:
            wandb.finish()

        print(
            f"Training completed for {self.config['run_name']} at {self.total_env_steps:,} steps. "
            f"Final checkpoint saved."
        )
