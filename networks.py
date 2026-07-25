"""
networks.py  —  SentryNet Phase 2
===================================
Policy and value neural networks for MAPPO training.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class PolicyNet(nn.Module):
    """Actor network with optional GRU/LSTM memory."""

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 1.0
    ACTION_EPS = 1e-6

    def __init__(
        self,
        obs_dim: int = 23,
        act_dim: int = 3,
        hidden_dim: int = 128,
        policy_type: str = "mlp",
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_dim = int(hidden_dim)
        self.policy_type = str(policy_type).lower()
        self.is_recurrent = self.policy_type in {"gru", "lstm"}

        self.obs_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.Tanh(),
        )
        if self.policy_type == "mlp":
            self.core = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
            )
        elif self.policy_type == "gru":
            self.core = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        elif self.policy_type == "lstm":
            self.core = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        else:
            raise ValueError(f"Unknown policy_type '{policy_type}'")

        self.mean_head = nn.Linear(self.hidden_dim, self.act_dim)
        self.log_std = nn.Parameter(torch.full((self.act_dim,), -0.7))

        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal initialization for stable PPO optimization."""
        for module in self.obs_encoder:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(module.bias)
        if self.policy_type == "mlp":
            for module in self.core:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                    nn.init.zeros_(module.bias)
        else:
            nn.init.orthogonal_(self.core.weight_ih, gain=np.sqrt(2.0))
            nn.init.orthogonal_(self.core.weight_hh, gain=np.sqrt(2.0))
            nn.init.zeros_(self.core.bias_ih)
            nn.init.zeros_(self.core.bias_hh)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

    def init_hidden(self, batch_size: int = 1, device: Optional[torch.device] = None):
        """Create zero hidden state for recurrent policies."""
        if not self.is_recurrent:
            return None
        device = device if device is not None else next(self.parameters()).device
        if self.policy_type == "gru":
            return torch.zeros(batch_size, self.hidden_dim, device=device)
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    def _prepare_hidden(self, hidden_state: Any, batch_size: int, device: torch.device):
        if not self.is_recurrent:
            return None
        if hidden_state is None:
            return self.init_hidden(batch_size=batch_size, device=device)
        if self.policy_type == "gru":
            hidden = hidden_state.to(device)
            if hidden.ndim == 1:
                hidden = hidden.unsqueeze(0)
            return hidden
        h, c = hidden_state
        h = h.to(device)
        c = c.to(device)
        if h.ndim == 1:
            h = h.unsqueeze(0)
        if c.ndim == 1:
            c = c.unsqueeze(0)
        return h, c

    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def _squash_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        return torch.tanh(raw_action)

    def _squashed_log_prob(self, dist: Normal, raw_action: torch.Tensor) -> torch.Tensor:
        squashed = self._squash_action(raw_action)
        correction = torch.log(1.0 - squashed.pow(2) + self.ACTION_EPS).sum(dim=-1)
        return dist.log_prob(raw_action).sum(dim=-1) - correction

    def forward(self, obs: torch.Tensor, hidden_state: Any = None) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """Return action distribution parameters for observations."""
        assert obs.ndim == 2 and obs.shape[-1] == self.obs_dim, f"Expected (batch, {self.obs_dim}), got {tuple(obs.shape)}"
        feat = self.obs_encoder(obs)
        next_hidden = None
        if self.policy_type == "mlp":
            feat = self.core(feat)
        elif self.policy_type == "gru":
            hidden = self._prepare_hidden(hidden_state, obs.shape[0], obs.device)
            next_hidden = self.core(feat, hidden)
            feat = next_hidden
        else:
            hidden = self._prepare_hidden(hidden_state, obs.shape[0], obs.device)
            next_h, next_c = self.core(feat, hidden)
            next_hidden = (next_h, next_c)
            feat = next_h
        mean = self.mean_head(feat)
        log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX).unsqueeze(0).expand_as(mean)
        std = torch.exp(log_std)
        return mean, std, next_hidden

    def distribution(self, obs: torch.Tensor, hidden_state: Any = None) -> Tuple[Normal, Any]:
        """Build a Normal action distribution for a batch of observations."""
        mean, std, next_hidden = self.forward(obs, hidden_state=hidden_state)
        return Normal(mean, std), next_hidden

    def sample_action_tensor(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        hidden_state: Any = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Sample actions for tensor observations."""
        dist, next_hidden = self.distribution(obs, hidden_state=hidden_state)
        raw_action = dist.mean if deterministic else dist.rsample()
        actions = self._squash_action(raw_action)
        log_probs = self._squashed_log_prob(dist, raw_action)
        entropy = dist.entropy().sum(dim=-1)
        return actions, log_probs, entropy, next_hidden

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, hidden_state: Any = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log-probabilities and entropy of provided actions."""
        assert actions.ndim == 2 and actions.shape[-1] == self.act_dim, f"Expected (batch, {self.act_dim}), got {tuple(actions.shape)}"
        dist, _ = self.distribution(obs, hidden_state=hidden_state)
        clipped_actions = torch.clamp(actions, -1.0 + self.ACTION_EPS, 1.0 - self.ACTION_EPS)
        raw_action = self._atanh(clipped_actions)
        log_probs = self._squashed_log_prob(dist, raw_action)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy

    def step(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False,
        hidden_state: Any = None,
    ) -> Tuple[np.ndarray, float, Any]:
        """Get one action plus the next recurrent state when enabled."""
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        elif isinstance(obs, torch.Tensor):
            obs_tensor = obs.float().unsqueeze(0) if obs.ndim == 1 else obs.float()
        else:
            raise TypeError(f"Unsupported observation type: {type(obs)}")

        assert obs_tensor.shape == (1, self.obs_dim), f"Expected (1, {self.obs_dim}), got {tuple(obs_tensor.shape)}"
        device = next(self.parameters()).device
        obs_tensor = obs_tensor.to(device)

        with torch.no_grad():
            action_t, log_prob_t, _, next_hidden = self.sample_action_tensor(
                obs_tensor,
                deterministic=deterministic,
                hidden_state=hidden_state,
            )

        action = action_t.squeeze(0).cpu().numpy().astype(np.float32)
        log_prob = float(log_prob_t.squeeze(0).cpu().item())
        if isinstance(next_hidden, tuple):
            next_hidden = tuple(h.detach() for h in next_hidden)
        elif isinstance(next_hidden, torch.Tensor):
            next_hidden = next_hidden.detach()
        return action, log_prob, next_hidden

    def get_action(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """Get one action for one observation."""
        action, log_prob, _ = self.step(obs, deterministic=deterministic, hidden_state=None)
        return action, log_prob


class ValueNet(nn.Module):
    """Centralized, agent-conditioned critic with flexible input dimension.

    Critic receives, per agent:  obs_dim * n_drones + n_drones
    - Joint state: every drone's observation concatenated (3 x 42 = 126 today)
    - Agent ID: one-hot naming which drone this value is for (3) -> 129

    The agent ID is what makes the critic agent-specific. The joint state is shared
    by all three drones, so without it the critic gets one input for three different
    per-drone returns and can only fit their mean (ADR-009).
    """

    def __init__(self, obs_dim: int = 129, hidden_dim: int = 128) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal initialization for critic stability."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs_all: torch.Tensor) -> torch.Tensor:
        """Return scalar values for centralized observations.

        Args:
            obs_all: Tensor shape (batch, 60).

        Returns:
            Tensor shape (batch, 1).
        """
        assert obs_all.ndim == 2 and obs_all.shape[-1] == self.obs_dim, f"Expected (batch, {self.obs_dim}), got {tuple(obs_all.shape)}"
        feat = self.net(obs_all)
        return self.value_head(feat)
