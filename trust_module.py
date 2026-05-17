"""
trust_module.py
===============
EMA-based per-sender trust scoring.

Specs:
  error    = || recv_pos - reference_pos ||
  accuracy = max(0, 1 - error / max_error)
  tau      = alpha * accuracy + (1-alpha) * tau   (on receive)
  tau      = decay_on_drop * tau                    (on drop)
  tau      ∈ [0, 1]                                 (bounded always)

  max_error = AdversarialChannel.MAX_SPOOF_ERROR = 5.0

The reference_pos is the receiver's own local sensor estimate — NOT
ground truth.  When the receiver has no fresh local estimate, trust
scores are only updated via drop-decay (no accuracy signal).
"""

import numpy as np
from typing import Optional


class TrustModule:
    """
    Maintains a per-sender EMA trust score for one agent.

    Each agent in the system has its OWN TrustModule tracking
    the trustworthiness of messages received from all other agents.

    Parameters
    ----------
    n_senders   : int   — number of other agents sending messages
    max_error   : float — normalisation constant for position error
    alpha       : float — EMA learning rate for accuracy signal (0.1 per PPT)
    decay_on_drop: float — multiplicative decay when message is dropped (0.95 per PPT)
    init_trust  : float — starting trust score for all senders (1.0 = full trust)
    """

    EMA_ALPHA      = 0.1    # PPT: tau = 0.1 * accuracy + 0.9 * tau
    DECAY_ON_DROP  = 0.95   # PPT: tau = 0.95 * tau on drop
    MAX_ERROR      = 5.0    # from AdversarialChannel.MAX_SPOOF_ERROR

    def __init__(
        self,
        n_senders: int,
        max_error: float = MAX_ERROR,
        alpha: float = EMA_ALPHA,
        decay_on_drop: float = DECAY_ON_DROP,
        init_trust: float = 1.0,
    ):
        assert 0 < alpha < 1,        "alpha must be in (0, 1)"
        assert 0 < decay_on_drop < 1,"decay_on_drop must be in (0, 1)"

        self.n_senders     = n_senders
        self.max_error     = max_error
        self.alpha         = alpha
        self.decay_on_drop = decay_on_drop

        # tau[j] = trust score for sender j ∈ [0, 1]
        self.tau = np.full(n_senders, init_trust, dtype=np.float64)

        # diagnostics
        self._update_count = 0
        # history for temporal-consistency checks: last received message per sender
        self._last_msgs = np.zeros((n_senders, 3), dtype=np.float64)
        self._last_valid = np.zeros(n_senders, dtype=bool)

    # ── public API ───────────────────────────────────────────────────────────

    def update(
        self,
        received_pos: np.ndarray,   # shape (n_senders, 3) — x,y,z from channel
        reference_pos: Optional[np.ndarray],  # shape (3,) — receiver's own local estimate or None
        dropped_mask: np.ndarray,   # shape (n_senders,)    — bool, True = dropped
    ):
        """
        Update trust scores for all senders after one communication round.

        Parameters
        ----------
        received_pos  : x,y,z portion of received messages (first 3 dims)
        reference_pos : receiver's own local sensor estimate (3D).
                        This must NOT be ground truth — only information
                        the receiver could physically obtain.
        dropped_mask  : True where channel dropped the packet
        """
        assert received_pos.ndim == 2 and received_pos.shape[0] == self.n_senders, \
            f"received_pos shape mismatch: {received_pos.shape}"
        if reference_pos is not None:
            assert reference_pos.ndim == 1, f"reference_pos shape mismatch: {reference_pos.shape}"
        assert dropped_mask.shape == (self.n_senders,), "dropped_mask shape mismatch"

        # Helper: weighted median for 1D array
        def _weighted_median_1d(values: np.ndarray, weights: np.ndarray) -> float:
            # values, weights are 1D and of same length
            idx = np.argsort(values)
            v = values[idx]
            w = weights[idx]
            cum = np.cumsum(w)
            half = 0.5 * cum[-1]
            i = int(np.searchsorted(cum, half))
            return float(v[min(i, len(v)-1)])

        # Helper: weighted median for 3D vectors (per-dim median)
        def _weighted_median_vec(arr: np.ndarray, weights: np.ndarray) -> np.ndarray:
            return np.array([
                _weighted_median_1d(arr[:, d], weights) for d in range(arr.shape[1])
            ], dtype=np.float64)

        # Consensus-based trust update (no access to ground truth).
        # For each sender j:
        #  - if packet dropped: apply multiplicative decay
        #  - else: compute accuracy signals from up to two references:
        #      * receiver's own fresh local estimate (`reference_pos`) if provided
        #      * consensus of other non-dropped senders (median)
        #    final accuracy = mean(available accuracies)
        for j in range(self.n_senders):
            if dropped_mask[j]:
                # ── packet dropped: decay trust ──────────────────────────
                self.tau[j] *= self.decay_on_drop
            else:
                accuracies = []
                # reference-based accuracy (receiver's own local estimate)
                if reference_pos is not None and reference_pos.size == 3:
                    error_ref = float(np.linalg.norm(received_pos[j] - reference_pos))
                    acc_ref = max(0.0, 1.0 - error_ref / self.max_error)
                    accuracies.append(acc_ref)

                # consensus-based accuracy: weighted median of other non-dropped senders
                other_idx = [k for k in range(self.n_senders) if k != j and (not dropped_mask[k])]
                if len(other_idx) > 0:
                    other_msgs = received_pos[other_idx]
                    weights = self.tau[other_idx]
                    if weights.sum() <= 1e-8:
                        # fallback to unweighted median
                        consensus = np.median(other_msgs, axis=0)
                    else:
                        consensus = _weighted_median_vec(other_msgs, weights)
                    error_cons = float(np.linalg.norm(received_pos[j] - consensus))
                    acc_cons = max(0.0, 1.0 - error_cons / self.max_error)
                    accuracies.append(acc_cons)

                # temporal consistency: compare to sender's own previous message
                if self._last_valid[j]:
                    error_temp = float(np.linalg.norm(received_pos[j] - self._last_msgs[j]))
                    # allow larger tolerance for temporal jumps; scale by 2*max_error
                    acc_temp = max(0.0, 1.0 - error_temp / (2.0 * self.max_error))
                    accuracies.append(acc_temp)

                if len(accuracies) > 0:
                    accuracy = float(sum(accuracies) / len(accuracies))
                    # EMA: tau = alpha * accuracy + (1-alpha) * tau
                    self.tau[j] = self.alpha * accuracy + (1.0 - self.alpha) * self.tau[j]
                else:
                    # No reference and no consensus available: leave tau unchanged
                    pass

            # Clamp to [0, 1]
            self.tau[j] = float(np.clip(self.tau[j], 0.0, 1.0))

            # update history for temporal checks when message received
            if not dropped_mask[j]:
                self._last_msgs[j] = received_pos[j]
                self._last_valid[j] = True

        self._update_count += 1

    def decay_on_drops(self, dropped_mask: np.ndarray):
        """Apply drop-decay only, without accuracy updates.

        Called when the receiver has no fresh local estimate and therefore
        cannot evaluate message quality.  Trust still decays for dropped
        packets but received messages leave trust unchanged.
        """
        assert dropped_mask.shape == (self.n_senders,), "dropped_mask shape mismatch"
        for j in range(self.n_senders):
            if dropped_mask[j]:
                self.tau[j] *= self.decay_on_drop
                self.tau[j] = float(np.clip(self.tau[j], 0.0, 1.0))
        self._update_count += 1

    def get_trust_scores(self) -> np.ndarray:
        """Return current trust scores, shape (n_senders,)."""
        return self.tau.copy()

    def reset(self, init_trust: float = 1.0):
        """Reset trust scores (call on env.reset())."""
        self.tau[:] = init_trust
        self._update_count = 0

    def get_stats(self) -> dict:
        return {
            "trust_scores":  self.tau.tolist(),
            "mean_trust":    float(np.mean(self.tau)),
            "min_trust":     float(np.min(self.tau)),
            "update_count":  self._update_count,
        }
