"""
GNSS at the *measurement* level.

We never model the RF signal. A node derives a self-position from GNSS and
broadcasts it; a spoofed node broadcasts a self-position that is honestly
reported but wrong. That is exactly why authentication cannot help — the node
is not lying about who it is, it is lying about where it is, and it may not
even know it.

Typical consumer-grade horizontal error is metre-scale, which is ~15x coarser
than the UWB ranging noise in sentrinet.sensing.uwb. That asymmetry is the
entire basis of the cross-check: the radio is far more precise than the thing
it is checking.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class GnssModel:
    """Isotropic Gaussian self-position error, in metres."""

    sigma: float = 1.5

    def noise(self, n_nodes: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, self.sigma, size=(n_nodes, 3))


def broadcast_claims(
    true_positions: np.ndarray,
    model: GnssModel,
    rng: np.random.Generator,
    attacker: Optional[int] = None,
    offset: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Return each node's claimed self-position, shape (n_nodes, 3).

    Honest nodes claim `true + N(0, sigma^2 I)`. If `attacker` is given, that
    node additionally claims `+ offset` — a spoof injected at the measurement
    level, with no assumption about how the attacker achieved it.
    """
    claims = np.asarray(true_positions, dtype=float) + model.noise(
        len(true_positions), rng
    )
    if attacker is not None:
        if offset is None:
            raise ValueError("attacker given without an offset")
        claims[attacker] = claims[attacker] + np.asarray(offset, dtype=float)
    return claims
