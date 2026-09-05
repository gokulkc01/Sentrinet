"""
The adaptive attacker: a spoofer that knows the detector exists.

Almost all of this literature assumes a naive adversary — a fixed offset, a
Gaussian perturbation — and then reports that the detector catches it. The
question a defence evaluator asks within ninety seconds is: *what if the
attacker is aware of the range cross-check and lies consistently?*

Geometrically, a node claiming a 3-D position must satisfy one distance
constraint per peer. Counting unknowns against constraints:

    peers = 2   under-determined  -> a whole circle of consistent lies
    peers = 3   generically two solutions: the truth and its mirror through the
                plane of the three peers
    peers >= 4  over-determined   -> generically only the truth ...

... *unless the peers are coplanar*, in which case the mirror solution survives
no matter how many of them there are. Drones holding a common altitude are
nearly coplanar, so altitude diversity is a security property. See
sentrinet.world.geometry.

The practical question is not whether an exact consistent lie exists but
whether the attacker can get its residuals **under the noise floor**, since
anything below that is indistinguishable from honest GNSS error. That is what
`attack_margin_curve` measures.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def consistency_cost(claim: np.ndarray, peer_claims: np.ndarray, peer_ranges: np.ndarray) -> float:
    """Sum of squared range residuals if the attacker broadcasts `claim`."""
    implied = np.linalg.norm(np.asarray(peer_claims, float) - np.asarray(claim, float), axis=1)
    return float(np.sum((implied - np.asarray(peer_ranges, float)) ** 2))


def _fibonacci_directions(n_dirs: int) -> np.ndarray:
    """Near-uniform unit vectors on the sphere, shape (n_dirs, 3)."""
    idx = np.arange(n_dirs, dtype=float) + 0.5
    polar = np.arccos(1.0 - 2.0 * idx / n_dirs)
    azim = np.pi * (1.0 + 5.0**0.5) * idx
    return np.stack(
        [np.cos(azim) * np.sin(polar), np.sin(azim) * np.sin(polar), np.cos(polar)],
        axis=1,
    )


def best_lie_at_displacement(
    true_position: np.ndarray,
    peer_claims: np.ndarray,
    peer_ranges: np.ndarray,
    displacement: float,
    n_dirs: int = 512,
    refine_top: int = 8,
) -> tuple[float, np.ndarray]:
    """
    Cheapest consistent lie that moves the claim exactly `displacement` metres.

    Searches directions on a sphere of that radius, then refines the best few
    with a local optimiser over the two angles. Constraining the displacement
    (rather than minimising cost freely) is what makes the result interpretable:
    it answers "how well can the attacker hide *while actually lying this far*",
    which is the quantity a defender cares about.

    Returns
    -------
    (rms_residual_metres, best_claim)
        RMS rather than the raw sum, so the number is directly comparable to the
        residual noise floor and across different peer counts.
    """
    true_position = np.asarray(true_position, dtype=float)
    peer_claims = np.asarray(peer_claims, dtype=float)
    peer_ranges = np.asarray(peer_ranges, dtype=float)
    n_peers = len(peer_ranges)
    if n_peers == 0:
        return 0.0, true_position.copy()

    dirs = _fibonacci_directions(n_dirs)
    candidates = true_position[None, :] + displacement * dirs
    costs = np.array([consistency_cost(c, peer_claims, peer_ranges) for c in candidates])

    def angular_cost(angles: np.ndarray) -> float:
        polar, azim = angles
        unit = np.array([np.cos(azim) * np.sin(polar), np.sin(azim) * np.sin(polar), np.cos(polar)])
        return consistency_cost(true_position + displacement * unit, peer_claims, peer_ranges)

    best_cost = float(costs.min())
    best_claim = candidates[int(np.argmin(costs))]
    for k in np.argsort(costs)[: max(1, refine_top)]:
        unit = dirs[k]
        start = np.array([np.arccos(np.clip(unit[2], -1.0, 1.0)), np.arctan2(unit[1], unit[0])])
        res = minimize(
            angular_cost,
            start,
            method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 400},
        )
        if float(res.fun) < best_cost:
            best_cost = float(res.fun)
            polar, azim = res.x
            best_claim = true_position + displacement * np.array(
                [np.cos(azim) * np.sin(polar), np.sin(azim) * np.sin(polar), np.cos(polar)]
            )
    return float(np.sqrt(best_cost / n_peers)), best_claim


def attack_margin_curve(
    true_position: np.ndarray,
    peer_claims: np.ndarray,
    peer_ranges: np.ndarray,
    displacements: np.ndarray,
    n_dirs: int = 512,
) -> np.ndarray:
    """
    Best achievable RMS residual at each displacement, shape (len(displacements),).

    Compare against the honest noise floor sqrt(2*sigma_g^2 + sigma_r^2): where
    the curve sits below it, the attacker is statistically invisible.
    """
    return np.array(
        [
            best_lie_at_displacement(
                true_position, peer_claims, peer_ranges, float(d), n_dirs=n_dirs
            )[0]
            for d in displacements
        ],
        dtype=float,
    )
