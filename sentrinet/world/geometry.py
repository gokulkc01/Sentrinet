"""
Node geometry: formations, links and true inter-node ranges.

`altitude_spread` is deliberately a first-class parameter rather than a detail.
The consistency constraints a spoofer must satisfy are sphere intersections
(see sentrinet.attacks.adaptive); when every peer sits in one plane, any
solution keeps its mirror image through that plane, so a *coplanar swarm
preserves a consistent lie no matter how many nodes it has*. Drones cruising at
a common altitude are very nearly coplanar, which makes altitude diversity a
security property, not an aesthetic one.
"""

from __future__ import annotations

import numpy as np

Link = tuple[int, int]


def link_list(n_nodes: int) -> list[Link]:
    """All undirected pairs (i, j), i < j — a fully connected mesh."""
    return [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]


def random_formation(
    n_nodes: int,
    rng: np.random.Generator,
    extent: float = 40.0,
    altitude_spread: float = 15.0,
    altitude_centre: float = 25.0,
    min_separation: float = 4.0,
    max_tries: int = 10_000,
) -> np.ndarray:
    """
    Sample `n_nodes` positions in a box, rejecting draws that crowd together.

    Parameters
    ----------
    extent           : horizontal side length (m) of the sampling box
    altitude_spread  : vertical extent (m). **0.0 gives a perfectly coplanar
                       swarm**, which is the degenerate case the mirror-solution
                       analysis cares about.
    min_separation   : reject a node closer than this to an existing one, so the
                       residual linearisation (||e|| << ||d||) stays valid.

    Returns
    -------
    (n_nodes, 3) array of true positions.
    """
    if n_nodes < 2:
        raise ValueError("need at least 2 nodes")

    half = 0.5 * extent
    pts: list[np.ndarray] = []
    tries = 0
    while len(pts) < n_nodes:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(
                f"could not place {n_nodes} nodes with min_separation="
                f"{min_separation} in extent={extent}; loosen the constraints"
            )
        z = altitude_centre + (
            0.0
            if altitude_spread == 0.0
            else rng.uniform(-0.5 * altitude_spread, 0.5 * altitude_spread)
        )
        cand = np.array([rng.uniform(-half, half), rng.uniform(-half, half), z])
        if all(np.linalg.norm(cand - p) >= min_separation for p in pts):
            pts.append(cand)
    return np.asarray(pts, dtype=float)


def true_ranges(positions: np.ndarray, links: list[Link]) -> np.ndarray:
    """Noise-free Euclidean distance for each link, shape (n_links,)."""
    return np.array(
        [np.linalg.norm(positions[i] - positions[j]) for i, j in links],
        dtype=float,
    )
