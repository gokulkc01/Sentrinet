"""
UWB two-way ranging.

Double-sided TWR is the practical choice on real hardware because it cancels
the clock-drift term that single-sided ranging leaves behind; here we model the
*outcome* — an unbiased, few-centimetre range in line of sight.

The NLoS term matters more than it looks. When the direct path is blocked the
signal arrives via a longer route, so the measured range is biased **positive
and never negative**. A spoofed position claim also produces a range/claim
disagreement, so "this node is behind a hill" and "this node is lying" are
genuinely confusable. Separating them is part of the research problem, not a
nuisance to engineer away.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from sentrinet.world.geometry import true_ranges

Link = Tuple[int, int]


@dataclass(frozen=True)
class UwbModel:
    """
    sigma       : LoS ranging standard deviation (m). ~0.10 is realistic for a
                  DW1000-class radio at short range.
    nlos_bias   : mean excess path length (m) added when a link is obstructed.
    nlos_sigma  : spread of that excess.
    """

    sigma: float = 0.10
    nlos_bias: float = 0.60
    nlos_sigma: float = 0.30


def measure_ranges(
    true_positions: np.ndarray,
    links: List[Link],
    model: UwbModel,
    rng: np.random.Generator,
    nlos_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Measured range per link, shape (n_links,).

    `nlos_mask` is a boolean per link; obstructed links get a strictly positive
    excess drawn from a folded normal, so the bias can never shorten a range.
    """
    rho = true_ranges(np.asarray(true_positions, dtype=float), links)
    rho = rho + rng.normal(0.0, model.sigma, size=rho.shape)
    if nlos_mask is not None:
        nlos_mask = np.asarray(nlos_mask, dtype=bool)
        excess = np.abs(rng.normal(model.nlos_bias, model.nlos_sigma, size=rho.shape))
        rho = rho + np.where(nlos_mask, excess, 0.0)
    return rho
