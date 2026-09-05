"""
Shared fixtures and simulation helpers for the test suite.

The repo root is put on sys.path so the suite runs whether or not the package
has been installed with `pip install -e .`.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import dataclass  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from sentrinet.integrity.residuals import (  # noqa: E402
    residual_covariance,
    residual_jacobian,
    residual_vector,
)
from sentrinet.sensing.gnss import GnssModel, broadcast_claims  # noqa: E402
from sentrinet.sensing.uwb import UwbModel, measure_ranges  # noqa: E402
from sentrinet.world.geometry import link_list, random_formation  # noqa: E402

GNSS = GnssModel(sigma=1.5)
UWB = UwbModel(sigma=0.10)


def noise_floor(gnss: GnssModel = GNSS, uwb: UwbModel = UWB) -> float:
    """Theoretical std of a single honest residual: sqrt(2 sigma_g^2 + sigma_r^2)."""
    return float(np.sqrt(2.0 * gnss.sigma**2 + uwb.sigma**2))


@dataclass
class Epoch:
    links: list
    truth: np.ndarray
    claims: np.ndarray
    ranges: np.ndarray
    residual: np.ndarray
    covariance: np.ndarray
    jacobian: np.ndarray


def simulate_epoch(
    n_nodes: int,
    rng: np.random.Generator,
    attacker: int | None = None,
    offset: np.ndarray | None = None,
    altitude_spread: float = 15.0,
) -> Epoch:
    """One measurement epoch: formation, claims, ranges, residual and covariance."""
    links = link_list(n_nodes)
    truth = random_formation(n_nodes, rng, altitude_spread=altitude_spread)
    claims = broadcast_claims(truth, GNSS, rng, attacker=attacker, offset=offset)
    ranges = measure_ranges(truth, links, UWB, rng)
    resid = residual_vector(claims, ranges, links)
    jac = residual_jacobian(claims, links)
    cov = residual_covariance(jac, GNSS.sigma, UWB.sigma)
    return Epoch(links, truth, claims, ranges, resid, cov, jac)


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded generator — every statistical assertion in this suite is deterministic."""
    return np.random.default_rng(0)
