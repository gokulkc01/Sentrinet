"""
Unit tests for the world and measurement models.

These are cheap invariants, but two of them encode facts the science depends on:
NLoS range bias is strictly positive (it can never shorten a range), and a
coplanar formation really is coplanar (it is the degenerate geometry the
security analysis turns on).
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import GNSS, UWB

from sentrinet.sensing.gnss import broadcast_claims
from sentrinet.sensing.uwb import measure_ranges
from sentrinet.world.geometry import link_list, random_formation, true_ranges


@pytest.mark.parametrize("n_nodes", [2, 3, 5, 9])
def test_link_list_is_every_unordered_pair(n_nodes):
    links = link_list(n_nodes)
    assert len(links) == n_nodes * (n_nodes - 1) // 2
    assert len(set(links)) == len(links)
    assert all(i < j for i, j in links)


def test_formation_respects_minimum_separation():
    """
    Crowded nodes break the first-order residual model, so separation is a
    correctness constraint rather than an aesthetic one.
    """
    rng = np.random.default_rng(0)
    pts = random_formation(8, rng, min_separation=4.0)
    d = [np.linalg.norm(pts[i] - pts[j]) for i in range(len(pts)) for j in range(i + 1, len(pts))]
    assert min(d) >= 4.0


def test_zero_altitude_spread_is_exactly_coplanar():
    """The degenerate geometry must actually be degenerate."""
    rng = np.random.default_rng(1)
    pts = random_formation(7, rng, altitude_spread=0.0)
    assert np.ptp(pts[:, 2]) == 0.0


def test_nonzero_altitude_spread_is_not_coplanar():
    rng = np.random.default_rng(1)
    pts = random_formation(7, rng, altitude_spread=15.0)
    assert np.ptp(pts[:, 2]) > 1.0


def test_impossible_formation_raises_rather_than_hanging():
    rng = np.random.default_rng(2)
    with pytest.raises(RuntimeError):
        random_formation(40, rng, extent=5.0, altitude_spread=0.0, min_separation=10.0)


def test_gnss_noise_is_zero_mean_at_the_configured_sigma():
    rng = np.random.default_rng(3)
    noise = GNSS.noise(4000, rng)
    assert abs(noise.mean()) < 0.05
    assert abs(noise.std() - GNSS.sigma) < 0.05


def test_attacker_offset_moves_only_the_attacker():
    rng = np.random.default_rng(4)
    truth = random_formation(5, rng)
    offset = np.array([10.0, 0.0, 0.0])
    a = broadcast_claims(truth, GNSS, np.random.default_rng(9), attacker=2, offset=offset)
    b = broadcast_claims(truth, GNSS, np.random.default_rng(9))
    np.testing.assert_allclose(a[2] - b[2], offset, atol=1e-9)
    np.testing.assert_allclose(np.delete(a, 2, axis=0), np.delete(b, 2, axis=0), atol=1e-9)


def test_attacker_without_offset_is_rejected():
    rng = np.random.default_rng(5)
    truth = random_formation(4, rng)
    with pytest.raises(ValueError):
        broadcast_claims(truth, GNSS, rng, attacker=1)


def test_line_of_sight_ranging_is_unbiased():
    rng = np.random.default_rng(6)
    truth = random_formation(6, rng)
    links = link_list(6)
    truth_d = true_ranges(truth, links)
    err = np.concatenate([measure_ranges(truth, links, UWB, rng) - truth_d for _ in range(300)])
    assert abs(err.mean()) < 0.02
    assert abs(err.std() - UWB.sigma) < 0.02


def test_nlos_bias_can_only_lengthen_a_range():
    """
    A blocked signal travels further, never shorter. If this inverts, occlusion
    starts masking spoofs instead of mimicking them.
    """
    rng = np.random.default_rng(7)
    truth = random_formation(6, rng)
    links = link_list(6)
    truth_d = true_ranges(truth, links)
    mask = np.ones(len(links), dtype=bool)
    excess = np.concatenate(
        [measure_ranges(truth, links, UWB, rng, nlos_mask=mask) - truth_d for _ in range(200)]
    )
    # Only the ranging noise may push a sample below the true distance.
    assert excess.mean() > UWB.nlos_bias * 0.7
    assert np.quantile(excess, 0.02) > -3 * UWB.sigma


def test_ranges_are_symmetric_and_positive():
    rng = np.random.default_rng(8)
    truth = random_formation(5, rng)
    links = link_list(5)
    d = true_ranges(truth, links)
    assert np.all(d > 0)
    for idx, (i, j) in enumerate(links):
        assert abs(d[idx] - np.linalg.norm(truth[j] - truth[i])) < 1e-12
