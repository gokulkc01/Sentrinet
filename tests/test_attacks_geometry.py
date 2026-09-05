"""
Tests for the geometric claims the project's rationale rests on.

If any of these break, the reasoning in ADR-011 and docs/DESIGN-v3.md is wrong
and needs revisiting — not the test.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import GNSS, UWB, noise_floor

from sentrinet.attacks.adaptive import best_lie_at_displacement, consistency_cost
from sentrinet.sensing.gnss import broadcast_claims
from sentrinet.sensing.uwb import measure_ranges
from sentrinet.world.geometry import link_list, random_formation


def _attacker_view(n_nodes, rng, altitude_spread=15.0, attacker=0):
    """True attacker position, its peers' claims, and the ranges it must satisfy."""
    links = link_list(n_nodes)
    truth = random_formation(n_nodes, rng, altitude_spread=altitude_spread)
    claims = broadcast_claims(truth, GNSS, rng)
    ranges = measure_ranges(truth, links, UWB, rng)
    peers = [k for k in range(n_nodes) if k != attacker]
    peer_ranges = np.array([ranges[m] for m, (i, j) in enumerate(links) if attacker in (i, j)])
    return truth[attacker], claims[peers], peer_ranges


def test_truth_itself_has_near_zero_consistency_cost():
    """Sanity: the honest claim should fit its own range measurements."""
    rng = np.random.default_rng(0)
    true_pos, peer_claims, peer_ranges = _attacker_view(9, rng)
    cost = consistency_cost(true_pos, peer_claims, peer_ranges)
    rms = np.sqrt(cost / len(peer_ranges))
    assert rms < noise_floor()


def test_three_nodes_admit_a_perfectly_consistent_lie():
    """
    With two peers there are 2 constraints on 3 unknowns, so a continuum of
    perfect lies exists. This is the structural reason the v2 trust mechanism
    could never have worked, and the motivation for the whole redesign.

    The claim is about the *typical* geometry, not every one. Constraining the
    lie to land at exactly 10 m adds a third constraint, so the sphere of that
    radius occasionally misses the circle of consistent lies. The median is what
    phase 0 measured (0.00 m) and what the argument rests on.
    """
    rng = np.random.default_rng(1)
    costs = []
    for _ in range(20):
        true_pos, peer_claims, peer_ranges = _attacker_view(3, rng)
        rms, claim = best_lie_at_displacement(true_pos, peer_claims, peer_ranges, 10.0)
        costs.append(rms)
        assert abs(np.linalg.norm(claim - true_pos) - 10.0) < 1e-6
    median = float(np.median(costs))
    assert median < 0.3, f"expected a free lie at N=3, median RMS was {median:.3f}"
    assert median < noise_floor() / 4, "an N=3 lie must sit far below the noise floor"


@pytest.mark.slow
def test_more_peers_shrink_the_attackers_reach():
    """
    Attacker capability must erode as the swarm grows. Note this is *monotone
    erosion*, not the sharp threshold at N=5 we originally predicted — phase 0
    measured a smooth decline and the prediction was corrected.
    """
    rng = np.random.default_rng(2)
    displacement = 20.0

    def median_rms(n_nodes, spread=15.0):
        vals = []
        for _ in range(10):
            true_pos, peer_claims, peer_ranges = _attacker_view(n_nodes, rng, spread)
            vals.append(
                best_lie_at_displacement(
                    true_pos, peer_claims, peer_ranges, displacement, n_dirs=192
                )[0]
            )
        return float(np.median(vals))

    small, large = median_rms(4), median_rms(9)
    assert large > small, f"N=9 ({large:.2f}) should cost more than N=4 ({small:.2f})"
    assert large > noise_floor(), "a 20 m lie should be visible at N=9 with altitude spread"


@pytest.mark.slow
def test_coplanar_peers_cancel_the_benefit_of_swarm_size():
    """
    With peers in one plane the mirror solution survives at any swarm size, so
    adding drones buys nothing. This is the finding that makes altitude
    diversity a security requirement rather than a preference.
    """
    rng = np.random.default_rng(3)
    displacement = 20.0

    def median_rms(spread):
        vals = []
        for _ in range(10):
            true_pos, peer_claims, peer_ranges = _attacker_view(9, rng, spread)
            # Lift the attacker off the peers' plane; an in-plane attacker is its
            # own mirror image, so that configuration tests nothing.
            if spread == 0.0:
                true_pos = true_pos + np.array([0.0, 0.0, 12.0])
            vals.append(
                best_lie_at_displacement(
                    true_pos, peer_claims, peer_ranges, displacement, n_dirs=192
                )[0]
            )
        return float(np.median(vals))

    assert median_rms(0.0) < median_rms(15.0), "coplanar peers must be easier to fool"


def test_consistency_cost_grows_with_displacement_when_over_determined():
    rng = np.random.default_rng(4)
    true_pos, peer_claims, peer_ranges = _attacker_view(9, rng)
    near, _ = best_lie_at_displacement(true_pos, peer_claims, peer_ranges, 5.0, n_dirs=192)
    far, _ = best_lie_at_displacement(true_pos, peer_claims, peer_ranges, 30.0, n_dirs=192)
    assert far > near


def test_attacker_with_no_peers_is_unconstrained():
    """Degenerate input must not raise — an isolated node simply cannot be checked."""
    rms, claim = best_lie_at_displacement(np.zeros(3), np.empty((0, 3)), np.empty(0), 10.0)
    assert rms == 0.0
    np.testing.assert_allclose(claim, np.zeros(3))
