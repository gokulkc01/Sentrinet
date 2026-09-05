"""
The tests that matter most in this repository.

Our largest technical risk is a *silent* modelling error: a statistic that does
not crash, it just quietly returns wrong numbers that look plausible. v2 ran for
months on a broken observation normalisation; phase 0 shipped two modelling
errors in its first draft.

No linter catches that class of bug. Asserting the null distribution does. Every
statistic here is checked against the distribution it is supposed to follow when
nothing is wrong, and every check is paired with a control that must FAIL — a
control that silently starts passing is itself a regression.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import GNSS, UWB, noise_floor, simulate_epoch
from scipy import stats

from sentrinet.integrity.chi2 import (
    global_nis,
    isolate_node,
    per_node_nis,
    threshold,
)


# ── the central claim ────────────────────────────────────────────────────────
@pytest.mark.slow
def test_per_node_nis_is_chi2_under_h0():
    """
    With no attacker, each node's statistic must be chi-squared at (N-1) dof.

    This is the claim the whole monitor rests on: if it holds, `alpha` really is
    the false-alarm probability and we can promise an operator a number.
    """
    n_nodes, trials = 9, 1200
    rng = np.random.default_rng(0)
    samples = np.array(
        [
            per_node_nis(e.residual, e.covariance, e.links, n_nodes)[0][0]
            for e in (simulate_epoch(n_nodes, rng) for _ in range(trials))
        ]
    )
    dof = n_nodes - 1
    assert abs(samples.mean() - dof) < 0.4, f"mean {samples.mean():.3f}, expected {dof}"
    assert stats.kstest(samples, "chi2", args=(dof,)).pvalue > 0.01


@pytest.mark.slow
def test_independence_model_fails_the_same_check():
    """
    CONTROL — must fail.

    Residuals sharing a node share that node's GNSS error, so they are
    correlated. Pretending otherwise gives the right *mean* and the wrong
    *distribution*, which is exactly the kind of error that hides. If this ever
    starts passing, the covariance has stopped doing its job.
    """
    n_nodes, trials = 9, 1200
    rng = np.random.default_rng(0)
    naive = np.array(
        [
            float(np.sum(simulate_epoch(n_nodes, rng).residual ** 2) / noise_floor() ** 2)
            for _ in range(trials)
        ]
    )
    n_links = n_nodes * (n_nodes - 1) // 2
    # The mean comes out right, which is what makes this failure mode dangerous.
    assert abs(naive.mean() - n_links) < 1.5
    # ... but the distribution does not.
    assert stats.kstest(naive, "chi2", args=(n_links,)).pvalue < 0.01


# ── the structural property we discovered ────────────────────────────────────
@pytest.mark.parametrize("n_nodes", [5, 6, 9])
def test_jacobian_rank_is_3n_minus_6(n_nodes):
    """
    Translating or rotating every claim together leaves all pairwise residuals
    unchanged, so the Jacobian carries a six-dimensional gauge null space.

    This is also the monitor's fundamental blind spot: a common-mode shift is
    invisible to any purely relative detector. Encoding it as a test keeps the
    limitation from being quietly "fixed" by a future change.
    """
    rng = np.random.default_rng(1)
    epoch = simulate_epoch(n_nodes, rng)
    expected = min(len(epoch.links), 3 * n_nodes - 6)
    assert np.linalg.matrix_rank(epoch.jacobian, tol=1e-8) == expected


def test_common_mode_shift_produces_no_residual_change():
    """A rigid translation of every claim must leave the residual untouched."""
    rng = np.random.default_rng(2)
    epoch = simulate_epoch(7, rng)
    from sentrinet.integrity.residuals import residual_vector

    shifted = epoch.claims + np.array([12.0, -5.0, 3.0])
    after = residual_vector(shifted, epoch.ranges, epoch.links)
    np.testing.assert_allclose(after, epoch.residual, atol=1e-9)


def test_global_nis_needs_rank_truncation():
    """
    CONTROL — the naive degrees of freedom are wrong.

    Using one dof per link inflates the network statistic several-fold because
    the residual only lives in a (3N-6)-dimensional subspace. Truncating to that
    rank brings the mean back to its expected value.
    """
    n_nodes, trials = 9, 300
    rng = np.random.default_rng(3)
    rank = 3 * n_nodes - 6
    full, truncated = [], []
    for _ in range(trials):
        e = simulate_epoch(n_nodes, rng)
        full.append(global_nis(e.residual, e.covariance)[0])
        truncated.append(global_nis(e.residual, e.covariance, rank=rank)[0])
    n_links = n_nodes * (n_nodes - 1) // 2
    assert np.mean(full) > 3 * n_links, "naive dof should be badly inflated"
    assert abs(np.mean(truncated) - rank) < 2.0


# ── calibration and sensitivity ──────────────────────────────────────────────
def test_residual_variance_matches_theory():
    """A single honest residual has standard deviation sqrt(2*sigma_g^2 + sigma_r^2)."""
    rng = np.random.default_rng(4)
    pooled = np.concatenate([simulate_epoch(6, rng).residual for _ in range(400)])
    assert abs(pooled.std() - noise_floor()) < 0.12


def test_honest_residuals_carry_a_small_positive_linearisation_bias():
    """
    Honest residuals are NOT zero-mean, and the offset is not noise.

    The first-order model drops a second-order term: expanding ||d + e|| about
    the true separation d leaves E[||e_perp||^2] / (2||d||), which for
    e ~ N(0, 2 sigma_g^2 I) evaluates to **2 sigma_g^2 / ||d||**. It is strictly
    positive and grows as links get shorter.

    This matters beyond bookkeeping: it is a *systematic* bias with the same
    sign as NLoS range bias, so the two are confounded, and it makes the
    residual model slightly optimistic on short links. Asserted here so the
    magnitude cannot drift unnoticed.
    """
    from sentrinet.world.geometry import true_ranges

    rng = np.random.default_rng(4)
    residuals, lengths = [], []
    for _ in range(400):
        e = simulate_epoch(6, rng)
        residuals.append(e.residual)
        lengths.append(true_ranges(e.truth, e.links))
    measured = float(np.concatenate(residuals).mean())
    predicted = float((2.0 * GNSS.sigma**2 / np.concatenate(lengths)).mean())

    assert measured > 0.0, "the linearisation bias is strictly positive"
    assert abs(measured / predicted - 1.0) < 0.2, (
        f"measured bias {measured:.4f} m vs predicted {predicted:.4f} m — "
        "the second-order term no longer explains it"
    )


@pytest.mark.slow
def test_false_alarm_rate_respects_alpha():
    """
    With one test per node, the network-level false-alarm rate is about N*alpha.

    A monitor that cannot keep this promise is worse than no monitor: operators
    stop believing it and switch it off.
    """
    n_nodes, trials, alpha = 9, 600, 1e-3
    rng = np.random.default_rng(5)
    alarms = sum(
        isolate_node(e.residual, e.covariance, e.links, n_nodes, alpha=alpha)[0] is not None
        for e in (simulate_epoch(n_nodes, rng) for _ in range(trials))
    )
    assert alarms / trials < 0.04, f"false-alarm rate {alarms / trials:.3f} too high"


@pytest.mark.slow
def test_large_offset_is_detected():
    """Sanity in the other direction: a 20 m error must not be missed."""
    n_nodes, trials = 9, 200
    rng = np.random.default_rng(6)
    detected = 0
    for _ in range(trials):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        e = simulate_epoch(n_nodes, rng, attacker=0, offset=20.0 * direction)
        accused, _ = isolate_node(e.residual, e.covariance, e.links, n_nodes)
        detected += accused is not None
    assert detected / trials > 0.9


def test_threshold_grows_with_dof_and_shrinks_with_alpha():
    assert threshold(8, 1e-3) > threshold(8, 1e-2)
    assert threshold(20, 1e-3) > threshold(8, 1e-3)
    assert not np.isfinite(threshold(0)), "a node with no links cannot be tested"


def test_isolate_returns_none_on_a_clean_network():
    """An unremarkable epoch should produce no accusation at all."""
    rng = np.random.default_rng(7)
    clean = 0
    for _ in range(50):
        e = simulate_epoch(9, rng)
        if isolate_node(e.residual, e.covariance, e.links, 9, alpha=1e-3)[0] is None:
            clean += 1
    assert clean >= 45


def test_per_node_dof_matches_link_count():
    """Degrees of freedom must track actual connectivity, not assumed connectivity."""
    rng = np.random.default_rng(8)
    e = simulate_epoch(6, rng)
    _, dof = per_node_nis(e.residual, e.covariance, e.links, 6)
    assert list(dof) == [5] * 6

    # Drop every link touching node 0; its dof must fall to zero and the rest to 4.
    kept = [i for i, (a, b) in enumerate(e.links) if 0 not in (a, b)]
    links = [e.links[i] for i in kept]
    _, dof2 = per_node_nis(e.residual[kept], e.covariance[np.ix_(kept, kept)], links, 6)
    assert dof2[0] == 0
    assert list(dof2[1:]) == [4] * 5


def test_sensor_models_are_configured_as_documented():
    """Guards against a silent change of the constants every number depends on."""
    assert GNSS.sigma == 1.5
    assert UWB.sigma == 0.10
    assert abs(noise_floor() - 2.1237) < 1e-3
