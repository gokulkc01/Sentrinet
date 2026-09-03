"""
Chi-squared integrity tests over the cross-modal residual.

This is the module that replaces the hand-tuned heuristic in the v1 trust
module, where trust fell out of `max(0, 1 - error / 5.0)`. There was no way to
state a false-alarm rate for that constant, and therefore no way to defend it.

Here the threshold comes from the model instead: under the no-attack hypothesis
the normalised innovation squared is chi-squared distributed, so `alpha` *is*
the false-alarm probability. That is the same machinery aviation RAIM uses to
certify integrity, applied one level up — to peer nodes rather than to
satellites.

Detection vs. isolation: a single spoofed node biases every residual it
participates in, so its honest peers' statistics inflate too. `isolate_node`
therefore attributes to the node with the largest normalised statistic rather
than flagging everything above threshold.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy import stats

Link = Tuple[int, int]


def _quadratic_form(residual: np.ndarray, covariance: np.ndarray) -> float:
    """r^T S^-1 r, solved rather than inverted."""
    sol = np.linalg.solve(covariance, residual)
    return float(residual @ sol)


def global_nis(
    residual: np.ndarray,
    covariance: np.ndarray,
    rank: Optional[int] = None,
) -> Tuple[float, int]:
    """
    Whole-network statistic and its degrees of freedom.

    Answers "is anything wrong?" but not "who?".

    **The naive degrees of freedom (one per link) are wrong.** Residuals are
    built from 3N claimed coordinates, and translating or rotating every claim
    together leaves all of them unchanged, so the Jacobian has a six-dimensional
    gauge null space and the residual actually lives in a subspace of dimension
    `rank(J) = 3N - 6` (measured empirically, exactly as predicted). The
    remaining M - rank(J) directions carry ranging noise only, where the
    first-order residual model is at its weakest — the neglected second-order
    term is O(sigma_g^2 / link_length) and swamps sigma_r on short links. Using
    them inflates the statistic by an order of magnitude.

    Passing `rank` projects onto the leading eigenvectors of the covariance and
    reports that many degrees of freedom. Use `rank=numpy.linalg.matrix_rank(J)`.

    The per-node statistic in `per_node_nis` does not suffer from this: its
    sub-covariance has no near-null directions, so it is well conditioned
    regardless of geometry. That is the statistic to prefer for attribution.
    """
    if rank is None:
        return _quadratic_form(residual, covariance), int(residual.shape[0])

    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1][:rank]
    projected = eigvecs[:, order].T @ residual
    return float(np.sum(projected ** 2 / eigvals[order])), int(rank)


def per_node_nis(
    residual: np.ndarray,
    covariance: np.ndarray,
    links: List[Link],
    n_nodes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-node statistic using only the residuals incident to each node.

    Returns
    -------
    (nis, dof) : each shape (n_nodes,). For a fully connected mesh dof is
                 n_nodes - 1; with occlusion-severed links it is lower, which is
                 exactly how reduced connectivity erodes detection power.
    """
    nis = np.zeros(n_nodes, dtype=float)
    dof = np.zeros(n_nodes, dtype=int)
    for k in range(n_nodes):
        idx = [m for m, (i, j) in enumerate(links) if i == k or j == k]
        if not idx:
            nis[k], dof[k] = 0.0, 0
            continue
        sub_r = residual[idx]
        sub_s = covariance[np.ix_(idx, idx)]
        nis[k] = _quadratic_form(sub_r, sub_s)
        dof[k] = len(idx)
    return nis, dof


def threshold(dof: int, alpha: float = 1e-3) -> float:
    """
    Chi-squared critical value. `alpha` is the per-test false-alarm probability.

    With one test per node per epoch, the network-level false-alarm rate is
    roughly n_nodes * alpha — budget accordingly rather than reusing a
    single-test alpha unchanged.
    """
    if dof <= 0:
        return float("inf")
    return float(stats.chi2.ppf(1.0 - alpha, df=dof))


def isolate_node(
    residual: np.ndarray,
    covariance: np.ndarray,
    links: List[Link],
    n_nodes: int,
    alpha: float = 1e-3,
) -> Tuple[Optional[int], np.ndarray]:
    """
    Flag the most anomalous node, or None if the network looks clean.

    Returns (accused, normalised) where `normalised` is each node's statistic
    divided by its own threshold, so values are comparable across nodes whose
    degrees of freedom differ (which happens as soon as occlusion cuts links).
    """
    nis, dof = per_node_nis(residual, covariance, links, n_nodes)
    normalised = np.zeros(n_nodes, dtype=float)
    for k in range(n_nodes):
        thr = threshold(int(dof[k]), alpha)
        normalised[k] = nis[k] / thr if np.isfinite(thr) and thr > 0 else 0.0
    if not np.any(normalised > 1.0):
        return None, normalised
    return int(np.argmax(normalised)), normalised
