"""
The cross-modal residual and its covariance.

For every link (i, j) we compare the distance implied by two *claimed* GNSS
positions against the distance the radio actually measured:

    r_ij = || c_i - c_j ||  -  rho_ij

Under honest operation, linearising about the true separation d = p_i - p_j
with unit vector u = d / ||d||:

    || c_i - c_j || ~= ||d|| + u . (n_i - n_j)
    r_ij            ~= u . (n_i - n_j) - nu_ij

so r_ij is zero-mean with variance 2*sigma_g^2 + sigma_r^2.

**The residuals are not independent.** Every residual incident to node i
carries that node's single GNSS error n_i, so residuals sharing a node are
correlated. Treating them as independent is the most natural mistake to make
here, and it makes the resulting statistic *not* chi-squared — which is
precisely what the premise test's Test 1 exists to catch.

Stacking the M residuals and differentiating with respect to the 3N claimed
coordinates gives a sparse Jacobian J, and

    S = sigma_g^2 * J J^T + sigma_r^2 * I_M

Note that with metre-scale GNSS noise and centimetre-scale ranging noise, the
sigma_g term dominates: detection power is set by how badly the *claims* are
known, not by the radio. The radio's job is to be trustworthy, not precise.
"""

from __future__ import annotations

import numpy as np

Link = tuple[int, int]


def residual_vector(
    claimed: np.ndarray, measured_ranges: np.ndarray, links: list[Link]
) -> np.ndarray:
    """Claim-implied distance minus measured range, shape (n_links,)."""
    claimed = np.asarray(claimed, dtype=float)
    implied = np.array([np.linalg.norm(claimed[i] - claimed[j]) for i, j in links], dtype=float)
    return implied - np.asarray(measured_ranges, dtype=float)


def residual_jacobian(claimed: np.ndarray, links: list[Link]) -> np.ndarray:
    """
    d r / d (claimed coordinates), shape (n_links, 3 * n_nodes).

    Row for link (i, j) holds +u_ij in node i's block and -u_ij in node j's.
    """
    claimed = np.asarray(claimed, dtype=float)
    n_nodes = claimed.shape[0]
    jac = np.zeros((len(links), 3 * n_nodes), dtype=float)
    for row, (i, j) in enumerate(links):
        diff = claimed[i] - claimed[j]
        dist = np.linalg.norm(diff)
        if dist < 1e-9:
            # Coincident claims: the direction is undefined. Leave the row zero;
            # the sigma_r term keeps S invertible.
            continue
        unit = diff / dist
        jac[row, 3 * i : 3 * i + 3] = unit
        jac[row, 3 * j : 3 * j + 3] = -unit
    return jac


def residual_covariance(jacobian: np.ndarray, sigma_gnss: float, sigma_range: float) -> np.ndarray:
    """
    S = sigma_g^2 J J^T + sigma_r^2 I, shape (n_links, n_links).

    The off-diagonal structure is the whole point — it is what makes the
    normalised statistic chi-squared instead of merely chi-squared-shaped.
    """
    n_links = jacobian.shape[0]
    return (sigma_gnss**2) * (jacobian @ jacobian.T) + (sigma_range**2) * np.eye(n_links)
