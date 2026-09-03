"""
Phase 0 - the premise test. The gate for the whole v3 plan.

Three questions, in dependency order:

  Test 1  Model validity.  With no attacker present, is the normalised
          innovation squared actually chi-squared at the claimed degrees of
          freedom? If not, the covariance is wrong and every number downstream
          is meaningless. Run against a deliberately wrong "independent
          residuals" model as a control.

  Test 2  Detectability.   Detection rate and false-alarm rate against a naive
          constant-offset spoofer, as a function of how far it lies.

  Test 3  The crux.        Against an *adaptive* spoofer that lies consistently,
          how does the achievable residual scale with swarm size? The
          constraint counting in sentrinet.attacks.adaptive predicts a threshold
          near 5 nodes, and predicts that a coplanar swarm never reaches it.

No RL, no reward function, no border_env. If Test 1 fails we fix the model
before going further; if Test 3 shows no threshold, the thesis needs rethinking
before a quarter is spent on it.

Usage:
    python -m experiments.premise_test              # all three
    python -m experiments.premise_test --test 3     # just the crux
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from sentrinet.attacks.adaptive import attack_margin_curve
from sentrinet.integrity.chi2 import global_nis, isolate_node, per_node_nis
from sentrinet.integrity.residuals import (
    residual_covariance,
    residual_jacobian,
    residual_vector,
)
from sentrinet.sensing.gnss import GnssModel, broadcast_claims
from sentrinet.sensing.uwb import UwbModel, measure_ranges
from sentrinet.world.geometry import link_list, random_formation

OUT_DIR = Path("results/premise")
GNSS = GnssModel(sigma=1.5)
UWB = UwbModel(sigma=0.10)


def noise_floor(gnss: GnssModel = GNSS, uwb: UwbModel = UWB) -> float:
    """Std of a single honest residual: sqrt(2 sigma_g^2 + sigma_r^2), in metres."""
    return float(np.sqrt(2.0 * gnss.sigma ** 2 + uwb.sigma ** 2))


def _one_epoch(n_nodes, rng, attacker=None, offset=None, altitude_spread=15.0):
    """Sample a formation, broadcast claims, measure ranges, build the residual."""
    links = link_list(n_nodes)
    truth = random_formation(n_nodes, rng, altitude_spread=altitude_spread)
    claims = broadcast_claims(truth, GNSS, rng, attacker=attacker, offset=offset)
    ranges = measure_ranges(truth, links, UWB, rng)
    resid = residual_vector(claims, ranges, links)
    jac = residual_jacobian(claims, links)
    cov = residual_covariance(jac, GNSS.sigma, UWB.sigma)
    return links, truth, claims, ranges, resid, cov, jac


# -- Test 1 -------------------------------------------------------------------
def test1_model_validity(n_nodes: int, trials: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    links = link_list(n_nodes)
    n_links = len(links)

    # rank(J) = 3N - 6: three translation and three rotation gauge freedoms
    # leave every pairwise residual unchanged.
    rank = min(n_links, 3 * n_nodes - 6)

    full, truncated, naive, per_node = [], [], [], []
    for _ in range(trials):
        _, _, _, _, resid, cov, _jac = _one_epoch(n_nodes, rng)
        full.append(global_nis(resid, cov)[0])
        truncated.append(global_nis(resid, cov, rank=rank)[0])
        # Control: pretend residuals are independent with the pairwise variance.
        naive.append(float(np.sum(resid ** 2) / noise_floor() ** 2))
        nis_k, _ = per_node_nis(resid, cov, links, n_nodes)
        per_node.append(nis_k[0])

    full = np.asarray(full)
    truncated = np.asarray(truncated)
    naive = np.asarray(naive)
    per_node = np.asarray(per_node)
    dof_node = n_nodes - 1

    rows = [
        dict(model="per-node (full covariance)", dof=dof_node,
             mean=per_node.mean(), expected_mean=dof_node,
             ks_p=float(stats.kstest(per_node, "chi2", args=(dof_node,)).pvalue)),
        dict(model="global, rank-truncated", dof=rank,
             mean=truncated.mean(), expected_mean=rank,
             ks_p=float(stats.kstest(truncated, "chi2", args=(rank,)).pvalue)),
        dict(model="global, naive dof (control)", dof=n_links,
             mean=full.mean(), expected_mean=n_links,
             ks_p=float(stats.kstest(full, "chi2", args=(n_links,)).pvalue)),
        dict(model="independence (control)", dof=n_links,
             mean=naive.mean(), expected_mean=n_links,
             ks_p=float(stats.kstest(naive, "chi2", args=(n_links,)).pvalue)),
    ]
    df = pd.DataFrame(rows)
    print("\n=== TEST 1 - model validity "
          "(N={}, {} trials) ===".format(n_nodes, trials))
    print(df.to_string(index=False, float_format=lambda v: "{:.4g}".format(v)))
    ok = df.loc[0, "ks_p"] > 0.01
    verdict = ("PASS - the attribution statistic is chi-squared" if ok
               else "FAIL - the covariance model is wrong")
    print("\n  per-node KS p        = {:.3g} -> {}".format(df.loc[0, "ks_p"], verdict))
    print("  rank-truncated KS p  = {:.3g}".format(df.loc[1, "ks_p"]))
    print("  controls (both expected to fail): naive dof {:.3g}, "
          "independence {:.3g}".format(df.loc[2, "ks_p"], df.loc[3, "ks_p"]))
    return df


# -- Test 2 -------------------------------------------------------------------
def test2_detectability(n_nodes: int, trials: int, seed: int,
                        alpha: float = 1e-3) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    offsets = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0])
    rows = []
    for mag in offsets:
        detected = correct_id = false_alarm = 0
        for _ in range(trials):
            attacker = int(rng.integers(n_nodes)) if mag > 0 else None
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            links, _, _, _, resid, cov, _ = _one_epoch(
                n_nodes, rng, attacker=attacker,
                offset=(mag * direction) if attacker is not None else None,
            )
            accused, _ = isolate_node(resid, cov, links, n_nodes, alpha=alpha)
            if mag == 0.0:
                false_alarm += int(accused is not None)
            else:
                detected += int(accused is not None)
                correct_id += int(accused == attacker)
        rows.append(dict(
            offset_m=float(mag),
            detection_rate=(detected / trials) if mag > 0 else np.nan,
            correct_isolation=(correct_id / trials) if mag > 0 else np.nan,
            false_alarm_rate=(false_alarm / trials) if mag == 0 else np.nan,
        ))
    df = pd.DataFrame(rows)
    print("\n=== TEST 2 - detectability (N={}, {} trials/point, alpha={}) ==="
          .format(n_nodes, trials, alpha))
    print("  honest residual noise floor: {:.2f} m".format(noise_floor()))
    print(df.to_string(index=False, float_format=lambda v: "{:.3f}".format(v)))
    return df


# -- Test 3 -------------------------------------------------------------------
def test3_adaptive_threshold(trials: int, seed: int,
                             node_counts=(3, 4, 5, 7, 9)) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 2)
    displacements = np.array([2.0, 5.0, 10.0, 15.0, 20.0, 30.0])
    floor = noise_floor()
    rows = []

    # The coplanar case must put the ATTACKER off the peers' plane. If every node
    # including the attacker is coplanar, the attacker's mirror image through that
    # plane is itself, so the mirror solution offers no lie at all -- which tests
    # nothing. Peers in a plane, attacker lifted out of it, is the real degenerate
    # geometry.
    geometries = (
        (15.0, 0.0, "3-D (altitude spread 15 m)"),
        (0.0, 12.0, "peers coplanar, attacker off-plane"),
    )

    for spread, attacker_lift, label in geometries:
        for n_nodes in node_counts:
            curves = []
            for _ in range(trials):
                links = link_list(n_nodes)
                truth = random_formation(n_nodes, rng, altitude_spread=spread)
                attacker = 0
                if attacker_lift:
                    truth = truth.copy()
                    truth[attacker, 2] += attacker_lift
                claims = broadcast_claims(truth, GNSS, rng)
                ranges = measure_ranges(truth, links, UWB, rng)
                peer_idx = [k for k in range(n_nodes) if k != attacker]
                peer_ranges = np.array([
                    ranges[m] for m, (i, j) in enumerate(links)
                    if attacker in (i, j)
                ])
                curves.append(attack_margin_curve(
                    truth[attacker], claims[peer_idx], peer_ranges,
                    displacements, n_dirs=256,
                ))
            curves = np.asarray(curves)
            for c, d in enumerate(displacements):
                med = float(np.median(curves[:, c]))
                rows.append(dict(
                    geometry=label, n_nodes=n_nodes, peers=n_nodes - 1,
                    displacement_m=float(d), median_rms_residual_m=med,
                    below_noise_floor=bool(med < floor),
                ))
    df = pd.DataFrame(rows)
    print("\n=== TEST 3 - adaptive attacker ({} trials/point) ===".format(trials))
    print("  honest residual noise floor: {:.2f} m".format(floor))
    print("  'below_noise_floor' = the attacker is statistically invisible.\n")
    for label in df["geometry"].unique():
        sub = df[df["geometry"] == label]
        piv = sub.pivot(index="n_nodes", columns="displacement_m",
                        values="median_rms_residual_m")
        print("  --- {} --- median achievable RMS residual (m) vs displacement"
              .format(label))
        print(piv.to_string(float_format=lambda v: "{:.2f}".format(v)))
        # The security-relevant quantity is not *whether* some lie is invisible
        # (a 2 m lie always is, and hardly matters) but how far the attacker can
        # move while staying under the noise floor.
        print("  max undetectable displacement per swarm size:")
        for n_nodes in sorted(sub.n_nodes.unique()):
            row = sub[sub.n_nodes == n_nodes].sort_values("displacement_m")
            invisible = row[row.median_rms_residual_m < floor]["displacement_m"]
            worst = float(invisible.max()) if len(invisible) else 0.0
            unbounded = len(invisible) == len(row)
            print("    N={:<2d} -> {}".format(
                int(n_nodes),
                ">= {:.0f} m (unbounded in tested range)".format(worst)
                if unbounded else "{:.0f} m".format(worst)))
        print("")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 0 premise test")
    ap.add_argument("--test", type=int, choices=[1, 2, 3], default=None,
                    help="run only one test (default: all)")
    ap.add_argument("--nodes", type=int, default=9)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--adaptive-trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run = {1: [1], 2: [2], 3: [3], None: [1, 2, 3]}[args.test]

    if 1 in run:
        test1_model_validity(args.nodes, args.trials, args.seed).to_csv(
            OUT_DIR / "test1_model_validity.csv", index=False)
    if 2 in run:
        test2_detectability(args.nodes, max(200, args.trials // 4), args.seed).to_csv(
            OUT_DIR / "test2_detectability.csv", index=False)
    if 3 in run:
        test3_adaptive_threshold(args.adaptive_trials, args.seed).to_csv(
            OUT_DIR / "test3_adaptive_threshold.csv", index=False)

    print("\nCSVs written to {}/".format(OUT_DIR))


if __name__ == "__main__":
    main()
