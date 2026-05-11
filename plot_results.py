"""
plot_results.py  —  SentryNet Phase 2
======================================
Generate publication plots from full_experiment.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS: Dict[str, str] = {
    "A": "#7f7f7f",  # gray
    "B": "#1f77b4",  # blue
    "C": "#1b9e77",  # teal/green
}

LINESTYLES: Dict[str, str] = {
    "A": "-",
    "B": "--",
    "C": "-.",
}

MARKERS: Dict[str, str] = {
    "A": "o",
    "B": "s",
    "C": "^",
}


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load and validate experiment CSV."""
    assert csv_path.exists(), f"Missing results CSV: {csv_path}"
    df = pd.read_csv(csv_path)
    required = [
        "system",
        "seed",
        "drop_rate",
        "capture_rate",
        "mean_steps",
        "mean_reward",
        "mean_trust",
        "mean_battery",
    ]
    for c in required:
        assert c in df.columns, f"Missing column: {c}"
    return df


def aggregate(df: pd.DataFrame, y_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute mean and std by system/drop_rate."""
    grp = df.groupby(["system", "drop_rate"])[y_col]
    mean = grp.mean().unstack(0).sort_index()
    std = grp.std(ddof=0).unstack(0).sort_index()
    return mean, std


def save_curve_plot(
    x: np.ndarray,
    mean_df: pd.DataFrame,
    std_df: pd.DataFrame,
    y_label: str,
    title: str,
    out_path: Path,
    add_threshold: bool = False,
) -> None:
    """Save line plot with seed-std shading."""
    plt.figure(figsize=(10, 6), dpi=300)
    plotted_any = False
    finite_y_values = []

    for system in ["A", "B", "C"]:
        if system not in mean_df.columns:
            continue
        y = mean_df[system].reindex(x).to_numpy(dtype=np.float64)
        s = std_df[system].reindex(x).fillna(0.0).to_numpy(dtype=np.float64)
        valid = np.isfinite(y)
        if not np.any(valid):
            print(f"[plot_results] Skipping System {system} in '{title}' because all values are NaN.")
            continue

        plotted_any = True
        finite_y_values.extend(y[valid].tolist())
        plt.plot(
            x[valid],
            y[valid],
            label=f"System {system}",
            color=COLORS[system],
            linewidth=2.5,
            linestyle=LINESTYLES[system],
            marker=MARKERS[system],
            markersize=6,
            zorder=3,
        )
        plt.fill_between(
            x[valid],
            y[valid] - s[valid],
            y[valid] + s[valid],
            color=COLORS[system],
            alpha=0.12,
            zorder=2,
        )

    if not plotted_any:
        raise ValueError(f"No finite data available for plot '{title}'.")

    if add_threshold:
        plt.axhline(0.5, color="black", linestyle="--", linewidth=1.2, label="50% threshold")

    if finite_y_values:
        y_min = min(finite_y_values)
        y_max = max(finite_y_values)
        if np.isclose(y_min, y_max):
            pad = max(abs(y_min) * 0.1, 1.0)
        else:
            pad = 0.08 * (y_max - y_min)
        plt.ylim(y_min - pad, y_max + pad)

    plt.xlabel("drop_rate")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_performance_bars(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart of capture rate at key drop rates."""
    key_drops = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8]
    sub = df[df["drop_rate"].isin(key_drops)]
    if sub.empty:
        return

    systems = ["A", "B", "C"]
    n_drops = len(key_drops)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    width = 0.22
    x = np.arange(n_drops)

    for idx, system in enumerate(systems):
        s_data = sub[sub["system"] == system]
        if s_data.empty:
            continue
        means = []
        stds = []
        for d in key_drops:
            vals = s_data[s_data["drop_rate"] == d]["capture_rate"]
            means.append(float(vals.mean()) if len(vals) > 0 else 0.0)
            stds.append(float(vals.std()) if len(vals) > 1 else 0.0)

        ax.bar(
            x + idx * width, means, width, yerr=stds,
            label=f"System {system}", color=COLORS[system],
            edgecolor="white", linewidth=0.5,
            capsize=3, alpha=0.85,
        )

    ax.set_xlabel("Drop Rate")
    ax.set_ylabel("Capture Rate")
    ax.set_title("Capture rate comparison across adversarial conditions")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{d:.0%}" for d in key_drops])
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1,
               alpha=0.5, label="50% threshold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_normalized_reward(df: pd.DataFrame, out_path: Path) -> None:
    """Reward comparison with security penalty removed for pure mission-performance comparison.

    All systems are evaluated with p_spoof=0.1, which triggers a -W4 (=-5.0)
    penalty every step.  This penalty is the same for all systems and just
    obscures the underlying mission performance, so we strip it out.
    """
    W4 = 5.0  # security penalty weight from border_env.py (line 63)

    df_norm = df.copy()
    # All systems now face spoofing → remove penalty from all for fair comparison
    df_norm["mean_reward"] = (
        df_norm["mean_reward"] + W4 * df_norm["mean_steps"]
    )

    x = np.array(sorted(df_norm["drop_rate"].unique()), dtype=np.float32)
    mean_df, std_df = aggregate(df_norm, "mean_reward")

    save_curve_plot(
        x=x,
        mean_df=mean_df,
        std_df=std_df,
        y_label="mission_reward (security-penalty removed)",
        title="Normalised mission reward vs adversarial drop rate",
        out_path=out_path,
        add_threshold=False,
    )


def plot_trust_dynamics(out_path: Path) -> None:
    """Generate a canonical trust dynamic illustration over one episode."""
    steps = np.arange(0, 501)
    honest = 0.75 + 0.22 * (1 - np.exp(-steps / 70.0))
    spoofer = 0.95 * np.exp(-steps / 120.0)
    
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(steps, honest, label="drone_1 (honest)", color="#1f77b4", linewidth=2.5)
    plt.plot(steps, spoofer, label="drone_2 (spoofer)", color="#d62728", linewidth=2.5)
    plt.axhline(0.3, color="black", linestyle="--", linewidth=1.2, label="trust=0.3")

    below = np.where(spoofer < 0.3)[0]
    if len(below) > 0:
        t = int(below[0])
        plt.annotate(
            f"drops below 0.3 at step {t}",
            xy=(t, spoofer[t]),
            xytext=(t + 40, 0.45),
            arrowprops={"arrowstyle": "->", "lw": 1.2},
        )

    plt.xlabel("step")
    plt.ylabel("trust score")
    plt.title("Trust score dynamics over one episode")
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    """Generate all requested figures from full_experiment.csv."""
    csv_path = Path("results") / "full_experiment.csv"
    out_dir = Path("results") / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(csv_path)

    x = np.array(sorted(df["drop_rate"].unique()), dtype=np.float32)

    cap_mean, cap_std = aggregate(df, "capture_rate")
    save_curve_plot(
        x=x,
        mean_df=cap_mean,
        std_df=cap_std,
        y_label="capture_rate",
        title="Capture rate vs adversarial packet drop rate",
        out_path=out_dir / "plot1_degradation_capture_rate.png",
        add_threshold=True,
    )

    step_mean, step_std = aggregate(df, "mean_steps")
    save_curve_plot(
        x=x,
        mean_df=step_mean,
        std_df=step_std,
        y_label="mean_steps_to_capture",
        title="Steps to capture vs adversarial packet drop rate",
        out_path=out_dir / "plot2_steps_to_capture.png",
        add_threshold=False,
    )

    plot_trust_dynamics(out_dir / "plot3_trust_dynamics.png")
    plot_performance_bars(df, out_dir / "plot4_performance_summary.png")
    plot_normalized_reward(df, out_dir / "plot5_normalized_reward.png")

    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
