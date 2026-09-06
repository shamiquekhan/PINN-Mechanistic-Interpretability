"""Figure 1 — Failure atlas reproducibility (10 seeds per regime).

Source: runs/failure_atlas/seed_matrix_stats.json
"""
from __future__ import annotations

import numpy as np

from _common import C, load_artifact, save_fig, start_fig


def main():
    stats = load_artifact("failure_atlas/seed_matrix_stats.json")
    regimes = ["success", "boundary_starvation", "gradient_conflict",
               "spectral_suppression", "collocation_starvation"]
    labels = ["success", "boundary\nstarvation", "gradient\nconflict",
              "spectral\nsuppression", "collocation\nstarvation"]

    fig, ax = start_fig(figsize=(6.4, 3.4))

    xs = np.arange(len(regimes))
    means, lows, highs, ns = [], [], [], []
    for r in regimes:
        d = stats[r]
        ci = d["rel_l2_ci"]
        means.append(ci["mean"])
        lows.append(ci["mean"] - ci["ci_lower"])
        highs.append(ci["ci_upper"] - ci["mean"])
        ns.append(d["n_seeds"])

    means = np.array(means)
    lows = np.array(lows)
    highs = np.array(highs)

    bars = ax.bar(xs, means, width=0.55, color=C["main"], alpha=0.85,
                  edgecolor="black", linewidth=0.5)
    ax.errorbar(xs, means, yerr=[lows, highs], fmt="none", ecolor="black",
                capsize=3, lw=1)

    # Failure threshold reference line (operational label definition).
    ax.axhline(0.05, color=C["alt"], ls="--", lw=1, alpha=0.8,
               label="failure threshold (rel $L_2$ = 0.05)")

    # Label counts above bars.
    for x, m, n, r in zip(xs, means, ns, regimes):
        top_label = stats[r]["label_counts"]
        dominant = max(top_label.items(), key=lambda kv: kv[1])[0] \
            if top_label else "?"
        ax.text(x, m + 0.08, f"n={n}\n{dominant.split('_')[0]}",
                ha="center", va="bottom", fontsize=7, color=C["grey"])

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel("final relative $L_2$ (log scale)")
    ax.set_title("Failure atlas: reproducibility across 10 seeds per regime")
    ax.legend(loc="upper right", frameon=False)
    save_fig(fig, "figure_01_failure_atlas")


if __name__ == "__main__":
    main()
