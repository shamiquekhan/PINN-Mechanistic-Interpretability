"""Figure 5 — Causal abstraction: interchange movement fractions per run.

Source: runs/causal_abstraction_results.json
"""
from __future__ import annotations

import numpy as np

from _common import C, load_artifact, save_fig, start_fig


def main():
    rep = load_artifact("causal_abstraction_results.json")

    runs = sorted(rep["per_run"].keys())
    methods = ["pca", "sae", "random"]
    colors = {"pca": C["main"], "sae": C["alt"], "random": C["grey"]}

    fig, ax = start_fig(figsize=(7.6, 3.4))

    xs = np.arange(len(runs))
    width = 0.26
    for i, m in enumerate(methods):
        movs = [rep["per_run"][r][m]["mean_movement_fraction"]
                for r in runs]
        ax.bar(xs + (i - 1) * width, movs, width=width,
               color=colors[m], label=m, alpha=0.85,
               edgecolor="black", linewidth=0.4)

    # Run-level consistency gate: a candidate must beat random on EVERY run.
    v = rep["verdict"]
    note = (f"PCA beats random every run: {v['pca']['beats_random_every_run']} "
            f"({v['pca']['n_runs_positive_diff']}/{v['pca']['n_runs']} positive)\n"
            f"SAE beats random every run: {v['sae']['beats_random_every_run']} "
            f"({v['sae']['n_runs_positive_diff']}/{v['sae']['n_runs']} positive)")
    ax.text(0.02, 0.97, note, transform=ax.transAxes, va="top",
            fontsize=8, bbox=dict(facecolor="white", alpha=0.8,
                                  edgecolor=C["grey"], boxstyle="round"))

    ax.set_xticks(xs)
    ax.set_xticklabels([r.replace("boundary_starvation_", "bs_")
                        .replace("failure_", "f_") for r in runs],
                       rotation=60, ha="right", fontsize=6.5)
    ax.set_ylabel("interchange movement fraction")
    ax.set_title("Partial interchanges: no alignment basis beats random "
                 "(causal-abstraction null)")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    save_fig(fig, "figure_05_causal_abstraction")


if __name__ == "__main__":
    main()
