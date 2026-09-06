"""Figure 7 — Early-warning monitor AUROCs with run-level bootstrap CIs.

Source: runs/monitor_report.json
"""
from __future__ import annotations

import numpy as np

from _common import C, load_artifact, save_fig, start_fig


def main():
    rep = load_artifact("monitor_report.json")

    monitors = ["loss_only_threshold", "conventional_logistic",
                "sae_plus_conventional"]
    labels = ["loss-only\nthreshold", "conventional\nlogistic",
              "SAE +\nconventional"]
    aurocs, lows, highs = [], [], []
    for m in monitors:
        r = rep[m]
        aurocs.append(r["auroc"])
        ci = r["auroc_ci"]
        lows.append(r["auroc"] - ci["ci_lower"])
        highs.append(ci["ci_upper"] - r["auroc"])

    fig, ax = start_fig(figsize=(5.6, 3.4))
    xs = np.arange(len(monitors))
    ax.bar(xs, aurocs, width=0.5, color=[C["light"], C["main"], C["alt"]],
           alpha=0.9, edgecolor="black", linewidth=0.5)
    ax.errorbar(xs, aurocs, yerr=[lows, highs], fmt="none",
                ecolor="black", capsize=4, lw=1.2)
    ax.axhline(0.5, color="black", ls=":", lw=1, label="chance (AUROC 0.5)")

    for x, a in zip(xs, aurocs):
        ax.text(x, 0.06, f"{a:.3f}", ha="center", fontsize=8)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC (held-out runs)")
    ax.set_title("Monitors: conventional signals suffice; SAE adds nothing "
                 "(CIs overlap)")
    ax.legend(frameon=False)
    save_fig(fig, "figure_07_monitor")


if __name__ == "__main__":
    main()
