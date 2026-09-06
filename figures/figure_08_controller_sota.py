"""Figure 8 — Controller rescue trajectories + SOTA baseline comparison.

Panels:
  A. rescue trajectories (controller / no-action / oracle) + the
     monitor-source ablation arms (SAE vs random feature monitors)
  B. final rel-L2 bar chart: controller, no-action, oracle, GradNorm,
     NTK-adaptive, RBA

Sources:
  runs/controller_demo/controller_comparison.json
  runs/sota_baselines/sota_baseline_report.json
"""
from __future__ import annotations

import numpy as np

from _common import C, load_artifact, save_fig, start_fig


def main():
    cc = load_artifact("controller_demo/controller_comparison.json")
    sota = load_artifact("sota_baselines/sota_baseline_report.json")

    fig, axes = start_fig(1, 2, figsize=(9.6, 3.4))

    # ---- Panel A: trajectories ----
    ax = axes[0]
    arms = [("controller", C["main"], "-"),
            ("no_action", C["grey"], ":"),
            ("oracle_reweight", C["third"], "--"),
            ("controller_sae_monitor", C["alt"], "-"),
            ("controller_random_monitor", C["fourth"], "-.")]
    for arm, color, style in arms:
        r = cc["arms"][arm]
        traj = r["traj_rel_l2"]
        steps = np.arange(len(traj))
        ax.plot(steps, traj, style, color=color, lw=1.2, label=arm.replace("_", " "))
    identical = cc["verdict"]["monitor_source_ablation"].get(
        "trajectories_bit_identical")
    if identical:
        ax.text(0.03, 0.95,
                "SAE monitor ≡ random monitor (bit-identical trajectories)",
                transform=ax.transAxes, fontsize=7.5, va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor=C["grey"],
                          boxstyle="round"))
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("relative $L_2$ (log)")
    ax.set_title("A  Closed-loop rescue of boundary starvation")
    ax.legend(frameon=False, fontsize=6.5)

    # ---- Panel B: finals with SOTA ----
    ax = axes[1]
    ref = sota["reference_arms"]
    names = ["no-action", "GradNorm", "RBA", "controller\n(ours)",
             "oracle", "NTK-adaptive"]
    vals = [ref["no_action_final"],
            sota["baselines"]["GradNorm"]["final_rel_l2"],
            sota["baselines"]["RBA"]["final_rel_l2"],
            ref["controller_final"],
            ref["oracle_final"],
            sota["baselines"]["NTK-adaptive"]["final_rel_l2"]]
    colors = [C["grey"], C["light"], C["light"], C["main"], C["third"],
              C["alt"]]
    xs = np.arange(len(vals))
    ax.bar(xs, vals, color=colors, alpha=0.9, edgecolor="black",
           linewidth=0.5)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.03, f"{v:.4f}" if v < 0.1 else f"{v:.2f}",
                ha="center", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("final relative $L_2$ (log)")
    ax.set_title("B  Optimization methods on boundary starvation")

    save_fig(fig, "figure_08_controller_sota")


if __name__ == "__main__":
    main()
