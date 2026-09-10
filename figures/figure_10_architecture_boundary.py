"""Figure 10 — The within-PINN architecture boundary (H18 centerpiece).

Panel A: activation participation ratio by architecture (depth sweep
2->6 tanh, Fourier-feature, FNO reference band) — depth lowers rank,
Fourier raises it into the operator neighborhood.
Panel B: SAE-vs-k-matched-PCA reconstruction ratio per architecture —
the Fourier PINN stands apart (25-56x; log scale).
Panel C: the phase diagram — PR/W (x) vs SAE advantage over PCA (y,
log scale) across every measured representation: tanh PINNs (all
widths/depths), Fourier PINN seeds, FNO. The preregistered rank
diagnostic predicted which runs would flip.

Sources:
  runs/architecture_boundary/architecture_boundary_report.json
  runs/operator_boundary/operator_boundary_report.json
  runs/effective_rank_analysis/effective_rank_report.json
  runs/sae_models_v2/sae_stage3_summary.json
"""
from __future__ import annotations

import numpy as np

from _common import C, RUNS, load_artifact, save_fig, start_fig


def main():
    arch = load_artifact("architecture_boundary/architecture_boundary_report.json")
    oper = load_artifact("operator_boundary/operator_boundary_report.json")
    eff = load_artifact("effective_rank_analysis/effective_rank_report.json")
    sae = load_artifact("sae_models_v2/sae_stage3_summary.json")

    # ---- Data ----
    depth_runs = [r for r in arch["per_run"] if r["arm"] == "depth"]
    fourier_runs = [r for r in arch["per_run"] if r["arm"] == "fourier"]

    depths = sorted({r["depth"] for r in depth_runs})
    pr_by_depth = [np.mean([r["participation_ratio"] for r in depth_runs
                            if r["depth"] == d]) for d in depths]
    fourier_prs = [r["participation_ratio"] for r in fourier_runs]
    fno_pr = oper["geometry"]["participation_ratio"]

    # Reconstruction ratios (PCA MSE / SAE MSE; >1 = SAE wins).
    # tanh PINN: SAE test MSE (replica mean) vs k-matched PCA train MSE
    # — the published stage-3 comparison (PCA ~50x BETTER there).
    pinn_sae = float(np.mean([r["test_recon_mse"] for r in sae["trained"]]))
    pinn_pca = sae["baselines"]["pca_k_matched_train_mse"]
    pinn_ratio = pinn_pca / pinn_sae          # ~0.02: PCA wins on tanh
    fno_ratio = oper["pca_k_recon"] / oper["sae"]["test_recon_mse"]  # 4.7
    fourier_ratios = [r["reconstruction"]["pca_over_sae_ratio"]
                      for r in fourier_runs if "reconstruction" in r]

    fig, (axA, axB, axC) = start_fig(figsize=(13.0, 4.0), ncols=3)

    # ---- Panel A: PR by architecture ----
    axA.plot(depths, pr_by_depth, "o-", color=C["main"], lw=2, ms=8,
             label="tanh depth sweep (mean of 3 seeds)")
    axA.scatter([0.0], [np.mean(fourier_prs)], s=110, color=C["alt"],
                marker="D", zorder=5, label=f"Fourier PINN (n_freq=32)")
    axA.axhline(fno_pr, color=C["grey"], ls="--", lw=1.5)
    axA.text(0.02, fno_pr + 0.25, f"FNO (PR {fno_pr:.1f})",
             color=C["grey"], fontsize=9)
    axA.axhspan(1.14, 2.51, color=C["grey"], alpha=0.12)
    axA.text(0.02, 2.7, "width-scaling envelope (tanh, widths 16-512)",
             color=C["grey"], fontsize=8)
    axA.set_xticks([0] + depths)
    axA.set_xticklabels(["Fourier"] + [str(d) for d in depths])
    axA.set_xlabel("architecture (width 64, fixed 1D Poisson)")
    axA.set_ylabel("activation participation ratio")
    axA.set_title("(A) Rank: depth lowers, Fourier raises", fontsize=10)
    axA.legend(fontsize=8, loc="upper right")

    # ---- Panel B: SAE advantage per architecture ----
    names = ["tanh\nPINNs", "Fourier\nPINN", "FNO"]
    ratios = [pinn_ratio, float(np.mean(fourier_ratios)), fno_ratio]
    colors = [C["grey"], C["alt"], C["main"]]
    bars = axB.bar(names, ratios, color=colors, width=0.55)
    axB.axhline(1.0, color="k", ls=":", lw=1)
    axB.set_yscale("log")
    axB.set_ylabel(r"PCA$_{k}$ MSE / SAE MSE  (log; $>$1 = SAE wins)")
    for bar, r in zip(bars, ratios):
        axB.text(bar.get_x() + bar.get_width() / 2, r * 1.25,
                 f"{r:.1f}x" if r >= 2 else f"{r:.2f}x",
                 ha="center", fontsize=9)
    axB.set_title("(B) Same TopK SAE, three regimes", fontsize=10)

    # ---- Panel C: phase diagram ----
    # tanh PINNs: one cloud point (mean PR, ratio ~1)
    axC.scatter([eff["summary"]["mean_participation_ratio"]
                 / eff["summary"]["dim"]],
                [pinn_ratio], s=90, color=C["grey"], marker="o",
                zorder=4, label="tanh PINNs (all widths/depths)")
    # R3 frequency sweep: the measured dose-response curve
    r3_path = RUNS / "r3_frequency_sweep" / "r3_report.json"
    if r3_path.exists():
        import json
        r3 = json.loads(r3_path.read_text())
        xs = [c["mean_rho"] for c in r3["curve_by_n_freq"].values()]
        ys = [c["topk_pca_over_sae_mean"]
              for c in r3["curve_by_n_freq"].values()]
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        axC.plot([xs[i] for i in order], [ys[i] for i in order],
                 "-", color=C["alt"], lw=1.6, alpha=0.8, zorder=3,
                 label="R3 frequency sweep (mean of 3 seeds)")
        axC.scatter(xs, ys, s=30, color=C["alt"], marker="o",
                    zorder=4, alpha=0.8)
    # R5 Burgers: the time-dependent replication
    r5_path = RUNS / "r5_burgers_boundary" / "r5_report.json"
    if r5_path.exists():
        import json
        r5 = json.loads(r5_path.read_text())
        agg = r5["aggregate"]
        axC.scatter([agg["fourier_mean_pr"] / 64],
                    [agg["fourier_mean_pca_over_sae"]],
                    s=110, color=C["third"], marker="^", zorder=6,
                    label="Burgers Fourier (t,x), R5a")
        axC.scatter([agg["tanh_mean_pr"] / 64],
                    [agg["tanh_mean_pca_over_sae"]],
                    s=80, color=C["fourth"], marker="v", zorder=6,
                    label="Burgers tanh (t,x)")
    # Fourier seeds
    for r in fourier_runs:
        if "reconstruction" in r:
            axC.scatter([r["pr_over_width"]],
                        [r["reconstruction"]["pca_over_sae_ratio"]],
                        s=70, color=C["alt"], marker="D", zorder=5)
    axC.scatter([fno_pr / oper["geometry"]["dim"]], [fno_ratio],
                s=110, color=C["main"], marker="s", zorder=5,
                label="FNO (function space)")
    axC.scatter([], [], s=70, color=C["alt"], marker="D",
                label="Fourier PINN (3 seeds)")
    axC.set_xscale("log")
    axC.set_yscale("log")
    axC.set_xlabel(r"$\rho$ = PR / width (log)")
    axC.set_ylabel("SAE advantage over $k$-matched PCA (log)")
    axC.set_title("(C) The measured phase diagram", fontsize=10)
    axC.axvline(3.0 / 64, color=C["grey"], ls="--", lw=1)
    axC.text(3.0 / 64 * 1.15, 0.4, "registered\nmove-bar", fontsize=7,
             color=C["grey"])
    axC.legend(fontsize=8, loc="upper left")

    save_fig(fig, "figure_10_architecture_boundary")


if __name__ == "__main__":
    main()
