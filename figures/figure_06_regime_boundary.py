"""Figure 6 — The regime boundary: representation rank vs SAE usefulness.

X: superposition ratio PR/W. Y: SAE-vs-PCA(k) reconstruction ratio
(SAE MSE / PCA MSE; < 1 means the SAE wins). PINN points from the
committed artifacts; FNO point from the operator boundary artifact.

Sources:
  runs/operator_boundary/operator_boundary_report.json
  runs/sae_models_v2/sae_stage3_summary.json
  runs/effective_rank_analysis/effective_rank_report.json
"""
from __future__ import annotations

import numpy as np

from _common import C, load_artifact, save_fig, start_fig


def main():
    oper = load_artifact("operator_boundary/operator_boundary_report.json")
    eff = load_artifact("effective_rank_analysis/effective_rank_report.json")
    sae = load_artifact("sae_models_v2/sae_stage3_summary.json")

    # PINN point: PR/64 from effective-rank summary; ratio SAE/PCA-k from
    # stage-3 summary (replica-mean SAE MSE vs k-matched PCA MSE).
    pinn_pr = eff["summary"]["mean_participation_ratio"]
    pinn_w = eff["summary"]["dim"]
    sae_mean = float(np.mean([r["test_recon_mse"]
                              for r in sae["trained"]]))
    pca_k = sae["baselines"]["pca_k_matched_train_mse"]
    pinn_ratio = sae_mean / pca_k

    # FNO point.
    fno_pr = oper["geometry"]["participation_ratio"]
    fno_w = oper["geometry"]["dim"]
    fno_ratio = oper["sae"]["test_recon_mse"] / oper["pca_k_recon"]

    fig, ax = start_fig(figsize=(6.4, 4.0))

    ax.scatter([pinn_pr / pinn_w], [pinn_ratio], s=90, color=C["main"],
               zorder=3, label="1D PINNs (Poisson/Adv/RD, W=64)")
    ax.scatter([fno_pr / fno_w], [fno_ratio], s=90, color=C["alt"],
               marker="s", zorder=3,
               label="FNO block states (Green's task, W=64)")

    # LLM literature anchor (qualitative; marked distinctly).
    ax.scatter([0.5], [0.5], s=70, color=C["grey"], marker="^", zorder=3,
               label="LLM residual streams (literature, qualitative)")
    ax.set_xscale("log")
    ax.axhline(1.0, color="black", ls="--", lw=1)
    ax.axvspan(0.09, 0.6, color=C["third"], alpha=0.08)
    ax.text(0.095, 0.28, "superposition-plausible\n(SAE appropriate)",
            fontsize=8, color=C["third"], va="bottom")
    ax.text(0.005, 8.0, "low-rank regime\n(SAE solves a nonexistent problem)",
            fontsize=8, color=C["main"])

    ax.annotate(f"SAE loses 12x\n(PR/W={pinn_pr/pinn_w:.3f})",
                (pinn_pr / pinn_w, pinn_ratio),
                textcoords="offset points", xytext=(-70, -6), fontsize=8)
    ax.annotate(f"SAE WINS 4.7x\n(PR/W={fno_pr/fno_w:.3f})",
                (fno_pr / fno_w, fno_ratio),
                textcoords="offset points", xytext=(-100, 12), fontsize=8)

    ax.set_yscale("log")
    ax.set_xlabel("superposition ratio  PR / W  (log)")
    ax.set_ylabel("recon MSE ratio  SAE / PCA(k=8)  (log)\n(<1: SAE wins)")
    ax.set_title("The regime boundary, measured on both sides")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    save_fig(fig, "figure_06_regime_boundary")


if __name__ == "__main__":
    main()
