"""Figure 2 — The activation manifold: geometry across the PDE suite.

Panels:
  A. per-run participation ratio distribution (1D suite, width 64)
  B. PR vs network width (1D Poisson, W = 16..512)
  C. mean PR by PDE family with tangent-rank annotations (the dimensional
     boundary), including the FNO positive control

Sources:
  runs/effective_rank_analysis/effective_rank_report.json
  runs/width_scaling/width_scaling_report.json
  runs/dimensional_boundary_expanded/dimensional_boundary_expanded_report.json
  runs/operator_boundary/operator_boundary_report.json
"""
from __future__ import annotations

import numpy as np

from _common import C, load_artifact, save_fig, start_fig


def main():
    eff = load_artifact("effective_rank_analysis/effective_rank_report.json")
    width = load_artifact("width_scaling/width_scaling_report.json")
    dexp = load_artifact(
        "dimensional_boundary_expanded/dimensional_boundary_expanded_report.json")
    oper = load_artifact("operator_boundary/operator_boundary_report.json")

    fig, axes = start_fig(1, 3, figsize=(9.6, 3.2))

    # ---- Panel A: PR distribution over 1D runs ----
    ax = axes[0]
    prs = [s["participation_ratio"]
           for s in eff["per_run"].values()]
    ax.hist(prs, bins=24, color=C["main"], alpha=0.85,
            edgecolor="black", linewidth=0.4)
    mean_pr = eff["summary"]["mean_participation_ratio"]
    dim = eff["summary"]["dim"]
    ax.axvline(mean_pr, color=C["alt"], ls="--", lw=1.2,
               label=f"mean = {mean_pr:.2f} / {dim} dims")
    ax.set_xlabel("covariance participation ratio")
    ax.set_ylabel("# runs")
    ax.set_title("A  1D suite: activations are low-rank")
    ax.legend(frameon=False)

    # ---- Panel B: PR vs width ----
    ax = axes[1]
    ws = [r["width"] for r in width["results"]]
    prs_w = [r["participation_ratio"] for r in width["results"]]
    ax.plot(ws, prs_w, "o-", color=C["main"], lw=1.2, ms=4,
            label="participation ratio")
    ax.axhline(2.0, color=C["grey"], ls=":", lw=1,
               label="d + 1 reference (d = 1)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ws)
    ax.set_xticklabels([str(w) for w in ws])
    ax.set_xlabel("hidden width $W$")
    ax.set_ylabel("participation ratio")
    ax.set_title("B  Width does not create superposition")
    ax.legend(frameon=False)

    # ---- Panel C: PR by family (+ FNO) ----
    ax = axes[2]
    fams = [("poisson 1D\n(ref, W=16–512)", 1.87, 1, C["main"]),
            ("poisson 2D\n(pilot)", 3.03, 2, C["main"]),
            ]
    for family, fr in dexp["families"].items():
        s = fr.get("summary", {})
        if s.get("mean_participation_ratio") is None:
            continue
        fams.append((family.replace("_", " ").replace(" 2d", " 2D")
                     .replace(" 1d", "\n(t, x)"), s["mean_participation_ratio"],
                    s["tangent_ranks"][0] if s["tangent_ranks"] else None,
                    C["main"]))
    fams.append(("FNO\n(function space)", oper["geometry"]["participation_ratio"],
                 None, C["alt"]))

    names = [f[0] for f in fams]
    vals = [f[1] for f in fams]
    colors = [f[3] for f in fams]
    xs = np.arange(len(fams))
    ax.bar(xs, vals, color=colors, alpha=0.85, edgecolor="black",
           linewidth=0.5)
    for x, (name, v, tr, _) in zip(xs, fams):
        tag = f"tangent={tr}" if tr else "PR/W=0.106"
        ax.text(x, v + 0.12, tag, ha="center", fontsize=6.5, color=C["grey"])
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("participation ratio")
    ax.set_title("C  Geometry across families + FNO boundary")
    ax.axhline(oper["geometry"]["participation_ratio"], color=C["alt"],
               ls=":", lw=0.8, alpha=0.5)

    save_fig(fig, "figure_02_activation_geometry")


if __name__ == "__main__":
    main()
