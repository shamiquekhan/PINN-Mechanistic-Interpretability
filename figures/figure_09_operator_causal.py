"""Figure 9 — The operator causal battery (stage 14): FNO vs PINN batteries.

Panels:
  A. per-feature sign-test p-values with Bonferroni threshold, across the
     three batteries (SAE-on-PINN, PCA-on-PINN, SAE-on-FNO) — the survivor
     asymmetry at a glance.
  B. beats-all-controls rate and median target/control ratio per battery.

Sources:
  runs/causal_intervention_results.json   (SAE on PINNs)
  runs/pca_causal_results.json            (PCA on PINNs)
  runs/operator_causal/operator_causal_report.json  (SAE on FNO)
"""
from __future__ import annotations

import numpy as np

from _common import C, load_artifact, save_fig, start_fig


def main():
    sae = load_artifact("causal_intervention_results.json")
    pca = load_artifact("pca_causal_results.json")
    op = load_artifact("operator_causal/operator_causal_report.json")

    fig, axes = start_fig(1, 2, figsize=(9.6, 3.4))

    # ---- Panel A: per-feature p-values ----
    ax = axes[0]
    batteries = [
        ("SAE / PINN", sae["multiple_comparisons"]["per_feature"], C["main"]),
        ("PCA / PINN", pca["multiple_comparisons"]["per_feature"], C["fourth"]),
        ("SAE / FNO", op["battery_summary"]["multiple_comparisons"]["per_feature"], C["alt"]),
    ]
    n_feat = max(len(b[1]) for b in batteries)
    width = 0.27
    for i, (name, feats, color) in enumerate(batteries):
        ps = [f["sign_test_p"] for f in feats]
        xs = np.arange(len(ps)) + (i - 1) * width
        ax.bar(xs, ps, width=width, color=color, label=name, alpha=0.9,
               edgecolor="black", linewidth=0.4)
    n_tests = sae["multiple_comparisons"]["n_features_tested"]
    ax.axhline(0.05 / max(n_tests, 1), color="black", ls="--", lw=1.2,
               label=f"Bonferroni $\\alpha$={0.05/n_tests:.4f}")
    ax.set_xlabel("candidate feature index (within battery)")
    ax.set_ylabel("sign-test $p$")
    ax.set_yscale("log")
    ax.set_title("A  MC-corrected survivors: FNO passes where PINNs fail")
    ax.legend(frameon=False, fontsize=7)

    # ---- Panel B: beats-all rate + ratio ----
    ax = axes[1]
    def _stats(rows):
        wins = sum(1 for r in rows
                   if abs(r["causal_score"]["target_effect"]) >
                   max(abs(r["causal_score"]["control_unrelated_effect"]),
                       abs(r["causal_score"]["control_random_effect"]),
                       abs(r["causal_score"]["control_probe_effect"])))
        ratios = [abs(r["causal_score"]["target_effect"]) /
                  max(abs(r["causal_score"]["control_unrelated_effect"]),
                      abs(r["causal_score"]["control_random_effect"]),
                      abs(r["causal_score"]["control_probe_effect"]))
                  for r in rows]
        return wins / len(rows), float(np.median(ratios))

    names = ["SAE / PINN", "PCA / PINN", "SAE / FNO"]
    rates = [_stats(sae["evaluations"])[0],
             _stats(pca["evaluations"])[0],
             _stats(op["evaluations"])[0]]
    ratios = [_stats(sae["evaluations"])[1],
              _stats(pca["evaluations"])[1],
              _stats(op["evaluations"])[1]]
    xs = np.arange(3)
    bars = ax.bar(xs, rates, color=[C["main"], C["fourth"], C["alt"]],
                  alpha=0.9, edgecolor="black", linewidth=0.5)
    for x, r, rt in zip(xs, rates, ratios):
        ax.text(x, r + 0.02, f"{r:.0%}\n(ratio {rt:.2f}x)", ha="center",
                fontsize=8)
    ax.axhline(1 / 8, color=C["grey"], ls=":", lw=1,
               label="chance-ish reference (1 feature/8)")
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylabel("target beats all 3 controls")
    ax.set_ylim(0, 1.0)
    ax.set_title("B  Control-beating rate by battery")
    ax.legend(frameon=False, fontsize=7)

    save_fig(fig, "figure_09_operator_causal")


if __name__ == "__main__":
    main()
