"""Figure 4 — The basis-independent causal null.

Panels:
  A. SAE battery: bootstrap distribution of E_T with 95% CI
  B. PCA battery: same (the v3 head-to-head)
  C. per-feature sign-test p-values with Bonferroni threshold
     (both batteries)
  D. positive control: targeted vs control deltas (pipeline validity)

Sources:
  runs/causal_intervention_results.json
  runs/pca_causal_results.json
  runs/positive_control.json
"""
from __future__ import annotations

import numpy as np

from _common import C, load_artifact, save_fig, start_fig


def _boot_replay(values, rng, n=1000):
    values = np.asarray(values, dtype=float)
    return [float(np.mean(rng.choice(values, len(values)))) for _ in range(n)]


def main():
    sae = load_artifact("causal_intervention_results.json")
    pca = load_artifact("pca_causal_results.json")
    pc = load_artifact("positive_control.json")

    fig, axes = start_fig(1, 4, figsize=(12.0, 2.9))

    # ---- Panel A: SAE E_T bootstrap ----
    ax = axes[0]
    et_sae = [e["causal_score"]["causal_strength"]
              for e in sae["evaluations"]]
    rng = np.random.default_rng(42)
    boots = _boot_replay(et_sae, rng)
    ax.hist(boots, bins=40, color=C["main"], alpha=0.85)
    ci = sae["causal_strength_ci"]
    ax.axvline(ci["ci_lower"], color=C["alt"], ls="--", lw=1.2,
               label=f"95% CI [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    ax.axvline(ci["ci_upper"], color=C["alt"], ls="--", lw=1.2)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("$E_T$ (bootstrap means)")
    ax.set_title("A  SAE causal strength: CI spans 0")
    ax.legend(frameon=False, fontsize=7)

    # ---- Panel B: PCA E_T bootstrap ----
    ax = axes[1]
    et_pca = [e["causal_score"]["causal_strength"]
              for e in pca["evaluations"]]
    boots = _boot_replay(et_pca, rng)
    ax.hist(boots, bins=40, color=C["alt"], alpha=0.85)
    ci = pca["causal_strength_ci"]
    ax.axvline(ci["ci_lower"], color="black", ls="--", lw=1.2,
               label=f"95% CI [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    ax.axvline(ci["ci_upper"], color="black", ls="--", lw=1.2)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("$E_T$ (bootstrap means)")
    ax.set_title("B  PCA battery: identical null")
    ax.legend(frameon=False, fontsize=7)

    # ---- Panel C: sign-test p-values vs Bonferroni ----
    ax = axes[2]
    mc_sae = sae["multiple_comparisons"]["per_feature"]
    mc_pca = pca["multiple_comparisons"]["per_feature"]
    n_feat = sae["multiple_comparisons"]["n_features_tested"]
    thr = 0.05 / max(n_feat, 1)  # Bonferroni threshold used by the battery
    p_sae = [f["sign_test_p"] for f in mc_sae]
    p_pca = [f["sign_test_p"] for f in mc_pca]
    xs_sae = np.arange(len(p_sae))
    xs_pca = np.arange(len(p_pca)) + 0.3
    ax.bar(xs_sae, p_sae, width=0.3, color=C["main"], label="SAE features")
    ax.bar(xs_pca, p_pca, width=0.3, color=C["alt"], label="PCA comps")
    ax.axhline(thr, color="black", ls="--", lw=1.2,
               label=f"Bonferroni $\\alpha$={thr:.4f}")
    ax.set_xlabel("candidate (feature / component)")
    ax.set_ylabel("sign-test $p$")
    ax.set_title("C  0/8 survivors in both batteries")
    ax.legend(frameon=False, fontsize=7)

    # ---- Panel D: positive control ----
    ax = axes[3]
    deltas = {
        "targeted\n(planted)": pc["targeted_delta"],
        "unrelated": pc["ctrl_unrelated_delta"],
        "random\n(median)": pc["ctrl_random_delta"],
        "probe": pc["ctrl_probe_delta"],
    }
    names = list(deltas.keys())
    vals = list(deltas.values())
    xs = np.arange(len(vals))
    colors = [C["third"]] + [C["light"]] * (len(vals) - 1)
    ax.bar(xs, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("$\\Delta$ target loss")
    pass_gate = "PASS" if pc["pipeline_pass"] else "FAIL"
    ax.set_title(f"D  Positive control: {pass_gate}")

    save_fig(fig, "figure_04_causal_null")


if __name__ == "__main__":
    main()
