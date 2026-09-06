"""Figure 3 — SAE failure modes with mandatory baselines.

Panels:
  A. test reconstruction MSE: TopK SAE replicas vs k-matched PCA vs
     full PCA vs random-encoder SAE (log scale)
  B. dead-feature fraction per SAE replica
  C. association strength vs random-direction control, per metric view

Sources:
  runs/sae_models_v2/sae_stage3_summary.json
  runs/feature_dictionary/random_baseline_report.json
"""
from __future__ import annotations

import numpy as np

from _common import C, load_artifact, save_fig, start_fig


def main():
    sae = load_artifact("sae_models_v2/sae_stage3_summary.json")
    assoc = load_artifact("feature_dictionary/random_baseline_report.json")

    replicas = [r for r in sae["trained"]]
    sae_mses = [r["test_recon_mse"] for r in replicas]
    dead = [r["dead_feature_frac"] for r in replicas]

    fig, axes = start_fig(1, 3, figsize=(9.6, 3.0))

    # ---- Panel A: reconstruction ----
    ax = axes[0]
    # Baselines from the stage-3 summary artifact.
    baselines = sae.get("baselines", {})
    pca_k = baselines.get("pca_k_matched_train_mse")
    pca_full = baselines.get("pca_full_train_mse")
    rand_sae = baselines.get("random_sae_test_mse")

    names = [f"TopK SAE\nreplica {i+1}" for i in range(len(sae_mses))]
    vals = list(sae_mses)
    colors = [C["main"]] * len(sae_mses)
    if pca_k is not None:
        names.append("PCA (k=8)"); vals.append(pca_k); colors.append(C["third"])
    if pca_full is not None:
        names.append("PCA (full=64)"); vals.append(pca_full); colors.append(C["third"])
    if rand_sae is not None:
        names.append("random SAE"); vals.append(rand_sae); colors.append(C["light"])

    xs = np.arange(len(vals))
    ax.bar(xs, vals, color=colors, edgecolor="black", linewidth=0.5, alpha=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=6.5, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("test reconstruction MSE (log)")
    ax.set_title("A  PCA(k-matched) dominates the SAE")

    # ---- Panel B: dead features ----
    ax = axes[1]
    xs = np.arange(len(dead))
    ax.bar(xs, dead, color=C["alt"], alpha=0.85, edgecolor="black",
           linewidth=0.5)
    ax.axhline(0.5, color=C["grey"], ls=":", lw=1, label="50%")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"replica {i+1}" for i in xs])
    ax.set_ylim(0, 1)
    ax.set_ylabel("dead latent fraction")
    ax.set_title("B  Most SAE latents never activate")
    ax.legend(frameon=False)

    # ---- Panel C: association vs random ----
    ax = axes[2]
    views = list(assoc.keys())
    real = [assoc[v]["max_real_abs_corr"] for v in views]
    rand = [assoc[v]["max_random_abs_corr"] for v in views]
    xs = np.arange(len(views))
    ax.bar(xs - 0.2, real, width=0.4, color=C["main"], label="real features")
    ax.bar(xs + 0.2, rand, width=0.4, color=C["light"], label="random dirs")
    ax.set_xticks(xs)
    ax.set_xticklabels([v.replace("_", "\n") for v in views], fontsize=6)
    ax.set_ylabel("max |r| with metric view")
    ax.set_title("C  Associations exist (geometry),\nnot causation")
    ax.legend(frameon=False)

    save_fig(fig, "figure_03_sae_vs_pca")


if __name__ == "__main__":
    main()
