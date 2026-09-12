#!/usr/bin/env python3
"""R8: hierarchical statistical analysis of the causal nulls (v4.2).

Preregistered in docs/preregistration.md §R8 (registered 2026-09-12,
BEFORE this implementation and any run). Analysis-only over the
committed stage-5/8 artifacts — no new training; the machinery gate is
the artifact checksums.

The pseudoreplication question: the batteries treat 88 (feature x
checkpoint) evaluations as independent, but they nest — 8 candidate
features within each of 11 trained models. R8 measures the nesting and
reports hierarchy-explicit uncertainty:

  1. one-way random-effects decomposition (ICC, design effect,
     effective n) of the paired PCA-SAE differences;
  2. cluster bootstrap (resample the 11 CHECKPOINTS, 5,000 draws) for
     (a) the paired difference and (b) each basis's mean E_T, compared
     against the naive row-level CIs;
  3. mixed-effects model (statsmodels MixedLM, difference ~ 1 + (1 |
     checkpoint)) — the SE inflation factor;
  4. the R7+ power preregistration: TOST n at delta=0.01 from the
     cluster-level SD (z-formula, checkpoint as the unit).

Output: runs/hierarchical_analysis/hierarchical_report.json.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS = PROJECT_ROOT / "runs"

DELTA = 0.01          # the R7 preregistered margin (E_T units)
N_BOOT = 5000
ALPHA = 0.05


def _percentile_ci(samples, alpha=ALPHA):
    lo = float(np.percentile(samples, 100 * (alpha / 2)))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return lo, hi


def run_r8() -> dict:
    print("=" * 64)
    print("R8: Hierarchical Statistical Analysis of the Causal Nulls")
    print("=" * 64)

    sae = json.loads((RUNS / "causal_intervention_results.json").read_text())
    pca = json.loads((RUNS / "pca_causal_results.json").read_text())
    pairs = pca["head_to_head_vs_sae"]["pairs"]

    # ---- L1 structure: checkpoints (the primary independent unit) ----
    by_run_diff = defaultdict(list)
    by_run_sae = defaultdict(list)
    by_run_pca = defaultdict(list)
    for pr in pairs:
        r = pr["run"]
        by_run_diff[r].append(pr["pca_causal_strength"]
                              - pr["sae_causal_strength"])
        by_run_sae[r].append(pr["sae_causal_strength"])
        by_run_pca[r].append(pr["pca_causal_strength"])
    runs = sorted(by_run_diff)
    m, k = len(runs), len(by_run_diff[runs[0]])
    diffs = np.array([pr["pca_causal_strength"] - pr["sae_causal_strength"]
                      for pr in pairs])
    print(f"  hierarchy: {m} checkpoints x {k} feature-pairs = {m*k} rows")

    # ---- 1. one-way random effects (ICC + design effect) ----
    grand = float(diffs.mean())
    within_ss = sum(float(np.sum((np.array(by_run_diff[r])
                                  - np.mean(by_run_diff[r])) ** 2))
                    for r in runs)
    between_ss = k * sum((np.mean(by_run_diff[r]) - grand) ** 2 for r in runs)
    ms_b = between_ss / (m - 1)
    ms_w = within_ss / (m * (k - 1))
    icc = float((ms_b - ms_w) / (ms_b + (k - 1) * ms_w))
    deff = 1 + (k - 1) * icc
    n_eff = m * k / deff
    print(f"\n  [1] random effects: ICC = {icc:.3f} | "
          f"DEFF = {deff:.2f} | n_eff = {n_eff:.1f} "
          f"(naive n = {m*k})")

    # ---- 2. cluster bootstrap (checkpoints as units) ----
    rng = np.random.default_rng(0)
    dbar = np.array([np.mean(by_run_diff[r]) for r in runs])
    sbar_sae = np.array([np.mean(by_run_sae[r]) for r in runs])
    sbar_pca = np.array([np.mean(by_run_pca[r]) for r in runs])
    boot_diff, boot_sae, boot_pca = [], [], []
    for _ in range(N_BOOT):
        sel = rng.choice(m, m, replace=True)
        boot_diff.append(float(dbar[sel].mean()))
        boot_sae.append(float(sbar_sae[sel].mean()))
        boot_pca.append(float(sbar_pca[sel].mean()))
    naive_se = float(diffs.std(ddof=1) / math.sqrt(m * k))
    cl_lo, cl_hi = _percentile_ci(boot_diff)
    sae_lo, sae_hi = _percentile_ci(boot_sae)
    pca_lo, pca_hi = _percentile_ci(boot_pca)
    # naive row-level 95% CI for comparison
    t95 = 1.96  # n=88; the normal approx is exact to 3 decimals here
    naive_ci = (grand - t95 * naive_se, grand + t95 * naive_se)
    inflation = float(np.std(boot_diff, ddof=1) / naive_se)
    print(f"\n  [2] cluster bootstrap (n_boot={N_BOOT}):")
    print(f"      paired difference: mean {grand:+.5f} | "
          f"cluster 95% CI [{cl_lo:+.5f}, {cl_hi:+.5f}] | "
          f"naive CI [{naive_ci[0]:+.5f}, {naive_ci[1]:+.5f}]")
    print(f"      SE inflation factor (cluster/naive): {inflation:.2f}")
    print(f"      SAE mean E_T cluster 95% CI [{sae_lo:+.5f}, {sae_hi:+.5f}]")
    print(f"      PCA mean E_T cluster 95% CI [{pca_lo:+.5f}, {pca_hi:+.5f}]")

    # ---- 3. mixed-effects model ----
    mixed = {"available": False}
    try:
        import pandas as pd
        from statsmodels.regression.mixed_linear_model import MixedLM
        df = pd.DataFrame({
            "diff": diffs,
            "checkpoint": [pr["run"] for pr in pairs],
        })
        fit = MixedLM(df["diff"], np.ones((len(df), 1)),
                      groups=df["checkpoint"]).fit(reml=True)
        fe = float(fit.params["const"])
        se = float(fit.bse["const"])
        mixed = {
            "available": True,
            "fixed_intercept": fe,
            "se": se,
            "ci90": [fe - 1.645 * se, fe + 1.645 * se],
            "ci95": [fe - 1.96 * se, fe + 1.96 * se],
            "group_var": float(fit.cov_re.iloc[0, 0]),
            "resid_var": float(fit.scale),
        }
        print(f"\n  [3] mixed model: intercept {fe:+.5f} (SE {se:.5f}) | "
              f"95% CI [{mixed['ci95'][0]:+.5f}, {mixed['ci95'][1]:+.5f}]")
    except Exception as e:  # pragma: no cover — environment-dependent
        print(f"\n  [3] mixed model unavailable: {e}")

    # ---- 4. the R7+ power preregistration ----
    s_cluster = float(dbar.std(ddof=1))
    z_a = 1.645  # one-sided 95% (TOST alpha=0.05)
    power_targets = {}
    for power, z_b in ((0.80, 0.8416), (0.90, 1.2816)):
        n_ckpt = (z_a + z_b) ** 2 * s_cluster ** 2 / DELTA ** 2
        power_targets[f"{int(power*100)}"] = {
            "n_checkpoints": int(math.ceil(n_ckpt)),
            "n_pairs_at_k8": int(math.ceil(n_ckpt) * k),
        }
    print(f"\n  [4] R7+ power (cluster SD s_c = {s_cluster:.5f}, "
          f"delta = {DELTA}):")
    for tgt, v in power_targets.items():
        print(f"      {tgt}% power -> {v['n_checkpoints']} checkpoints "
              f"({v['n_pairs_at_k8']} pairs)")

    # ---- registered outcome classification ----
    # Outcome 1 (null survives): the difference's cluster CI straddles
    # zero the same way (still indistinguishable) AND both per-basis
    # cluster CIs remain negative-side (exclude zero on the negative
    # side). Outcome 2: any of those readings changes.
    diff_indistinguishable = bool(cl_lo <= 0.0 <= cl_hi)
    sae_negative_side = bool(sae_hi < 0.0)
    pca_negative_side = bool(pca_hi < 0.0)
    naive_diff_straddle = bool(naive_ci[0] <= 0.0 <= naive_ci[1])
    if diff_indistinguishable and sae_negative_side and pca_negative_side:
        outcome = "R8a"
        note = ("The hierarchy does not change any recorded reading: "
                "the paired difference remains indistinguishable from "
                "zero under checkpoint-clustered resampling, and both "
                "bases' mean E_T remain negative-side (the "
                "representational signature). The recorded conclusions "
                "upgrade to hierarchy-explicit wording; naive CIs are "
                "annotated with the inflation factor.")
    else:
        outcome = "R8b"
        changed = []
        if diff_indistinguishable != naive_diff_straddle:
            changed.append("paired-difference reading")
        if not sae_negative_side:
            changed.append("SAE negative-side reading")
        if not pca_negative_side:
            changed.append("PCA negative-side reading")
        note = ("The hierarchy CHANGES the reading of: "
                + ", ".join(changed)
                + " — the affected claims are reworded at their new "
                "strength everywhere (no silent absorption).")
    print(f"\n  OUTCOME: {outcome}")
    print(f"    {note}")

    report = {
        "stage": "r8_hierarchical_analysis",
        "preregistration": "docs/preregistration.md §R8 (registered "
                           "2026-09-12 before run)",
        "hierarchy": {"n_checkpoints": m, "k_features_per_ckpt": k,
                      "n_rows": m * k,
                      "primary_unit": "trained model (checkpoint)"},
        "icc": icc, "design_effect": deff, "n_effective": n_eff,
        "paired_difference": {
            "mean": grand,
            "cluster_ci95": [cl_lo, cl_hi],
            "naive_ci95": list(naive_ci),
            "se_inflation_factor": inflation,
        },
        "sae_mean_et": {"cluster_ci95": [sae_lo, sae_hi]},
        "pca_mean_et": {"cluster_ci95": [pca_lo, pca_hi]},
        "mixed_model": mixed,
        "r7plus_power_preregistration": {
            "delta": DELTA, "cluster_sd": s_cluster,
            "targets": power_targets,
            "scope": "new checkpoints drawn from the same "
                     "boundary-starvation regime (seed-level "
                     "exchangeability); checkpoint = the unit",
        },
        "outcome": outcome,
        "outcome_note": note,
    }
    out_dir = RUNS / "hierarchical_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hierarchical_report.json").write_text(
        json.dumps(report, indent=2))
    print(f"\n  report: {out_dir / 'hierarchical_report.json'}")
    return report


if __name__ == "__main__":
    run_r8()
