#!/usr/bin/env python3
"""R10: geometry predictors of the sparse-compression advantage (v4.2).

Preregistered in docs/preregistration.md §R10 (registered 2026-09-12,
BEFORE this implementation and any run). Analysis-only over committed
artifacts: per-run rows carrying BOTH geometry statistics and the TopK
k-matched-PCA reconstruction ratio — H18 (architecture_boundary), R3
(frequency sweep), R5 (Burgers), R9 (reaction-diffusion).

Registered predictors: PR, stable rank, PCA-95, PR/width.
Registered response: log10(PCA/SAE reconstruction ratio).
Registered analyses:
  1. Spearman per predictor, bootstrap CIs clustered by experiment arm;
  2. univariate OLS on the log-ratio, leave-one-ARM-out CV R2;
  3. the model-comparison verdict (which predictor best explains the
     advantage; honest residual statement).

Output: runs/geometry_predictors/geometry_predictors_report.json.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS = PROJECT_ROOT / "runs"

PREDICTORS = ["participation_ratio", "stable_rank",
              "n_components_95pct", "pr_over_width"]
ARMS_HELD_OUT = True  # leave-one-ARM-out CV (the registered design)


def _collect_rows() -> list:
    """Per-run (arm, PR, stable_rank, pca95, rho, ratio) rows from the
    four committed artifacts."""
    rows = []

    def _add(run_name, arm, stats, recon, rho=None):
        # R3/H18 nest the ratio under "topk_vs_pca" (both-family
        # protocol); R5/R9 carry it at the top level. Handle both.
        if not isinstance(recon, dict):
            return
        ratio = recon.get("pca_over_sae_ratio")
        if ratio is None and isinstance(
                recon.get("topk_vs_pca"), dict):
            ratio = recon["topk_vs_pca"].get("pca_over_sae_ratio")
        if ratio is None or ratio <= 0:
            return
        rows.append({
            "run": run_name, "arm": arm,
            "participation_ratio": stats["participation_ratio"],
            "stable_rank": stats.get("stable_rank"),
            "n_components_95pct": stats.get("n_components_95pct"),
            "pr_over_width": rho if rho is not None
            else stats.get("pr_over_width",
                           stats["participation_ratio"] / 64),
            "ratio": ratio,
        })

    # H18: architecture boundary (tanh depths 2-6 + fourier n_freq=32)
    h18 = json.loads(
        (RUNS / "architecture_boundary" /
         "architecture_boundary_report.json").read_text())
    for r in h18["per_run"]:
        _add(r["run"], f"h18:{r.get('arm', 'depth')}", r,
             r.get("reconstruction"), rho=r.get("pr_over_width"))

    # R3: frequency sweep
    r3 = json.loads(
        (RUNS / "r3_frequency_sweep" / "r3_report.json").read_text())
    for r in r3["per_run"]:
        _add(r["run"], f"r3:nf{r['n_freq']}", r, r.get("reconstruction"),
             rho=r.get("pr_over_width"))

    # R5: Burgers
    r5 = json.loads(
        (RUNS / "r5_burgers_boundary" / "r5_report.json").read_text())
    for r in r5["per_run"]:
        _add(r["run"], f"r5:{r['arm']}", r, r.get("reconstruction"),
             rho=r.get("pr_over_width"))

    # R9: reaction-diffusion
    r9 = json.loads(
        (RUNS / "r9_rd_boundary" / "r9_report.json").read_text())
    for r in r9["per_run"]:
        _add(r["run"], f"r9:{r['arm']}", r, r.get("reconstruction"),
             rho=r.get("pr_over_width"))

    # drop rows with missing fields
    rows = [r for r in rows
            if all(r.get(p) is not None for p in PREDICTORS)
            and r["ratio"] is not None and r["ratio"] > 0]
    return rows


def _spearman(x, y):
    """Spearman rho without scipy dependency (rank correlation)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def run_r10() -> dict:
    print("=" * 64)
    print("R10: Geometry Predictors of the Sparse-Compression Advantage")
    print("=" * 64)

    rows = _collect_rows()
    arms = sorted({r["arm"] for r in rows})
    print(f"  {len(rows)} runs across {len(arms)} arms: {arms}")

    y_all = np.array([math.log10(r["ratio"]) for r in rows])

    # ---- 1. Spearman per predictor, arm-clustered bootstrap CIs ----
    rng = np.random.default_rng(0)
    arm_index = {a: i for i, a in enumerate(arms)}
    spearman_report = {}
    for pred in PREDICTORS:
        x_all = np.array([r[pred] for r in rows], float)
        rho = _spearman(x_all, y_all)
        # arm-clustered bootstrap
        arm_of = np.array([arm_index[r["arm"]] for r in rows])
        boot = []
        for _ in range(5000):
            sel_arms = rng.choice(len(arms), len(arms), replace=True)
            idx = np.concatenate([
                np.where(arm_of == a)[0] for a in sel_arms])
            boot.append(_spearman(x_all[idx], y_all[idx]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        spearman_report[pred] = {
            "rho": rho,
            "ci95": [float(lo), float(hi)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
        }
        print(f"\n  Spearman({pred}, log10 ratio) = {rho:+.3f} "
              f"CI [{lo:+.3f}, {hi:+.3f}]")

    # ---- 2. univariate OLS with leave-one-ARM-out CV R2 ----
    def _ols_cv(pred):
        x = np.array([r[pred] for r in rows], float)
        y = y_all
        # leave-one-arm-out
        preds = np.empty_like(y)
        for a in arms:
            mask_in = np.array([r["arm"] != a for r in rows])
            mask_out = ~mask_in
            xi, yi = x[mask_in], y[mask_in]
            A = np.vstack([np.ones_like(xi), xi]).T
            coef, *_ = np.linalg.lstsq(A, yi, rcond=None)
            preds[mask_out] = coef[0] + coef[1] * x[mask_out]
        ss_res = float(np.sum((y - preds) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2_cv = 1.0 - ss_res / ss_tot
        # in-sample
        A = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2_in = 1.0 - float(np.sum(
            (y - (coef[0] + coef[1] * x)) ** 2)) / ss_tot
        return {"r2_in_sample": r2_in, "r2_leave_one_arm_out": r2_cv}

    ols_report = {p: _ols_cv(p) for p in PREDICTORS}
    for p, v in ols_report.items():
        print(f"\n  OLS({p}): R2 in-sample {v['r2_in_sample']:.3f} | "
              f"leave-one-arm-out {v['r2_leave_one_arm_out']:.3f}")

    # ---- 3. the model-comparison verdict ----
    best_pred = max(ols_report,
                    key=lambda p: ols_report[p]["r2_leave_one_arm_out"])
    best_r2 = ols_report[best_pred]["r2_leave_one_arm_out"]
    competitors = {p: ols_report[p]["r2_leave_one_arm_out"]
                   for p in PREDICTORS if p != best_pred}
    if best_r2 >= 0.5 and all(
            best_r2 - v > 0.05 for v in competitors.values()):
        outcome = "R10a"
        note = (f"PR-family predictor '{best_pred}' wins decisively "
                f"(CV R2 = {best_r2:.2f}); the rank diagnostic is "
                f"validated as predictive")
    elif best_r2 >= 0.3:
        outcome = "R10b"
        note = (f"'{best_pred}' is the best predictor (CV R2 = "
                f"{best_r2:.2f}) but the margin over competitors is "
                f"modest — the geometry gate is well described by the "
                f"PR family, with a recorded residual")
    else:
        outcome = "R10c"
        note = (f"All single second-moment predictors are weak (best CV "
                f"R2 = {best_r2:.2f}, '{best_pred}') — the compression "
                f"advantage is not well explained by any single "
                f"covariance statistic, the strongest version of the R3 "
                f"residual finding")
    print(f"\n  OUTCOME: {outcome}")
    print(f"    {note}")

    report = {
        "stage": "r10_geometry_predictors",
        "preregistration": "docs/preregistration.md §R10 (registered "
                           "2026-09-12 before run)",
        "n_runs": len(rows), "n_arms": len(arms), "arms": arms,
        "response": "log10(PCA/SAE k-matched reconstruction ratio)",
        "spearman": spearman_report,
        "ols": ols_report,
        "best_predictor": best_pred,
        "outcome": outcome,
        "outcome_note": note,
        "rows": rows,
    }
    out_dir = RUNS / "geometry_predictors"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "geometry_predictors_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(f"\n  report: {out_dir / 'geometry_predictors_report.json'}")
    return report


if __name__ == "__main__":
    run_r10()
