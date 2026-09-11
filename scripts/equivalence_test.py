#!/usr/bin/env python3
"""R7: TOST equivalence testing of the basis-independence null (v4.2).

Preregistered in docs/preregistration.md §R7 (registered 2026-09-12,
BEFORE this implementation and any run). Analysis-only stage over the
COMMITTED stage-5/stage-8 artifacts — no new training, no new
interventions; the machinery gate is the artifact checksums.

Design (registered):
  Primary endpoint: the paired per-evaluation difference
    d_i = E_T^PCA_i - E_T^SAE_i over the 88 (feature, checkpoint) pairs
    of the identical battery.
  Margin: delta = 0.01 E_T units — preregistered BEFORE computing,
    calibrated as ~2.6x smaller than the planted positive control's
    demonstrated minimum detectable effect (+0.026), i.e. a
    conservative bar for "no basis-specific difference".
  Procedure: TOST via the paired t-distribution — equivalence at
    alpha=0.05 iff the 90% CI of the mean difference lies entirely
    within (-delta, +delta). Also reported: the TOST p-value, the
    95% CI, and the design-honest power note (if the 95% CI is wider
    than 2*delta, the test is underpowered for the margin).
  Secondary (pre-specified): each basis against the zero criterion
    (|E_T| < delta, one-sample TOST per basis).

Output: runs/equivalence_test/equivalence_report.json (+ stdout).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS = PROJECT_ROOT / "runs"

DELTA = 0.01          # preregistered equivalence margin (E_T units)
ALPHA = 0.05


def _t_cdf(t: float, df: int) -> float:
    """t CDF: scipy when available; a compact regularized-incomplete-beta
    fallback otherwise (Lentz continued fraction)."""
    try:
        from scipy import stats
        return float(stats.t.cdf(t, df))
    except ImportError:
        pass

    def _betacf(a, b, x):
        MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        if abs(d) < FPMIN:
            d = FPMIN
        d = 1.0 / d
        h = d
        for m in range(1, MAXIT + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            h *= d * c
            if abs(aa) < EPS:
                break
        return h

    def _betainc(a, b, x):
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b))
        front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
        if x < (a + 1.0) / (a + b + 2.0):
            return front * _betacf(a, b, x) / a
        return 1.0 - front * _betacf(b, a, 1.0 - x) / b

    x = df / (df + t * t)
    p = 0.5 * _betainc(df / 2.0, 0.5, x)
    return 1.0 - p if t > 0 else p


def _t_ppf(p: float, df: int) -> float:
    """t quantile by bisection on _t_cdf (df is small; 60 iterations
    more than exhaust double precision)."""
    try:
        from scipy import stats
        return float(stats.t.ppf(p, df))
    except ImportError:
        pass
    lo, hi = -50.0, 50.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def tost_paired(diffs, delta: float, alpha: float = ALPHA) -> dict:
    """Two-one-sided-tests on the mean of paired differences.

    H0: |mean| >= delta; H1: |mean| < delta (equivalence).
    Equivalence is concluded iff p_tost <= alpha, equivalently iff the
    (1-2*alpha) CI — the 90% CI at alpha=0.05 — lies inside (-delta,
    +delta).
    """
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    df = n - 1
    t_crit = _t_ppf(1.0 - alpha, df)          # one-sided 95% quantile
    half90 = t_crit * se                       # 90% CI half-width
    ci90_lo, ci90_hi = mean - half90, mean + half90
    # 95% CI for the power note
    t_crit95 = _t_ppf(1.0 - ALPHA / 2.0, df)
    half95 = t_crit95 * se
    ci95_lo, ci95_hi = mean - half95, mean + half95
    # one-sided p-values: H0a mean <= -delta; H0b mean >= +delta
    if se <= 0:
        p_low = 1.0 if mean <= -delta else 0.0
        p_high = 1.0 if mean >= delta else 0.0
    else:
        t_low = (mean - (-delta)) / se
        t_high = (delta - mean) / se
        p_low = 1.0 - _t_cdf(t_low, df)
        p_high = 1.0 - _t_cdf(t_high, df)
    p_tost = max(p_low, p_high)
    ci_within = bool(ci90_lo > -delta and ci90_hi < delta)
    return {
        "n": n,
        "mean": mean,
        "se": se,
        "ci90": [ci90_lo, ci90_hi],
        "ci95": [ci95_lo, ci95_hi],
        "p_tost": p_tost,
        "equivalent": bool(p_tost <= alpha),
        "ci90_within_margin": ci_within,
        "underpowered_for_margin": bool((ci95_hi - ci95_lo) > 2 * delta),
    }


def run_r7() -> dict:
    print("=" * 64)
    print("R7: TOST Equivalence Test of the Basis-Independence Null")
    print("=" * 64)
    print(f"  preregistered margin delta = {DELTA} E_T units")

    sae = json.loads((RUNS / "causal_intervention_results.json").read_text())
    pca = json.loads((RUNS / "pca_causal_results.json").read_text())

    # The preregistered pairing unit is the CHECKPOINT (the batteries
    # test 8 candidates against 8 components on the same 11 held-out
    # boundary-starvation checkpoints; feature indices are
    # basis-specific and do not pair across bases). The stage-8
    # artifact's committed head_to_head_vs_sae.pairs encodes exactly
    # this pairing — consume it rather than re-deriving.
    pairs = pca["head_to_head_vs_sae"]["pairs"]
    diffs = [pr["pca_causal_strength"] - pr["sae_causal_strength"]
             for pr in pairs]
    sae_ets = [pr["sae_causal_strength"] for pr in pairs]
    pca_ets = [pr["pca_causal_strength"] for pr in pairs]
    print(f"  paired checkpoint evaluations: {len(pairs)} "
          f"(the stage-8 committed pairing)")

    primary = tost_paired(diffs, DELTA)
    print(f"\n  PRIMARY (paired PCA-SAE E_T difference):")
    print(f"    mean = {primary['mean']:+.5f} | 90% CI "
          f"[{primary['ci90'][0]:+.5f}, {primary['ci90'][1]:+.5f}]")
    print(f"    TOST p = {primary['p_tost']:.4f} -> "
          f"{'EQUIVALENT' if primary['equivalent'] else 'NOT ESTABLISHED'}"
          f" at delta={DELTA}")
    if primary["underpowered_for_margin"]:
        print("    design note: 95% CI wider than 2*delta — the test is")
        print("    underpowered for this margin (recorded, not hidden)")

    # secondary: each basis vs the zero criterion
    secondary = {}
    for name, vals in (("sae", sae_ets), ("pca", pca_ets)):
        res = tost_paired(vals, DELTA)
        secondary[name] = res
        print(f"\n  SECONDARY ({name} vs zero criterion):")
        print(f"    mean E_T = {res['mean']:+.5f} | 90% CI "
              f"[{res['ci90'][0]:+.5f}, {res['ci90'][1]:+.5f}]")
        print(f"    TOST p = {res['p_tost']:.4f} -> "
              f"{'EQUIVALENT to zero' if res['equivalent'] else 'NOT ESTABLISHED'}")

    # pre-written outcome classification (rule corrected pre-verdict —
    # see preregistration §R7: the trichotomy is decided by the 90% CI
    # vs the margin: inside => equivalent; entirely outside => a
    # difference >= delta is established; overlapping the boundary =>
    # inconclusive at this n)
    lo90, hi90 = primary["ci90"]
    if primary["equivalent"]:
        outcome = "R7a"
        note = ("Equivalence confirmed: the SAE-PCA difference is "
                "statistically equivalent to zero at the preregistered "
                "margin — the basis-independence claim upgrades from "
                "'difference spans zero' to formal equivalence.")
    elif lo90 <= DELTA and hi90 >= -DELTA and not (
            lo90 > -DELTA and hi90 < DELTA):
        # CI overlaps the margin boundary (and is not inside it):
        # neither equivalence nor a delta-difference is established
        outcome = "R7b"
        note = ("Inconclusive: the 90% CI overlaps the equivalence "
                "margin (point estimate inside, CI straddling the "
                "boundary) — the data cannot distinguish equivalence "
                "from a small difference at this n; recorded with the "
                "power note.")
    else:
        outcome = "R7c"
        note = ("Equivalence rejected: the 90% CI lies entirely "
                "outside the margin — the bases differ by at least "
                "delta; the basis-independence claim weakens to "
                "'direction-specificity absent for both bases (0/8 "
                "each)'.")
    print(f"\n  OUTCOME: {outcome} — {note}")

    report = {
        "stage": "r7_equivalence_test",
        "preregistration": "docs/preregistration.md §R7 (registered "
                           "2026-09-12 before run)",
        "delta": DELTA,
        "alpha": ALPHA,
        "delta_calibration": "planted positive control's demonstrated "
                             "minimum detectable effect = +0.026; delta "
                             "= 0.01 is ~2.6x smaller (conservative)",
        "n_pairs": len(pairs),
        "pairing": "checkpoint-matched (the stage-8 committed head_to_head_vs_sae.pairs)",
        "primary_pca_minus_sae": primary,
        "secondary_vs_zero": secondary,
        "outcome": outcome,
        "outcome_note": note,
    }
    out_dir = RUNS / "equivalence_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "equivalence_report.json").write_text(
        json.dumps(report, indent=2))
    print(f"\n  report: {out_dir / 'equivalence_report.json'}")
    return report


if __name__ == "__main__":
    run_r7()
