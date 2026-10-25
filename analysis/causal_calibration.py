"""Prospective calibration utilities for the causal intervention statistic.

The frozen campaign's sign test is retained unchanged.  This module is an
audit harness for asking whether that test is calibrated for the statistic it
actually receives:

    d = |target_effect| - max(|control_effects|)

The max over controls is deliberately included in the synthetic nulls.  A
sign-flip randomization test is exposed only for sign-exchangeable paired
differences; it is not applied to the asymmetric max-control null.
"""
from __future__ import annotations

from math import comb
from typing import Dict, Iterable, Optional

import numpy as np


CALIBRATION_SCHEMA_VERSION = 1


def causal_differences(
    target_effects: Iterable[float],
    control_effects: np.ndarray,
) -> np.ndarray:
    """Compute the exact per-run statistic used by the frozen battery."""
    target = np.abs(np.asarray(list(target_effects), dtype=np.float64))
    controls = np.abs(np.asarray(control_effects, dtype=np.float64))
    if controls.ndim != 2 or controls.shape[0] != target.size:
        raise ValueError("controls must have shape (n_runs, n_controls)")
    if controls.shape[1] == 0:
        raise ValueError("at least one control is required")
    return target - controls.max(axis=1)


def exact_sign_test_pvalue(differences: Iterable[float]) -> float:
    """One-sided exact sign-test p-value, excluding zero ties."""
    values = np.asarray(list(differences), dtype=np.float64)
    n_pos = int(np.count_nonzero(values > 0))
    n = int(np.count_nonzero(values != 0))
    if n == 0:
        return 1.0
    return float(sum(comb(n, k) for k in range(n_pos, n + 1)) / (2 ** n))


def sign_flip_randomization_pvalue(
    differences: Iterable[float],
    *,
    n_resamples: int = 20_000,
    seed: int = 0,
) -> float:
    """One-sided paired sign-flip p-value under sign-exchangeability.

    This is valid only when the paired differences have exchangeable signs
    under the null.  The caller must not use it for the max-control
    exchangeability null merely because the control effects are identically
    generated: ``target - max(controls)`` is not sign-exchangeable in that
    design.
    """
    values = np.asarray(list(differences), dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_resamples, values.size))
    null_means = (signs * np.abs(values)).mean(axis=1)
    return float((1 + np.count_nonzero(null_means >= observed)) / (n_resamples + 1))


def _simulate_case(
    case: str,
    n_runs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate paired differences for a named calibration case."""
    if case == "symmetric_difference_null":
        # Sign-exchangeable null: sign-flips are justified here.
        return rng.normal(0.0, 1.0, size=n_runs)
    if case == "exchangeable_max_control_null":
        # Target and controls are identically distributed, but the statistic
        # compares one draw with the maximum of three draws.  This is the
        # null most directly relevant to the current battery.
        effects = np.abs(rng.normal(0.0, 1.0, size=(n_runs, 4)))
        return effects[:, 0] - effects[:, 1:].max(axis=1)
    if case == "asymmetric_median_null":
        # Median-zero but skewed differences: sign-test calibration depends on
        # the median/sign condition, not on a symmetric magnitude distribution.
        positive = rng.exponential(1.0, size=n_runs)
        negative = rng.exponential(2.0, size=n_runs)
        signs = rng.integers(0, 2, size=n_runs)
        return np.where(signs == 1, positive, -negative)
    if case == "zero_effect_ties":
        return np.zeros(n_runs, dtype=np.float64)
    if case == "planted_alternative":
        return rng.normal(0.8, 1.0, size=n_runs)
    raise ValueError(f"unknown calibration case: {case}")


def calibrate_case(
    case: str,
    *,
    n_runs: int = 11,
    n_trials: int = 2_000,
    alpha: float = 0.05,
    seed: int = 0,
    randomization_resamples: int = 2_000,
) -> Dict[str, object]:
    """Estimate rejection/power behavior for one synthetic case."""
    rng = np.random.default_rng(seed)
    sign_pvalues = []
    flip_pvalues = []
    positive_rates = []
    flip_applicable = case == "symmetric_difference_null"
    for trial in range(n_trials):
        differences = _simulate_case(case, n_runs, rng)
        sign_pvalues.append(exact_sign_test_pvalue(differences))
        positive_rates.append(float(np.mean(differences > 0)))
        if flip_applicable:
            flip_pvalues.append(sign_flip_randomization_pvalue(
                differences,
                n_resamples=randomization_resamples,
                seed=seed + trial,
            ))

    result: Dict[str, object] = {
        "case": case,
        "n_runs": n_runs,
        "n_trials": n_trials,
        "alpha": alpha,
        "sign_test_rejection_rate": float(np.mean(np.asarray(sign_pvalues) <= alpha)),
        "mean_positive_rate": float(np.mean(positive_rates)),
        "ties_expected": case == "zero_effect_ties",
        "randomization_test_applicable": flip_applicable,
    }
    if flip_applicable:
        result["sign_flip_rejection_rate"] = float(
            np.mean(np.asarray(flip_pvalues) <= alpha)
        )
    else:
        result["sign_flip_note"] = (
            "Not applied: the case is not sign-exchangeable under the "
            "specified data-generating process."
        )
    return result


def run_calibration_suite(
    *,
    n_runs: int = 11,
    n_trials: int = 2_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, object]:
    """Run the preregistered synthetic calibration cases."""
    cases = [
        "symmetric_difference_null",
        "exchangeable_max_control_null",
        "asymmetric_median_null",
        "zero_effect_ties",
        "planted_alternative",
    ]
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "statistic": "abs(target) - max(abs(controls))",
        "sign_test_null": (
            "independent paired differences with P(d > 0) = 0.5 after "
            "excluding exact zero ties"
        ),
        "randomization_test_null": "sign-exchangeable paired differences",
        "cases": [
            calibrate_case(
                case,
                n_runs=n_runs,
                n_trials=n_trials,
                alpha=alpha,
                seed=seed + i * 100,
            )
            for i, case in enumerate(cases)
        ],
    }
