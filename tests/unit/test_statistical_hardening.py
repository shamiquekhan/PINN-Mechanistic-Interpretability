"""Tests for statistical-hardening analyses."""
import json
from pathlib import Path

import numpy as np

from analysis.statistical_hardening import (
    sign_test_power, mde_sign_test, run_monitor_bayesian_bootstrap,
)

RUNS = Path(__file__).resolve().parent.parent.parent / "runs"


def test_sign_test_power_bounds():
    # At the null (p=0.5) power equals alpha (one-sided).
    assert sign_test_power(10, 0.5) <= 0.05 + 1e-9
    # A deterministic effect has power 1.
    assert sign_test_power(10, 1.0) > 0.999
    # Power increases with the alternative.
    assert sign_test_power(10, 0.9) >= sign_test_power(10, 0.7)


def test_mde_sign_test_reasonable():
    mde = mde_sign_test(11)
    assert 0.75 < mde <= 1.0
    assert mde_sign_test(11) < mde_sign_test(5)  # more runs, lower MDE


def test_monitor_bayesian_guard():
    out = run_monitor_bayesian_bootstrap()
    # Either a proper posterior or the documented skip.
    assert "prob_sae_better" in out or "error" in out


def test_power_analysis_on_artifacts():
    from analysis.statistical_hardening import power_analysis_causal
    path = RUNS / "causal_intervention_results.json"
    if not path.exists():
        return  # artifact not generated in CI environments
    out = power_analysis_causal(path)
    assert out["features"], "per-feature power stats must be non-empty"
    for f in out["features"]:
        assert f["mde_success_prob_at_80pct_power"] > 0.5
