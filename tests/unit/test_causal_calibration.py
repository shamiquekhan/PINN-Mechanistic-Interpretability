"""Tests for the prospective causal-statistic calibration audit."""
import numpy as np
import pytest

from analysis.causal_calibration import (
    causal_differences,
    exact_sign_test_pvalue,
    run_calibration_suite,
    sign_flip_randomization_pvalue,
)


def test_causal_differences_uses_max_absolute_control():
    target = [0.5, -0.2]
    controls = np.array([[0.1, -0.4, 0.2], [0.1, 0.1, 0.1]])
    assert np.allclose(causal_differences(target, controls), [0.1, 0.1])


def test_exact_sign_test_excludes_ties():
    assert exact_sign_test_pvalue([1.0, 0.0, -1.0]) == pytest.approx(0.75)
    assert exact_sign_test_pvalue([0.0, 0.0]) == pytest.approx(1.0)


def test_sign_flip_requires_no_unjustified_max_control_claim():
    p = sign_flip_randomization_pvalue([1.0, -1.0, 1.0], n_resamples=2_000)
    assert 0.0 <= p <= 1.0


def test_calibration_suite_separates_randomization_applicability():
    report = run_calibration_suite(n_runs=11, n_trials=100, seed=4)
    cases = {case["case"]: case for case in report["cases"]}
    assert cases["symmetric_difference_null"]["randomization_test_applicable"] is True
    assert "sign_flip_rejection_rate" in cases["symmetric_difference_null"]
    assert cases["exchangeable_max_control_null"]["randomization_test_applicable"] is False
    assert "sign_flip_rejection_rate" not in cases["exchangeable_max_control_null"]
    assert cases["zero_effect_ties"]["sign_test_rejection_rate"] == 0.0
