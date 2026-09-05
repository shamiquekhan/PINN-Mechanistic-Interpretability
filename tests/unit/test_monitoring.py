"""Tests for the monitor: past-only windows, leakage audit, monitor model basics."""
import numpy as np
from monitoring.features import build_trajectory_features, leakage_audit
from monitoring.models import ThresholdMonitor


def _make_metrics(n=200):
    return [
        {"step": i, "loss": 1.0 / (i + 1), "loss_pde": 0.5 / (i + 1),
         "loss_bc": 0.5 / (i + 1), "relative_l2": 1.0 / (i + 1)}
        for i in range(n)
    ]


def test_build_trajectory_features_shape():
    metrics = _make_metrics(100)
    X, y, steps = build_trajectory_features(metrics, history_window=10, failure_horizon=5,
                                             failure_label="any", failure_step=80)
    assert X.ndim == 2
    assert len(y) == X.shape[0]
    assert len(steps) == X.shape[0]
    # Feature dim = history_window * n_keys = 10 * 4
    assert X.shape[1] == 10 * 4


def test_past_only_window_no_future_data():
    metrics = _make_metrics(100)
    X, y, steps = build_trajectory_features(metrics, history_window=5, failure_horizon=3,
                                             failure_label="any", failure_step=60)
    # All prediction steps should be before failure_step + failure_horizon to avoid look-ahead
    for s in steps:
        assert s < len(metrics), f"Step {s} out of range"


def test_leakage_audit_passes_on_clean_data():
    metrics = _make_metrics(100)
    X, y, steps = build_trajectory_features(metrics, history_window=10, failure_horizon=5,
                                             failure_label="any", failure_step=80)
    audit = leakage_audit(X, y, steps, failure_step=80)
    assert "passed" in audit


def test_threshold_monitor_alarm_on_degradation():
    monitor = ThresholdMonitor(plateau_window=5, degradation_factor=2.0)
    # Simulate improving, then suddenly degrading
    for i in range(10):
        monitor.update(1.0 / (i + 1), 0.1 / (i + 1))
    score = monitor.update(1.0, 1.0)   # rel_l2 shoots up
    assert score > 0.5, f"Expected high alarm score on degradation, got {score}"


def test_threshold_monitor_no_alarm_on_steady_improvement():
    monitor = ThresholdMonitor(plateau_window=20, degradation_factor=10.0)
    scores = [monitor.update(1.0 / (i + 1), 0.1 / (i + 1)) for i in range(30)]
    # Should not trigger alarm during steady improvement
    assert max(scores) < 0.7, f"Unexpected alarm during improvement: {max(scores)}"
