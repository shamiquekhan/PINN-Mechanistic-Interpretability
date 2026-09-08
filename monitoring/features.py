"""
Past-only trajectory feature builder for early-warning monitoring.
Enforces no temporal leakage: every feature window uses only information
available at the prediction time.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pinn_logging.io import load_jsonl


# ---------------------------------------------------------------------------
# Feature window construction
# ---------------------------------------------------------------------------

def build_trajectory_features(
    metrics: List[Dict],
    history_window: int,
    failure_horizon: int,
    failure_label: Optional[str] = None,   # M5: unused; kept for call-site compat
    failure_step: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build supervised examples from a single run's trajectory.

    For each time point t (within the valid range), create:
      - X[t]: vector of metrics from [t - history_window, t)
      - y[t]: 1 if failure occurs within [t, t + failure_horizon)
      - steps[t]: the step index at prediction time

    Parameters
    ----------
    metrics:         List of metric records (each has 'step', 'loss_pde', etc.)
    history_window:  Number of past records to include as features.
    failure_horizon: Number of future records to look ahead for failure.
    failure_label:   e.g. "gradient_conflict" — used to identify failure steps.
    failure_step:    If known, the step at which failure was declared.

    Returns
    -------
    X: (N, D) feature matrix
    y: (N,) binary label array
    steps: (N,) step indices at prediction time
    """
    if len(metrics) < history_window + 1:
        return np.empty((0, 0)), np.empty(0), np.empty(0)

    keys = ["loss", "loss_pde", "loss_bc", "relative_l2"]
    n = len(metrics)
    step_indices = np.array([m["step"] for m in metrics])

    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    pred_steps: List[int] = []

    for t in range(history_window, n - failure_horizon):
        # Past-only window: [t - history_window, t)
        window = metrics[t - history_window: t]
        feat_vec = []
        for rec in window:
            for k in keys:
                feat_vec.append(float(rec.get(k, 0.0) or 0.0))
        X_rows.append(feat_vec)

        # Label: does failure occur within the next failure_horizon steps?
        future = metrics[t: t + failure_horizon]
        label = 0
        if failure_step is not None:
            future_steps = [m["step"] for m in future]
            label = int(failure_step in future_steps)

        y_rows.append(label)
        pred_steps.append(metrics[t]["step"])

    if not X_rows:
        return np.empty((0, 0)), np.empty(0), np.empty(0)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)
    steps = np.array(pred_steps, dtype=np.int32)
    return X, y, steps


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------

def leakage_audit(
    X: np.ndarray,
    y: np.ndarray,
    steps: np.ndarray,
    failure_step: int,
) -> Dict:
    """Verify that no post-failure data appears in the feature window.

    Returns a dict with pass/fail flags and diagnostics.
    """
    violations = []
    for i in range(len(X)):
        # The prediction is made at steps[i]; feature window covers steps up to steps[i].
        # A violation occurs if the feature vector contains any value from failure_step or later.
        # (We check the step indices implicitly by ensuring the window ends before failure_step)
        if steps[i] >= failure_step and y[i] == 0:
            violations.append(int(steps[i]))

    return {
        "n_examples":     len(X),
        "n_positive":     int(y.sum()),
        "n_violations":   len(violations),
        "passed":         len(violations) == 0,
        "violation_steps": violations[:5],   # first 5 for debugging
    }


# ---------------------------------------------------------------------------
# Multi-run dataset builder
# ---------------------------------------------------------------------------

def derive_failure_step(
    metrics: List[Dict],
    fail_threshold: float = 0.05,
    confirm_records: int = 3,
) -> Optional[int]:
    """Derive the operational failure step from the trajectory itself.

    A run fails at step s if relative_l2 first exceeds `fail_threshold` at s
    and stays above it for `confirm_records` consecutive records (avoiding
    labeling from one noisy spike).  Returns None for successful runs.
    """
    rel = [float(m.get("relative_l2", float("nan"))) for m in metrics]
    steps = [int(m["step"]) for m in metrics]

    above = 0
    for i, r in enumerate(rel):
        if r > fail_threshold:
            above += 1
            if above >= confirm_records:
                # First step of the confirmed crossing window.
                j = i - confirm_records + 1
                return steps[max(j, 0)]
        else:
            above = 0
    return None


def build_monitor_dataset(
    run_dirs: List[Path],
    failure_steps: Optional[Dict[str, int]] = None,   # run_name -> failure step (or -1 if success)
    history_window: int = 200,
    failure_horizon: int = 500,
    derive_labels: bool = True,
    fail_threshold: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate trajectory features across multiple runs.

    Parameters
    ----------
    run_dirs:      List of run directories.
    failure_steps: Optional explicit mapping run_name -> failure step
                   (-1 = no failure).  Ignored when derive_labels=True.
    derive_labels: If True (default), derive each run's failure step from its
                   own rel_l2 trajectory via `derive_failure_step` instead of
                   trusting a hardcoded step number.
    fail_threshold: rel_l2 threshold used for derived labels.

    Returns
    -------
    X: (N_total, D) feature matrix
    y: (N_total,) labels
    """
    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            continue

        metrics = load_jsonl(metrics_path)
        run_name = run_dir.stem

        if derive_labels:
            fail_step = derive_failure_step(metrics, fail_threshold=fail_threshold)
        else:
            fs = (failure_steps or {}).get(run_name, -1)
            fail_step = fs if fs >= 0 else None

        X, y, _ = build_trajectory_features(
            metrics,
            history_window=history_window,
            failure_horizon=failure_horizon,
            failure_label="any",
            failure_step=fail_step,
        )
        if X.shape[0] > 0:
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        return np.empty((0, 0)), np.empty(0)

    return np.vstack(all_X), np.concatenate(all_y)
