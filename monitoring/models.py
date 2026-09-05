"""
Early-warning monitor models for PINN failure prediction.

Three monitors (per the implementation plan):
  1. loss_only      — threshold rule on loss plateau / relative_l2 spike
  2. conventional   — gradient norms + cosine + residual + coverage features
  3. sae_monitor    — validated SAE features + conventional features

Evaluation: AUROC, AUPRC, calibration, recall at fixed FPR, median lead time.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.calibration import calibration_curve
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

from pinn_logging.io import load_jsonl


# ---------------------------------------------------------------------------
# Threshold (loss-only) monitor
# ---------------------------------------------------------------------------

class ThresholdMonitor:
    """Simple threshold rule: alarm when relative_l2 exceeds a fraction of its
    running minimum (indicating deterioration), or when loss plateaus.

    Parameters
    ----------
    plateau_window:  Number of consecutive steps with < min_improvement to declare plateau.
    min_improvement: Minimum fractional improvement to not count as plateau.
    degradation_factor: Relative L2 must exceed min_so_far * this factor to alarm.
    """

    def __init__(
        self,
        plateau_window: int = 50,
        min_improvement: float = 1e-4,
        degradation_factor: float = 2.0,
    ):
        self.plateau_window = plateau_window
        self.min_improvement = min_improvement
        self.degradation_factor = degradation_factor
        self._loss_history: List[float] = []
        self._best_l2: float = float("inf")

    def update(self, loss: float, rel_l2: float) -> float:
        """Update monitor state and return alarm score in [0, 1]."""
        self._loss_history.append(loss)
        self._best_l2 = min(self._best_l2, rel_l2)

        score = 0.0

        # Plateau detection
        if len(self._loss_history) >= self.plateau_window:
            window = self._loss_history[-self.plateau_window:]
            improvement = (window[0] - window[-1]) / (abs(window[0]) + 1e-10)
            if improvement < self.min_improvement:
                score = max(score, 0.6)

        # Degradation detection
        if rel_l2 > self._best_l2 * self.degradation_factor:
            score = max(score, 0.9)

        return score

    def reset(self):
        self._loss_history.clear()
        self._best_l2 = float("inf")


# ---------------------------------------------------------------------------
# Logistic regression monitor
# ---------------------------------------------------------------------------

class LogisticMonitor:
    """Logistic regression trained on past-only trajectory features.

    Works for both the 'conventional' (gradient + residual features) and
    'sae_monitor' (+ SAE feature trajectories) variants.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        if not _SKLEARN:
            raise ImportError("scikit-learn is required for LogisticMonitor. pip install scikit-learn")
        self.C = C
        self.model = LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced", solver="lbfgs")
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self._fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Monitor not fitted.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def evaluate_monitor(
    y_true: np.ndarray,
    y_score: np.ndarray,
    steps: np.ndarray,
    failure_step: Optional[int] = None,
    fpr_target: float = 0.1,
) -> Dict:
    """Compute AUROC, AUPRC, calibration, recall@FPR, and lead time.

    Parameters
    ----------
    y_true:       Binary labels (1 = failure imminent).
    y_score:      Predicted probability scores.
    steps:        Step index at each prediction.
    failure_step: Actual step where failure occurred (for lead-time computation).
    fpr_target:   False-positive rate at which to report recall.
    """
    results: Dict = {}

    if len(np.unique(y_true)) < 2:
        return {"error": "Only one class in y_true — cannot compute AUROC"}

    if _SKLEARN:
        results["auroc"]  = float(roc_auc_score(y_true, y_score))
        results["auprc"]  = float(average_precision_score(y_true, y_score))

        # Recall at target FPR
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        idx = np.searchsorted(fpr, fpr_target)
        results[f"recall_at_fpr{fpr_target}"] = float(tpr[min(idx, len(tpr) - 1)])

    # Lead time: earliest step where score > 0.5 and a failure is labelled positive
    if failure_step is not None:
        alarm_steps = steps[(y_score > 0.5) & (y_true == 1)]
        if len(alarm_steps) > 0:
            earliest_alarm = int(alarm_steps.min())
            results["lead_time_steps"] = int(failure_step - earliest_alarm)
        else:
            results["lead_time_steps"] = -1   # no alarm fired

    results["n_examples"]  = int(len(y_true))
    results["n_positive"]  = int(y_true.sum())
    results["prevalence"]  = float(y_true.mean())
    return results


def tune_threshold_for_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_recall: float = 0.8,
) -> float:
    """Find the lowest threshold achieving at least target_recall."""
    thresholds = np.linspace(0, 1, 101)
    for t in reversed(thresholds):
        preds = (y_score >= t).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        recall = tp / (tp + fn + 1e-10)
        if recall >= target_recall:
            return float(t)
    return 0.0   # fallback
