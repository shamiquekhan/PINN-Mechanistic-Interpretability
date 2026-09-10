"""Stage 19: Monitor label provenance audit (H19).

Preregistered in docs/preregistration.md §H19 (v3.5, committed BEFORE this
run). Two registered parts:

(a) RD2D label provenance: trace every RD2D run's label derivation; runs
    labeled "failure" for the wrong reason are excluded from monitor
    training and the audit is recorded in the artifact.

(b) A loss-only trajectory monitor (logistic on past-loss windows, same
    splits/CIs) replaces the single-threshold straw man as the floor
    baseline; if it closes much of the 0.468→0.875 gap, the monitor
    section says so and repositions the conventional arm's value honestly.

Acceptance: the monitor section states its label provenance explicitly
and the floor baseline is no longer a threshold rule alone.

Audit findings recorded PRE-VERDICT (traced before any re-training):
  1. The stage-10 RD2D runs (rel L2 ~1.93 — the v3.4 reference) live in
     runs/dimensional_boundary_expanded/reaction_diffusion_2d/ and are
     structurally EXCLUDED from the stage-6 monitor pool (the pool scans
     runs/*/metrics.jsonl only): no RD2D run ever entered monitor
     training. The registered mislabeling concern is therefore moot for
     those runs — recorded, not inferred.
  2. The monitor pool DOES contain runs/  reaction_diffusion_baseline
     (1D RD), whose derived failure step fs=0 is an INIT-TRANSIENT
     artifact: the run converges to 0.0015. It is excluded from
     monitor training per the registered rule (labeled "failure" for
     the wrong reason).
  3. gradient_conflict_seed2026 derives fs=0 but its final rel L2 is
     0.0006 (it recovers to success mid-run after the confirm window).
     Excluded per the same rule (label does not describe the outcome).
  4. Collocation-starvation runs (10/10) derive NO failure step under the
     operational labeler (their failure mode is not a rel-L2 crossing) —
     recorded; consistent with the existing label-reproducibility
     exclusion of collocation from failure-class claims.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from monitoring.features import (
    build_trajectory_features,
    derive_failure_step,
)
from monitoring.models import LogisticMonitor, evaluate_monitor
from pinn_logging.io import load_jsonl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS = PROJECT_ROOT / "runs"

HISTORY_WINDOW = 10
FAILURE_HORIZON = 5
FINAL_SUCCESS_BAR = 0.05  # a run whose final rel L2 ends below this bar
#                       # recovered — its "failure" label is an artifact.


def _audit_run(d: Path) -> Dict:
    """Trace one run's label derivation and outcome."""
    metrics = load_jsonl(d / "metrics.jsonl")
    fs = derive_failure_step(metrics)
    rels = [float(m["relative_l2"]) for m in metrics]
    final, best = rels[-1], min(rels)
    labeled_failure = fs is not None
    ends_failed = final > FINAL_SUCCESS_BAR
    artifact_label = labeled_failure and not ends_failed
    return {
        "run": d.name,
        "derived_failure_step": fs,
        "final_rel_l2": final,
        "best_rel_l2": best,
        "labeled_failure": labeled_failure,
        "ends_failed": ends_failed,
        "label_is_artifact": bool(artifact_label),
        "verdict": (
            "EXCLUDE: recovered run labeled failed (init transient / "
            "recovery after confirm window)"
            if artifact_label else
            "keep: genuine failure (label matches outcome)"
            if labeled_failure else
            "keep: success (no derived failure step)"
        ),
    }


def run_label_provenance_audit() -> Tuple[Dict, List[Path], List[Path]]:
    """(a) Trace labels; return (audit_record, train_pool, test_pool).

    Pool = stage-6 pool minus artifact-labeled runs (registered exclusion).
    Split = stage-6 protocol (same rng, 70/30) reproduced on the cleaned
    pool so the floor baseline is comparable to the published arms.
    """
    run_dirs = [d for d in sorted(RUNS.iterdir())
                if d.is_dir() and (d / "metrics.jsonl").exists()]

    per_run = [_audit_run(d) for d in run_dirs]
    excluded = [r for r in per_run if r["label_is_artifact"]]

    # RD2D structural-exclusion trace (finding 1, above).
    rd2d_stage10 = (RUNS / "dimensional_boundary_expanded"
                    / "reaction_diffusion_2d")
    rd2d_trace = []
    if rd2d_stage10.exists():
        for sub in sorted(rd2d_stage10.iterdir()):
            mp = sub / "metrics.jsonl"
            if sub.is_dir() and mp.exists():
                m = load_jsonl(mp)
                rels = [float(x["relative_l2"]) for x in m]
                rd2d_trace.append({
                    "run": sub.name,
                    "in_monitor_pool": False,
                    "reason": "subdirectory not scanned by the stage-6 pool",
                    "derived_failure_step": derive_failure_step(m),
                    "final_rel_l2": rels[-1],
                })

    cleaned = [d for d, r in zip(run_dirs, per_run)
               if not r["label_is_artifact"]]

    audit = {
        "pool_size_before": len(run_dirs),
        "pool_size_after_exclusion": len(cleaned),
        "excluded_runs": [r["run"] for r in excluded],
        "excluded_details": excluded,
        "collocation_note": (
            "collocation_starvation runs (10/10) derive no failure step "
            "under the operational rel-L2 labeler — the monitor never "
            "tests collocation failures; consistent with the existing "
            "label-reproducibility exclusion"
        ),
        "rd2d_stage10_trace": rd2d_trace,
        "rd2d_finding": (
            "The stage-10 RD2D runs (rel L2 ~1.93) never entered monitor "
            "training: they live under runs/dimensional_boundary_expanded/"
            "reaction_diffusion_2d/, a subdirectory the stage-6 pool does "
            "not scan. The registered mislabeling concern is structurally "
            "moot for those runs — recorded, not inferred."
        ),
    }

    # Stage-6 split protocol reproduced verbatim (same rng/seed/shuffle).
    rng = np.random.default_rng(0)
    shuffled = list(cleaned)
    rng.shuffle(shuffled)
    n_train = max(1, int(0.7 * len(shuffled)))
    return audit, shuffled[:n_train], shuffled[n_train:]


def _build_dataset(runs: List[Path]):
    Xs, ys = [], []
    for d in runs:
        metrics = load_jsonl(d / "metrics.jsonl")
        fs = derive_failure_step(metrics)
        X, y, _ = build_trajectory_features(
            metrics, HISTORY_WINDOW, FAILURE_HORIZON, "any", fs)
        if X.shape[0] > 0:
            Xs.append(X)
            ys.append(y)
    if not Xs:
        return np.empty((0, 0)), np.empty(0)
    return np.vstack(Xs), np.concatenate(ys)


def _run_level_bootstrap_auroc(y, scores, run_slices, n_boot=2000):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    aurocs = []
    n_runs = len(run_slices)
    if n_runs < 2:
        return None
    for _ in range(n_boot):
        picks = rng.integers(0, n_runs, n_runs)
        yb = np.concatenate([y[run_slices[i]] for i in picks])
        sb = np.concatenate([scores[run_slices[i]] for i in picks])
        if len(np.unique(yb)) < 2:
            continue
        try:
            aurocs.append(roc_auc_score(yb, sb))
        except Exception:
            continue
    if not aurocs:
        return None
    lo, hi = np.percentile(aurocs, [2.5, 97.5])
    return {"ci_lower": float(lo), "ci_upper": float(hi),
            "n_boot_valid": len(aurocs)}


def run_h19_monitor_audit() -> Dict:
    print("\n=======================================================")
    print("STAGE 19: Monitor Label Provenance Audit (H19)")
    print("=======================================================")

    audit, train_runs, test_runs = run_label_provenance_audit()
    print(f"Pool: {audit['pool_size_before']} runs -> "
          f"{audit['pool_size_after_exclusion']} after exclusion")
    print(f"Excluded (artifact labels): {audit['excluded_runs']}")
    print(f"Split: {len(train_runs)} train / {len(test_runs)} test")

    X_train, y_train = _build_dataset(train_runs)
    X_test, y_test = _build_dataset(test_runs)
    print(f"Examples: {X_train.shape[0]} train / {X_test.shape[0]} test "
          f"({int(y_test.sum())} positive)")

    if X_test.shape[0] == 0 or len(np.unique(y_test)) < 2:
        out = {"stage": "monitor_label_audit", "audit": audit,
               "status": "insufficient diversity"}
        out_path = RUNS / "monitor_audit" / "monitor_audit_report.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, default=str))
        return out

    # ------------------------------------------------------------------
    # (b) Loss-only trajectory logistic floor: PAST-ONLY loss channels
    # (loss, loss_pde, loss_bc) WITHOUT validation-grid rel_l2 and
    # without gradients.  The published threshold floor conflated "the
    # features are weak" with "the rule is a straw man"; this arm
    # separates them.
    # ------------------------------------------------------------------
    KEYS = ["loss", "loss_pde", "loss_bc"]  # rel_l2 deliberately excluded

    def _loss_only_features(runs: List[Path]):
        Xs, ys = [], []
        for d in runs:
            metrics = load_jsonl(d / "metrics.jsonl")
            fs = derive_failure_step(metrics)
            n = len(metrics)
            rows, labels = [], []
            for t in range(HISTORY_WINDOW, n - FAILURE_HORIZON):
                window = metrics[t - HISTORY_WINDOW: t]
                rows.append([float(rec.get(k, 0.0) or 0.0)
                             for rec in window for k in KEYS])
                future = metrics[t: t + FAILURE_HORIZON]
                future_steps = [m["step"] for m in future]
                labels.append(int(fs in future_steps) if fs is not None
                              else 0)
            if rows:
                Xs.append(np.array(rows, dtype=np.float32))
                ys.append(np.array(labels, dtype=np.int32))
        if not Xs:
            return np.empty((0, 0)), np.empty(0)
        return np.vstack(Xs), np.concatenate(ys)

    XL_train, yl_train = _loss_only_features(train_runs)
    XL_test, yl_test = _loss_only_features(test_runs)

    floor = LogisticMonitor()
    floor.fit(XL_train, yl_train)
    floor_scores = floor.predict_proba(XL_test)

    # Run slices for the bootstrap (aligned with the loss-only builder).
    run_slices = []
    cursor = 0
    for d in test_runs:
        metrics = load_jsonl(d / "metrics.jsonl")
        fs = derive_failure_step(metrics)
        n = len(metrics)
        n_rows = max(0, n - FAILURE_HORIZON - HISTORY_WINDOW)
        if n_rows > 0:
            run_slices.append(slice(cursor, cursor + n_rows))
            cursor += n_rows
        else:
            run_slices.append(slice(cursor, cursor))

    floor_eval = evaluate_monitor(yl_test, floor_scores,
                                  np.arange(yl_test.shape[0]))
    ci = _run_level_bootstrap_auroc(yl_test, floor_scores, run_slices)
    if ci is not None:
        floor_eval["auroc_ci"] = ci

    # Reference arms (published stage-6 numbers, unchanged protocol).
    reference = {
        "loss_only_threshold_auroc": 0.468,
        "conventional_logistic_auroc": 0.875,
        "sae_plus_conventional_auroc": 0.877,
    }

    gap_threshold = 0.468
    gap_conv = 0.875
    closes_gap = bool(
        floor_eval["auroc"] - gap_threshold
        >= 0.5 * (gap_conv - gap_threshold))

    report = {
        "stage": "monitor_label_audit",
        "hypotheses": "H19 (docs/preregistration.md §H19; v3.5)",
        "audit": audit,
        "protocol": {
            "history_window": HISTORY_WINDOW,
            "failure_horizon": FAILURE_HORIZON,
            "split": "stage-6 protocol reproduced (rng seed 0, 70/30) on "
                     "the cleaned pool",
            "floor_features": "past-only loss channels (loss, loss_pde, "
                              "loss_bc); rel_l2 and gradients excluded",
        },
        "loss_trajectory_logistic": floor_eval,
        "reference_arms_published": reference,
        "gap_closure": {
            "threshold_floor": gap_threshold,
            "conventional": gap_conv,
            "loss_trajectory_floor_auroc": floor_eval["auroc"],
            "fraction_of_gap_closed": float(
                (floor_eval["auroc"] - gap_threshold)
                / (gap_conv - gap_threshold)),
            "closes_half_or_more": closes_gap,
        },
    }

    out_path = RUNS / "monitor_audit" / "monitor_audit_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))

    print(f"Loss-only trajectory logistic AUROC: "
          f"{floor_eval['auroc']:.3f} "
          f"(CI [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}])"
          if ci else "")
    print(f"Gap closure: {report['gap_closure']['fraction_of_gap_closed']:.1%}"
          f" of the 0.468->0.875 threshold->conventional gap")
    print(f"Report: {out_path}")
    return report


if __name__ == "__main__":
    run_h19_monitor_audit()
