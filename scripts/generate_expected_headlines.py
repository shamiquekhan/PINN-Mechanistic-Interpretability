#!/usr/bin/env python3
"""C1 fix companion: regenerate runs/expected_headlines.json from the
committed artifacts. Expectations are DERIVED, never hand-edited, so the
reproduce-script gate can never silently contradict the artifacts the way
the hard-coded AUROC 0.872 (stale v3.0 value) did against the fresh 0.859.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def load(rel: str):
    return json.loads((RUNS / rel).read_text())


def main() -> int:
    m = load("monitor_report.json")
    c = load("controller_demo/controller_comparison.json")
    sae = load("causal_intervention_results.json")
    pca = load("pca_causal_results.json")
    op = load("operator_boundary/operator_boundary_report.json")
    oc = load("operator_causal/operator_causal_report.json")
    sota = load("sota_baselines/sota_baseline_report.json")
    er = load("effective_rank_analysis/effective_rank_report.json")
    ntk = load("ntk_bridge/ntk_bridge_report.json")

    expected = {
        "_generator": "scripts/generate_expected_headlines.py — derive from artifacts, never hand-edit",
        "monitor_conventional_auroc": {
            "expected": round(m["conventional_logistic"]["auroc"], 3),
            "range": [round(m["conventional_logistic"]["auroc_ci"]["ci_lower"] - 0.01, 3),
                      round(m["conventional_logistic"]["auroc_ci"]["ci_upper"] + 0.01, 3)],
        },
        "sae_et_ci_spans_zero": {"expected": bool(sae["causal_strength_ci"]["ci_lower"] <= 0 <= sae["causal_strength_ci"]["ci_upper"])},
        "sae_bonferroni_survivors": {"expected": sae["multiple_comparisons"]["bonferroni_n_survivors"]},
        "pca_bonferroni_survivors": {"expected": pca["multiple_comparisons"]["bonferroni_n_survivors"]},
        "fno_sae_beats_pca": {"expected": op["verdict"]["sae_beats_k_matched_pca"]},
        "geometry_mean_pr": {"expected": round(er["summary"]["mean_participation_ratio"], 3),
                             "range": [1.0, 2.5]},
        "controller_final": {"expected": round(c["verdict"]["controller_final"], 4),
                             "range": [0.0, 0.10]},
        "no_action_final": {"expected": round(c["verdict"]["no_action_final"], 4)},
        "ntk_adaptive_final": {"expected": round(sota["baselines"]["NTK-adaptive"]["final_rel_l2"], 4)},
        "stage14_bonferroni_survivors": {"expected": oc["verdict"]["bonferroni_survivors"]},
        "stage14_recorded": {"expected": "H14b" if oc["verdict"]["h14b_broader_null"] else "H14a"},
        "stage15_recorded": {"expected": ntk["decision_rule"]["recorded"]},
    }
    out = RUNS / "expected_headlines.json"
    out.write_text(json.dumps(expected, indent=2) + "\n")
    print(f"wrote {out} ({len(expected) - 1} headlines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
