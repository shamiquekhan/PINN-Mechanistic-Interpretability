#!/usr/bin/env bash
# Reproduce the main published results (stages 2-13) from the committed
# runs/ artifacts — no retraining. Verifies headline numbers against the
# expectations documented in docs/reproducibility.md §5.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Reproducing aggregate stages 2-13 (reads committed runs/)"
for stage in 2 3 4 5 6 7 8 9 10 11 12 13 14; do
  echo "---- stage $stage ----"
  python -m experiments.run_pipeline --stages "$stage"
done

echo "==> Verifying headline numbers"
python - <<'EOF'
import json, math, sys
from pathlib import Path

checks = [
    ("positive control passes", "runs/positive_control.json",
     lambda d: d["pipeline_pass"] is True),
    ("SAE causal CI spans zero", "runs/causal_intervention_results.json",
     lambda d: d["causal_strength_ci"]["ci_lower"] <= 0 <= d["causal_strength_ci"]["ci_upper"]),
    ("SAE 0/8 Bonferroni", "runs/causal_intervention_results.json",
     lambda d: d["multiple_comparisons"]["bonferroni_n_survivors"] == 0),
    ("PCA 0/8 Bonferroni", "runs/pca_causal_results.json",
     lambda d: d["multiple_comparisons"]["bonferroni_n_survivors"] == 0),
    ("abstraction: PCA fails gate", "runs/causal_abstraction_results.json",
     lambda d: d["verdict"]["pca"]["beats_random_every_run"] is False),
    ("geometry PR low-rank (fresh pool ~1.3-1.9)", "runs/effective_rank_analysis/effective_rank_report.json",
     lambda d: 1.0 < d["summary"]["mean_participation_ratio"] < 2.5),
    ("FNO: SAE beats PCA", "runs/operator_boundary/operator_boundary_report.json",
     lambda d: d["verdict"]["sae_beats_k_matched_pca"] is True),
    ("monitor AUROC ~ 0.87", "runs/monitor_report.json",
     lambda d: abs(d["conventional_logistic"]["auroc"] - 0.872) < 0.01),
    ("controller rescue ~ 0.017", "runs/controller_demo/controller_comparison.json",
     lambda d: abs(d["verdict"]["controller_final"] - 0.0166) < 0.005),
    ("NTK-adaptive ~ 0.0064", "runs/sota_baselines/sota_baseline_report.json",
     lambda d: abs(d["baselines"]["NTK-adaptive"]["final_rel_l2"] - 0.0064) < 0.003),
    ("stage-14 machinery gate passes", "runs/operator_causal/operator_causal_report.json",
     lambda d: d["positive_control"]["pipeline_pass"] is True),
    ("stage-14 MC survivors > PINN (>=4)", "runs/operator_causal/operator_causal_report.json",
     lambda d: d["battery_summary"]["multiple_comparisons"]["bonferroni_n_survivors"] >= 4),
]

failed = 0
for name, path, pred in checks:
    p = Path(path)
    if not p.exists():
        print(f"  MISSING  {name}  ({path})"); failed += 1; continue
    try:
        ok = pred(json.loads(p.read_text()))
    except Exception as e:
        ok = False
        print(f"  ERROR    {name}: {e}")
    status = "PASS  " if ok else "FAIL  "
    if not ok:
        failed += 1
    print(f"  {status}{name}")

if failed:
    print(f"{failed} check(s) failed — investigate before publication.")
    sys.exit(1)
print("All headline checks passed.")
EOF

echo "==> Regenerating figures and tables"
bash scripts/generate_figures.sh
bash scripts/generate_tables.sh

echo "==> MAIN RESULTS REPRODUCED"
