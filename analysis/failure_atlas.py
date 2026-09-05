"""
Failure atlas generator — aggregates all run directories, assigns operational
failure labels, generates comparative summary plots, and outputs the index JSON.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from pinn_logging.io import load_jsonl


def classify_run_failure(summary: Dict, metrics: List[Dict], grad_logs: List[Dict]) -> Dict[str, float | str]:
    """Assign operational failure label based on quantitative criteria:

    - success              : rel_l2 < 0.01
    - boundary_starvation  : loss_bc > 10 * loss_pde and rel_l2 > 0.05
    - gradient_conflict   : mean_gradient_cosine < -0.2 persistent over last 50% steps
    - spectral_suppression: rel_l2 > 0.1 despite loss < 1e-3
    - collocation_starvation: rel_l2 > 0.1 and loss_pde < 1e-4
    """
    rel_l2 = summary.get("validation_relative_l2", float("nan"))
    final_pde = summary.get("final_loss_pde", 0.0) or 0.0
    final_bc = summary.get("final_loss_bc", 0.0) or 0.0

    # Gradient conflict score from grad logs
    cosines = []
    for g in grad_logs:
        c = g.get("gradient_cosines", {}).get("pde_vs_bc")
        if c is not None:
            cosines.append(c)

    mean_cos = float(np.mean(cosines[-len(cosines)//2:])) if len(cosines) >= 4 else 0.0

    label = "unlabeled"
    if not np.isnan(rel_l2) and rel_l2 < 0.01:
        label = "success"
    elif final_bc > 10.0 * (final_pde + 1e-8) and rel_l2 > 0.05:
        label = "boundary_starvation"
    elif mean_cos < -0.2 and rel_l2 > 0.05:
        label = "gradient_conflict"
    elif final_pde < 1e-4 and rel_l2 > 0.1:
        label = "collocation_starvation"
    elif rel_l2 > 0.1:
        label = "spectral_suppression"

    return {
        "operational_label": label,
        "declared_label":    summary.get("failure_label", "unlabeled"),
        "mean_grad_cosine":  mean_cos,
        "final_rel_l2":      rel_l2,
    }


def aggregate_failure_atlas(runs_dir: Path, out_dir: Path) -> List[Dict]:
    runs_dir = Path(runs_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    atlas_entries: List[Dict] = []

    for run_path in sorted(runs_dir.iterdir()):
        if not run_path.is_dir():
            continue
        summary_path = run_path / "run_summary.json"
        if not summary_path.exists():
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        metrics = load_jsonl(run_path / "metrics.jsonl")
        grad_logs = load_jsonl(run_path / "gradients.jsonl")

        classification = classify_run_failure(summary, metrics, grad_logs)

        entry = {
            "run_dir":       str(run_path),
            "run_name":      summary.get("run_name"),
            "seed":          summary.get("seed"),
            **classification,
            "summary":       summary,
        }
        atlas_entries.append(entry)

    with open(out_dir / "failure_atlas_index.json", "w") as f:
        json.dump(atlas_entries, f, indent=2)

    print(f"Failure Atlas generated: {len(atlas_entries)} runs indexed at {out_dir / 'failure_atlas_index.json'}")
    return atlas_entries
