"""Stage 8: PCA causal battery — the 'what works instead' head-to-head.

Runs the SAME intervention protocol (3 controls, bootstrap CIs,
multiple-comparison corrections) on the leading PCA components of the
activation manifold, on the SAME held-out boundary-starvation checkpoints
used by the SAE battery, then compares methods.

Hypotheses (preregistered in docs/preregistration.md):
  H8a (null): PCA components also fail the causal battery — the low-rank
       manifold carries no direction-specific causal signal, and the
       correct conclusion is 'no feature-based causal interpretability
       in this benchmark, linear or sparse'.
  H8b (linear): PCA components beat all three controls with MC-corrected
       survivors — linear coordinates of the manifold are the causal
       features and SAEs merely fragmented them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from analysis.pca_interpretability import fit_pca
from experiments.run_pipeline import (
    DEVICE, RUNS, LAYER, _remap_legacy_state,
    build_multiview_metrics, find_logged_runs,
)
from interventions.pca_battery import (
    run_pca_causal_battery, summarize_battery, compare_pca_vs_sae,
    train_pca_probe_direction, pca_basis_to_device,
)
from pinn.config import load_config
from pinn.model import MLP
from pinn.pdes import make_pde


def run_pca_causal_experiment(
    n_components: int = 8,
    layer_index: int = 1,
) -> Dict:
    print("\n=======================================================")
    print("STAGE 8: PCA Causal Battery (head-to-head vs SAE)")
    print("=======================================================")

    # 1. Discovery data: identical to the SAE battery (all logged 64-wide
    #    runs, layer layers.1) — fit the PCA basis + probe on it.
    run_dirs = find_logged_runs(width=64)
    if not run_dirs:
        print("No logged runs found; skipping.")
        return {"status": "skipped", "reason": "no logged runs"}
    samples, labels, run_ids, _metrics = build_multiview_metrics(run_dirs)
    data = torch.tensor(samples, dtype=torch.float32)
    basis_cpu = fit_pca(data, n_components=n_components)
    print(f"PCA basis: {basis_cpu.n_components} components on {basis_cpu.input_dim}-d "
          f"activations from {len(run_dirs)} runs, {data.shape[0]} samples")

    probe_dir = train_pca_probe_direction(basis_cpu, data, labels, torch.device("cpu"))
    print("PCA probe direction trained (failure labels).")
    basis = pca_basis_to_device(basis_cpu, DEVICE)

    # 2. Evaluation checkpoints: the SAME boundary-starvation runs as stage 5.
    eval_runs = [d for d in RUNS.iterdir()
                 if d.is_dir() and "boundary_starvation" in d.name
                 and list(d.glob("checkpoint_*.pt"))]
    if not eval_runs:
        eval_runs = [RUNS / "failure_boundary_starvation"]
    eval_runs = sorted(eval_runs, key=lambda d: d.name)
    print(f"PCA causal eval on {len(eval_runs)} boundary-starvation runs")

    # 3. Battery on each checkpoint.
    candidates = list(range(n_components))
    all_results: List[Dict] = []
    for run_dir in eval_runs:
        cfg = load_config(run_dir / "config.json")
        pde = make_pde(cfg.pde)
        model = MLP(cfg.model.input_dim, cfg.model.output_dim,
                    cfg.model.hidden_layers, cfg.model.activation).to(DEVICE)
        ckpts = sorted(run_dir.glob("checkpoint_*.pt"))
        if not ckpts:
            continue
        ckpt = torch.load(ckpts[-1], map_location=DEVICE, weights_only=False)
        state = ckpt["model"]
        if any(k.startswith("net.") for k in state):
            state = _remap_legacy_state(state)
        model.load_state_dict(state)
        dtype = getattr(torch, cfg.run.dtype)

        results = run_pca_causal_battery(
            model=model, basis=basis, pde=pde, device=DEVICE, dtype=dtype,
            candidate_components=candidates,
            probe_direction=probe_dir,
            target_loss="pde", layer_index=layer_index,
            alpha=0.0,
        )
        for r in results:
            r["run"] = run_dir.name
        all_results.extend(results)

    if not all_results:
        print("No checkpoints evaluated; aborting.")
        return {"status": "skipped", "reason": "no checkpoints"}

    pca_summary = summarize_battery(all_results)

    # 4. Head-to-head vs the SAE battery (same runs, same protocol).
    sae_path = RUNS / "causal_intervention_results.json"
    head_to_head = None
    sae_rows_for_compare = None
    if sae_path.exists():
        sae_report = json.loads(sae_path.read_text())
        sae_rows = sae_report.get("evaluations", [])
        # Align by (run, slot): SAE battery used 8 features per run; PCA
        # battery uses 8 components per run. Both are ordered run-major.
        sae_rows_aligned = [r for r in sae_rows
                            if r.get("run") in {rr["run"] for rr in all_results}]
        # Reorder SAE rows to match PCA run order (run-major, feature-minor).
        order = {name: i for i, name in enumerate(
            sorted({rr["run"] for rr in all_results}))}
        sae_rows_aligned.sort(key=lambda r: (order.get(r.get("run"), 99),
                                             r.get("feature_idx", 0)))
        all_results_sorted = sorted(
            all_results,
            key=lambda r: (order.get(r["run"], 99), r["feature_idx"]),
        )
        if len(sae_rows_aligned) == len(all_results_sorted):
            # SAE feature ids are large (e.g., 9..205); PCA ids are 0..7 —
            # pair positionally within each run's 8-slot block.
            sae_rows_for_compare = sae_rows_aligned
            head_to_head = compare_pca_vs_sae(
                all_results_sorted, sae_rows_for_compare)
        else:
            head_to_head = {
                "error": "row count mismatch",
                "n_pca": len(all_results_sorted),
                "n_sae": len(sae_rows_aligned),
            }

    # 5. Write artifact.
    out = {
        "status": "complete",
        "method": "PCA components (n={})".format(n_components),
        "protocol": ("3 controls (unrelated component, random coefficient "
                     "direction, supervised probe), bootstrap CIs, "
                     "Bonferroni + BH-FDR across components"),
        "n_evaluations": pca_summary["n_evaluations"],
        "n_beats_all_controls": pca_summary["n_beats_all_controls"],
        "verification_rate": pca_summary["verification_rate"],
        "causal_strength_ci": pca_summary["causal_strength_ci"],
        "specificity_ci": pca_summary["specificity_ci"],
        "sign_diagnostic": pca_summary["sign_diagnostic"],
        "probe_control": pca_summary["probe_control"],
        "multiple_comparisons": pca_summary["multiple_comparisons"],
        "head_to_head_vs_sae": head_to_head,
        "evaluations": all_results,
    }
    out_path = RUNS / "pca_causal_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Evaluations: {out['n_evaluations']} "
          f"({out['n_beats_all_controls']} beat all 3 controls)")
    print(f"PCA causal strength E_T: "
          f"{out['causal_strength_ci']['mean']:.4f} "
          f"[{out['causal_strength_ci']['ci_lower']:.4f}, "
          f"{out['causal_strength_ci']['ci_upper']:.4f}]")
    print(f"MC correction: Bonferroni "
          f"{pca_summary['multiple_comparisons']['bonferroni_n_survivors']}/"
          f"{n_components} | BH-FDR "
          f"{pca_summary['multiple_comparisons']['bh_fdr_n_survivors']}/"
          f"{n_components}")
    if head_to_head and "error" not in (head_to_head or {}):
        print(f"Head-to-head: PCA E_T mean "
              f"{head_to_head['mean_pca_causal_strength']:.4f} vs SAE "
              f"{head_to_head['mean_sae_causal_strength']:.4f} "
              f"(paired diff CI "
              f"[{head_to_head['pca_minus_sae_causal_strength']['ci_lower']:.4f},"
              f" {head_to_head['pca_minus_sae_causal_strength']['ci_upper']:.4f}])")
    return out


if __name__ == "__main__":
    result = run_pca_causal_experiment()
    status = result.get("multiple_comparisons", {})
    if "error" in result:
        print(json.dumps(result, indent=2))
