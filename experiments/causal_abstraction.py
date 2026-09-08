"""Stage 9: Causal-abstraction battery — do ANY candidate alignments satisfy
the interchange criterion on real PINN checkpoints? (Phase 11)

Compares three alignment candidates under identical partial-interchange
interventions on held-out boundary-starvation checkpoints:

  1. PCA basis  — leading components of the low-rank activation manifold
  2. SAE decoder directions — the discovered sparse dictionary
  3. Random orthonormal basis — chance control

High-level causal model H (preregistered): "the leading aligned directions
encode the failure-relevant loss information."  A candidate alignment is a
causal abstraction iff interchange accuracy beats the random-basis control
with run-level consistency.

This is the formal test of the guide's Phase 11 claim — with the honest
possibility (supported by stages 5 and 8) that NO candidate alignment
passes, which itself is a strong statement: neither linear nor sparse
feature alignments causally abstract this PINN.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from analysis.pca_interpretability import fit_pca
from experiments.run_pipeline import (
    DEVICE, RUNS, LAYER, _remap_legacy_state,
    build_multiview_metrics, find_logged_runs,
)
from interventions.interchange import run_alignment_battery, random_basis
from pinn.config import load_config
from pinn.model import MLP
from pinn.pdes import make_pde
from sae.model import SparseAutoencoder


def run_causal_abstraction_experiment(
    n_swapped: int = 1,
    n_pairs: int = 64,
    layer_index: int = 1,
) -> Dict:
    print("\n=======================================================")
    print("STAGE 9: Causal Abstraction (partial interchange battery)")
    print("=======================================================")

    # ---- Load SAE + fit PCA on the shared discovery data ----
    sae_paths = sorted(RUNS.glob("sae_models_v2/*/sae.pt"))
    run_dirs = find_logged_runs(width=64)
    if not run_dirs:
        print("No logged runs; skipping.")
        return {"status": "skipped", "reason": "no logged runs"}
    samples, labels, run_ids, _m = build_multiview_metrics(run_dirs)
    data = torch.tensor(samples, dtype=torch.float32)

    pca_full = fit_pca(data, n_components=8)
    pca_basis = pca_full.components.to(DEVICE)          # (8, 64)
    pca_mean = pca_full.mean.to(DEVICE).flatten()

    sae_basis = None
    sae_mean = None
    if sae_paths:
        sae = SparseAutoencoder.load(sae_paths[0], DEVICE)
        with torch.no_grad():
            # SAE alignment: top-8 decoder columns by total latent activation.
            z, _ = sae(data.to(DEVICE))
            activity = z.abs().sum(dim=0)
            top = torch.argsort(activity, descending=True)[:8]
            sae_basis = sae.W_d.weight.detach().T[top]   # (8, 64) rows
            sae_mean = sae.b_a.detach().flatten().to(DEVICE)
        print(f"SAE basis: top-8 decoder columns by activity "
              f"(activity range {activity[top].min():.2f}..{activity[top].max():.2f})")

    rand_basis = random_basis(64, 8, seed=0).to(DEVICE)
    # For the random control, centering convention must match PCA's
    # (data mean), so the only difference is the basis itself.
    rand_mean = pca_mean

    # ---- Held-out checkpoints ----
    eval_runs = [d for d in RUNS.iterdir()
                 if d.is_dir() and "boundary_starvation" in d.name
                 and list(d.glob("checkpoint_*.pt"))]
    if not eval_runs:
        eval_runs = [RUNS / "failure_boundary_starvation"]
    eval_runs = sorted(eval_runs, key=lambda d: d.name)

    probe_grid_cache: Dict[str, torch.Tensor] = {}
    per_run: Dict[str, Dict] = {}

    for run_dir in eval_runs:
        cfg = load_config(run_dir / "config.json")
        pde = make_pde(cfg.pde)
        model = MLP(cfg.model.input_dim, cfg.model.output_dim,
                    cfg.model.hidden_layers, cfg.model.activation).to(DEVICE)
        ckpt = torch.load(sorted(run_dir.glob("checkpoint_*.pt"))[-1],
                          map_location=DEVICE, weights_only=True)
        state = ckpt["model"]
        if any(k.startswith("net.") for k in state):
            state = _remap_legacy_state(state)
        model.load_state_dict(state)
        dtype = getattr(torch, cfg.run.dtype)

        row = {}
        row["pca"] = run_alignment_battery(
            model, pde, pca_basis, pca_mean,
            layer_index=layer_index, n_swapped=n_swapped,
            n_trials=n_pairs, seed=0,
        )
        row["random"] = run_alignment_battery(
            model, pde, rand_basis, rand_mean,
            layer_index=layer_index, n_swapped=n_swapped,
            n_trials=n_pairs, seed=0,
        )
        if sae_basis is not None:
            row["sae"] = run_alignment_battery(
                model, pde, sae_basis, sae_mean,
                layer_index=layer_index, n_swapped=n_swapped,
                n_trials=n_pairs, seed=0,
            )
        per_run[run_dir.name] = row
        movs = {k: v["mean_movement_fraction"] for k, v in row.items()}
        print(f"  {run_dir.name}: " +
              " ".join(f"{k}_mv={v:+.3f}" for k, v in movs.items()))

    # ---- Aggregate ----
    methods = ["pca", "random"] + (["sae"] if sae_basis is not None else [])
    summary = {}
    for m in methods:
        movs = [per_run[r][m]["mean_movement_fraction"] for r in per_run]
        pos = [per_run[r][m]["frac_movement_positive"] for r in per_run]
        summary[m] = {
            "mean_movement_fraction": float(np.nanmean(movs)),
            "min_movement": float(np.nanmin(movs)),
            "max_movement": float(np.nanmax(movs)),
            "mean_frac_trials_positive_movement": float(np.nanmean(pos)),
            "n_runs": len(movs),
        }

    # Verdict: a candidate alignment is a causal abstraction of H_boundary
    # only if its movement beats the random-basis control on EVERY run —
    # the same run-level consistency standard as the causal battery.
    verdict = {}
    for m in methods:
        if m == "random":
            continue
        diffs = [per_run[r][m]["mean_movement_fraction"]
                 - per_run[r]["random"]["mean_movement_fraction"]
                 for r in per_run]
        verdict[m] = {
            "beats_random_every_run": bool(all(d > 0 for d in diffs)),
            "mean_diff_vs_random": float(np.nanmean(diffs)),
            "n_runs_positive_diff": int(sum(1 for d in diffs if d > 0)),
            "n_runs": len(diffs),
        }

    out = {
        "status": "complete",
        "n_swapped": n_swapped,
        "n_trials_per_run": n_pairs,
        "high_level_model": (
            "H_boundary: region identity {interior, boundary_proximal} -> "
            "PDE loss.  A partial interchange of the aligned subspace "
            "should move the source (interior) loss toward the donor "
            "(boundary-proximal) loss if the alignment carries region "
            "identity."
        ),
        "criterion": (
            "movement fraction = (log L_swap - log L_src)/(log L_don - log L_src); "
            "causal abstraction iff beats random basis on every run"
        ),
        "summary": summary,
        "verdict": verdict,
        "per_run": {
            run: {m: {"mean_movement_fraction": rec[m]["mean_movement_fraction"],
                      "frac_movement_positive": rec[m]["frac_movement_positive"]}
                  for m in methods}
            for run, rec in per_run.items()
        },
        "records_sample": {
            run: {m: rec[m]["records"][:3] for m in methods}
            for run, rec in list(per_run.items())[:1]
        },
    }
    out_path = RUNS / "causal_abstraction_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\nCausal abstraction summary (movement toward donor):")
    for m, s in summary.items():
        print(f"  {m:8s} movement={s['mean_movement_fraction']:+.3f} "
              f"[{s['min_movement']:+.3f},{s['max_movement']:+.3f}] "
              f"pos-trials={s['mean_frac_trials_positive_movement']:.2f}")
    for m, v in verdict.items():
        print(f"  VERDICT {m}: beats random every run = "
              f"{v['beats_random_every_run']} "
              f"(mean diff {v['mean_diff_vs_random']:+.3f}, "
              f"{v['n_runs_positive_diff']}/{v['n_runs']} runs positive)")
    return out


if __name__ == "__main__":
    run_causal_abstraction_experiment()
