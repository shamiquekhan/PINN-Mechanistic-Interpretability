"""Stage 18: Scope boundary probe — Fourier-feature PINN + depth (H18).

Preregistered in docs/preregistration.md §H18 (v3.5, committed BEFORE this
run): the strongest reviewer attack on the low-rank null is "width-16-512
tanh MLPs — of course no superposition." The FNO demonstrates the positive
side of the boundary; within-PINN architectural variation was missing.

Registered design:
  - Fixed 1D Poisson task (the width-scaling control), width 64:
    (a) Fourier-feature PINN (fourier_embed=true, n_freq=32), 3 seeds;
    (b) depth sweep 2 -> 6 hidden layers, 3 seeds each;
  - Rank diagnostic (PR/W, tangent rank, PCA-95% components) for every run;
  - SAE-vs-k-matched-PCA reconstruction wherever PR moves.

Pre-written outcomes (both strengthen the paper):
  - H18a: Fourier-feature activation PR rises toward superposition and
    the SAE's relative advantage over PCA appears within PINNs -> the
    regime boundary gains a within-PINN demonstration, rank diagnostic
    as predictor.
  - H18b: PR stays ~2 and the SAE stays null under the geometry-changing
    trick -> the null is robust to architectural perturbation; scope is
    honestly drawn at operator/function-space representations.

Machinery note (pre-run, recorded before any verdict): the fourier_embed
config field existed in the schema but was dead in the training entry
point (experiments/train.py never passed it to MLP) — fixed at the same
commit as this file, BEFORE the run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from analysis.activation_manifold import local_tangent_rank
from analysis.effective_rank import (
    analyze_run_activations,
    participation_ratio,
    pca_components_for_energy,
    stable_rank,
)
from pinn_logging.io import load_jsonl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS = PROJECT_ROOT / "runs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WIDTH = 64
SEEDS = [7, 42, 123]
STEPS = 2000  # matches the width-scaling control regime (logged PR there
#              saturated by ~1000 steps; 2000 gives margin)

# Registered: PR moves "meaningfully" if it departs the width-scaling
# envelope.  The width-scaling runs measured PR 1.14-2.51 (max 2.51)
# across widths 16-512 on this task family; the pre-written reading uses
# the same envelope: PR > 3 (well above the measured max, below FNO 6.8)
# counts as "moved" for the reconstruction comparison.
PR_MOVE_THRESHOLD = 3.0


def build_h18_config(name: str, hidden_layers: List[int], seed: int,
                     fourier: bool, output_root: Path,
                     n_freq: int = 32) -> Dict:
    """Fixed 1D Poisson task (the width-scaling control), width 64."""
    return {
        "run": {
            "name": name,
            "output_dir": str(output_root),
            "seed": seed,
            "deterministic": True,
            "device": "cuda" if DEVICE.type == "cuda" else "cpu",
            "dtype": "float32",
            "failure_label": "unlabeled",
        },
        "pde": {
            "name": "poisson_1d",
            "domain": [-1.0, 1.0],
            "source": 1.0,
            "forcing": 0.0,
            "diffusion": 0.01,
            "boundary_values": [0.0, 0.0],
            "validation_points": 201,
        },
        "model": {
            "input_dim": 1,
            "output_dim": 1,
            "hidden_layers": hidden_layers,
            "activation": "tanh",
            "init": "xavier",
            "fourier_embed": fourier,
            "fourier_n_freq": n_freq,
            "fourier_scale": 1.0,
        },
        "training": {
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "steps": STEPS,
            "interior_points": 128,
            "boundary_points": 2,
            "log_every": 100,
            "checkpoint_every": STEPS,
            "lambda_pde": 1.0,
            "lambda_bc": 1.0,
            "resample_every": 0,
            "spatial_bias": 0.0,
        },
        "logging": {
            "save_activations": True,
            "activation_layers": [1],
            "save_pointwise_residuals": True,
            "log_gradients": False,
            "grad_log_every": 100,
            "log_diagnostics": False,
            "diag_log_every": 100,
        },
    }


def _tangent_rank_for_run(run_dir: Path, layer_name: str = "layers.1") -> int:
    records = load_jsonl(run_dir / "activations.jsonl")
    arrays = []
    for record in records:
        raw = record.get("activations", {}).get(layer_name, {}).get("raw")
        if raw is not None:
            arrays.append(np.asarray(raw, dtype=np.float64))
    if not arrays:
        return 0
    data = np.concatenate(arrays, axis=0)
    coordinates = np.linspace(-1.0, 1.0, data.shape[0])
    return local_tangent_rank(data, coordinates=coordinates)


def _collect_representations(run_dir: Path,
                             layer_name: str = "layers.1") -> torch.Tensor:
    """All logged raw activations of the canonical SAE site (layers.1)."""
    records = load_jsonl(run_dir / "activations.jsonl")
    arrays = []
    for record in records:
        raw = record.get("activations", {}).get(layer_name, {}).get("raw")
        if raw is not None:
            arrays.append(np.asarray(raw, dtype=np.float32))
    if not arrays:
        raise ValueError(f"no raw activations in {run_dir}")
    return torch.from_numpy(np.concatenate(arrays, axis=0))


def _train_one(config: Dict, steps: int) -> Path:
    """Run experiments.train as a subprocess (the canonical training path)."""
    import os
    import subprocess
    import sys
    import tempfile

    out_root = Path(config["run"]["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    cfg_path = out_root / f"{config['run']['name']}.yaml"
    import yaml
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False))
    subprocess.run(
        [sys.executable, "-m", "experiments.train",
         "--config", str(cfg_path),
         "--steps", str(steps),
         "--log-activations", "--act-save-raw"],
        cwd=PROJECT_ROOT,
        env={**os.environ, "MKL_THREADING_LAYER": "GNU"},
        check=True,
    )
    return out_root / config["run"]["name"]


def _reconstruction_comparison(reps: torch.Tensor, seed: int) -> Dict:
    """SAE vs k-matched PCA on one run's representations (stage-11 protocol).

    Split: 80/20 by sample (the operator boundary used the same
    within-representation split); k = topk = 8 (the PINN SAE config).
    """
    from experiments.operator_boundary import (k_matched_pca_reconstruction,
                                                train_topk_sae)

    n = reps.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_train = int(0.8 * n)
    train_reps = reps[perm[:n_train]]
    test_reps = reps[perm[n_train:]]

    k = 8
    sae_out = train_topk_sae(train_reps, test_reps, expansion=4, topk=k,
                             steps=1500, seed=seed)
    pca_mse = k_matched_pca_reconstruction(train_reps, test_reps, k)
    sae_mse = sae_out["test_recon_mse"]
    return {
        "sae_test_recon_mse": sae_mse,
        "pca_k_matched_test_recon_mse": pca_mse,
        "pca_over_sae_ratio": float(pca_mse / max(sae_mse, 1e-12)),
        "sae_beats_pca": bool(sae_mse < pca_mse),
        "dead_feature_fraction": sae_out["dead_feature_fraction"],
        "mean_l0": sae_out["mean_l0"],
    }


def _analyze_run(run_dir: Path, label: Dict) -> Dict:
    stats = analyze_run_activations(run_dir, "layers.1")
    stats["local_tangent_rank"] = _tangent_rank_for_run(run_dir, "layers.1")
    stats["pr_over_width"] = (stats["participation_ratio"] / WIDTH
                              if stats.get("participation_ratio") else None)
    stats.update(label)
    return stats


def run_architecture_boundary(seeds: List[int] = None,
                              steps: int = STEPS,
                              reconstruction: bool = True) -> Dict:
    print("\n=======================================================")
    print("STAGE 18: Architecture Boundary Probe (H18)")
    print("=======================================================")

    if seeds is None:
        seeds = list(SEEDS)

    out_root = RUNS / "architecture_boundary"
    out_root.mkdir(parents=True, exist_ok=True)

    variants = []
    # (a) Fourier-feature PINN, width 64, depth 3 (the width-scaling
    #     control shape), 3 seeds.
    for seed in seeds:
        variants.append({
            "name": f"fourier_w64_seed{seed}",
            "hidden_layers": [WIDTH] * 3,
            "fourier": True, "seed": seed,
            "arm": "fourier", "depth": 3,
        })
    # (b) Depth sweep 2 -> 6 hidden layers, width 64, 3 seeds each.
    for depth in [2, 3, 4, 5, 6]:
        for seed in seeds:
            variants.append({
                "name": f"depth{depth}_w64_seed{seed}",
                "hidden_layers": [WIDTH] * depth,
                "fourier": False, "seed": seed,
                "arm": "depth", "depth": depth,
            })

    results = []
    by_arm: Dict[str, List[Dict]] = {}
    for v in variants:
        print(f"\n--- {v['name']} ---")
        config = build_h18_config(v["name"], v["hidden_layers"], v["seed"],
                                  v["fourier"], out_root)
        run_dir = _train_one(config, steps)
        label = {"run": v["name"], "arm": v["arm"], "depth": v["depth"],
                 "fourier_embed": v["fourier"], "seed": v["seed"]}
        stats = _analyze_run(run_dir, label)
        pr = stats.get("participation_ratio")
        pr_moved = bool(pr is not None and pr > PR_MOVE_THRESHOLD)
        stats["pr_moved"] = pr_moved

        if reconstruction and pr_moved:
            print(f"  PR {pr:.2f} > {PR_MOVE_THRESHOLD} -> running "
                  f"SAE-vs-PCA reconstruction comparison")
            reps = _collect_representations(run_dir)
            stats["reconstruction"] = _reconstruction_comparison(reps,
                                                                 v["seed"])
        else:
            print(f"  PR {pr if pr is not None else float('nan'):.2f} "
                  f"({'moved' if pr_moved else 'within width-scaling envelope'})")

        results.append(stats)
        by_arm.setdefault(v["arm"], []).append(stats)

    # ------------------------------------------------------------------
    # Aggregation per arm
    # ------------------------------------------------------------------
    agg = {}
    for arm, runs in by_arm.items():
        prs = [r["participation_ratio"] for r in runs]
        entry = {
            "n_runs": len(runs),
            "mean_pr": float(np.mean(prs)),
            "min_pr": float(np.min(prs)),
            "max_pr": float(np.max(prs)),
            "mean_pr_over_width": float(np.mean(prs) / WIDTH),
            "mean_tangent_rank": float(np.mean(
                [r["local_tangent_rank"] for r in runs])),
            "mean_pca95": float(np.mean(
                [r["n_components_95pct"] for r in runs])),
        }
        if arm == "depth":
            by_depth = {}
            for r in runs:
                by_depth.setdefault(r["depth"], []).append(
                    r["participation_ratio"])
            entry["mean_pr_by_depth"] = {
                str(d): float(np.mean(v)) for d, v in sorted(by_depth.items())}
        if any("reconstruction" in r for r in runs):
            recon = [r["reconstruction"] for r in runs
                     if "reconstruction" in r]
            entry["reconstruction"] = {
                "n_runs_with_comparison": len(recon),
                "mean_pca_over_sae_ratio": float(np.mean(
                    [x["pca_over_sae_ratio"] for x in recon])),
                "sae_beats_pca_runs": int(sum(
                    x["sae_beats_pca"] for x in recon)),
            }
        agg[arm] = entry

    # ------------------------------------------------------------------
    # Pre-written outcome reading
    # ------------------------------------------------------------------
    fourier = agg.get("fourier", {})
    fourier_pr_moved = bool(
        fourier.get("max_pr", 0.0) > PR_MOVE_THRESHOLD)
    if fourier_pr_moved:
        outcome = "H18a"
        outcome_note = (
            "Fourier-feature PINN activation PR rises above the "
            "width-scaling envelope toward the superposition regime; "
            "reconstruction comparisons run wherever PR moved. The regime "
            "boundary gains a within-PINN demonstration with the rank "
            "diagnostic as predictor."
        )
    else:
        outcome = "H18b"
        outcome_note = (
            "PR stays within the width-scaling envelope (max "
            f"{fourier.get('max_pr', float('nan')):.2f} vs envelope 2.51) "
            "under the geometry-changing Fourier embedding AND the depth "
            "sweep; the null is robust to within-PINN architectural "
            "perturbation. Scope is drawn at operator/function-space "
            "representations, as pre-written."
        )

    report = {
        "stage": "architecture_boundary",
        "hypotheses": "H18a/H18b (docs/preregistration.md §H18; v3.5)",
        "preregistration": "committed before this run (v3.5 section)",
        "design": {"width": WIDTH, "seeds": seeds, "steps": steps,
                   "fourier_n_freq": 32,
                   "depth_sweep": [2, 3, 4, 5, 6],
                   "pr_move_threshold": PR_MOVE_THRESHOLD,
                   "width_scaling_envelope_pr": [1.14, 2.51],
                   "reconstruction_protocol": "SAE(topk=8, exp=4) vs "
                                              "k-matched PCA(k=8), 80/20 "
                                              "sample split, run wherever "
                                              "PR moved"},
        "per_run": results,
        "aggregate": agg,
        "outcome": outcome,
        "outcome_note": outcome_note,
    }

    out_path = out_root / "architecture_boundary_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nOutcome: {outcome}")
    print(f"Report: {out_path}")
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Stage 18: architecture boundary probe (H18, preregistered)")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seeds", type=str, default=None,
                    help="comma list (default 7,42,123)")
    ap.add_argument("--smoke", action="store_true",
                    help="1 seed, depth arm only, no reconstruction")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    if args.smoke:
        run_architecture_boundary(seeds=[7], steps=200,
                                  reconstruction=False)
    else:
        run_architecture_boundary(seeds=seeds, steps=args.steps)
