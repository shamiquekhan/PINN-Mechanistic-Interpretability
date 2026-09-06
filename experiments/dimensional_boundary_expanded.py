"""Dimensional boundary experiments for the expanded PDE suite (Phase 9A/9B).

Trains small PINNs on each new PDE family with activation logging, then
measures the same geometry statistics as the width study and the 2D pilot:

  - covariance participation ratio (PR)
  - local tangent rank (Jacobian rank of the input->activation map)
  - PCA energy concentration

PDE families and expected input dimensions:
  poisson_1d            d=1  (reference, PR ~ 1.7-2.0, tangent rank 1)
  poisson_2d            d=2  (pilot: PR ~ 3.0, tangent rank 2)
  advection_diffusion_2d d=2 (new: does operator structure change geometry?)
  reaction_diffusion_2d d=2  (new)
  burgers_1d            d=2  (time+space input: does TIME raise effective dim?)
  allen_cahn_1d         d=2  (time+space input)

The scientifically interesting comparison: Burgers/Allen-Cahn have 2-D
INPUTS (t, x).  If the effective activation rank rises to the 2D level,
time acts as a second input dimension and the manifold is a surface; if it
stays near the 1D level, the solutions explored are effectively
1-parameter families.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from analysis.activation_manifold import local_tangent_rank
from analysis.effective_rank import analyze_run_activations
from experiments.run_pipeline import PROJECT_ROOT, RUNS
from pinn_logging.io import load_jsonl

import subprocess
import sys


def train_and_analyze(
    config_name: str,
    seeds: List[int],
    steps: int,
    output_root: Path,
    device: str = "cuda",
) -> List[Dict]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    template = yaml.safe_load(
        (PROJECT_ROOT / "configs" / config_name).read_text()
    )
    results = []
    for seed in seeds:
        config = json.loads(json.dumps(template))
        run_name = f"{template['run']['name']}_seed{seed}"
        config["run"].update({"name": run_name, "output_dir": str(output_root),
                              "seed": seed, "device": device})
        config["training"]["steps"] = steps
        config_path = output_root / f"{run_name}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        env = {**dict(__import__('os').environ), "MKL_THREADING_LAYER": "GNU"}
        subprocess.run(
            [sys.executable, "-m", "experiments.train",
             "--config", str(config_path), "--steps", str(steps),
             "--log-activations", "--act-save-raw"],
            cwd=PROJECT_ROOT, env=env, check=True,
            capture_output=True,
        )
        run_dir = output_root / run_name
        stats = analyze_run_activations(run_dir, "layers.1")
        records = load_jsonl(run_dir / "activations.jsonl")
        arrays = [
            np.asarray(r["activations"]["layers.1"]["raw"], dtype=np.float64)
            for r in records
            if r.get("activations", {}).get("layers.1", {}).get("raw") is not None
        ]
        if not arrays:
            results.append({"seed": seed, "error": "no activation logs"})
            continue
        # Reconstruct probe coordinates from the config's PDE family.
        pde_name = template["pde"]["name"]
        if pde_name in ("poisson_2d", "advection_diffusion_2d",
                        "reaction_diffusion_2d"):
            axis = np.linspace(0.0, 1.0, 20)
            gx, gy = np.meshgrid(axis, axis, indexing="ij")
            coords = np.stack([gx.ravel(), gy.ravel()], axis=1)
        elif pde_name in ("burgers_1d", "allen_cahn_1d"):
            t_axis = np.linspace(0.0, 1.0, 10)
            x_axis = np.linspace(-1.0, 1.0, 20)
            gt, gx = np.meshgrid(t_axis, x_axis, indexing="ij")
            coords = np.stack([gt.ravel(), gx.ravel()], axis=1)
        else:
            coords = np.linspace(-1.0, 1.0, 50)[:, None]
        spatial_snapshot = arrays[0]
        metrics = load_jsonl(run_dir / "metrics.jsonl")
        stats.update({
            "seed": seed,
            "final_relative_l2": metrics[-1]["relative_l2"] if metrics else None,
            "local_tangent_rank": local_tangent_rank(spatial_snapshot,
                                                     coordinates=coords),
        })
        results.append(stats)
        print(f"  seed {seed}: PR={stats.get('participation_ratio', float('nan')):.2f} "
              f"tangent_rank={stats.get('local_tangent_rank')} "
              f"final_l2={stats.get('final_relative_l2')}")
    return results


def run_dimensional_boundary_expanded(
    steps_2d: int = 1000,
    steps_time: int = 2000,
    seeds: List[int] = None,
) -> Dict:
    if seeds is None:
        seeds = [7, 42, 123]
    print("\n=======================================================")
    print("STAGE 10: Dimensional Boundary (2D suite + time-dependent)")
    print("=======================================================")

    families = [
        ("advection_diffusion_2d_baseline.yaml", "advection_diffusion_2d",
         steps_2d, "2D input (steady advection-diffusion)"),
        ("reaction_diffusion_2d_baseline.yaml", "reaction_diffusion_2d",
         steps_2d, "2D input (steady reaction-diffusion)"),
        ("burgers_1d_baseline.yaml", "burgers_1d",
         steps_time, "time-space input (viscous Burgers)"),
        ("allen_cahn_1d_baseline.yaml", "allen_cahn_1d",
         steps_time, "time-space input (Allen-Cahn)"),
    ]

    report: Dict = {"families": {}, "reference": {
        "poisson_1d": {"pr_mean": 1.87, "tangent_rank": 1,
                       "note": "width study W=16..512 (runs/width_scaling)"},
        "poisson_2d": {"pr_mean": 3.03, "tangent_rank": 2,
                       "note": "5-seed pilot (runs/dimensional_boundary_2d)"},
    }}

    for config_name, family, steps, description in families:
        print(f"\n--- {family}: {description} ---")
        out_root = RUNS / "dimensional_boundary_expanded" / family
        try:
            results = train_and_analyze(config_name, seeds, steps, out_root)
        except subprocess.CalledProcessError as e:
            print(f"  TRAINING FAILED for {family}: "
                  f"{e.stderr.decode()[:200] if e.stderr else 'unknown'}")
            report["families"][family] = {"error": "training failed"}
            continue
        prs = [r["participation_ratio"] for r in results
               if "participation_ratio" in r]
        tangents = sorted({r["local_tangent_rank"] for r in results
                           if "local_tangent_rank" in r})
        report["families"][family] = {
            "description": description,
            "seeds": seeds,
            "steps": steps,
            "results": results,
            "summary": {
                "mean_participation_ratio": float(np.mean(prs)) if prs else None,
                "pr_range": [float(np.min(prs)), float(np.max(prs))] if prs else None,
                "tangent_ranks": tangents,
                "mean_final_relative_l2": float(np.mean(
                    [r["final_relative_l2"] for r in results
                     if r.get("final_relative_l2") is not None])) if any(
                    r.get("final_relative_l2") is not None for r in results) else None,
            },
        }

    # Comparison table.
    print("\nDimensional boundary summary:")
    print(f"  {'family':28s} {'mean PR':>8s} {'tangent':>8s}")
    print(f"  {'poisson_1d (ref)':28s} {1.87:8.2f} {1:8d}")
    print(f"  {'poisson_2d (ref)':28s} {3.03:8.2f} {2:8d}")
    for family, fr in report["families"].items():
        if "summary" in fr and fr["summary"]["mean_participation_ratio"]:
            print(f"  {family:28s} "
                  f"{fr['summary']['mean_participation_ratio']:8.2f} "
                  f"{str(fr['summary']['tangent_ranks']):>8s}")

    out_path = RUNS / "dimensional_boundary_expanded" / \
        "dimensional_boundary_expanded_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {out_path}")
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 123])
    ap.add_argument("--steps-2d", type=int, default=1000)
    ap.add_argument("--steps-time", type=int, default=2000)
    args = ap.parse_args()
    run_dimensional_boundary_expanded(
        steps_2d=args.steps_2d, steps_time=args.steps_time, seeds=args.seeds)
