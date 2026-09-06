"""Run a small 2D Poisson effective-rank boundary matrix."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from analysis.activation_manifold import local_tangent_rank
from analysis.effective_rank import analyze_run_activations
from pinn_logging.io import load_jsonl

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_dimensional_boundary(
    seeds: list[int],
    *,
    steps: int = 1000,
    output_root: Path = PROJECT_ROOT / "runs" / "dimensional_boundary_2d",
    device: str = "cuda",
) -> dict:
    """Train and analyze one 2D Poisson PINN per seed."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    template = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "poisson_2d_boundary.yaml").read_text()
    )
    results = []

    for seed in seeds:
        config = json.loads(json.dumps(template))
        run_name = f"poisson_2d_seed{seed}"
        config["run"].update({"name": run_name, "output_dir": str(output_root), "seed": seed, "device": device})
        config["training"]["steps"] = steps
        config_path = output_root / f"{run_name}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        env = {**os.environ, "MKL_THREADING_LAYER": "GNU"}
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.train",
                "--config",
                str(config_path),
                "--steps",
                str(steps),
                "--log-activations",
                "--act-save-raw",
            ],
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
        )

        run_dir = output_root / run_name
        stats = analyze_run_activations(run_dir, "layers.1")
        records = load_jsonl(run_dir / "activations.jsonl")
        arrays = [
            np.asarray(record["activations"]["layers.1"]["raw"], dtype=np.float64)
            for record in records
            if record.get("activations", {}).get("layers.1", {}).get("raw") is not None
        ]
        activation_data = np.concatenate(arrays, axis=0)
        axis = np.linspace(0.0, 1.0, 20)
        grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
        probe_coordinates = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        # Tangent rank is a spatial-manifold diagnostic. Use one fixed probe
        # snapshot so repeated training-time snapshots do not duplicate
        # coordinates and corrupt local Jacobian estimates.
        spatial_snapshot = arrays[0]
        metrics = load_jsonl(run_dir / "metrics.jsonl")
        stats.update(
            {
                "seed": seed,
                "final_relative_l2": metrics[-1]["relative_l2"] if metrics else None,
                "local_tangent_rank": local_tangent_rank(spatial_snapshot, coordinates=probe_coordinates),
            }
        )
        results.append(stats)

    report = {
        "dimension": 2,
        "pde": "poisson_2d",
        "seeds": seeds,
        "steps": steps,
        "results": results,
        "summary": {
            "mean_participation_ratio": float(np.mean([r["participation_ratio"] for r in results])),
            "min_participation_ratio": float(np.min([r["participation_ratio"] for r in results])),
            "max_participation_ratio": float(np.max([r["participation_ratio"] for r in results])),
            "all_tangent_ranks": sorted({r["local_tangent_rank"] for r in results}),
            "mean_final_relative_l2": float(np.mean([r["final_relative_l2"] for r in results])),
        },
        "interpretation": (
            "This is a dimensional-boundary pilot. Compare 2D PR with the 1D "
            "width study; do not infer a covariance-rank theorem from input dimension."
        ),
    }
    (output_root / "dimensional_boundary_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 123, 2024, 2025])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "dimensional_boundary_2d")
    args = parser.parse_args()
    print(json.dumps(run_dimensional_boundary(args.seeds, steps=args.steps, output_root=args.out, device=args.device), indent=2))
