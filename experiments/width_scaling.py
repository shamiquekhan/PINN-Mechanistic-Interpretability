"""Measure activation geometry as 1D Poisson width changes.

This is a preregistered follow-up scaffold, not a claim that width alone
creates superposition. It records participation ratio, stable rank, PCA energy,
and local tangent rank for each width so the result can distinguish intrinsic
manifold dimension from covariance rank.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

from analysis.activation_manifold import local_tangent_rank
from analysis.effective_rank import analyze_run_activations
from pinn_logging.io import load_jsonl

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_config(width: int, output_root: Path, *, seed: int, device: str) -> dict:
    """Build a fixed 1D Poisson config for one hidden width."""
    return {
        "run": {
            "name": f"width_{width}_seed{seed}",
            "output_dir": str(output_root),
            "seed": seed,
            "deterministic": True,
            "device": device,
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
            "hidden_layers": [width, width, width],
            "activation": "tanh",
            "init": "xavier",
        },
        "training": {
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "steps": 1000,
            "interior_points": 128,
            "boundary_points": 2,
            "log_every": 100,
            "checkpoint_every": 1000,
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


def run_width_scaling(
    widths: Iterable[int],
    *,
    output_root: Path = PROJECT_ROOT / "runs" / "width_scaling",
    seed: int = 0,
    steps: int = 1000,
    device: str = "cuda",
) -> dict:
    """Train each width and write a compact geometry report."""
    widths = [int(width) for width in widths]
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for width in widths:
        if width < 1:
            raise ValueError("widths must be positive")
        config = build_config(width, output_root, seed=seed, device=device)
        config["training"]["steps"] = steps
        config_path = output_root / f"width_{width}_seed{seed}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
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
            env={**os.environ, "MKL_THREADING_LAYER": "GNU"},
            check=True,
        )
        run_dir = output_root / f"width_{width}_seed{seed}"
        stats = analyze_run_activations(run_dir, "layers.1")
        stats.update({"width": width, "local_tangent_rank": _tangent_rank_for_run(run_dir)})
        results.append(stats)

    report = {
        "input_dim": 1,
        "widths": widths,
        "seed": seed,
        "steps": steps,
        "results": results,
        "interpretation": (
            "Compare PR and local tangent rank empirically; input dimension does "
            "not impose a covariance-rank bound on curved activation manifolds."
        ),
    }
    (output_root / "width_scaling_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--widths", nargs="+", type=int, default=[16, 32, 64, 128, 256, 512])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "width_scaling")
    args = parser.parse_args()
    print(json.dumps(run_width_scaling(args.widths, output_root=args.out, seed=args.seed, steps=args.steps, device=args.device), indent=2))
