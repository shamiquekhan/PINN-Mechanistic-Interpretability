"""
Effective-rank / participation-ratio analysis of PINN hidden activations
(v2.1 hardening, point 3).

Motivation
----------
The TopK SAE lost to k-matched PCA by ~12x on reconstruction, and only 1/256
features were stable across SAE seeds.  One specific hypothesis explains both:
the PINN's hidden activations may not be in superposition at all.  SAEs are
designed for the regime where a network encodes MORE features than it has
neurons (superposition).  A 64-unit tanh MLP trained on a single 1D PDE may
simply never need to pack more concepts than dimensions — its activations
would then live on a genuinely low-rank, smooth manifold, where PCA is the
CORRECT tool and an overcomplete sparse dictionary is solving a nonexistent
problem.

This module measures that directly:

  - Participation ratio (PR) of the activation covariance:
        PR = (sum lambda_i)^2 / sum lambda_i^2
    PR ~ effective number of dimensions with nonneglible variance.

  - Stable rank (SR): ||C||_F^2 / ||C||_2^2

  - PCA cumulative-energy curve: how many components reach 95%/99%/99.9%
    of total variance.

If PR is small (close to the PCA-explained rank) across BOTH success and
failure runs, Gate 3's failure is *explained*: the activations are
low-rank, not superposed, and the SAE null follows mechanically.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from pinn_logging.io import load_jsonl


def participation_ratio(cov: np.ndarray) -> float:
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0, None)
    total = eig.sum()
    if total <= 0:
        return 0.0
    return float(total ** 2 / (eig ** 2).sum())


def stable_rank(cov: np.ndarray) -> float:
    spectral_norm = np.linalg.norm(cov, 2)
    if spectral_norm <= 0:
        return 0.0
    return float(np.linalg.norm(cov, "fro") ** 2 / spectral_norm ** 2)


def pca_components_for_energy(cov: np.ndarray, target: float) -> int:
    eig = np.sort(np.clip(np.linalg.eigvalsh(cov), 0, None))[::-1]
    total = eig.sum()
    if total <= 0:
        return 0
    cum = np.cumsum(eig) / total
    return int(np.searchsorted(cum, target) + 1)


def analyze_run_activations(run_dir: Path, layer_name: str = "layers.1") -> Dict:
    """Compute effective-rank statistics for one run's activation logs."""
    recs = load_jsonl(Path(run_dir) / "activations.jsonl")
    vecs: List[np.ndarray] = []
    for rec in recs:
        layer_data = rec.get("activations", {}).get(layer_name, {})
        raw = layer_data.get("raw")
        if raw is not None:
            vecs.append(np.asarray(raw, dtype=np.float64))
    if not vecs:
        return {}
    data = np.concatenate(vecs, axis=0)  # (N, d)
    centered = data - data.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / (centered.shape[0] - 1)

    return {
        "n_samples": int(data.shape[0]),
        "dim": int(data.shape[1]),
        "participation_ratio": participation_ratio(cov),
        "stable_rank": stable_rank(cov),
        "n_components_95pct": pca_components_for_energy(cov, 0.95),
        "n_components_99pct": pca_components_for_energy(cov, 0.99),
        "n_components_999pct": pca_components_for_energy(cov, 0.999),
    }


def run_effective_rank_analysis(runs_root: Path, layer_name: str = "layers.1",
                                width: int = 64) -> Dict:
    """Effective-rank analysis across all logged runs (success + failures)."""
    runs_root = Path(runs_root)
    per_run: Dict[str, Dict] = {}
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir() or not (d / "activations.jsonl").exists():
            continue
        stats = analyze_run_activations(d, layer_name)
        if stats and stats.get("dim") == width:
            per_run[d.name] = stats

    if not per_run:
        return {"error": "no logged runs found"}

    prs = [s["participation_ratio"] for s in per_run.values()]
    ranks95 = [s["n_components_95pct"] for s in per_run.values()]
    dims = [s["dim"] for s in per_run.values()]

    superposition_evidence = {
        "n_runs": len(per_run),
        "mean_participation_ratio": float(np.mean(prs)),
        "min_participation_ratio": float(np.min(prs)),
        "max_participation_ratio": float(np.max(prs)),
        "mean_components_95pct": float(np.mean(ranks95)),
        "dim": int(dims[0]),
        "superposition_ratio_mean_to_dim": float(np.mean(prs) / dims[0]),
    }
    verdict = (
        "LOW-RANK / NOT IN SUPERPOSITION — the activation manifold has far "
        "fewer effective dimensions than neurons; PCA is the appropriate "
        "tool and the SAE null result is mechanically explained"
        if superposition_evidence["superposition_ratio_mean_to_dim"] < 0.5
        else "potentially superposed — SAE failure is NOT explained by rank "
             "alone and needs further investigation"
    )

    report = {
        "per_run": per_run,
        "summary": superposition_evidence,
        "verdict": verdict,
    }

    out_dir = runs_root / "effective_rank_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "effective_rank_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--layer", default="layers.1")
    args = ap.parse_args()
    rep = run_effective_rank_analysis(args.runs, args.layer)
    print(json.dumps(rep["summary"], indent=2))
    print("\nVERDICT:", rep["verdict"])