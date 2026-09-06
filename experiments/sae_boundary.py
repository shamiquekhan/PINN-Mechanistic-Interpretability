"""Train a small SAE/PCA comparison on the completed 2D boundary pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from analysis.pca_interpretability import fit_pca, reconstruction_error
from sae.dataset import ActivationDataset, build_dataloaders, make_run_splits
from sae.model import SparseAutoencoder
from sae.train import _eval_recon, train_sae

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_boundary_sae(
    *,
    run_root: Path = PROJECT_ROOT / "runs" / "dimensional_boundary_2d",
    out_dir: Path = PROJECT_ROOT / "runs" / "dimensional_boundary_2d_sae",
    steps: int = 500,
    device: str = "cuda",
) -> dict:
    run_dirs = sorted(
        d for d in Path(run_root).iterdir()
        if d.is_dir() and (d / "activations.jsonl").exists()
    )
    if len(run_dirs) < 3:
        raise ValueError("at least three logged 2D runs are required")
    train_dirs, val_dirs, test_dirs = make_run_splits(run_dirs, seed=0)
    torch_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, _ = build_dataloaders(
        train_dirs, val_dirs, test_dirs, "layers.1", batch_size=256, device=torch_device
    )
    train_data = torch.cat([batch for batch in train_loader], dim=0).to(torch_device)
    test_data = torch.cat([batch for batch in test_loader], dim=0).to(torch_device)
    basis = fit_pca(train_data, n_components=4)
    pca_error = reconstruction_error(test_data, basis)
    sae = SparseAutoencoder(
        input_dim=32, latent_expansion=4, activation_mode="topk", topk=4
    ).to(torch_device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_result = train_sae(
        sae, train_loader, val_loader, steps=steps, learning_rate=1e-3,
        device=torch_device, out_dir=out_dir,
    )
    sae_error = _eval_recon(sae, test_loader, torch_device)
    sae.save(out_dir / "sae.pt")
    report = {
        "dimension": 2,
        "n_runs": len(run_dirs),
        "train_runs": [d.name for d in train_dirs],
        "test_runs": [d.name for d in test_dirs],
        "pca_components": 4,
        "pca_test_reconstruction_mse": pca_error,
        "sae_test_reconstruction_mse": sae_error,
        "sae_summary": sae.feature_summary(),
        "sae_training_final": train_result.get("final", {}),
        "interpretation": "Reconstruction comparison only; causal value requires a preregistered 2D intervention battery.",
    }
    (out_dir / "boundary_sae_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=PROJECT_ROOT / "runs" / "dimensional_boundary_2d")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "dimensional_boundary_2d_sae")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run_boundary_sae(run_root=args.runs, out_dir=args.out, steps=args.steps, device=args.device), indent=2))
