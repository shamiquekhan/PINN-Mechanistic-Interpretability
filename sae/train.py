"""
SAE training loop with hyperparameter sweeping and PCA baseline comparison.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from sae.model import SparseAutoencoder
from sae.dataset import build_dataloaders, make_run_splits, ActivationDataset
from pinn_logging.io import append_jsonl


# ---------------------------------------------------------------------------
# PCA reconstruction baseline
# ---------------------------------------------------------------------------

def pca_reconstruction_error(
    data: torch.Tensor,
    n_components: int,
) -> float:
    """Compute mean squared reconstruction error of a truncated PCA on `data`."""
    X = data - data.mean(dim=0, keepdim=True)
    # Use torch SVD (GPU-friendly)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    # Truncate to n_components
    k = min(n_components, S.shape[0])
    X_recon = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
    return float(((X - X_recon) ** 2).mean())


# ---------------------------------------------------------------------------
# Single SAE training run
# ---------------------------------------------------------------------------

def train_sae(
    sae: SparseAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    steps: int,
    learning_rate: float = 1e-3,
    device: torch.device = torch.device("cuda"),
    out_dir: Optional[Path] = None,
    dead_feature_window: int = 200,
    log_every: int = 100,
) -> Dict:
    """Train SAE and return metrics dict.

    Parameters
    ----------
    sae:          Initialised SparseAutoencoder (moved to device externally).
    train_loader: DataLoader yielding activation batches.
    val_loader:   DataLoader for held-out validation.
    steps:        Total gradient steps to run.
    """
    sae.to(device)
    sae.train()
    opt = optim.Adam(sae.parameters(), lr=learning_rate)

    step = 0
    train_iter = iter(train_loader)
    history: List[Dict] = []

    while step < steps:
        # ---- Get batch ----
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = batch.to(device)
        opt.zero_grad()
        total_loss, info = sae.loss(batch)
        total_loss.backward()
        opt.step()

        if sae.decoder_normalize:
            sae._normalise_decoder()

        # Track feature use
        with torch.no_grad():
            z, _ = sae(batch)
            sae.update_feature_use(z)

        # Reset dead-feature counter periodically
        if step > 0 and step % dead_feature_window == 0:
            sae.reset_feature_use()

        # ---- Logging ----
        if step % log_every == 0 or step == steps - 1:
            val_recon = _eval_recon(sae, val_loader, device)
            rec = {
                "step": step,
                **info,
                "val_recon": val_recon,
                "dead_feature_frac": sae.dead_feature_fraction(),
            }
            history.append(rec)
            if out_dir is not None:
                append_jsonl(Path(out_dir) / "sae_training.jsonl", rec)
            print(f"  SAE step {step:5d} | recon={info['recon_loss']:.4e} | "
                  f"sparsity={info['sparsity_loss']:.4e} | "
                  f"dead={sae.dead_feature_fraction():.2%}")

        step += 1

    sae.eval()
    return {"history": history, "final": history[-1] if history else {}}


def _eval_recon(sae: SparseAutoencoder, loader: DataLoader, device: torch.device) -> float:
    sae.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, a_hat = sae(batch)
            total += float(((batch - a_hat) ** 2).mean()) * len(batch)
            n += len(batch)
    sae.train()
    return total / n if n > 0 else float("nan")


# ---------------------------------------------------------------------------
# Hyperparameter sweep
# ---------------------------------------------------------------------------

def sweep_sae(
    run_dirs: List[Path],
    layer_name: str,
    expansion_values: List[int],
    sparsity_values: List[float],
    steps: int,
    device: torch.device,
    out_dir: Path,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> List[Dict]:
    """Grid sweep over (expansion, sparsity) — all runs on GPU.

    Returns
    -------
    List of result dicts, each containing config + final metrics.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dirs, val_dirs, test_dirs = make_run_splits(run_dirs, seed=seed)

    # Fit normalisation once on train set
    try:
        train_ds = ActivationDataset(train_dirs, layer_name, normalise=True)
    except ValueError as e:
        print(f"Warning: {e}")
        return []

    fit_stats = train_ds.get_fit_stats()
    input_dim = train_ds.data.shape[1]

    # PCA baseline
    pca_n_comp = input_dim // 2
    pca_err = pca_reconstruction_error(train_ds.data.to(device), pca_n_comp)
    print(f"PCA baseline ({pca_n_comp} components): MSE = {pca_err:.4e}")

    results: List[Dict] = []

    for expansion in expansion_values:
        for sparsity in sparsity_values:
            tag = f"exp{expansion}_sp{sparsity:.0e}"
            print(f"\n=== SAE sweep: {tag} ===")

            # DataLoaders (re-use fit stats)
            train_loader, val_loader, test_loader, _ = build_dataloaders(
                train_dirs, val_dirs, test_dirs, layer_name,
                batch_size=batch_size, device=device,
            )

            sae = SparseAutoencoder(
                input_dim=input_dim,
                latent_expansion=expansion,
                sparsity_coeff=sparsity,
                decoder_normalize=True,
            ).to(device)

            run_out = out_dir / tag
            run_out.mkdir(exist_ok=True)

            result = train_sae(
                sae, train_loader, val_loader,
                steps=steps,
                learning_rate=learning_rate,
                device=device,
                out_dir=run_out,
            )

            # Test set evaluation
            test_recon = _eval_recon(sae, test_loader, device)
            sae.save(run_out / "sae.pt")

            entry = {
                "expansion":     expansion,
                "sparsity_coeff": sparsity,
                "tag":           tag,
                "pca_baseline_mse": pca_err,
                "test_recon":    test_recon,
                "dead_frac":     sae.dead_feature_fraction(),
                **result.get("final", {}),
            }
            results.append(entry)
            print(f"  Test recon: {test_recon:.4e} | PCA: {pca_err:.4e} | "
                  f"Dead: {sae.dead_feature_fraction():.2%}")

    # Save sweep summary
    with open(out_dir / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
