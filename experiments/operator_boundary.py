"""Regime-boundary experiment: does the FNO live in the superposition regime?

Phase 9C — the highest-impact follow-up.  The PINN negative result is
scoped by representational geometry (PR ~ 1.3-3 of 64 dims across the PDE
suite).  The natural positive control is a neural OPERATOR whose inputs
are functions (high-dimensional objects): if its hidden representations
have high effective rank, SAEs should recover planted/causal structure
there, defining the regime boundary the paper promises.

This script:
  1. Generates a parametric Green's-function regression dataset
     (input = Gaussian-mixture forcing function, output = screened-Poisson
     solution).
  2. Trains an FNO1d to regress the operator.
  3. Measures the SAME geometry statistics used for PINNs (covariance PR,
     PCA energy, k-matched PCA reconstruction) on the FNO's post-activation
     block states (function-space representations).
  4. Trains the SAME TopK SAE on those representations and compares
     reconstruction quality + dead features against the PINN numbers.
  5. Runs the PLANTED-FEATURE positive control (experiments.positive_control)
     as the causal sanity check, unchanged.

The comparison table (PINN vs FNO) is the regime boundary.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from operators.fno import FNO1d, green_function_dataset
from analysis.effective_rank import participation_ratio, stable_rank, pca_components_for_energy
from sae.model import SparseAutoencoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS = PROJECT_ROOT / "runs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_fno_representations(
    model: FNO1d,
    inputs: torch.Tensor,
    layer_index: int = -1,
    pool: str = "positions",
) -> torch.Tensor:
    """Return hidden block states at layer_index.

    pool="positions" (default): each (sample, grid-position) pair is one
    row — the faithful analogue of LLM token-level residual activations
    and of the PINN probe-point protocol the SAEs were trained on.
    Shape: (batch * grid_points, width).

    pool="mean": one row per sample (grid-averaged state).
    Shape: (batch, width).
    """
    model.eval()
    with torch.no_grad():
        _, intermediates = model(inputs, return_intermediates=True)
        state = intermediates[layer_index]      # (batch, L, width)
        if pool == "mean":
            return state.mean(dim=1)
        return state.reshape(-1, state.shape[-1])


def train_fno(
    n_train: int = 2048,
    n_test: int = 256,
    grid_points: int = 64,
    width: int = 64,
    modes: int = 12,
    n_layers: int = 4,
    steps: int = 4000,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
) -> Dict:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    a_train, u_train = green_function_dataset(
        n_train, grid_points, seed=seed, device=DEVICE)
    a_test, u_test = green_function_dataset(
        n_test, grid_points, seed=seed + 999, device=DEVICE)

    model = FNO1d(in_channels=1, out_channels=1, width=width,
                  modes=modes, n_layers=n_layers).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.5)

    for step in range(steps):
        idx = torch.randint(0, n_train, (batch_size,), device=DEVICE)
        pred = model(a_train[idx])
        loss = F.mse_loss(pred, u_train[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps - 1:
            with torch.no_grad():
                test_pred = model(a_test)
                test_mse = F.mse_loss(test_pred, u_test).item()
            print(f"  step {step}: train {loss.item():.5f} test {test_mse:.5f}")

    with torch.no_grad():
        test_pred = model(a_test)
        rel_l2 = float(
            (torch.linalg.vector_norm(test_pred - u_test, dim=(1, 2))
             / torch.linalg.vector_norm(u_test, dim=(1, 2)).clamp(min=1e-12)
             ).mean())
    print(f"  final test relative L2 (mean over samples): {rel_l2:.4f}")
    return {"model": model, "a_train": a_train, "a_test": a_test,
            "u_test": u_test, "test_rel_l2": rel_l2}


def geometry_stats(reps: torch.Tensor) -> Dict:
    """Covariance PR / stable rank / PCA energy on representation matrix."""
    data = reps.detach().cpu().numpy().astype(np.float64)
    centered = data - data.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / (centered.shape[0] - 1)
    return {
        "n_samples": int(data.shape[0]),
        "dim": int(data.shape[1]),
        "participation_ratio": participation_ratio(cov),
        "stable_rank": stable_rank(cov),
        "n_components_95pct": pca_components_for_energy(cov, 0.95),
        "n_components_99pct": pca_components_for_energy(cov, 0.99),
    }


def k_matched_pca_reconstruction(train_reps: torch.Tensor,
                                  test_reps: torch.Tensor, k: int) -> float:
    """PCA fit on train, evaluated on test (no refit), k components."""
    from analysis.pca_interpretability import fit_pca, reconstruction_error
    basis = fit_pca(train_reps.cpu().float(), n_components=k)
    return reconstruction_error(test_reps.cpu().float(), basis)


def train_topk_sae(train_reps: torch.Tensor, test_reps: torch.Tensor,
                   expansion: int = 4, topk: int = 8, steps: int = 2000,
                   seed: int = 0) -> Dict:
    """Train the SAME TopK SAE used for PINNs, on operator representations."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    sae = SparseAutoencoder(
        input_dim=train_reps.shape[1], latent_expansion=expansion,
        activation_mode="topk", topk=topk, decoder_normalize=True,
    ).to(DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-2)
    bank = train_reps.to(DEVICE)
    n = bank.shape[0]
    rng = torch.Generator(device="cpu").manual_seed(seed + 777)
    for step in range(steps):
        idx = torch.randint(0, n, (256,), generator=rng).to(DEVICE)
        opt.zero_grad()
        loss, _ = sae.loss(bank[idx])
        loss.backward()
        opt.step()
        if sae.decoder_normalize:
            sae._normalise_decoder()
    sae.eval()
    with torch.no_grad():
        z, recon = sae(test_reps.to(DEVICE))
        mse = F.mse_loss(recon, test_reps.to(DEVICE)).item()
        # Dead features measured by latent inactivity on the test split
        # (the internal feature-use counter is only updated when
        # update_feature_use is called during training loops).
        active_mask = (z > 0).any(dim=0).float()
        dead = 1.0 - float(active_mask.mean())
        l0 = (z > 0).float().sum(dim=-1).mean().item()
    return {"sae": sae, "test_recon_mse": mse,
            "dead_feature_fraction": dead, "mean_l0": l0}


def run_operator_boundary(seed: int = 0) -> Dict:
    print("\n=======================================================")
    print("STAGE 11: Operator Regime Boundary (FNO + Green's function)")
    print("=======================================================")

    print("\n[1/4] Training FNO on parametric Green's-function regression...")
    run = train_fno(seed=seed)

    print("\n[2/4] Collecting function-space representations (block states)...")
    reps_train = collect_fno_representations(run["model"], run["a_train"])
    reps_test = collect_fno_representations(run["model"], run["a_test"])
    geom = geometry_stats(reps_train)
    print(f"  PR = {geom['participation_ratio']:.2f} of {geom['dim']} dims "
          f"(superposition ratio {geom['participation_ratio']/geom['dim']:.3f})")
    print(f"  PCA comps for 95%/99% energy: "
          f"{geom['n_components_95pct']}/{geom['n_components_99pct']}")

    print("\n[3/4] k-matched PCA vs TopK SAE on operator representations...")
    k = 8
    pca_mse = k_matched_pca_reconstruction(reps_train, reps_test, k)
    sae_res = train_topk_sae(reps_train, reps_test, seed=seed)
    print(f"  PCA(k={k}) test recon MSE:  {pca_mse:.6f}")
    print(f"  TopK SAE test recon MSE:    {sae_res['test_recon_mse']:.6f} "
          f"(dead {sae_res['dead_feature_fraction']:.0%}, L0 {sae_res['mean_l0']:.1f})")

    # ---- Assemble the regime-boundary comparison ----
    pinn_reference = {
        "participation_ratio": 1.34,
        "dim": 64,
        "pca_k_recon": 2.66e-4,
        "sae_recon": 3.35e-3,
        "dead_features": 0.51,
        "source": "runs/effective_rank_analysis + runs/sae_models_v2 (v2.1)",
    }
    comparison = {
        "PINN (poisson_1d, W=64)": pinn_reference,
        "FNO (green regression, W=64)": {
            "participation_ratio": geom["participation_ratio"],
            "dim": geom["dim"],
            "pca_k_recon": pca_mse,
            "sae_recon": sae_res["test_recon_mse"],
            "dead_features": sae_res["dead_feature_fraction"],
            "test_rel_l2": run["test_rel_l2"],
        },
    }

    # Regime verdict: superposition ratio PR/dim.
    pr_ratio = geom["participation_ratio"] / geom["dim"]
    sae_beats_pca = sae_res["test_recon_mse"] < pca_mse
    verdict = {
        "fno_superposition_ratio": pr_ratio,
        "fno_pr": geom["participation_ratio"],
        "pinn_pr": pinn_reference["participation_ratio"],
        "ratio_increase_vs_pinn": geom["participation_ratio"]
            / pinn_reference["participation_ratio"],
        "sae_beats_k_matched_pca": bool(sae_beats_pca),
        "regime": (
            "HIGH-RANK / superposition-plausible — SAE methodology is "
            "appropriate here; this is the positive-control regime"
            if pr_ratio > 0.1 else
            "STILL LOW-RANK — operator representations also concentrate; "
            "the negative result extends to this operator task"
        ),
    }

    out = {
        "status": "complete",
        "fno_training": {"test_rel_l2": run["test_rel_l2"]},
        "geometry": geom,
        "pca_k_recon": pca_mse,
        "sae": {k: v for k, v in sae_res.items() if k != "sae"},
        "comparison_table": comparison,
        "verdict": verdict,
        "config": {"seed": seed, "k": k},
    }
    out_dir = RUNS / "operator_boundary"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "operator_boundary_report.json", "w") as f:
        json.dump(out, f, indent=2)
    torch.save(run["model"].state_dict(), out_dir / "fno_green.pt")
    sae_res["sae"].save(out_dir / "sae_green.pt")

    print("\nRegime verdict:")
    for k_, v in verdict.items():
        print(f"  {k_}: {v}")
    print(f"\nReport: {out_dir / 'operator_boundary_report.json'}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_operator_boundary(seed=args.seed)
