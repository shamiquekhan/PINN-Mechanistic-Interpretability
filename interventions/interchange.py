"""Partial interchange interventions for causal-abstraction testing (Phase 11).

Theory (Geiger et al. 2021, 2022)
--------------------------------
A high-level causal model H is an *abstraction* of a neural network N with
alignment map phi if interventions on H variables correspond, through phi,
to interventions on N's internal states.  The canonical test is the
INTERCHANGE INTERVENTION: run input A, record its state; run input B, record
its state; replace (part of) A's state with B's and check whether the
network output matches the H-level prediction built from the interchange.

This module implements REGION-LEVEL partial interchange interventions at a
chosen hidden layer.  The high-level variable under test is the *spatial
region identity* of a collocation batch — the physically meaningful
variable for boundary starvation:

  H_boundary: { region in {interior, boundary_proximal} } -> PDE loss

The interchange prediction: replacing the aligned subspace of an interior
batch's activations with a boundary-proximal batch's coefficients should
move the PDE loss toward the boundary-proximal batch's loss — IF the
aligned subspace encodes region identity.  If not (random-baseline
movement), the alignment is not a causal abstraction of H_boundary.

The "part" being swapped is defined by a linear alignment basis (PCA
components, SAE decoder directions, or a random orthonormal basis), with
the donor's REGION-MEAN coefficients swapped into every source row.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


def capture_activation(model: torch.nn.Module, x: torch.Tensor,
                       layer_index: int) -> torch.Tensor:
    """Forward pass returning the hidden activation at layer_index."""
    captured: Dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["a"] = output.detach()

    h = model.acts[layer_index].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(x)
    finally:
        h.remove()
    return captured["a"]


def partial_interchange(
    source_act: torch.Tensor,
    donor_act: torch.Tensor,
    basis: torch.Tensor,          # (k, width), rows are basis directions
    mean: torch.Tensor,           # (width,)
    n_swapped: int,
) -> torch.Tensor:
    """Replace the leading n_swapped coefficients of source with donor's.

    The donor_act is expected to be a REGION MEAN (1, width).  Every source
    row receives the donor's leading coefficients; the rest keep the
    source's values.  Returns the decoded, interchanged activations.
    """
    cs = (source_act - mean) @ basis.T
    cd = (donor_act - mean) @ basis.T
    cs = torch.cat([cd.expand(cs.shape[0], -1)[:, :n_swapped],
                    cs[:, n_swapped:]], dim=1)
    return cs @ basis + mean


def make_interchange_hook(
    donor_act: torch.Tensor,
    basis: torch.Tensor,
    mean: torch.Tensor,
    n_swapped: int,
):
    """Forward hook performing the partial interchange on the incoming
    (graph-attached) activation output.  donor_act is a region mean (1, width)."""
    cd = ((donor_act - mean) @ basis.T)[:, :n_swapped]

    def hook(_module, _inputs, output):
        cs = (output - mean) @ basis.T
        cs = torch.cat([cd.expand(cs.shape[0], -1), cs[:, n_swapped:]], dim=1)
        return cs @ basis + mean

    return hook


def _losses_with_hook(model, pde, x, hook_fn, layer_index: int):
    """Compute losses with an arbitrary forward hook active at layer_index."""
    h = model.acts[layer_index].register_forward_hook(hook_fn)
    try:
        r = pde.residual(model, x)
        lp = float((r ** 2).mean())
        lb = float(pde.boundary_residual(model).mean())
    finally:
        h.remove()
    return lp, lb


def region_split(pde, n_total: int, boundary_band: float,
                 device, dtype, seed: int = 0) -> Dict[str, torch.Tensor]:
    """Split collocation points into interior and boundary-proximal regions.

    boundary_band: fraction of the domain width on each edge counted as
    'boundary-proximal' (default 15%).
    """
    lo, hi = pde.domain
    x = pde.sample_interior(n_total, device, dtype, seed=seed)
    width = hi - lo
    dist = torch.minimum(x[:, 0] - lo, hi - x[:, 0])
    mask_b = dist < boundary_band * width
    return {
        "boundary_proximal": x[mask_b],
        "interior": x[~mask_b],
    }


def interchange_alignment_accuracy(
    model: torch.nn.Module,
    pde,
    source_x: torch.Tensor,
    donor_x: torch.Tensor,
    basis: torch.Tensor,
    mean: torch.Tensor,
    layer_index: int = 1,
    n_swapped: int = 1,
) -> Dict:
    """Region-level partial interchange with the H_boundary alignment test.

    Returns the source/donor/swapped losses and the ALIGNMENT METRIC:
    movement = (log L_swapped - log L_source) / (log L_donor - log L_source).
    movement ~ 1  -> the aligned subspace fully carries region identity
    movement ~ 0  -> the subspace carries none of it
    movement <= 0 -> the subspace is anti-aligned / noise
    """
    model.eval()
    with torch.no_grad():
        a_donor_mean = capture_activation(model, donor_x, layer_index).mean(
            dim=0, keepdim=True)

    # Baseline losses (autograd graph required by pde.residual).
    r_s = pde.residual(model, source_x)
    lp_s = float((r_s ** 2).mean())
    r_d = pde.residual(model, donor_x)
    lp_d = float((r_d ** 2).mean())

    # Losses under interchange: graph-preserving hook.
    lp_i, lb_i = _losses_with_hook(
        model, pde, source_x,
        make_interchange_hook(a_donor_mean, basis, mean, n_swapped),
        layer_index,
    )

    eps = 1e-12
    denom = np.log(lp_d + eps) - np.log(lp_s + eps)
    numer = np.log(lp_i + eps) - np.log(lp_s + eps)
    movement = float(numer / denom) if abs(denom) > 1e-9 else float("nan")

    return {
        "source_loss_pde": lp_s,
        "donor_loss_pde": lp_d,
        "swapped_loss_pde": lp_i,
        "swapped_loss_bc": lb_i,
        "log_change_pde": float(numer),
        "log_target_pde": float(denom),
        "movement_fraction": movement,
        "agree_sign": bool(np.sign(numer) == np.sign(denom)),
    }


def run_alignment_battery(
    model: torch.nn.Module,
    pde,
    basis: torch.Tensor,
    mean: torch.Tensor,
    layer_index: int = 1,
    n_swapped: int = 1,
    n_trials: int = 16,
    n_batch: int = 128,
    boundary_band: float = 0.15,
    seed: int = 0,
) -> Dict:
    """Average region-interchange alignment over resampled region draws.

    Each trial draws a fresh interior batch (source) and boundary-proximal
    batch (donor), performs the partial interchange, and measures how far
    the swapped loss moves toward the donor loss (movement fraction).
    """
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    records = []
    for t in range(n_trials):
        regions = region_split(pde, n_batch * 4, boundary_band,
                               device, dtype, seed=seed + t)
        if regions["interior"].shape[0] < 8 or \
                regions["boundary_proximal"].shape[0] < 8:
            continue
        idx = torch.randperm(regions["interior"].shape[0])[:n_batch]
        source_x = regions["interior"][idx]
        donor_x = regions["boundary_proximal"]
        rec = interchange_alignment_accuracy(
            model, pde, source_x, donor_x,
            basis, mean, layer_index=layer_index, n_swapped=n_swapped,
        )
        records.append(rec)
    movements = [r["movement_fraction"] for r in records
                 if np.isfinite(r["movement_fraction"])]
    agrees = [r["agree_sign"] for r in records]
    return {
        "n_trials": len(records),
        "mean_movement_fraction": float(np.mean(movements))
        if movements else float("nan"),
        "frac_movement_positive": float(
            np.mean([m > 0 for m in movements])) if movements else float("nan"),
        "alignment_accuracy_sign": float(np.mean(agrees)),
        "records": records,
    }


def random_basis(width: int, k: int, seed: int = 0) -> torch.Tensor:
    """Orthonormalized random basis (k rows, width cols) — chance control."""
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(k, width, generator=g)
    q, _ = torch.linalg.qr(raw.T)
    return q.T[:k]
