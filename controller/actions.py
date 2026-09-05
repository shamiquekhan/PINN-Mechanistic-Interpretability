"""
Controller action implementations: loss reweighting, GradNorm-style balancing,
adaptive resampling, and Fourier feature injection.
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Loss-weight actions
# ---------------------------------------------------------------------------

def increase_lambda_bc(
    lambda_bc: float,
    factor: float = 2.0,
    max_val: float = 100.0,
) -> float:
    """Double BC weight, capped at max_val."""
    return min(lambda_bc * factor, max_val)


def gradnorm_rebalance(
    lambda_pde: float,
    lambda_bc: float,
    grad_norm_pde: Optional[float] = None,
    grad_norm_bc: Optional[float] = None,
    max_val: float = 100.0,
) -> Tuple[float, float]:
    """GradNorm-style rebalancing.

    If gradient norms are provided, weight each loss inversely proportional
    to its gradient norm so all components contribute equally to the total
    gradient magnitude.  Otherwise, fall back to equal weight.
    """
    if grad_norm_pde is not None and grad_norm_bc is not None and grad_norm_pde > 0 and grad_norm_bc > 0:
        total_norm = grad_norm_pde + grad_norm_bc
        new_pde = min(total_norm / grad_norm_pde, max_val)
        new_bc  = min(total_norm / grad_norm_bc, max_val)
    else:
        total = lambda_pde + lambda_bc
        new_pde = new_bc = min(total / 2.0, max_val)
    return new_pde, new_bc


# ---------------------------------------------------------------------------
# Adaptive resampling
# ---------------------------------------------------------------------------

def resample_high_residual(
    current_points: torch.Tensor,
    residuals: torch.Tensor,
    n_new: int,
    domain: Tuple[float, float],
    device: torch.device,
    dtype: torch.dtype,
    top_frac: float = 0.2,
) -> torch.Tensor:
    """Replace the top-residual fraction of collocation points with new samples
    drawn preferentially near high-residual regions.

    Parameters
    ----------
    current_points: (N, 1) current collocation points.
    residuals:      (N, 1) PDE residual at each point.
    n_new:          Number of points to add/replace.
    domain:         (lo, hi) domain bounds.
    top_frac:       Fraction of high-residual points to replace.
    """
    lo, hi = domain
    res_flat = residuals.abs().flatten()
    k = max(1, int(len(res_flat) * top_frac))
    top_k_idx = torch.topk(res_flat, k).indices

    # New random points (uniform for now; can be biased by residual magnitude)
    new_pts = torch.rand(n_new, 1, device=device, dtype=dtype) * (hi - lo) + lo

    # Replace top-residual points with new ones (cycle if n_new > k)
    replace_pts = current_points.clone()
    for i, new_pt in enumerate(new_pts):
        replace_pts[top_k_idx[i % len(top_k_idx)]] = new_pt

    return replace_pts


# ---------------------------------------------------------------------------
# Fourier feature injection (scheduled architecture change)
# ---------------------------------------------------------------------------

def inject_fourier_features(
    model: nn.Module,
    n_freq: int = 16,
    scale: float = 1.0,
    seed: int = 0,
) -> nn.Module:
    """Prepend a Fourier embedding to the existing model.

    This wraps the model in a Sequential that first applies the Fourier
    embedding, then passes through the original model.

    Note: this changes the model's input processing.  The caller is responsible
    for updating the optimizer to include the new parameters.

    Returns the wrapped model.
    """
    from pinn.model import FourierEmbedding

    input_dim = 1   # 1D PDE
    embed = FourierEmbedding(input_dim, n_freq, scale, seed).to(
        next(model.parameters()).device,
        dtype=next(model.parameters()).dtype,
    )

    wrapped = nn.Sequential(embed, model)
    return wrapped
