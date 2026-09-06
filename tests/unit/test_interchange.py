"""Tests for the partial-interchange causal-abstraction machinery."""
import numpy as np
import torch

from interventions.interchange import (
    capture_activation, partial_interchange, make_interchange_hook,
    interchange_alignment_accuracy, run_alignment_battery, random_basis,
    region_split, _losses_with_hook,
)
from pinn.model import MLP
from pinn.pdes import Poisson1D

DEVICE = torch.device("cpu")
DTYPE = torch.float32


def _setup():
    torch.manual_seed(0)
    pde = Poisson1D()
    model = MLP(1, 1, [16, 16], activation="tanh").to(device=DEVICE, dtype=DTYPE)
    return pde, model


def test_capture_activation_shape():
    pde, model = _setup()
    x = pde.validation_grid(10, DEVICE, DTYPE)
    a = capture_activation(model, x, layer_index=1)
    assert a.shape == (10, 16)


def test_partial_interchange_changes_leading_coeffs():
    a_s = torch.randn(5, 16)
    a_d = torch.randn(1, 16)
    basis = torch.linalg.qr(torch.randn(16, 4))[0].T
    mean = torch.zeros(16)
    out = partial_interchange(a_s, a_d, basis, mean, n_swapped=2)
    assert out.shape == a_s.shape
    assert not torch.allclose(out, a_s)


def test_interchange_hook_preserves_batch():
    pde, model = _setup()
    x = pde.validation_grid(8, DEVICE, DTYPE)
    a_d = capture_activation(model, x[:4], layer_index=1).mean(
        dim=0, keepdim=True)
    basis = random_basis(16, 4)
    mean = torch.zeros(16)
    hook = make_interchange_hook(a_d, basis, mean, 1)
    lp, lb = _losses_with_hook(model, pde, x, hook, layer_index=1)
    assert np.isfinite(lp) and np.isfinite(lb)


def test_region_split_two_regions():
    pde, _ = _setup()
    regions = region_split(pde, 512, 0.15, DEVICE, DTYPE, seed=0)
    assert regions["interior"].shape[0] > 0
    assert regions["boundary_proximal"].shape[0] > 0
    lo, hi = pde.domain
    b = regions["boundary_proximal"]
    dist = torch.minimum(b[:, 0] - lo, hi - b[:, 0])
    assert (dist < 0.15 * (hi - lo) + 1e-6).all()


def test_alignment_accuracy_record():
    pde, model = _setup()
    x = pde.sample_interior(32, DEVICE, DTYPE, seed=0)
    regions = region_split(pde, 128, 0.15, DEVICE, DTYPE, seed=1)
    basis = random_basis(16, 4)
    mean = torch.zeros(16)
    rec = interchange_alignment_accuracy(
        model, pde, regions["interior"][:32], regions["boundary_proximal"],
        basis, mean, layer_index=1, n_swapped=2,
    )
    assert np.isfinite(rec["source_loss_pde"])
    assert np.isfinite(rec["swapped_loss_pde"])
    assert "movement_fraction" in rec


def test_run_alignment_battery():
    pde, model = _setup()
    basis = random_basis(16, 4)
    mean = torch.zeros(16)
    out = run_alignment_battery(
        model, pde, basis, mean, layer_index=1, n_swapped=2,
        n_trials=4, n_batch=32,
    )
    assert out["n_trials"] == 4
    assert "mean_movement_fraction" in out


def test_random_basis_orthonormal():
    B = random_basis(16, 4, seed=3)
    G = B @ B.T
    assert torch.allclose(G, torch.eye(4), atol=1e-5)
