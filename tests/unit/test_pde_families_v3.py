"""Tests for the new PDE families (2D suite + time-dependent)."""
import torch

from pinn.pdes import (
    Burgers1D, AllenCahn1D, AdvectionDiffusion2D, ReactionDiffusion2D, make_pde,
)
from pinn.config import PDEConfig

DEVICE = torch.device("cpu")
DTYPE = torch.float32


class _ExactModel(torch.nn.Module):
    """Wraps a PDE's exact() so it acts like a model (with a parameter)."""

    def __init__(self, pde):
        super().__init__()
        self.pde = pde
        self._p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.pde.exact(x)


def test_advection_diffusion_2d_manufactured_solution():
    pde = AdvectionDiffusion2D()
    x = pde.sample_interior(64, DEVICE, DTYPE, seed=0)
    r = pde.residual(_ExactModel(pde), x)
    assert r.abs().max().item() < 1e-4
    bc = pde.boundary_residual(_ExactModel(pde))
    assert bc.abs().max().item() < 1e-4


def test_reaction_diffusion_2d_manufactured_solution():
    pde = ReactionDiffusion2D()
    x = pde.sample_interior(64, DEVICE, DTYPE, seed=0)
    r = pde.residual(_ExactModel(pde), x)
    assert r.abs().max().item() < 1e-4


def test_burgers_reference_finite():
    pde = Burgers1D()
    x, u = pde.reference_solution()
    assert torch.isfinite(u).all()
    assert u.abs().max() > 0.1  # non-trivial solution


def test_allen_cahn_reference_finite():
    pde = AllenCahn1D()
    x, u = pde.reference_solution()
    assert torch.isfinite(u).all()


def test_burgers_residual_and_bc():
    pde = Burgers1D()
    model = torch.nn.Sequential()
    # Use a tiny trainable MLP so autograd flows.
    from pinn.model import MLP
    m = MLP(2, 1, [8, 8], activation="tanh")
    x = pde.sample_interior(32, DEVICE, DTYPE, seed=0)
    r = pde.residual(m, x)
    assert r.shape == (32, 1)
    bc = pde.boundary_residual(m)
    # 2 spatial edges x 33 times + 33 IC points.
    assert bc.shape[0] == 2 * 33 + 33


def test_allen_cahn_residual_and_bc():
    pde = AllenCahn1D()
    from pinn.model import MLP
    m = MLP(2, 1, [8, 8], activation="tanh")
    x = pde.sample_interior(32, DEVICE, DTYPE, seed=0)
    r = pde.residual(m, x)
    assert r.shape == (32, 1)
    bc = pde.boundary_residual(m)
    assert bc.shape[0] == 2 * 33 + 33


def test_make_pde_new_families():
    for name, kwargs, cls in [
        ("advection_diffusion_2d",
         {"domain": [0, 1], "domain_y": [0, 1], "source": 1.0,
          "forcing": 0.5, "diffusion": 0.05, "boundary_values": [0, 0]},
         AdvectionDiffusion2D),
        ("reaction_diffusion_2d",
         {"domain": [0, 1], "domain_y": [0, 1], "source": 2.0,
          "forcing": 0.0, "diffusion": 0.1, "boundary_values": [0, 0]},
         ReactionDiffusion2D),
        ("burgers_1d",
         {"domain": [-1, 1], "time_domain": [0, 1], "diffusion": 0.01,
          "source": 1.0, "forcing": 0.0, "boundary_values": [0, 0]},
         Burgers1D),
        ("allen_cahn_1d",
         {"domain": [-1, 1], "time_domain": [0, 1], "diffusion": 0.05,
          "source": 1.0, "forcing": 0.0, "boundary_values": [0, 0]},
         AllenCahn1D),
    ]:
        cfg = PDEConfig(name=name, **kwargs)
        pde = make_pde(cfg)
        assert isinstance(pde, cls), f"{name} factory mismatch"


def test_burgers_time_dependent_validation_grid():
    pde = Burgers1D()
    g = pde.validation_grid(100, DEVICE, DTYPE)
    assert g.shape[1] == 2
    assert g[:, 0].min() >= 0.0 and g[:, 0].max() <= 1.0
