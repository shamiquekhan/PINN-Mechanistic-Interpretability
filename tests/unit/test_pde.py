"""Tests for PDE implementations: residual correctness, boundary evaluation, AD vs FD."""
import pytest
import torch
from pinn.pdes import Poisson1D, Advection1D, ReactionDiffusion1D, make_pde
from pinn.model import MLP


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64   # use float64 for numerical derivative tests


def _make_model_from_pde(pde, hidden=[16, 16]) -> MLP:
    model = MLP(1, 1, hidden, activation="tanh").to(device=DEVICE, dtype=DTYPE)
    return model


# ---------------------------------------------------------------------------
# Poisson 1D
# ---------------------------------------------------------------------------

class TestPoisson1D:
    def setup_method(self):
        self.pde = Poisson1D(left=-1.0, right=1.0, source=1.0, left_bc=0.0, right_bc=0.0)

    def test_exact_satisfies_boundary(self):
        pde = self.pde
        # At left boundary
        xl = torch.tensor([[pde.left]], dtype=DTYPE, device=DEVICE)
        assert abs(float(pde.exact(xl)) - pde.left_bc) < 1e-6, "Left BC not satisfied by exact"
        # At right boundary
        xr = torch.tensor([[pde.right]], dtype=DTYPE, device=DEVICE)
        assert abs(float(pde.exact(xr)) - pde.right_bc) < 1e-6, "Right BC not satisfied by exact"

    def test_exact_satisfies_pde_residual(self):
        """u'' - source should be ~0 for the analytic solution."""
        pde = self.pde
        x = torch.linspace(pde.left + 0.01, pde.right - 0.01, 50, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
        x.requires_grad_(True)
        u = pde.exact(x)
        ux = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        uxx = torch.autograd.grad(ux, x, torch.ones_like(ux))[0]
        res = uxx - pde.source
        assert res.abs().max().item() < 1e-4, f"Exact solution does not satisfy PDE: max residual = {res.abs().max()}"

    def test_validation_grid_shape(self):
        n = 101
        xv = self.pde.validation_grid(n, DEVICE, DTYPE)
        assert xv.shape == (n, 1)
        assert float(xv[0]) == pytest.approx(self.pde.left)
        assert float(xv[-1]) == pytest.approx(self.pde.right)

    def test_boundary_points(self):
        xb = self.pde.boundary_points(DEVICE, DTYPE)
        assert xb.shape == (2, 1)

    def test_residual_on_mlp(self):
        model = _make_model_from_pde(self.pde)
        x = self.pde.sample_interior(32, DEVICE, DTYPE)
        res = self.pde.residual(model, x)
        assert res.shape == (32, 1)
        assert res.requires_grad

    def test_sample_interior_in_domain(self):
        x = self.pde.sample_interior(500, DEVICE, DTYPE)
        assert (x >= self.pde.left).all()
        assert (x <= self.pde.right).all()

    def test_biased_sampling_honors_seed(self):
        first = self.pde.sample_interior(128, DEVICE, DTYPE, seed=17, spatial_bias=0.8)
        second = self.pde.sample_interior(128, DEVICE, DTYPE, seed=17, spatial_bias=0.8)
        assert torch.equal(first, second)

    def test_to_dict_serialisable(self):
        d = self.pde.to_dict()
        assert d["name"] == "poisson_1d"
        assert "source" in d


# ---------------------------------------------------------------------------
# Advection 1D
# ---------------------------------------------------------------------------

class TestAdvection1D:
    def setup_method(self):
        self.pde = Advection1D(left=0.0, right=1.0, speed=2.0, source=0.0, left_bc=1.0)

    def test_exact_satisfies_inlet_bc(self):
        pde = self.pde
        xl = torch.tensor([[pde.left]], dtype=DTYPE, device=DEVICE)
        assert abs(float(pde.exact(xl)) - pde.left_bc) < 1e-6

    def test_exact_satisfies_pde(self):
        pde = self.pde
        x = torch.linspace(pde.left, pde.right, 50, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
        x.requires_grad_(True)
        u = pde.exact(x)
        ux = torch.autograd.grad(u, x, torch.ones_like(u))[0]
        res = pde.speed * ux - pde.source
        assert res.abs().max().item() < 1e-5

    def test_validation_grid(self):
        xv = self.pde.validation_grid(100, DEVICE, DTYPE)
        assert xv.shape == (100, 1)


# ---------------------------------------------------------------------------
# Reaction-Diffusion 1D
# ---------------------------------------------------------------------------

class TestReactionDiffusion1D:
    def setup_method(self):
        self.pde = ReactionDiffusion1D(
            left=-1.0, right=1.0,
            diffusion=0.1, reaction=1.0,
            source=0.0, left_bc=0.0, right_bc=0.0,
        )

    def test_exact_satisfies_boundary(self):
        pde = self.pde
        for x_val, expected_bc in [(pde.left, pde.left_bc), (pde.right, pde.right_bc)]:
            x = torch.tensor([[x_val]], dtype=DTYPE, device=DEVICE)
            u = pde.exact(x)
            if u is not None:
                assert abs(float(u) - expected_bc) < 1e-4

    def test_exact_not_none_for_nondegenerate(self):
        pde = self.pde
        x = torch.tensor([[0.0]], dtype=DTYPE, device=DEVICE)
        u = pde.exact(x)
        assert u is not None

    def test_validation_grid(self):
        xv = self.pde.validation_grid(50, DEVICE, DTYPE)
        assert xv.shape == (50, 1)


# ---------------------------------------------------------------------------
# PDE factory
# ---------------------------------------------------------------------------

class TestMakePDE:
    def test_poisson_factory(self):
        class Cfg:
            name = "poisson_1d"
            domain = [-1.0, 1.0]
            source = 1.0
            forcing = 0.0
            diffusion = 0.01
            boundary_values = [0.0, 0.0]
        pde = make_pde(Cfg())
        assert pde.name == "poisson_1d"

    def test_advection_factory(self):
        class Cfg:
            name = "advection_1d"
            domain = [0.0, 1.0]
            source = 2.0   # speed
            forcing = 0.0
            diffusion = 0.01
            boundary_values = [1.0, 0.0]
        pde = make_pde(Cfg())
        assert pde.name == "advection_1d"
