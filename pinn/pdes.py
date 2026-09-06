"""
PDE interfaces and implementations for the PINN mechanistic interpretability benchmark.

Canonical PDE suite (Stage A per the implementation plan):
  - Poisson1D          : smooth solution, Dirichlet BC — pipeline qualification
  - Advection1D        : transport + spectral sensitivity
  - ReactionDiffusion1D: competing spatial/reaction scales

All PDEs implement the BasePDE interface so trainers and diagnostics are PDE-agnostic.
"""
from __future__ import annotations
import abc
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple
import math
import torch


# ---------------------------------------------------------------------------
# Abstract base interface
# ---------------------------------------------------------------------------

class BasePDE(abc.ABC):
    """Common interface every PDE must satisfy."""

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def domain(self) -> Tuple[float, float]: ...

    @abc.abstractmethod
    def exact(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Analytic/reference solution at coordinates x.  None if unavailable."""

    @abc.abstractmethod
    def residual(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """PDE interior residual (should be zero for the exact solution)."""

    @abc.abstractmethod
    def boundary_residual(self, model: torch.nn.Module) -> torch.Tensor:
        """Boundary-condition residual (scalar or mean-squared error tensor)."""

    @abc.abstractmethod
    def boundary_points(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Canonical boundary collocation points."""

    @abc.abstractmethod
    def validation_grid(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Frozen evaluation grid of n uniformly spaced interior + boundary points."""

    def sample_interior(
        self,
        n: int,
        device: torch.device,
        dtype: torch.dtype,
        seed: Optional[int] = None,
        spatial_bias: Optional[float] = None,
    ) -> torch.Tensor:
        """Uniform or biased random interior sample.
        spatial_bias in [0,1]: probability mass concentrated near boundaries.
        """
        lo, hi = self.domain
        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(seed)
        if spatial_bias is None or spatial_bias == 0.0:
            return torch.rand(n, 1, device=device, dtype=dtype, generator=gen) * (hi - lo) + lo
        # Use only generator-aware tensor operations so seeded sampling stays
        # reproducible on both CPU and CUDA.
        boundary_mask = torch.rand(
            n, 1, device=device, dtype=dtype, generator=gen
        ) < spatial_bias
        side = torch.rand(n, 1, device=device, dtype=dtype, generator=gen) < 0.5
        distance = torch.rand(n, 1, device=device, dtype=dtype, generator=gen).square()
        boundary_samples = torch.where(side, distance, 1.0 - distance)
        uniform_samples = torch.rand(n, 1, device=device, dtype=dtype, generator=gen)
        samples = torch.where(boundary_mask, boundary_samples, uniform_samples)
        return samples * (hi - lo) + lo

    def to_dict(self) -> Dict:
        """Serialise PDE parameters for run manifests."""
        return {"name": self.name, "domain": list(self.domain)}


# ---------------------------------------------------------------------------
# 1D Poisson: u'' = f,  u(left)=left_bc,  u(right)=right_bc
# Exact solution for uniform source:  u = (f/2)(x-L)(x-R) + linear BCs
# ---------------------------------------------------------------------------

@dataclass
class Poisson1D(BasePDE):
    left: float = -1.0
    right: float = 1.0
    source: float = 1.0
    left_bc: float = 0.0
    right_bc: float = 0.0

    @property
    def name(self) -> str:
        return "poisson_1d"

    @property
    def domain(self) -> Tuple[float, float]:
        return (self.left, self.right)

    def exact(self, x: torch.Tensor) -> torch.Tensor:
        L = self.right - self.left
        u_part = (self.source / 2.0) * (x - self.left) * (x - self.right)
        bc_part = self.left_bc + (self.right_bc - self.left_bc) * (x - self.left) / L
        return u_part + bc_part

    def residual(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        u = model(x)
        ux = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
        return uxx - self.source

    def boundary_residual(self, model: torch.nn.Module) -> torch.Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        xb = self.boundary_points(device, dtype)
        ub = model(xb)
        target = torch.tensor([[self.left_bc], [self.right_bc]], device=device, dtype=dtype)
        return (ub - target) ** 2

    def boundary_points(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor([[self.left], [self.right]], device=device, dtype=dtype)

    def validation_grid(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.linspace(self.left, self.right, n, device=device, dtype=dtype).reshape(-1, 1)

    def to_dict(self) -> Dict:
        return {**super().to_dict(), "source": self.source, "left_bc": self.left_bc, "right_bc": self.right_bc}


@dataclass
class Poisson2D(BasePDE):
    """Manufactured 2D Poisson benchmark on a rectangular domain.

    The exact solution is ``source * sin(pi*x) * sin(pi*y)`` after mapping
    both coordinates to [0, 1]. The residual is the corresponding homogeneous
    Helmholtz form, which gives a smooth zero-Dirichlet 2D boundary problem.
    """
    left: float = 0.0
    right: float = 1.0
    bottom: float = 0.0
    top: float = 1.0
    source: float = 1.0

    @property
    def name(self) -> str:
        return "poisson_2d"

    @property
    def domain(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return ((self.left, self.right), (self.bottom, self.top))

    def exact(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x[:, 0:1] - self.left) / (self.right - self.left)
        y_norm = (x[:, 1:2] - self.bottom) / (self.top - self.bottom)
        return self.source * torch.sin(math.pi * x_norm) * torch.sin(math.pi * y_norm)

    def residual(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        u = model(x)
        grad_u = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(grad_u[:, 0:1], x, torch.ones_like(grad_u[:, 0:1]), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(grad_u[:, 1:2], x, torch.ones_like(grad_u[:, 1:2]), create_graph=True)[0][:, 1:2]
        x_norm = (x[:, 0:1] - self.left) / (self.right - self.left)
        y_norm = (x[:, 1:2] - self.bottom) / (self.top - self.bottom)
        forcing = 2.0 * math.pi ** 2 * self.source * torch.sin(math.pi * x_norm) * torch.sin(math.pi * y_norm)
        return u_xx + u_yy + forcing

    def boundary_residual(self, model: torch.nn.Module) -> torch.Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        points = self.boundary_points(device, dtype)
        return model(points) ** 2

    def boundary_points(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        edge = torch.linspace(self.left, self.right, 20, device=device, dtype=dtype)
        side = torch.linspace(self.bottom, self.top, 20, device=device, dtype=dtype)
        return torch.cat([
            torch.stack([edge, torch.full_like(edge, self.bottom)], dim=1),
            torch.stack([edge, torch.full_like(edge, self.top)], dim=1),
            torch.stack([torch.full_like(side, self.left), side], dim=1),
            torch.stack([torch.full_like(side, self.right), side], dim=1),
        ], dim=0)

    def sample_interior(self, n: int, device: torch.device, dtype: torch.dtype, seed: Optional[int] = None, spatial_bias: Optional[float] = None) -> torch.Tensor:
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        x = torch.rand(n, 2, device=device, dtype=dtype, generator=generator)
        x[:, 0] = x[:, 0] * (self.right - self.left) + self.left
        x[:, 1] = x[:, 1] * (self.top - self.bottom) + self.bottom
        return x

    def validation_grid(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        side = max(2, int(math.ceil(math.sqrt(n))))
        x = torch.linspace(self.left, self.right, side, device=device, dtype=dtype)
        y = torch.linspace(self.bottom, self.top, side, device=device, dtype=dtype)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    def to_dict(self) -> Dict:
        return {**super().to_dict(), "domain_y": [self.bottom, self.top], "source": self.source}


# ---------------------------------------------------------------------------
# 1D Advection: u_t + c·u_x = 0  (steady: c·u_x = f)
# Steady formulation: c·u' = f,  u(left) = u_bc
# Exact: u(x) = u_bc + (f/c)·(x - left)
# ---------------------------------------------------------------------------

@dataclass
class Advection1D(BasePDE):
    """Steady 1D linear advection: c·u' = f, u(left) = left_bc.

    This introduces transport and spectral sensitivity without time stepping.
    High advection speed c creates boundary-layer-like behaviour that challenges
    PINNs via spectral bias.
    """
    left: float = 0.0
    right: float = 1.0
    speed: float = 1.0        # advection coefficient c
    source: float = 0.0       # forcing term f
    left_bc: float = 1.0      # inlet value

    @property
    def name(self) -> str:
        return "advection_1d"

    @property
    def domain(self) -> Tuple[float, float]:
        return (self.left, self.right)

    def exact(self, x: torch.Tensor) -> torch.Tensor:
        # c·u' = f  →  u(x) = left_bc + (f/c)·(x - left)
        if abs(self.speed) < 1e-12:
            return torch.full_like(x, self.left_bc)
        return self.left_bc + (self.source / self.speed) * (x - self.left)

    def residual(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        u = model(x)
        ux = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        return self.speed * ux - self.source

    def boundary_residual(self, model: torch.nn.Module) -> torch.Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        xb = self.boundary_points(device, dtype)
        ub = model(xb)
        target = torch.tensor([[self.left_bc]], device=device, dtype=dtype)
        return (ub - target) ** 2

    def boundary_points(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor([[self.left]], device=device, dtype=dtype)

    def validation_grid(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.linspace(self.left, self.right, n, device=device, dtype=dtype).reshape(-1, 1)

    def to_dict(self) -> Dict:
        return {**super().to_dict(), "speed": self.speed, "source": self.source, "left_bc": self.left_bc}


# ---------------------------------------------------------------------------
# 1D Reaction-Diffusion: -ε·u'' + μ·u = f,  u(left)=left_bc, u(right)=right_bc
# Exact solution via characteristic roots when f=0:
#   u = A·exp(r·x) + B·exp(-r·x), r = sqrt(μ/ε)
# ---------------------------------------------------------------------------

@dataclass
class ReactionDiffusion1D(BasePDE):
    """Steady 1D reaction-diffusion: -ε·u'' + μ·u = f.

    Competing spatial (diffusion) and reaction scales.  Small ε creates stiff
    boundary layers that reliably induce spectral-suppression failure in PINNs.
    """
    left: float = -1.0
    right: float = 1.0
    diffusion: float = 0.01       # ε — small values → stiff BL
    reaction: float = 1.0         # μ
    source: float = 0.0           # f (constant forcing)
    left_bc: float = 0.0
    right_bc: float = 0.0

    @property
    def name(self) -> str:
        return "reaction_diffusion_1d"

    @property
    def domain(self) -> Tuple[float, float]:
        return (self.left, self.right)

    def exact(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Analytic solution for constant source f and Dirichlet BCs."""
        eps = self.diffusion
        mu = self.reaction
        f = self.source
        lo, hi = self.left, self.right

        if abs(mu) < 1e-12:
            # Pure diffusion: -ε u'' = f
            return (f / (2 * eps)) * (x - lo) * (hi - x) + self.left_bc + (self.right_bc - self.left_bc) * (x - lo) / (hi - lo)

        r = (mu / eps) ** 0.5
        # Homogeneous solution + particular solution (u_p = f/mu)
        u_p = f / mu
        # BCs: A·exp(r·lo) + B·exp(-r·lo) + u_p = left_bc
        #       A·exp(r·hi) + B·exp(-r·hi) + u_p = right_bc
        import math
        e_lo_p = math.exp(r * lo)
        e_lo_m = math.exp(-r * lo)
        e_hi_p = math.exp(r * hi)
        e_hi_m = math.exp(-r * hi)
        rhs_lo = self.left_bc - u_p
        rhs_hi = self.right_bc - u_p
        det = e_lo_p * e_hi_m - e_lo_m * e_hi_p
        if abs(det) < 1e-30:
            return None   # degenerate — caller should fall back to numerical
        A = (rhs_lo * e_hi_m - rhs_hi * e_lo_m) / det
        B = (rhs_hi * e_lo_p - rhs_lo * e_hi_p) / det

        x_np = x.detach().cpu().numpy()
        import numpy as np
        u_np = A * np.exp(r * x_np) + B * np.exp(-r * x_np) + u_p
        return torch.tensor(u_np, device=x.device, dtype=x.dtype)

    def residual(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        u = model(x)
        ux = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
        return -self.diffusion * uxx + self.reaction * u - self.source

    def boundary_residual(self, model: torch.nn.Module) -> torch.Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        xb = self.boundary_points(device, dtype)
        ub = model(xb)
        target = torch.tensor([[self.left_bc], [self.right_bc]], device=device, dtype=dtype)
        return (ub - target) ** 2

    def boundary_points(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor([[self.left], [self.right]], device=device, dtype=dtype)

    def validation_grid(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.linspace(self.left, self.right, n, device=device, dtype=dtype).reshape(-1, 1)

    def to_dict(self) -> Dict:
        return {
            **super().to_dict(),
            "diffusion": self.diffusion,
            "reaction": self.reaction,
            "source": self.source,
            "left_bc": self.left_bc,
            "right_bc": self.right_bc,
        }


# ---------------------------------------------------------------------------
# 2D Advection-Diffusion (steady): c·grad u = ε·nabla²u + f on [left,right]²
# Manufactured solution: u = sin(pi*x_norm)*sin(pi*y_norm); the forcing is
# the analytic residual of that profile under the operator.
# ---------------------------------------------------------------------------

@dataclass
class AdvectionDiffusion2D(BasePDE):
    """Steady 2D advection-diffusion with a manufactured exact solution.

    Adds a transport term to the 2D Poisson prototype so 2D runs differ in
    operator structure, not merely in forcing.  The exact solution is the
    separable sine profile; the forcing is derived analytically.
    """

    left: float = 0.0
    right: float = 1.0
    bottom: float = 0.0
    top: float = 1.0
    speed_x: float = 1.0
    speed_y: float = 0.5
    diffusion: float = 0.05
    amplitude: float = 1.0

    @property
    def name(self) -> str:
        return "advection_diffusion_2d"

    @property
    def domain(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return ((self.left, self.right), (self.bottom, self.top))

    def _normed(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xn = (x[:, 0:1] - self.left) / (self.right - self.left)
        yn = (x[:, 1:2] - self.bottom) / (self.top - self.bottom)
        return xn, yn

    def exact(self, x: torch.Tensor) -> torch.Tensor:
        xn, yn = self._normed(x)
        return self.amplitude * torch.sin(math.pi * xn) * torch.sin(math.pi * yn)

    def residual(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        u = model(x)
        grad_u = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(grad_u[:, 0:1], x, torch.ones_like(grad_u[:, 0:1]), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(grad_u[:, 1:2], x, torch.ones_like(grad_u[:, 1:2]), create_graph=True)[0][:, 1:2]
        adv = self.speed_x * grad_u[:, 0:1] + self.speed_y * grad_u[:, 1:2]
        xn, yn = self._normed(x)
        prof = torch.sin(math.pi * xn) * torch.sin(math.pi * yn)
        dprof_x = math.pi * torch.cos(math.pi * xn) * torch.sin(math.pi * yn)
        dprof_y = math.pi * torch.sin(math.pi * xn) * torch.cos(math.pi * yn)
        # Forcing for L u = f with L = c·grad - eps·nabla²:
        f = self.amplitude * (self.speed_x * dprof_x + self.speed_y * dprof_y
                              + 2.0 * math.pi ** 2 * self.diffusion * prof)
        return adv - self.diffusion * (u_xx + u_yy) - f

    def boundary_residual(self, model: torch.nn.Module) -> torch.Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        points = self.boundary_points(device, dtype)
        return model(points) ** 2

    def boundary_points(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        edge = torch.linspace(self.left, self.right, 20, device=device, dtype=dtype)
        side = torch.linspace(self.bottom, self.top, 20, device=device, dtype=dtype)
        return torch.cat([
            torch.stack([edge, torch.full_like(edge, self.bottom)], dim=1),
            torch.stack([edge, torch.full_like(edge, self.top)], dim=1),
            torch.stack([torch.full_like(side, self.left), side], dim=1),
            torch.stack([torch.full_like(side, self.right), side], dim=1),
        ], dim=0)

    def sample_interior(self, n: int, device: torch.device, dtype: torch.dtype, seed: Optional[int] = None, spatial_bias: Optional[float] = None) -> torch.Tensor:
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        x = torch.rand(n, 2, device=device, dtype=dtype, generator=generator)
        x[:, 0] = x[:, 0] * (self.right - self.left) + self.left
        x[:, 1] = x[:, 1] * (self.top - self.bottom) + self.bottom
        return x

    def validation_grid(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        side = max(2, int(math.ceil(math.sqrt(n))))
        x = torch.linspace(self.left, self.right, side, device=device, dtype=dtype)
        y = torch.linspace(self.bottom, self.top, side, device=device, dtype=dtype)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    def to_dict(self) -> Dict:
        return {**super().to_dict(), "domain_y": [self.bottom, self.top],
                "speed_x": self.speed_x, "speed_y": self.speed_y,
                "diffusion": self.diffusion, "amplitude": self.amplitude}


# ---------------------------------------------------------------------------
# 2D Reaction-Diffusion (steady): eps·nabla²u + R(u) = f, R(u) = mu·u(1-u)
# Manufactured solution: u = sin(pi*x_norm)*sin(pi*y_norm) with derived f.
# ---------------------------------------------------------------------------

@dataclass
class ReactionDiffusion2D(BasePDE):
    """Steady 2D reaction-diffusion (logistic reaction) with manufactured f."""

    left: float = 0.0
    right: float = 1.0
    bottom: float = 0.0
    top: float = 1.0
    diffusion: float = 0.1
    reaction: float = 2.0
    amplitude: float = 1.0

    @property
    def name(self) -> str:
        return "reaction_diffusion_2d"

    @property
    def domain(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return ((self.left, self.right), (self.bottom, self.top))

    def _normed(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xn = (x[:, 0:1] - self.left) / (self.right - self.left)
        yn = (x[:, 1:2] - self.bottom) / (self.top - self.bottom)
        return xn, yn

    def exact(self, x: torch.Tensor) -> torch.Tensor:
        xn, yn = self._normed(x)
        return self.amplitude * torch.sin(math.pi * xn) * torch.sin(math.pi * yn)

    def residual(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        u = model(x)
        grad_u = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(grad_u[:, 0:1], x, torch.ones_like(grad_u[:, 0:1]), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(grad_u[:, 1:2], x, torch.ones_like(grad_u[:, 1:2]), create_graph=True)[0][:, 1:2]
        xn, yn = self._normed(x)
        prof = torch.sin(math.pi * xn) * torch.sin(math.pi * yn)
        u_exact = self.amplitude * prof
        # Forcing for eps·nabla²u + mu·u(1-u) = f evaluated at u_exact.
        lap = -2.0 * math.pi ** 2 * u_exact
        react = self.reaction * u_exact * (1.0 - u_exact)
        f = self.diffusion * lap + react
        return self.diffusion * (u_xx + u_yy) + self.reaction * u * (1.0 - u) - f

    def boundary_residual(self, model: torch.nn.Module) -> torch.Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        points = self.boundary_points(device, dtype)
        return model(points) ** 2

    def boundary_points(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        edge = torch.linspace(self.left, self.right, 20, device=device, dtype=dtype)
        side = torch.linspace(self.bottom, self.top, 20, device=device, dtype=dtype)
        return torch.cat([
            torch.stack([edge, torch.full_like(edge, self.bottom)], dim=1),
            torch.stack([edge, torch.full_like(edge, self.top)], dim=1),
            torch.stack([torch.full_like(side, self.left), side], dim=1),
            torch.stack([torch.full_like(side, self.right), side], dim=1),
        ], dim=0)

    def sample_interior(self, n: int, device: torch.device, dtype: torch.dtype, seed: Optional[int] = None, spatial_bias: Optional[float] = None) -> torch.Tensor:
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        x = torch.rand(n, 2, device=device, dtype=dtype, generator=generator)
        x[:, 0] = x[:, 0] * (self.right - self.left) + self.left
        x[:, 1] = x[:, 1] * (self.top - self.bottom) + self.bottom
        return x

    def validation_grid(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        side = max(2, int(math.ceil(math.sqrt(n))))
        x = torch.linspace(self.left, self.right, side, device=device, dtype=dtype)
        y = torch.linspace(self.bottom, self.top, side, device=device, dtype=dtype)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    def to_dict(self) -> Dict:
        return {**super().to_dict(), "domain_y": [self.bottom, self.top],
                "diffusion": self.diffusion, "reaction": self.reaction,
                "amplitude": self.amplitude}


# ---------------------------------------------------------------------------
# Time-dependent PDEs (input = (t, x)): the PINN takes (t,x) and returns u.
# The "domain" tuple convention: domain = spatial interval; t in [t0, tT].
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 1D Burgers (viscous): u_t + u·u_x = nu·u_xx,  u(0,x)=u0(x), u(t,±1)=0
# Reference: spectral solution computed once on a fixed grid (cached).
# ---------------------------------------------------------------------------

class Burgers1D(BasePDE):
    """Viscous 1D Burgers on x in [-1,1], t in [0,1].

    The initial condition is u0(x) = -sin(pi x).  The reference solution is
    computed by a deterministic pseudo-spectral solver (Fourier + ETDRK4 is
    overkill here; a simple explicit RK4 in the rotated Fourier basis is
    sufficient at nu=0.01 and cached to disk on first use).

    Boundary treatment: the network input is (t, x) padded with the spatial
    interval; Dirichlet u=0 at x=±1.
    """

    def __init__(self, viscosity: float = 0.01, t0: float = 0.0, tT: float = 1.0,
                 left: float = -1.0, right: float = 1.0,
                 n_spectral: int = 256, n_timesteps: int = 200):
        self.viscosity = viscosity
        self.t0 = t0
        self.tT = tT
        self.left = left
        self.right = right
        self.n_spectral = n_spectral
        self.n_timesteps = n_timesteps
        self._reference = None

    @property
    def name(self) -> str:
        return "burgers_1d"

    @property
    def domain(self) -> Tuple[float, float]:
        return (self.left, self.right)

    @property
    def time_domain(self) -> Tuple[float, float]:
        return (self.t0, self.tT)

    def _spectral_reference(self) -> tuple:
        """Deterministic pseudo-spectral solve with an integrating factor.

        The linear diffusion term is handled exactly via the integrating
        factor E = exp(-nu k^2 dt); the nonlinear advection term is stepped
        with Heun (RK2) inside the rotated variable.  This is stable at
        nu=0.01 / dt=1/200 where naive explicit stepping blows up.
        """
        import numpy as np
        N = self.n_spectral
        L = self.right - self.left
        x = np.linspace(self.left, self.right, N, endpoint=False)
        k = 2 * np.pi * np.fft.rfftfreq(N, d=L / N)
        u = -np.sin(np.pi * x)
        dt = (self.tT - self.t0) / self.n_timesteps
        nu = self.viscosity
        E = np.exp(-nu * k ** 2 * dt)
        E2 = np.exp(-nu * k ** 2 * dt / 2.0)
        u_hat = np.fft.rfft(u)
        for _ in range(self.n_timesteps):
            # Integrating-factor Heun step (Kassam--Trefethen style).
            def g(uh):
                uu = np.fft.irfft(uh, N)
                return -0.5j * k * np.fft.rfft(uu ** 2)
            a = E2 * u_hat
            b = E2 * (u_hat + 0.5 * dt * g(a))
            u_hat = E * u_hat + 0.5 * dt * (E * g(a) + g(b))
        u_final = np.fft.irfft(u_hat, N)
        return x, u_final

    def reference_solution(self) -> tuple:
        if self._reference is None:
            x, u = self._spectral_reference()
            self._reference = (torch.tensor(x, dtype=torch.float32),
                               torch.tensor(u, dtype=torch.float32))
        return self._reference

    def exact(self, x_tx: torch.Tensor) -> Optional[torch.Tensor]:
        """Reference evaluated at final time on the spectral grid (interp)."""
        x_ref, u_ref = self.reference_solution()
        x_ref = x_ref.to(x_tx.device, x_tx.dtype)
        u_ref = u_ref.to(x_tx.device, x_tx.dtype)
        t = x_tx[:, 0]
        xq = x_tx[:, 1]
        at_final = torch.abs(t - self.tT) < 1e-6
        # Linear interpolation of the reference profile.
        lo, hi = self.left, self.right
        pos = (xq - lo) / (hi - lo) * (x_ref.shape[0] - 1)
        i0 = pos.floor().clamp(0, x_ref.shape[0] - 2).long()
        frac = pos - i0
        u_interp = u_ref[i0] * (1 - frac) + u_ref[i0 + 1] * frac
        return torch.where(at_final, u_interp, torch.zeros_like(u_interp))

    def residual(self, model: torch.nn.Module, x_tx: torch.Tensor) -> torch.Tensor:
        x_tx = x_tx.requires_grad_(True)
        u = model(x_tx)
        grad_u = torch.autograd.grad(u, x_tx, torch.ones_like(u), create_graph=True)[0]
        u_t = grad_u[:, 0:1]
        u_x = grad_u[:, 1:2]
        u_xx = torch.autograd.grad(u_x, x_tx, torch.ones_like(u_x), create_graph=True)[0][:, 1:2]
        return u_t + u * u_x - self.viscosity * u_xx

    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        """u0(x) = -sin(pi x) — satisfies the zero Dirichlet BCs at x=±1."""
        return -torch.sin(math.pi * x)

    def boundary_residual(self, model: torch.nn.Module) -> torch.Tensor:
        """Spatial Dirichlet (u=0 at x=±1) + initial condition (u=u0 at t=0).

        Without the IC term a time-dependent PINN converges to the trivial
        u≡0 solution (PDE + spatial BCs are satisfied by zero); the IC is
        what makes the problem well-posed.
        """
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        t_grid = torch.linspace(self.t0, self.tT, 33, device=device, dtype=dtype)
        x_grid = torch.linspace(self.left, self.right, 33, device=device, dtype=dtype)
        spatial = torch.cat([
            torch.stack([t_grid, torch.full_like(t_grid, self.left)], dim=1),
            torch.stack([t_grid, torch.full_like(t_grid, self.right)], dim=1),
        ])
        u_spatial = model(spatial) ** 2
        ic = torch.stack([torch.zeros_like(x_grid), x_grid], dim=1)
        u_ic = (model(ic) - self.initial_condition(x_grid).reshape(-1, 1)) ** 2
        return torch.cat([u_spatial, u_ic])

    def boundary_points(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t_grid = torch.linspace(self.t0, self.tT, 33, device=device, dtype=dtype)
        x_grid = torch.linspace(self.left, self.right, 33, device=device, dtype=dtype)
        return torch.cat([
            torch.stack([t_grid, torch.full_like(t_grid, self.left)], dim=1),
            torch.stack([t_grid, torch.full_like(t_grid, self.right)], dim=1),
            torch.stack([torch.zeros_like(x_grid), x_grid], dim=1),
        ])

    def sample_interior(self, n: int, device: torch.device, dtype: torch.dtype,
                        seed: Optional[int] = None, spatial_bias: Optional[float] = None) -> torch.Tensor:
        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(seed)
        t = torch.rand(n, 1, device=device, dtype=dtype, generator=gen) * (self.tT - self.t0) + self.t0
        x = torch.rand(n, 1, device=device, dtype=dtype, generator=gen) * (self.right - self.left) + self.left
        return torch.cat([t, x], dim=1)

    def validation_grid(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        n_side = max(2, int(math.ceil(math.sqrt(n))))
        t = torch.linspace(self.t0, self.tT, n_side, device=device, dtype=dtype)
        x = torch.linspace(self.left, self.right, n_side, device=device, dtype=dtype)
        gt, gx = torch.meshgrid(t, x, indexing="ij")
        return torch.stack([gt.flatten(), gx.flatten()], dim=1)

    def to_dict(self) -> Dict:
        return {"name": self.name, "domain": [self.left, self.right],
                "time_domain": [self.t0, self.tT],
                "viscosity": self.viscosity}


# ---------------------------------------------------------------------------
# 1D Allen-Cahn: u_t = eps²·u_xx + u - u³ (no-flux/periodic-style IC)
# Reference: same pseudo-spectral machinery, reaction handled explicitly.
# ---------------------------------------------------------------------------

class AllenCahn1D(BasePDE):
    """1D Allen-Cahn on x in [-1,1], t in [0,1], u0 = tanh-like profile.

    u0(x) = 0.5*(1 + tanh(2x/eps)) * 1 - 0.5 ... we use the standard smooth
    double-well IC: u0(x) = x³? No — the canonical PINN IC is
    u0(x) = u0h(x) with u0h(x) = x³ (from the original PINN paper's AC
    example domain x in [-1,1], t in [0,0.05] with eps=0.003).  Here we use
    a numerically tractable setting: eps=0.05, t up to 1.0, IC = x³
    rescaled, periodic in the spectral solver.
    """

    def __init__(self, epsilon: float = 0.05, t0: float = 0.0, tT: float = 1.0,
                 left: float = -1.0, right: float = 1.0,
                 n_spectral: int = 256, n_timesteps: int = 400):
        self.epsilon = epsilon
        self.t0 = t0
        self.tT = tT
        self.left = left
        self.right = right
        self.n_spectral = n_spectral
        self.n_timesteps = n_timesteps
        self._reference = None

    @property
    def name(self) -> str:
        return "allen_cahn_1d"

    @property
    def domain(self) -> Tuple[float, float]:
        return (self.left, self.right)

    @property
    def time_domain(self) -> Tuple[float, float]:
        return (self.t0, self.tT)

    def _spectral_reference(self) -> tuple:
        import numpy as np
        N = self.n_spectral
        L = self.right - self.left
        x = np.linspace(self.left, self.right, N, endpoint=False)
        k = 2 * np.pi * np.fft.rfftfreq(N, d=L / N)
        u = x ** 3
        dt = (self.tT - self.t0) / self.n_timesteps
        eps2 = self.epsilon ** 2
        u_hat = np.fft.rfft(u)
        for _ in range(self.n_timesteps):
            def rhs_total(uh):
                uu = np.fft.irfft(uh, N)
                reaction = np.fft.rfft(uu - uu ** 3)
                return -eps2 * (k ** 2) * uh + reaction
            k1 = rhs_total(u_hat)
            u_mid = u_hat + 0.5 * dt * k1
            k2 = rhs_total(u_mid)
            u_hat = u_hat + dt * k2
        u_final = np.fft.irfft(u_hat, N)
        return x, u_final

    def reference_solution(self) -> tuple:
        if self._reference is None:
            x, u = self._spectral_reference()
            self._reference = (torch.tensor(x, dtype=torch.float32),
                               torch.tensor(u, dtype=torch.float32))
        return self._reference

    def exact(self, x_tx: torch.Tensor) -> Optional[torch.Tensor]:
        x_ref, u_ref = self.reference_solution()
        x_ref = x_ref.to(x_tx.device, x_tx.dtype)
        u_ref = u_ref.to(x_tx.device, x_tx.dtype)
        t = x_tx[:, 0]
        xq = x_tx[:, 1]
        at_final = torch.abs(t - self.tT) < 1e-6
        lo, hi = self.left, self.right
        pos = (xq - lo) / (hi - lo) * (x_ref.shape[0] - 1)
        i0 = pos.floor().clamp(0, x_ref.shape[0] - 2).long()
        frac = pos - i0
        u_interp = u_ref[i0] * (1 - frac) + u_ref[i0 + 1] * frac
        return torch.where(at_final, u_interp, torch.zeros_like(u_interp))

    def residual(self, model: torch.nn.Module, x_tx: torch.Tensor) -> torch.Tensor:
        x_tx = x_tx.requires_grad_(True)
        u = model(x_tx)
        grad_u = torch.autograd.grad(u, x_tx, torch.ones_like(u), create_graph=True)[0]
        u_t = grad_u[:, 0:1]
        u_x = grad_u[:, 1:2]
        u_xx = torch.autograd.grad(u_x, x_tx, torch.ones_like(u_x), create_graph=True)[0][:, 1:2]
        return u_t - self.epsilon ** 2 * u_xx - u + u ** 3

    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        """u0(x) = x³ (canonical Allen-Cahn PINN initial condition)."""
        return x ** 3

    def boundary_residual(self, model: torch.nn.Module) -> torch.Tensor:
        """Spatial Dirichlet (u=0 at x=±1) + initial condition (u=x³ at t=0)."""
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        t_grid = torch.linspace(self.t0, self.tT, 33, device=device, dtype=dtype)
        x_grid = torch.linspace(self.left, self.right, 33, device=device, dtype=dtype)
        spatial = torch.cat([
            torch.stack([t_grid, torch.full_like(t_grid, self.left)], dim=1),
            torch.stack([t_grid, torch.full_like(t_grid, self.right)], dim=1),
        ])
        u_spatial = model(spatial) ** 2
        ic = torch.stack([torch.zeros_like(x_grid), x_grid], dim=1)
        u_ic = (model(ic) - self.initial_condition(x_grid).reshape(-1, 1)) ** 2
        return torch.cat([u_spatial, u_ic])

    def boundary_points(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t_grid = torch.linspace(self.t0, self.tT, 33, device=device, dtype=dtype)
        x_grid = torch.linspace(self.left, self.right, 33, device=device, dtype=dtype)
        return torch.cat([
            torch.stack([t_grid, torch.full_like(t_grid, self.left)], dim=1),
            torch.stack([t_grid, torch.full_like(t_grid, self.right)], dim=1),
            torch.stack([torch.zeros_like(x_grid), x_grid], dim=1),
        ])

    def sample_interior(self, n: int, device: torch.device, dtype: torch.dtype,
                        seed: Optional[int] = None, spatial_bias: Optional[float] = None) -> torch.Tensor:
        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(seed)
        t = torch.rand(n, 1, device=device, dtype=dtype, generator=gen) * (self.tT - self.t0) + self.t0
        x = torch.rand(n, 1, device=device, dtype=dtype, generator=gen) * (self.right - self.left) + self.left
        return torch.cat([t, x], dim=1)

    def validation_grid(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        n_side = max(2, int(math.ceil(math.sqrt(n))))
        t = torch.linspace(self.t0, self.tT, n_side, device=device, dtype=dtype)
        x = torch.linspace(self.left, self.right, n_side, device=device, dtype=dtype)
        gt, gx = torch.meshgrid(t, x, indexing="ij")
        return torch.stack([gt.flatten(), gx.flatten()], dim=1)

    def to_dict(self) -> Dict:
        return {"name": self.name, "domain": [self.left, self.right],
                "time_domain": [self.t0, self.tT], "epsilon": self.epsilon}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_pde(cfg) -> BasePDE:
    """Instantiate a PDE from a PDEConfig object."""
    name = cfg.name
    if name == "poisson_1d":
        return Poisson1D(
            left=cfg.domain[0],
            right=cfg.domain[1],
            source=cfg.source,
            left_bc=cfg.boundary_values[0],
            right_bc=cfg.boundary_values[1],
        )
    elif name == "poisson_2d":
        if cfg.domain_y is None:
            raise ValueError("poisson_2d requires domain_y=[bottom, top]")
        return Poisson2D(
            left=cfg.domain[0], right=cfg.domain[1],
            bottom=cfg.domain_y[0], top=cfg.domain_y[1],
            source=cfg.source,
        )
    elif name == "advection_diffusion_2d":
        if cfg.domain_y is None:
            raise ValueError("advection_diffusion_2d requires domain_y")
        return AdvectionDiffusion2D(
            left=cfg.domain[0], right=cfg.domain[1],
            bottom=cfg.domain_y[0], top=cfg.domain_y[1],
            speed_x=cfg.source,
            speed_y=getattr(cfg, "forcing", 0.5),
            diffusion=getattr(cfg, "diffusion", 0.05),
        )
    elif name == "reaction_diffusion_2d":
        if cfg.domain_y is None:
            raise ValueError("reaction_diffusion_2d requires domain_y")
        return ReactionDiffusion2D(
            left=cfg.domain[0], right=cfg.domain[1],
            bottom=cfg.domain_y[0], top=cfg.domain_y[1],
            diffusion=getattr(cfg, "diffusion", 0.1),
            reaction=cfg.source,
        )
    elif name == "burgers_1d":
        return Burgers1D(
            viscosity=getattr(cfg, "diffusion", 0.01) or 0.01,
            t0=getattr(cfg, "time_domain", [0.0, 1.0])[0],
            tT=getattr(cfg, "time_domain", [0.0, 1.0])[1],
            left=cfg.domain[0], right=cfg.domain[1],
        )
    elif name == "allen_cahn_1d":
        return AllenCahn1D(
            epsilon=getattr(cfg, "diffusion", 0.05) or 0.05,
            t0=getattr(cfg, "time_domain", [0.0, 1.0])[0],
            tT=getattr(cfg, "time_domain", [0.0, 1.0])[1],
            left=cfg.domain[0], right=cfg.domain[1],
        )
    elif name == "advection_1d":
        return Advection1D(
            left=cfg.domain[0],
            right=cfg.domain[1],
            speed=cfg.source,   # reuse 'source' field as primary PDE param
            source=getattr(cfg, "forcing", 0.0),
            left_bc=cfg.boundary_values[0],
        )
    elif name == "reaction_diffusion_1d":
        return ReactionDiffusion1D(
            left=cfg.domain[0],
            right=cfg.domain[1],
            diffusion=getattr(cfg, "diffusion", 0.01),
            reaction=cfg.source,
            source=getattr(cfg, "forcing", 0.0),
            left_bc=cfg.boundary_values[0],
            right_bc=cfg.boundary_values[1],
        )
    else:
        raise ValueError(f"Unknown PDE '{name}'")
