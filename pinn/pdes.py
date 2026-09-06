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
