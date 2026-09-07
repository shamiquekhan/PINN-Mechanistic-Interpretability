"""Fourier Neural Operator (1D) — minimal, dependency-free implementation.

Architecture (Li et al. 2021):
    lift:  Linear(in_channels -> width)
    layers: [width -> width] via spectral convolution + skip, each followed
            by a pointwise nonlinearity
    project: Linear(width -> width) + GELU + Linear(width -> out_channels)

The spectral convolution multiplies the (truncated) Fourier modes of the
signal by a complex weight tensor per layer.  This module exists for the
regime-boundary experiment: FNO hidden representations are function-space
embeddings expected to have HIGH effective rank (superposition regime),
in contrast to the low-rank PINN activations.

Notes
-----
* Mode truncation (`modes`) follows the reference implementation.
* Everything runs on either CPU or CUDA with plain torch.
"""
from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """Multiply truncated Fourier modes by learned complex weights."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, length)
        n = x.shape[-1]
        x_f = torch.fft.rfft(x, dim=-1)
        m = min(self.modes, x_f.shape[-1])
        out_f = torch.zeros(x.shape[0], self.out_channels, x_f.shape[-1],
                            dtype=x_f.dtype, device=x.device)
        # einsum over complex weights: (b, i, m) x (i, o, m) -> (b, o, m)
        prod = torch.einsum("bim,iom->bom", x_f[..., :m],
                            self.weights[..., :m].to(x_f.dtype))
        out_f[..., :m] = prod
        return torch.fft.irfft(out_f, n=n, dim=-1)


class FNO1d(nn.Module):
    """1D Fourier Neural Operator.

    Parameters
    ----------
    in_channels:  input function channels (e.g., 1 for a scalar forcing).
    out_channels: output function channels.
    width:       hidden representation width (the layer analyzed by the
                 rank/SAE experiments).
    modes:       number of retained Fourier modes per spectral conv.
    n_layers:    number of spectral-conv blocks.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 64,
        modes: int = 12,
        n_layers: int = 4,
        activation: str = "gelu",
    ):
        super().__init__()
        self.width = width
        self.n_layers = n_layers
        self.lift = nn.Linear(in_channels, width)
        self.spectral_layers = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(n_layers)]
        )
        self.skips = nn.ModuleList(
            [nn.Linear(width, width) for _ in range(n_layers)]
        )
        self.project1 = nn.Linear(width, width)
        self.project2 = nn.Linear(width, out_channels)
        self.act = nn.GELU() if activation == "gelu" else F.gelu

        # Named accessors so the interchange/hook machinery works uniformly.
        # Block states = spectral layer + skip outputs (post-activation).
        self.acts = nn.ModuleList([nn.Identity() for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, return_intermediates: bool = False):
        """x: (batch, length, in_channels) -> (batch, length, out_channels).

        Returns intermediates (post-activation block states) when requested —
        each has shape (batch, length, width), i.e. the FUNCTION-SPACE
        representation this architecture is designed around.

        Block states are routed through ``self.acts[i]`` (Identity modules)
        so forward hooks registered on them fire — the same convention the
        MLP uses (``model.acts[layer_index]``), letting the intervention /
        interchange machinery operate on FNOs uniformly.
        """
        x = self.lift(x)  # (b, L, width)
        intermediates: List[torch.Tensor] = []
        for i in range(self.n_layers):
            x_t = x.permute(0, 2, 1)               # (b, width, L)
            x_spec = self.spectral_layers[i](x_t)  # (b, width, L)
            x_spec = x_spec.permute(0, 2, 1)       # (b, L, width)
            x = self.act(x_spec + self.skips[i](x))
            x = self.acts[i](x)                    # hook point (Identity)
            if return_intermediates:
                intermediates.append(x)
        x = self.act(self.project1(x))
        out = self.project2(x)
        if return_intermediates:
            return out, intermediates
        return out


def green_function_dataset(
    n_samples: int,
    grid_points: int = 64,
    kernel_scale: float = 1.0,
    n_kernel_params: int = 8,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
    nonlinear: bool = True,
):
    """Parametric Green's-function regression dataset (operator learning).

    Inputs  a(x) = sum_j c_j * exp(-(x - mu_j)^2 / (2 s_j^2))
            — random Gaussian-mixture forcing functions parameterized by
            (c_j, mu_j, s_j) triples; the input space is genuinely
            n_kernel_params*3-dimensional in parameters, i.e. FUNCTION SPACE.
    Outputs u(x) = solution of the screened Poisson / Helmholtz-type
            integral  u = G * a  with G(k) = 1/(k^2 + m^2), computed
            spectrally with mass m ~ kernel_scale, followed by the
            NONLINEAR output map  u <- tanh(2 u)  when ``nonlinear``.

    The nonlinearity is essential for the regime-boundary experiment: the
    linear screened-Poisson operator is solved exactly by one spectral
    layer (train MSE -> 0 in a few hundred steps), which produces
    degenerate representations.  tanh(2u) forces the network to combine
    modes nonlinearly and keeps test error away from machine zero.

    Returns (a, u) tensors of shape (n_samples, grid_points, 1).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.linspace(0.0, 1.0, grid_points)
    # Random mixture parameters per sample.
    c = torch.randn(n_samples, n_kernel_params, generator=g) * kernel_scale
    mu = torch.rand(n_samples, n_kernel_params, generator=g)
    s = 0.05 + 0.15 * torch.rand(n_samples, n_kernel_params, generator=g)

    # a(x) = sum_j c_j exp(-(x-mu_j)^2 / 2 s_j^2): (n, L)
    a = torch.zeros(n_samples, grid_points)
    for j in range(n_kernel_params):
        a += c[:, j:j + 1] * torch.exp(
            -((x[None, :] - mu[:, j:j + 1]) ** 2) / (2 * s[:, j:j + 1] ** 2))

    # Green's operator applied spectrally (periodic convention).
    k = 2 * torch.pi * torch.fft.rfftfreq(grid_points, d=1.0 / grid_points)
    mass = kernel_scale
    response = 1.0 / (k ** 2 + mass ** 2)
    response[0] = 0.0  # zero mode: no DC response
    a_f = torch.fft.rfft(a, dim=-1)
    u_f = a_f * response[None, :]
    u = torch.fft.irfft(u_f, n=grid_points, dim=-1)

    if nonlinear:
        # Strong nonlinearity so a single spectral layer cannot solve the
        # task: square the energy in a few bands, then tanh.  This forces
        # genuine mode-mixing (the operator is no longer diagonal in the
        # Fourier basis).
        u_f = torch.fft.rfft(u, dim=-1)
        mag = u_f.abs()
        phase = torch.angle(u_f)
        # Mode mixing: each output mode gets a contribution from the
        # magnitude of neighboring modes (a simple non-diagonal map).
        mixed = 0.5 * torch.roll(mag, 1, dims=-1) + 0.5 * mag
        u_f = mixed * torch.exp(1j * phase)
        u = torch.fft.irfft(u_f, n=grid_points, dim=-1)
        u = torch.tanh(3.0 * u)

    return (a.unsqueeze(-1).to(device),
            u.unsqueeze(-1).to(device))
