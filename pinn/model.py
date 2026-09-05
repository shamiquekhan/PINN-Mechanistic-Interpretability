"""
MLP PINN model with named layer access, Fourier embedding, explicit init,
and optional intermediate-activation returns.
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Fourier feature embedding (optional input preprocessing)
# ---------------------------------------------------------------------------

class FourierEmbedding(nn.Module):
    """Random Fourier feature embedding: maps x -> [sin(Bx), cos(Bx)]."""

    def __init__(self, input_dim: int, n_frequencies: int, scale: float = 1.0, seed: int = 0):
        super().__init__()
        gen = torch.Generator()
        gen.manual_seed(seed)
        B = torch.randn(input_dim, n_frequencies, generator=gen) * scale
        self.register_buffer("B", B)
        self.output_dim = 2 * n_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * x @ self.B          # (batch, n_freq)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

_ACTIVATIONS = {
    "tanh":  nn.Tanh,
    "relu":  nn.ReLU,
    "gelu":  nn.GELU,
    "silu":  nn.SiLU,
    "sin":   lambda: _SinActivation(),
}


class _SinActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class MLP(nn.Module):
    """Configurable fully-connected PINN network.

    Layers are stored as ``self.layers`` (nn.ModuleList of Linear modules)
    and ``self.acts`` (nn.ModuleList of activation modules), making hooks easy
    to register by index.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int],
        activation: str = "tanh",
        init: str = "xavier",
        fourier_embed: bool = False,
        fourier_n_freq: int = 32,
        fourier_scale: float = 1.0,
        fourier_seed: int = 0,
    ):
        super().__init__()
        self.fourier_embed_module: Optional[FourierEmbedding] = None
        effective_input = input_dim
        if fourier_embed:
            self.fourier_embed_module = FourierEmbedding(
                input_dim, fourier_n_freq, fourier_scale, fourier_seed
            )
            effective_input = self.fourier_embed_module.output_dim

        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(_ACTIVATIONS)}")

        dims = [effective_input] + list(hidden_layers) + [output_dim]

        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.acts = nn.ModuleList(
            [_ACTIVATIONS[activation]() for _ in range(len(hidden_layers))]
        )
        self.n_hidden = len(hidden_layers)
        self._apply_init(init, activation)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _apply_init(self, scheme: str, activation: str):
        nonlinearity = "tanh" if activation in ("tanh", "sin") else "relu"
        for layer in self.layers:
            if scheme == "xavier":
                nn.init.xavier_uniform_(layer.weight)
            elif scheme == "kaiming":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        if self.fourier_embed_module is not None:
            x = self.fourier_embed_module(x)

        intermediates: List[torch.Tensor] = []
        h = x
        for i, layer in enumerate(self.layers[:-1]):   # hidden layers
            h = layer(h)
            h = self.acts[i](h)
            if return_intermediates:
                intermediates.append(h)

        out = self.layers[-1](h)   # output layer (no activation)

        if return_intermediates:
            return out, intermediates
        return out

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def parameter_count(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def architecture_summary(self) -> Dict:
        return {
            "n_hidden_layers": self.n_hidden,
            "hidden_widths": [layer.out_features for layer in self.layers[:-1]],
            "output_dim": self.layers[-1].out_features,
            "activation": type(self.acts[0]).__name__ if self.acts else "none",
            "fourier_embed": self.fourier_embed_module is not None,
            **self.parameter_count(),
        }

    def hidden_layer_names(self) -> List[str]:
        return [f"layers.{i}" for i in range(self.n_hidden)]