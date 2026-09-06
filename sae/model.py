"""
Sparse Autoencoder (SAE) for PINN activation discovery.

Architecture options:
  1. TopK SAE (Gao et al., 2024):
       z = TopK_k(ReLU(W_e(a - b_a)))   # Exact k-sparse latents without L1 shrinkage
       â = W_d z + b_d                   # Decoder reconstruction
       L = ‖a - â‖²                      # Pure MSE loss (k controls sparsity directly)

  2. Standard ReLU + L1 SAE (Anthropic 2023):
       z = ReLU(W_e(a - b_a))
       â = W_d z + b_d
       L = ‖a - â‖² + β‖z‖₁
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_topk(x: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the top-k activation values along dim=-1, zeroing the rest."""
    if k >= x.shape[-1]:
        return x
    topk_vals, topk_indices = torch.topk(x, k, dim=-1)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask.scatter_(-1, topk_indices, True)
    return torch.where(mask, x, torch.zeros_like(x))


class SparseAutoencoder(nn.Module):
    """Overcomplete SAE for interpreting PINN hidden-layer activations.

    Parameters
    ----------
    input_dim:        Dimension of the activation vector (hidden layer width).
    latent_expansion: Overcomplete factor (latent_dim = input_dim * expansion).
    sparsity_coeff:   L1 sparsity penalty coefficient β (used if activation_mode == "relul1").
    activation_mode:  "topk" (Gao 2024) or "relul1" (legacy Anthropic 2023).
    topk:             k value for TopK activation (default 8).
    decoder_normalize: If True, normalise decoder columns to unit norm after each step.
    """

    def __init__(
        self,
        input_dim: int,
        latent_expansion: int = 4,
        sparsity_coeff: float = 1e-3,
        activation_mode: str = "topk",
        topk: int = 8,
        decoder_normalize: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = input_dim * latent_expansion
        self.sparsity_coeff = sparsity_coeff
        self.activation_mode = activation_mode
        self.topk = topk
        self.decoder_normalize = decoder_normalize

        # Pre-encoder bias (subtracted before encoding)
        self.b_a = nn.Parameter(torch.zeros(input_dim))
        # Encoder
        self.W_e = nn.Linear(input_dim, self.latent_dim, bias=True)
        # Decoder
        self.W_d = nn.Linear(self.latent_dim, input_dim, bias=True)

        self._init_weights()

        # Dead-feature tracking
        self.register_buffer("_feature_use", torch.zeros(self.latent_dim))
        self._steps_since_reset = 0

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_e.weight)
        nn.init.zeros_(self.W_e.bias)
        nn.init.xavier_uniform_(self.W_d.weight)
        nn.init.zeros_(self.W_d.bias)

    def _normalise_decoder(self):
        """Normalise each decoder column to unit norm (in-place)."""
        with torch.no_grad():
            norms = self.W_d.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
            self.W_d.weight.div_(norms)

    def encode(self, a: torch.Tensor) -> torch.Tensor:
        """Return sparse latent code z."""
        raw = F.relu(self.W_e(a - self.b_a))
        if self.activation_mode == "topk":
            return apply_topk(raw, k=self.topk)
        return raw

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.W_d(z)

    def forward(self, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (z, a_hat)."""
        z = self.encode(a)
        a_hat = self.decode(z)
        return z, a_hat

    def loss(self, a: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute SAE loss and return detailed breakdown."""
        z, a_hat = self(a)
        recon_loss = F.mse_loss(a_hat, a)

        if self.activation_mode == "topk":
            sparsity_loss = torch.tensor(0.0, device=a.device)
            total = recon_loss
        else:
            sparsity_loss = self.sparsity_coeff * z.abs().mean()
            total = recon_loss + sparsity_loss

        info = {
            "recon_loss": recon_loss.item(),
            "sparsity_loss": sparsity_loss.item(),
            "total_loss": total.item(),
            "mean_l0": (z > 0).float().sum(dim=-1).mean().item(),
            "mean_l1": z.abs().mean().item(),
        }
        return total, info

    def update_feature_use(self, z: torch.Tensor):
        """Call after each forward step to track which features ever activate."""
        with torch.no_grad():
            activated = (z > 0).float().sum(dim=0)
            self._feature_use += activated
            self._steps_since_reset += 1

    def dead_feature_mask(self, window: int = 200) -> torch.Tensor:
        """Return boolean mask of features that never activated."""
        return self._feature_use <= 0.0

    def reset_feature_use(self):
        self._feature_use.zero_()
        self._steps_since_reset = 0

    def dead_feature_fraction(self) -> float:
        return self.dead_feature_mask().float().mean().item()

    def save(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "latent_expansion": self.latent_dim // self.input_dim,
            "sparsity_coeff": self.sparsity_coeff,
            "activation_mode": self.activation_mode,
            "topk": self.topk,
            "decoder_normalize": self.decoder_normalize,
        }, path)

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "SparseAutoencoder":
        ck = torch.load(path, map_location=device, weights_only=False)
        has_mode_key = "activation_mode" in ck
        # Legacy checkpoints (trained with ReLU+L1) lack the activation_mode
        # key.  Defaulting them to topk silently retrains the SAE's *forward
        # semantics* on weights that were trained under different sparsity —
        # a checkpoint/code version-skew bug.  Infer the mode from the tag
        # instead: relul1-era checkpoints always carry a nonzero sparsity_coeff.
        if has_mode_key:
            mode = ck["activation_mode"]
        elif float(ck.get("sparsity_coeff", 0.0) or 0.0) > 0.0:
            mode = "relul1"
        else:
            mode = "topk"
        sae = cls(
            input_dim=ck["input_dim"],
            latent_expansion=ck.get("latent_expansion", 4),
            sparsity_coeff=ck.get("sparsity_coeff", 1e-3),
            activation_mode=mode,
            topk=ck.get("topk", 8),
            decoder_normalize=ck.get("decoder_normalize", True),
        ).to(device)
        sae.load_state_dict(ck["state_dict"])
        return sae

    def feature_summary(self) -> Dict:
        return {
            "latent_dim": self.latent_dim,
            "input_dim": self.input_dim,
            "activation_mode": self.activation_mode,
            "topk": self.topk,
            "sparsity_coeff": self.sparsity_coeff,
            "dead_feature_fraction": self.dead_feature_fraction(),
            "steps_tracked": self._steps_since_reset,
        }
