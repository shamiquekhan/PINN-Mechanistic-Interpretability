"""
Sparse Autoencoder (SAE) for PINN activation discovery.

Architecture:
    z = ReLU(W_e(a - b_a) + b_z)     # encoder: overcomplete sparse code
    â = W_d z + b_d                   # decoder: reconstruction
    L = ‖a - â‖² + β‖z‖₁             # loss: reconstruction + L1 sparsity
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """Overcomplete SAE for interpreting PINN hidden-layer activations.

    Parameters
    ----------
    input_dim:        Dimension of the activation vector (hidden layer width).
    latent_expansion: Overcomplete factor (latent_dim = input_dim * expansion).
    sparsity_coeff:   L1 sparsity penalty coefficient β.
    decoder_normalize: If True, normalise decoder columns to unit norm after each step.
    """

    def __init__(
        self,
        input_dim: int,
        latent_expansion: int = 4,
        sparsity_coeff: float = 1e-3,
        decoder_normalize: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = input_dim * latent_expansion
        self.sparsity_coeff = sparsity_coeff
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

    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(self, a: torch.Tensor) -> torch.Tensor:
        """Return sparse latent code z (all non-negative via ReLU)."""
        return F.relu(self.W_e(a - self.b_a))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.W_d(z)

    def forward(self, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (z, a_hat)."""
        z = self.encode(a)
        a_hat = self.decode(z)
        return z, a_hat

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(self, a: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute SAE loss and return detailed breakdown."""
        z, a_hat = self(a)
        recon_loss = F.mse_loss(a_hat, a)
        sparsity_loss = self.sparsity_coeff * z.abs().mean()
        total = recon_loss + sparsity_loss

        info = {
            "recon_loss": recon_loss.item(),
            "sparsity_loss": sparsity_loss.item(),
            "total_loss": total.item(),
            "mean_l0": (z > 0).float().mean().item(),
            "mean_l1": z.abs().mean().item(),
        }
        return total, info

    # ------------------------------------------------------------------
    # Dead-feature diagnostics
    # ------------------------------------------------------------------

    def update_feature_use(self, z: torch.Tensor):
        """Call after each forward to track which features ever activate."""
        with torch.no_grad():
            activated = (z > 0).float().sum(dim=0)
            self._feature_use += activated
            self._steps_since_reset += 1

    def dead_feature_mask(self, window: int = 200) -> torch.Tensor:
        """Return boolean mask of features that never activated in the window."""
        threshold = 0.0
        return self._feature_use <= threshold

    def reset_feature_use(self):
        self._feature_use.zero_()
        self._steps_since_reset = 0

    def dead_feature_fraction(self) -> float:
        return self.dead_feature_mask().float().mean().item()

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "latent_expansion": self.latent_dim // self.input_dim,
            "sparsity_coeff": self.sparsity_coeff,
            "decoder_normalize": self.decoder_normalize,
        }, path)

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "SparseAutoencoder":
        ck = torch.load(path, map_location=device, weights_only=False)
        sae = cls(
            input_dim=ck["input_dim"],
            latent_expansion=ck["latent_expansion"],
            sparsity_coeff=ck["sparsity_coeff"],
            decoder_normalize=ck["decoder_normalize"],
        ).to(device)
        sae.load_state_dict(ck["state_dict"])
        return sae

    def feature_summary(self) -> Dict:
        return {
            "latent_dim":            self.latent_dim,
            "input_dim":             self.input_dim,
            "sparsity_coeff":        self.sparsity_coeff,
            "dead_feature_fraction": self.dead_feature_fraction(),
            "steps_tracked":         self._steps_since_reset,
        }
