"""PCA directions as a low-rank interpretability baseline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PCABasis:
    mean: torch.Tensor
    components: torch.Tensor
    explained_variance: torch.Tensor

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.components.shape[1])

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        return (activations - self.mean) @ self.components.T

    def decode(self, coefficients: torch.Tensor) -> torch.Tensor:
        return coefficients @ self.components + self.mean


def fit_pca(activations: torch.Tensor, n_components: int) -> PCABasis:
    """Fit a centered PCA basis using torch SVD."""
    if activations.ndim != 2:
        raise ValueError("activations must have shape (samples, width)")
    if not 1 <= n_components <= min(activations.shape):
        raise ValueError("n_components must fit within the activation matrix")
    mean = activations.mean(dim=0, keepdim=True)
    centered = activations - mean
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    variance = singular_values.square() / max(activations.shape[0] - 1, 1)
    return PCABasis(mean, vh[:n_components], variance[:n_components])


def reconstruction_error(data: torch.Tensor, basis: PCABasis) -> float:
    """Evaluate a train-fitted PCA basis without refitting on evaluation data."""
    reconstructed = basis.decode(basis.encode(data))
    return float((data - reconstructed).square().mean())


def pca_component_intervention(
    activations: torch.Tensor,
    basis: PCABasis,
    component_idx: int,
    alpha: float = 0.0,
) -> torch.Tensor:
    """Decode activations after setting one PCA coefficient to alpha times itself."""
    if not 0 <= component_idx < basis.n_components:
        raise IndexError("component_idx is outside the fitted PCA basis")
    coefficients = basis.encode(activations).clone()
    coefficients[:, component_idx] *= alpha
    return basis.decode(coefficients)


def pca_intervention_hook(
    basis: PCABasis,
    component_idx: int,
    alpha: float = 0.0,
):
    """Return a forward-hook function that intervenes on a hidden activation."""
    def hook(_module, _inputs, output):
        return pca_component_intervention(output, basis, component_idx, alpha)
    return hook


def compare_pca_vs_sae_causal_specificity(
    pca_results: list[Dict], sae_results: list[Dict]
) -> Dict[str, float]:
    """Summarize paired causal-strength and specificity differences."""
    pca_strength = np.asarray([row["causal_strength"] for row in pca_results], dtype=float)
    sae_strength = np.asarray([row["causal_strength"] for row in sae_results], dtype=float)
    pca_specificity = np.asarray([row["specificity"] for row in pca_results], dtype=float)
    sae_specificity = np.asarray([row["specificity"] for row in sae_results], dtype=float)
    if not (len(pca_strength) and len(pca_strength) == len(sae_strength)):
        raise ValueError("PCA and SAE results must have equal nonzero length")
    return {
        "mean_pca_causal_strength": float(pca_strength.mean()),
        "mean_sae_causal_strength": float(sae_strength.mean()),
        "mean_pca_specificity": float(pca_specificity.mean()),
        "mean_sae_specificity": float(sae_specificity.mean()),
        "pca_minus_sae_strength": float((pca_strength - sae_strength).mean()),
        "pca_minus_sae_specificity": float((pca_specificity - sae_specificity).mean()),
    }
