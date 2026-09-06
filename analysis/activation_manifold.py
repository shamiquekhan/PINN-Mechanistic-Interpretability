"""Activation-manifold diagnostics for PINN hidden states.

This module keeps two notions separate:

* A smooth map from a d-dimensional input domain has local tangent rank at
  most d (where the Jacobian exists).
* The covariance matrix of sampled points on that manifold is not generally
  rank-bounded by d. A curved one-dimensional manifold can have full affine
  span, so participation ratio remains an empirical diagnostic.

The distinction matters for the PINN study: low observed PR is evidence about
the sampled representation, not a theorem forced by input dimension alone.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def local_tangent_rank_bound(input_dim: int) -> int:
    """Return the Jacobian-rank bound for a differentiable input map."""
    if not isinstance(input_dim, (int, np.integer)) or input_dim < 0:
        raise ValueError("input_dim must be a non-negative integer")
    return int(input_dim)


def affine_covariance_rank_bound(n_samples: int, width: int) -> int:
    """Return the generic finite-sample covariance-rank bound.

    This is the only unconditional rank bound available from sample count and
    activation width alone: centering removes at most one degree of freedom.
    """
    if n_samples < 0 or width < 0:
        raise ValueError("n_samples and width must be non-negative")
    return min(int(width), max(int(n_samples) - 1, 0))


def local_tangent_rank(
    activations: np.ndarray,
    *,
    coordinates: Optional[np.ndarray] = None,
    tolerance: float = 1e-6,
) -> int:
    """Estimate local tangent rank from ordered activation samples.

    For scalar coordinates this uses first differences. For multi-dimensional
    coordinates it uses a least-squares local Jacobian at each interior point.
    The samples must be ordered consistently with the coordinate grid.
    """
    values = np.asarray(activations, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("activations must have shape (n_samples, width)")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    if coordinates is None:
        differences = np.diff(values, axis=0)
        singular_values = np.linalg.svd(differences, compute_uv=False)
    else:
        coords = np.asarray(coordinates, dtype=np.float64)
        if coords.ndim == 1:
            if len(coords) != len(values):
                raise ValueError("coordinates and activations must have equal length")
            differences = np.diff(values, axis=0)
            steps = np.diff(coords)
            if np.any(np.isclose(steps, 0.0)):
                raise ValueError("coordinates must be strictly separated")
            derivatives = differences / steps[:, None]
            # A scalar input has a one-column Jacobian at every point. Its
            # local rank is therefore 1 whenever the derivative is nonzero,
            # even when the tangent direction rotates along a curved path.
            return int(np.any(np.linalg.norm(derivatives, axis=1) > 0.0))
        elif coords.ndim == 2 and coords.shape[0] == values.shape[0]:
            # Centered local neighborhoods provide a coordinate-to-activation
            # least-squares Jacobian without requiring a symbolic model.
            jacobians = []
            for index in range(len(values)):
                distances = np.linalg.norm(coords - coords[index], axis=1)
                neighbors = np.argsort(distances)[1 : min(len(values), 2 * coords.shape[1] + 2)]
                delta_coords = coords[neighbors] - coords[index]
                delta_values = values[neighbors] - values[index]
                jacobians.append(np.linalg.lstsq(delta_coords, delta_values, rcond=None)[0])
            singular_values = np.linalg.svd(np.concatenate(jacobians), compute_uv=False)
        else:
            raise ValueError("coordinates must have shape (n_samples,) or (n_samples, input_dim)")

    if singular_values.size == 0 or singular_values[0] <= 0:
        return 0
    return int(np.sum(singular_values > singular_values[0] * tolerance))


def pca_projection(activations: np.ndarray, n_components: int = 3) -> np.ndarray:
    """Project activations onto their leading centered PCA components."""
    values = np.asarray(activations, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("activations must have shape (n_samples, width)")
    if n_components < 1:
        raise ValueError("n_components must be positive")
    centered = values - values.mean(axis=0, keepdims=True)
    _, _, right_singular_vectors = np.linalg.svd(centered, full_matrices=False)
    n_available = min(n_components, right_singular_vectors.shape[0])
    projected = centered @ right_singular_vectors[:n_available].T
    if n_available < n_components:
        projected = np.pad(
            projected,
            ((0, 0), (0, n_components - n_available)),
        )
    return projected


def visualize_activation_manifold(
    activations: np.ndarray,
    output_path: Path,
    *,
    color_by: Optional[np.ndarray] = None,
    title: str = "PINN activation manifold (PCA)",
) -> Path:
    """Save a 3D PCA projection, padding lower-dimensional data with zeros."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    projected = pca_projection(activations, n_components=min(3, np.asarray(activations).shape[1]))
    if projected.shape[1] < 3:
        projected = np.pad(projected, ((0, 0), (0, 3 - projected.shape[1])))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(7, 6))
    axis = figure.add_subplot(111, projection="3d")
    colors = color_by if color_by is not None else np.arange(len(projected))
    scatter = axis.scatter(projected[:, 0], projected[:, 1], projected[:, 2], c=colors, s=8)
    axis.set(xlabel="PC1", ylabel="PC2", zlabel="PC3", title=title)
    if color_by is not None:
        figure.colorbar(scatter, ax=axis, shrink=0.7, label="color value")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path