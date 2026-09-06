"""
SAE feature statistics, physical association analysis, cross-seed matching,
and Physics-Feature Dictionary generation.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import torch

from sae.model import SparseAutoencoder
from pinn_logging.io import load_jsonl


# ---------------------------------------------------------------------------
# Feature activation statistics
# ---------------------------------------------------------------------------

def compute_feature_stats(
    sae: SparseAutoencoder,
    data: torch.Tensor,
    device: torch.device,
) -> Dict:
    """Compute per-feature activation statistics over a dataset.

    Returns
    -------
    Dict with arrays of shape (latent_dim,) for each stat.
    """
    sae.eval()
    data = data.to(device)
    with torch.no_grad():
        z, _ = sae(data)

    z_np = z.cpu().numpy()

    return {
        "activation_frequency": (z_np > 0).mean(axis=0).tolist(),
        "mean_activation": z_np.mean(axis=0).tolist(),
        "max_activation": z_np.max(axis=0).tolist(),
        "std_activation": z_np.std(axis=0).tolist(),
        "l1_norm": np.abs(z_np).mean(axis=0).tolist(),
    }


# ---------------------------------------------------------------------------
# Multi-View Physical Association Analysis
# ---------------------------------------------------------------------------

def associate_features_with_metrics(
    sae: SparseAutoencoder,
    activation_data: torch.Tensor,
    metric_arrays: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, List[float]]:
    """Compute Pearson correlation between each SAE feature and multi-view metrics.

    Supported Metric Views:
    - loss_pde: Interior PDE loss component.
    - loss_bc: Boundary condition loss component.
    - grad_cosine: Cosine similarity between loss gradients.
    - boundary_dist: Spatial distance to boundaries |x ∓ 1|.
    - high_freq_power: Spatial high-frequency power spectral density.
    - data_std: Activation standard deviation across probe points.

    Returns
    -------
    Dict mapping metric_name -> list of correlation coefficients (length = latent_dim).
    """
    sae.eval()
    with torch.no_grad():
        z, _ = sae(activation_data.to(device))
    z_np = z.cpu().numpy()

    correlations: Dict[str, List[float]] = {}
    for metric_name, metric_vals in metric_arrays.items():
        metric_vals = np.asarray(metric_vals, dtype=np.float64)
        if len(metric_vals) != z_np.shape[0]:
            continue
        corrs = []
        for feat_idx in range(z_np.shape[1]):
            feat_vals = z_np[:, feat_idx]
            if feat_vals.std() < 1e-10 or metric_vals.std() < 1e-10:
                corrs.append(0.0)
            else:
                corr = float(np.corrcoef(feat_vals, metric_vals)[0, 1])
                corrs.append(corr if np.isfinite(corr) else 0.0)
        correlations[metric_name] = corrs

    return correlations


def compare_associations_with_random_baseline(
    real_correlations: Dict[str, List[float]],
    latent_dim: int,
    n_samples: int,
    n_random_trials: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Compare real SAE feature correlations against random-direction baselines.

    Returns
    -------
    Dict mapping metric_name -> {
        "max_real_abs_corr": float,
        "max_random_abs_corr": float,
        "significance_ratio": float,
    }
    """
    results: Dict[str, Dict[str, float]] = {}
    rng = np.random.default_rng(42)

    for metric_name, real_corrs in real_correlations.items():
        if not real_corrs:
            continue
        max_real = float(np.max(np.abs(real_corrs)))

        # Simulate random direction latents
        random_maxes = []
        for _ in range(n_random_trials):
            rand_z = rng.standard_normal((n_samples, latent_dim))
            rand_z = np.maximum(0, rand_z)
            dummy_corrs = []
            for j in range(latent_dim):
                if rand_z[:, j].std() > 1e-10:
                    c = float(np.corrcoef(rand_z[:, j], rng.standard_normal(n_samples))[0, 1])
                    dummy_corrs.append(abs(c) if np.isfinite(c) else 0.0)
            random_maxes.append(np.max(dummy_corrs) if dummy_corrs else 0.0)

        max_rand = float(np.mean(random_maxes))
        ratio = max_real / max(max_rand, 1e-6)
        results[metric_name] = {
            "max_real_abs_corr": max_real,
            "max_random_abs_corr": max_rand,
            "significance_ratio": ratio,
        }

    return results


# ---------------------------------------------------------------------------
# Cross-Seed Feature Matching
# ---------------------------------------------------------------------------

def match_features_across_saes(
    sae_list: List[SparseAutoencoder],
    threshold: float = 0.8,
) -> List[Dict]:
    """Match features across multiple SAE seeds by decoder cosine similarity."""
    if len(sae_list) < 2:
        return []

    def get_decoder_cols(sae: SparseAutoencoder) -> np.ndarray:
        W = sae.W_d.weight.detach().cpu().numpy()
        norms = np.linalg.norm(W, axis=0, keepdims=True).clip(min=1e-8)
        return (W / norms).T

    decoders = [get_decoder_cols(s) for s in sae_list]
    base = decoders[0]
    families = []

    for feat_i in range(base.shape[0]):
        family: Dict = {"sae_0_feature": feat_i, "matches": [], "stable": True}
        for sae_j_idx, other in enumerate(decoders[1:], start=1):
            cosines = (other @ base[feat_i]).clip(-1, 1)
            best_match = int(np.argmax(cosines))
            best_cosine = float(cosines[best_match])
            family["matches"].append({
                "sae_idx": sae_j_idx,
                "feature_idx": best_match,
                "cosine": best_cosine,
            })
            if best_cosine < threshold:
                family["stable"] = False
        families.append(family)

    return families


# ---------------------------------------------------------------------------
# Physics-Feature Dictionary
# ---------------------------------------------------------------------------

_ANNOTATION_TEMPLATE = {
    "feature_id": None,
    "layer": None,
    "sae_version": None,
    "candidate_interpretation": "unassigned",
    "primary_associated_metric": "none",
    "primary_correlation": 0.0,
    "supporting_spatial_evidence": "",
    "supporting_temporal_evidence": "",
    "candidate_failure_mode": "unknown",
    "predicted_intervention_direction": "unknown",
    "confounds_tested": [],
    "causal_test_result": "not_tested",
    "confidence_tier": "candidate",
}


def build_feature_dictionary(
    feature_stats: Dict,
    correlations: Dict[str, List[float]],
    families: List[Dict],
    layer_name: str,
    sae_version: str,
) -> List[Dict]:
    """Assemble the Physics-Feature Dictionary from multi-view stats."""
    freq = np.array(feature_stats.get("activation_frequency", []))
    n_features = len(freq)

    stable_ids = set()
    for fam in families:
        if fam.get("stable"):
            stable_ids.add(fam["sae_0_feature"])

    corr_summary: Dict[int, Dict[str, float]] = {i: {} for i in range(n_features)}
    for metric, corr_list in correlations.items():
        for i, c in enumerate(corr_list):
            if i < n_features:
                corr_summary[i][metric] = float(c)

    dictionary: List[Dict] = []
    for i in range(n_features):
        entry = dict(_ANNOTATION_TEMPLATE)
        entry["feature_id"] = i
        entry["layer"] = layer_name
        entry["sae_version"] = sae_version
        entry["activation_frequency"] = float(freq[i]) if i < len(freq) else 0.0
        entry["correlations"] = corr_summary.get(i, {})
        entry["stable_across_seeds"] = i in stable_ids

        best_metric, best_corr = max(
            entry["correlations"].items(),
            key=lambda kv: abs(kv[1]),
            default=("none", 0.0),
        )
        entry["primary_associated_metric"] = best_metric
        entry["primary_correlation"] = best_corr
        entry["candidate_interpretation"] = (
            f"correlates with {best_metric} (r={best_corr:.3f})"
            if best_metric != "none" else "unassigned"
        )
        dictionary.append(entry)

    dictionary.sort(key=lambda e: e["activation_frequency"], reverse=True)
    return dictionary


def save_feature_dictionary(dictionary: List[Dict], out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dictionary, f, indent=2)
    print(f"Physics-Feature Dictionary saved: {out_path} ({len(dictionary)} features)")
