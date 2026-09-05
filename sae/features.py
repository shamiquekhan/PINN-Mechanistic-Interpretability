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
        "activation_frequency":  (z_np > 0).mean(axis=0).tolist(),   # fraction of samples
        "mean_activation":       z_np.mean(axis=0).tolist(),
        "max_activation":        z_np.max(axis=0).tolist(),
        "std_activation":        z_np.std(axis=0).tolist(),
        "l1_norm":               np.abs(z_np).mean(axis=0).tolist(),
    }


# ---------------------------------------------------------------------------
# Physical association analysis
# ---------------------------------------------------------------------------

def associate_features_with_metrics(
    sae: SparseAutoencoder,
    activation_data: torch.Tensor,
    metric_arrays: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, List[float]]:
    """Compute Pearson correlation between each SAE feature and physical metrics.

    Parameters
    ----------
    activation_data: (N, D) tensor of activations.
    metric_arrays:   Dict mapping metric name -> (N,) array (e.g. pde_loss per sample).

    Returns
    -------
    Dict mapping metric_name -> list of correlation coefficients (length = latent_dim).
    """
    sae.eval()
    with torch.no_grad():
        z, _ = sae(activation_data.to(device))
    z_np = z.cpu().numpy()   # (N, latent_dim)

    correlations: Dict[str, List[float]] = {}
    for metric_name, metric_vals in metric_arrays.items():
        if len(metric_vals) != z_np.shape[0]:
            continue
        corrs = []
        for feat_idx in range(z_np.shape[1]):
            feat_vals = z_np[:, feat_idx]
            if feat_vals.std() < 1e-10:
                corrs.append(0.0)
            else:
                corr = float(np.corrcoef(feat_vals, metric_vals)[0, 1])
                corrs.append(corr if np.isfinite(corr) else 0.0)
        correlations[metric_name] = corrs

    return correlations


# ---------------------------------------------------------------------------
# Cross-seed feature matching
# ---------------------------------------------------------------------------

def match_features_across_saes(
    sae_list: List[SparseAutoencoder],
    threshold: float = 0.8,
) -> List[Dict]:
    """Match features across multiple SAE seeds by decoder cosine similarity.

    Returns
    -------
    List of feature families, each being a dict:
        {
          "sae_0_feature": int,    # feature index in first SAE
          "matches": [             # matches in other SAEs
              {"sae_idx": int, "feature_idx": int, "cosine": float}
          ],
          "stable": bool           # True if matched in all SAEs above threshold
        }
    """
    if len(sae_list) < 2:
        return []

    # Normalise all decoder weight matrices
    def get_decoder_cols(sae: SparseAutoencoder) -> np.ndarray:
        W = sae.W_d.weight.detach().cpu().numpy()   # (input_dim, latent_dim)
        norms = np.linalg.norm(W, axis=0, keepdims=True).clip(min=1e-8)
        return (W / norms).T  # (latent_dim, input_dim)

    decoders = [get_decoder_cols(s) for s in sae_list]
    base = decoders[0]  # (L0, D)
    families = []

    for feat_i in range(base.shape[0]):
        family: Dict = {"sae_0_feature": feat_i, "matches": [], "stable": True}
        for sae_j_idx, other in enumerate(decoders[1:], start=1):
            # Cosine similarity between feat_i in base and all features in other
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
    "supporting_spatial_evidence": "",
    "supporting_temporal_evidence": "",
    "candidate_failure_mode": "unknown",
    "predicted_intervention_direction": "unknown",
    "confounds_tested": [],
    "causal_test_result": "not_tested",
    "confidence_tier": "candidate",    # candidate | verified | rejected | unresolved
}


def build_feature_dictionary(
    feature_stats: Dict,
    correlations: Dict[str, List[float]],
    families: List[Dict],
    layer_name: str,
    sae_version: str,
    top_k: int = 10,
) -> List[Dict]:
    """Assemble the Physics-Feature Dictionary from computed stats.

    Returns
    -------
    List of feature annotation dicts, one per feature in the SAE,
    sorted by activation frequency descending.
    """
    freq = np.array(feature_stats.get("activation_frequency", []))
    n_features = len(freq)

    # Build stable-feature lookup
    stable_ids = set()
    for fam in families:
        if fam.get("stable"):
            stable_ids.add(fam["sae_0_feature"])

    # Identify top correlated metrics for each feature
    corr_summary: Dict[int, Dict[str, float]] = {i: {} for i in range(n_features)}
    for metric, corr_list in correlations.items():
        for i, c in enumerate(corr_list):
            corr_summary[i][metric] = c

    dictionary: List[Dict] = []
    for i in range(n_features):
        entry = dict(_ANNOTATION_TEMPLATE)
        entry["feature_id"]   = i
        entry["layer"]        = layer_name
        entry["sae_version"]  = sae_version
        entry["activation_frequency"] = float(freq[i]) if i < len(freq) else 0.0
        entry["correlations"] = corr_summary.get(i, {})
        entry["stable_across_seeds"] = i in stable_ids

        # Heuristic interpretation hints (placeholder — human annotator fills later)
        best_metric, best_corr = max(
            entry["correlations"].items(),
            key=lambda kv: abs(kv[1]),
            default=("none", 0.0),
        )
        entry["candidate_interpretation"] = (
            f"correlates with {best_metric} (r={best_corr:.3f})"
            if best_metric != "none" else "unassigned"
        )
        dictionary.append(entry)

    # Sort by activation frequency
    dictionary.sort(key=lambda e: e["activation_frequency"], reverse=True)
    return dictionary


def save_feature_dictionary(dictionary: List[Dict], out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dictionary, f, indent=2)
    print(f"Physics-Feature Dictionary saved: {out_path} ({len(dictionary)} features)")
