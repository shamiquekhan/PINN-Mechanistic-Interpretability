"""
Batch causal intervention experiments and causal scoring.

Implements the two causal regimes from the implementation plan:
  1. Inference-time counterfactuals (frozen weights)
  2. Training-time interventions (applied for a fixed window, then continued)

Causal score:
  E_T = ΔL_T^target − ΔL_T^control
  S   = |ΔL_T| / (ε + Σ_{j≠T} |ΔL_j|)   (specificity)
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from pinn.model import MLP
from pinn.pdes import BasePDE
from sae.model import SparseAutoencoder
from interventions.engine import (
    SAEInterventionHook,
    measure_intervention_effect,
)
from pinn_logging.io import append_jsonl, save_checkpoint


# ---------------------------------------------------------------------------
# Causal score
# ---------------------------------------------------------------------------

def compute_causal_score(
    target_effect: float,
    control_effect: float,
    non_target_effects: List[float],
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Compute causal strength (E_T) and specificity (S).

    Parameters
    ----------
    target_effect:      ΔL_T under targeted intervention.
    control_effect:     ΔL_T under matched control (unrelated feature / random).
    non_target_effects: [ΔL_j for j ≠ T] under targeted intervention.
    """
    E_T = target_effect - control_effect
    S = abs(target_effect) / (eps + sum(abs(e) for e in non_target_effects))
    return {
        "causal_strength":  E_T,
        "specificity":      S,
        "target_effect":    target_effect,
        "control_effect":   control_effect,
    }


# ---------------------------------------------------------------------------
# Batch inference-time experiments
# ---------------------------------------------------------------------------

def run_inference_interventions(
    model: MLP,
    sae: SparseAutoencoder,
    pde: BasePDE,
    device: torch.device,
    dtype: torch.dtype,
    candidate_features: List[int],
    target_loss: str = "pde",          # "pde" or "bc"
    layer_index: int = 1,
    n_interior: int = 256,
    alpha: float = 1.5,
    lambda_pde: float = 1.0,
    lambda_bc: float = 1.0,
    out_dir: Optional[Path] = None,
) -> List[Dict]:
    """Run targeted and control interventions for each candidate feature.

    For each feature, runs: ablate, amplify, unrelated_control, random_direction.
    Computes causal score for each.
    """
    results: List[Dict] = []

    for feat_idx in candidate_features:
        print(f"  Evaluating feature {feat_idx}...")

        # Targeted intervention (ablate as primary)
        targeted = measure_intervention_effect(
            model, sae, pde, device, dtype,
            feature_idx=feat_idx, mode="ablate",
            n_interior=n_interior, alpha=alpha,
            layer_index=layer_index,
            lambda_pde=lambda_pde, lambda_bc=lambda_bc,
        )

        # Amplified intervention
        amplified = measure_intervention_effect(
            model, sae, pde, device, dtype,
            feature_idx=feat_idx, mode="amplify",
            n_interior=n_interior, alpha=alpha,
            layer_index=layer_index,
        )

        # Control: unrelated feature
        ctrl_unrelated = measure_intervention_effect(
            model, sae, pde, device, dtype,
            feature_idx=feat_idx, mode="unrelated_control",
            n_interior=n_interior,
            layer_index=layer_index,
        )

        # Control: random direction
        ctrl_random = measure_intervention_effect(
            model, sae, pde, device, dtype,
            feature_idx=feat_idx, mode="random_direction",
            n_interior=n_interior,
            layer_index=layer_index,
        )

        # Causal scoring (ablate vs unrelated-feature control)
        if target_loss == "pde":
            t_eff   = targeted["delta_pde"]
            ctrl_eff = ctrl_unrelated["delta_pde"]
            nontarget = [targeted["delta_bc"]]
        else:
            t_eff   = targeted["delta_bc"]
            ctrl_eff = ctrl_unrelated["delta_bc"]
            nontarget = [targeted["delta_pde"]]

        score = compute_causal_score(t_eff, ctrl_eff, nontarget)

        entry = {
            "feature_idx":    feat_idx,
            "target_loss":    target_loss,
            "targeted_ablate": targeted,
            "targeted_amplify": amplified,
            "ctrl_unrelated": ctrl_unrelated,
            "ctrl_random":    ctrl_random,
            "causal_score":   score,
        }
        results.append(entry)

        if out_dir is not None:
            append_jsonl(Path(out_dir) / "inference_interventions.jsonl", entry)

    return results


# ---------------------------------------------------------------------------
# Training-time intervention
# ---------------------------------------------------------------------------

def run_training_intervention(
    model: MLP,
    sae: SparseAutoencoder,
    pde: BasePDE,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dtype: torch.dtype,
    feature_idx: int,
    mode: str,
    intervention_window: int,
    total_steps: int,
    cfg,
    out_dir: Path,
    layer_index: int = 1,
) -> List[Dict]:
    """Apply an SAE intervention for `intervention_window` training steps,
    then continue without intervention.

    This tests whether the intervention rescues or worsens the trajectory.
    """
    from pinn_logging.io import append_jsonl, save_checkpoint

    tcfg = cfg.training
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []

    # Set up intervention hook
    hook = SAEInterventionHook(sae, mode=mode, feature_idx=feature_idx)
    hook.register(model, layer_index)
    hook_active = True

    for step in range(total_steps):
        # Deactivate hook after window
        if hook_active and step >= intervention_window:
            hook.remove()
            hook_active = False

        x = pde.sample_interior(tcfg.interior_points, device, dtype)
        r = pde.residual(model, x)
        lp = (r ** 2).mean()
        bc_res = pde.boundary_residual(model)
        lb = bc_res.mean()
        loss = tcfg.lambda_pde * lp + tcfg.lambda_bc * lb

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rec = {
            "step": step,
            "loss": float(loss),
            "loss_pde": float(lp),
            "loss_bc": float(lb),
            "intervention_active": hook_active,
        }
        records.append(rec)
        append_jsonl(out_dir / "training_intervention.jsonl", rec)

    if hook_active:
        hook.remove()

    return records
