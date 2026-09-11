"""PCA causal battery — head-to-head causal comparison of PCA components vs SAE features.

Protocol (mirrors the SAE battery of `interventions.causal` exactly):

  For each candidate PCA component k and each held-out boundary-starvation
  checkpoint:
    1. Targeted ablation     — set PCA coefficient c_k <- alpha * c_k
    2. Unrelated control     — same ablation on a different component
    3. Random-direction ctrl — matched-norm perturbation of a random
                               direction in the PCA coefficient space
    4. Probe control         — supervised logistic-probe direction
                               trained on PCA coefficients (failure labels)

  Scoring is the SAME compute_causal_score / per_feature_causal_pvalues
  machinery used for the SAE battery, so the comparison is apples-to-apples.

Rationale
---------
k-matched PCA beats the TopK SAE 12x on reconstruction, and the
effective-rank analysis (PR ~ 1.3 / 64) says the activations are
low-rank.  The open causal question is whether PCA components — the
natural coordinates of a low-rank manifold — carry direction-specific
causal information about the PDE losses that SAE latents lack.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from analysis.pca_interpretability import PCABasis, fit_pca
from interventions.causal import (
    compute_causal_score,
    compute_bootstrap_ci,
    per_feature_causal_pvalues,
)
from pinn.model import MLP
from pinn.pdes import BasePDE


# ---------------------------------------------------------------------------
# PCA intervention hook (matches SAEInterventionHook semantics)
# ---------------------------------------------------------------------------

class PCAInterventionHook:
    """Forward hook that encodes a hidden activation through a frozen PCA
    basis, modifies one coefficient, and decodes back.

    Modes
    -----
    ablate_component    : c_k <- alpha * c_k      (alpha=0 → full ablation)
    amplify_component   : c_k <- alpha * c_k      (alpha>1 → amplification)
    unrelated_component : same ablation on a different (seed-chosen) component
    random_direction    : add a random unit perturbation in coefficient
                          space with per-row magnitude matched to |c_k|
    natural             : identity (baseline)
    """

    _MODES = {
        "natural", "ablate_component", "amplify_component",
        "unrelated_component", "random_direction", "probe_direction",
    }

    def __init__(
        self,
        basis: PCABasis,
        mode: str = "ablate_component",
        component_idx: Optional[int] = None,
        alpha: float = 0.0,
        random_seed: int = 0,
        probe_direction: Optional[torch.Tensor] = None,
        control_idx: Optional[int] = None,
    ):
        if mode not in self._MODES:
            raise ValueError(f"Unknown mode '{mode}'. Choose from {sorted(self._MODES)}")
        if mode == "probe_direction" and probe_direction is None:
            raise ValueError("probe_direction mode requires a probe_direction tensor")
        self.basis = basis
        self.mode = mode
        self.component_idx = component_idx
        self.alpha = alpha
        self.random_seed = random_seed
        # R6 control-matching correction: explicitly designated control
        # component (closest mean |coefficient|); None keeps sampled
        # behavior.
        self.control_idx = control_idx
        d = probe_direction.detach().cpu() if probe_direction is not None else None
        self.probe_direction = d / d.norm().clamp(min=1e-8) if d is not None else None
        self._hook = None

    def _pick_control(self, coeffs: torch.Tensor, k: int) -> int:
        """Control component for the control modes: the explicitly
        designated one (R6 correction) or a uniform non-target draw."""
        if self.control_idx is not None and self.control_idx != k:
            return self.control_idx
        rng = torch.Generator(device=coeffs.device)
        rng.manual_seed(self.random_seed)
        choices = [i for i in range(coeffs.shape[1]) if i != k]
        return choices[torch.randint(len(choices), (1,), generator=rng,
                                      device=coeffs.device).item()]

    def _intervene(self, coeffs: torch.Tensor) -> torch.Tensor:
        mode = self.mode
        k = self.component_idx
        if mode == "natural":
            return coeffs

        if mode in ("ablate_component", "amplify_component"):
            coeffs = coeffs.clone()
            coeffs[:, k] = coeffs[:, k] * self.alpha
            return coeffs

        if mode == "unrelated_component":
            coeffs = coeffs.clone()
            ctrl = self._pick_control(coeffs, k)
            coeffs[:, ctrl] = coeffs[:, ctrl] * self.alpha
            return coeffs

        if mode == "random_direction":
            # C3b fix (external review): the previous control INJECTED a
            # random direction while the target ABLATED — arms not matched
            # in kind. On a dense PCA basis every coefficient is "active",
            # so the matched-deletion control scales a random non-target
            # component by the same alpha the target arm uses.
            ctrl = self._pick_control(coeffs, k)
            coeffs = coeffs.clone()
            coeffs[:, ctrl] = coeffs[:, ctrl] * self.alpha
            return coeffs

        if mode == "probe_direction":
            # Supervised probe direction, per-row magnitude matched to |c_k|
            # (same scaling as the random control).
            per_row_mag = coeffs[:, k].abs() if k is not None else \
                coeffs.abs().mean(dim=1)
            d = self.probe_direction.to(coeffs.device, coeffs.dtype)
            return coeffs.clone() + d.unsqueeze(0) * per_row_mag.unsqueeze(1)

        return coeffs

    def _hook_fn(self, _module, _inputs, output: torch.Tensor):
        if self.mode == "natural":
            return output
        basis = self.basis
        device = output.device
        dtype = output.dtype
        mean = basis.mean.to(device=device, dtype=dtype)
        components = basis.components.to(device=device, dtype=dtype)
        coeffs = (output - mean) @ components.T
        coeffs = self._intervene(coeffs)
        return coeffs @ components + mean

    def register(self, model: MLP, layer_index: int):
        target = model.acts[layer_index]
        self._hook = target.register_forward_hook(self._hook_fn)

    def remove(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()


def pca_basis_to_device(basis: PCABasis, device: torch.device) -> PCABasis:
    """Return a copy of the basis with tensors on the target device."""
    return PCABasis(
        mean=basis.mean.to(device),
        components=basis.components.to(device),
        explained_variance=basis.explained_variance.to(device),
    )


def measure_pca_intervention_effect(
    model: MLP,
    basis: PCABasis,
    pde: BasePDE,
    device: torch.device,
    dtype: torch.dtype,
    component_idx: int,
    mode: str,
    n_interior: int = 256,
    alpha: float = 0.0,
    layer_index: int = 1,
    seed: Optional[int] = None,
    control_idx: Optional[int] = None,
) -> Dict:
    """Frozen-weight loss deltas for one PCA-component intervention.

    Mirrors `measure_intervention_effect` in interventions/engine.py:
    baseline = natural pass (PCA bypassed), intervened = hooked pass.
    control_idx: R6 correction — explicitly designated control
    component; None keeps the sampled behavior.
    """
    if seed is None:
        seed = 0
    model.eval()

    x = pde.sample_interior(n_interior, device, dtype, seed=seed)

    def _losses(hook: Optional[PCAInterventionHook]):
        if hook is not None:
            hook.register(model, layer_index)
        try:
            r = pde.residual(model, x)
            lp = float((r ** 2).mean())
            lb = float(pde.boundary_residual(model).mean())
        finally:
            if hook is not None:
                hook.remove()
        return lp, lb

    lp_base, lb_base = _losses(None)
    hook = PCAInterventionHook(
        basis, mode=mode, component_idx=component_idx,
        alpha=alpha, random_seed=seed, control_idx=control_idx,
    )
    lp_int, lb_int = _losses(hook)

    model.train()
    return {
        "component_idx": component_idx,
        "mode": mode,
        "layer_index": layer_index,
        "delta_pde": lp_int - lp_base,
        "delta_bc": lb_int - lb_base,
        "baseline_pde": lp_base,
        "baseline_bc": lb_base,
        "intervened_pde": lp_int,
        "intervened_bc": lb_int,
    }


def train_pca_probe_direction(
    basis: PCABasis,
    activations: torch.Tensor,
    labels: np.ndarray,
    device: torch.device,
    C: float = 1.0,
) -> torch.Tensor:
    """Supervised logistic probe on PCA coefficients; unit-norm direction.

    This is the PCA-space analogue of `train_failure_probe` in
    interventions/causal.py (which probes SAE latents).  The probe direction
    lives in PCA-coefficient space and is the strongest supervised LINEAR
    signal for predicting failure from activations.
    """
    from sklearn.linear_model import LogisticRegression

    coeffs = basis.encode(activations).cpu().numpy()
    probe = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")
    probe.fit(coeffs, labels)
    direction = torch.tensor(probe.coef_.flatten(), dtype=torch.float32,
                             device=device)
    return direction / direction.norm().clamp(min=1e-8)


def measure_pca_probe_effect(
    model: MLP,
    basis: PCABasis,
    probe_direction: torch.Tensor,
    pde: BasePDE,
    device: torch.device,
    dtype: torch.dtype,
    component_idx: int,
    n_interior: int = 256,
    layer_index: int = 1,
    seed: Optional[int] = None,
) -> Dict:
    """Perturb PCA coefficients along the supervised probe direction.

    Magnitude is matched per-row to |c_k| so the control is norm-matched to
    the targeted ablation, exactly like the SAE probe control.
    """
    if seed is None:
        seed = 0
    model.eval()

    x = pde.sample_interior(n_interior, device, dtype, seed=seed)

    def _losses(hook: Optional[PCAInterventionHook]):
        if hook is not None:
            hook.register(model, layer_index)
        try:
            r = pde.residual(model, x)
            lp = float((r ** 2).mean())
            lb = float(pde.boundary_residual(model).mean())
        finally:
            if hook is not None:
                hook.remove()
        return lp, lb

    lp_base, lb_base = _losses(None)
    hook = PCAInterventionHook(
        basis, mode="probe_direction", component_idx=component_idx,
        probe_direction=probe_direction,
    )
    lp_int, lb_int = _losses(hook)
    model.train()

    return {
        "component_idx": component_idx,
        "mode": "probe_direction",
        "layer_index": layer_index,
        "delta_pde": lp_int - lp_base,
        "delta_bc": lb_int - lb_base,
        "baseline_pde": lp_base,
        "baseline_bc": lb_base,
        "intervened_pde": lp_int,
        "intervened_bc": lb_int,
    }


# ---------------------------------------------------------------------------
# Battery runner
# ---------------------------------------------------------------------------

def run_pca_causal_battery(
    model: MLP,
    basis: PCABasis,
    pde: BasePDE,
    device: torch.device,
    dtype: torch.dtype,
    candidate_components: List[int],
    probe_direction: torch.Tensor,
    target_loss: str = "pde",
    layer_index: int = 1,
    n_interior: int = 256,
    alpha: float = 0.0,
    seed: int = 0,
) -> List[Dict]:
    """Run targeted + 3-control interventions for each candidate component."""
    results: List[Dict] = []
    for comp in candidate_components:
        targeted = measure_pca_intervention_effect(
            model, basis, pde, device, dtype,
            component_idx=comp, mode="ablate_component",
            n_interior=n_interior, alpha=alpha, layer_index=layer_index,
            seed=seed,
        )
        amplified = measure_pca_intervention_effect(
            model, basis, pde, device, dtype,
            component_idx=comp, mode="amplify_component",
            n_interior=n_interior, alpha=1.5 if alpha == 0.0 else 1.5,
            layer_index=layer_index, seed=seed,
        )
        ctrl_unrelated = measure_pca_intervention_effect(
            model, basis, pde, device, dtype,
            component_idx=comp, mode="unrelated_component",
            n_interior=n_interior, alpha=alpha, layer_index=layer_index,
            seed=seed,
        )
        ctrl_random = measure_pca_intervention_effect(
            model, basis, pde, device, dtype,
            component_idx=comp, mode="random_direction",
            n_interior=n_interior, layer_index=layer_index, seed=seed,
        )
        ctrl_probe = measure_pca_probe_effect(
            model, basis, probe_direction, pde, device, dtype,
            component_idx=comp, n_interior=n_interior,
            layer_index=layer_index, seed=seed,
        )

        if target_loss == "pde":
            t_eff = targeted["delta_pde"]
            c_u = ctrl_unrelated["delta_pde"]
            c_r = ctrl_random["delta_pde"]
            c_p = ctrl_probe["delta_pde"]
            nontarget = [targeted["delta_bc"]]
        else:
            t_eff = targeted["delta_bc"]
            c_u = ctrl_unrelated["delta_bc"]
            c_r = ctrl_random["delta_bc"]
            c_p = ctrl_probe["delta_bc"]
            nontarget = [targeted["delta_pde"]]

        score = compute_causal_score(
            target_effect=t_eff,
            control_unrelated_effect=c_u,
            control_random_effect=c_r,
            control_probe_effect=c_p,
            non_target_effects=nontarget,
        )
        results.append({
            "feature_idx": comp,   # same key as SAE rows for shared scoring
            "component_idx": comp,
            "target_loss": target_loss,
            "targeted_ablate": targeted,
            "targeted_amplify": amplified,
            "ctrl_unrelated": ctrl_unrelated,
            "ctrl_random": ctrl_random,
            "ctrl_probe": ctrl_probe,
            "causal_score": score,
        })
    return results


def summarize_battery(all_results: List[Dict]) -> Dict:
    """Aggregate a battery (PCA or SAE rows) with CIs, MC corrections, signs."""
    if not all_results:
        return {"error": "no results"}
    strengths = [r["causal_score"]["causal_strength"] for r in all_results]
    specificities = [r["causal_score"]["specificity"] for r in all_results]
    verified = 0
    for r in all_results:
        cs = r["causal_score"]
        t = abs(cs["target_effect"])
        beats_all = (
            t > abs(cs["control_unrelated_effect"])
            and t > abs(cs["control_random_effect"])
            and t > abs(cs["control_probe_effect"])
        )
        r["beats_all_controls"] = bool(beats_all)
        verified += int(beats_all)
    mc = per_feature_causal_pvalues(all_results)
    signs = [1 if r["causal_score"]["target_effect"] > 0 else -1
             for r in all_results]
    probe_beats = sum(
        1 for r in all_results
        if abs(r["causal_score"]["control_probe_effect"])
        > abs(r["causal_score"]["target_effect"])
    )
    return {
        "n_evaluations": len(all_results),
        "n_beats_all_controls": verified,
        "verification_rate": verified / len(all_results),
        "causal_strength_ci": compute_bootstrap_ci(strengths),
        "specificity_ci": compute_bootstrap_ci(specificities),
        "sign_diagnostic": {
            "n_target_delta_positive": sum(1 for s in signs if s > 0),
            "n_target_delta_negative": sum(1 for s in signs if s < 0),
        },
        "probe_control": {
            "n_probe_beats_target": probe_beats,
            "probe_beats_target_rate": probe_beats / len(all_results),
        },
        "multiple_comparisons": {
            "n_features_tested": mc["n_features_tested"],
            "bonferroni_survivors": mc["bonferroni"]["survivors"],
            "bonferroni_n_survivors": mc["bonferroni"]["n_survivors"],
            "bh_fdr_survivors": mc["bh_fdr"]["survivors"],
            "bh_fdr_n_survivors": mc["bh_fdr"]["n_survivors"],
            "chance_expected_hits_at_alpha": mc["chance_expected_hits_at_alpha"],
            "per_feature": mc["per_feature"],
        },
    }


def compare_pca_vs_sae(pca_rows: List[Dict], sae_rows: List[Dict]) -> Dict:
    """Head-to-head causal comparison matched on (feature-slot, run).

    Both row lists must contain a "run" key and a feature index
    ("feature_idx").  Rows are paired on (run, slot position); the SAE
    feature identity within a slot may differ from the PCA component
    identity — the comparison is at the method level (top-8 candidates).
    """
    if not pca_rows or not sae_rows:
        return {"error": "empty battery"}
    # Pair by (run, slot position): both batteries iterate runs outer and
    # 8 candidates inner, so positional pairing aligns method-level slots.
    n = min(len(pca_rows), len(sae_rows))
    pairs = []
    for i in range(n):
        p, s = pca_rows[i], sae_rows[i]
        pairs.append({
            "run": p.get("run", s.get("run", "?")),
            "pca_causal_strength": p["causal_score"]["causal_strength"],
            "sae_causal_strength": s["causal_score"]["causal_strength"],
            "pca_specificity": p["causal_score"]["specificity"],
            "sae_specificity": s["causal_score"]["specificity"],
            "pca_target_effect": p["causal_score"]["target_effect"],
            "sae_target_effect": s["causal_score"]["target_effect"],
            "pca_max_control": p["causal_score"]["max_control_effect"],
            "sae_max_control": s["causal_score"]["max_control_effect"],
            "pca_beats_all": p.get("beats_all_controls", False),
            "sae_beats_all": s.get("beats_all_controls", False),
        })
    pca_strength = np.array([pp["pca_causal_strength"] for pp in pairs])
    sae_strength = np.array([pp["sae_causal_strength"] for pp in pairs])
    diff = pca_strength - sae_strength
    return {
        "n_pairs": len(pairs),
        "mean_pca_causal_strength": float(pca_strength.mean()),
        "mean_sae_causal_strength": float(sae_strength.mean()),
        "pca_minus_sae_causal_strength": {
            **compute_bootstrap_ci(diff.tolist()),
        },
        "n_pca_beats_all": sum(1 for pp in pairs if pp["pca_beats_all"]),
        "n_sae_beats_all": sum(1 for pp in pairs if pp["sae_beats_all"]),
        "pairs": pairs,
    }
