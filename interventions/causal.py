"""
Batch causal intervention experiments and causal scoring.

Implements the causal scoring protocol with 3 mandatory controls:
  1. Unrelated Feature Control
  2. Random Direction Control
  3. Linear Probe Direction Control

Causal strength & specificity formulas:
  E_T = ΔL_T^target − max(ΔL_T^unrelated, ΔL_T^random, ΔL_T^probe)
  S_spec = |ΔL_T^target| / (ε + Σ_{j≠T} |ΔL_j^target|)
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from pinn.model import MLP
from pinn.pdes import BasePDE
from sae.model import SparseAutoencoder
from interventions.engine import (
    SAEInterventionHook,
    measure_intervention_effect,
    measure_probe_direction_effect,
)
from pinn_logging.io import append_jsonl, save_checkpoint


# ---------------------------------------------------------------------------
# Extended Causal score with 3 Controls & Bootstrap CIs
# ---------------------------------------------------------------------------

def compute_causal_score(
    target_effect: float,
    control_unrelated_effect: float,
    control_random_effect: float,
    control_probe_effect: float,
    non_target_effects: List[float],
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Compute causal strength (E_T) and specificity (S_spec) against 3 controls.

    Parameters
    ----------
    target_effect:             ΔL_T under targeted feature ablation.
    control_unrelated_effect:  ΔL_T under unrelated feature ablation.
    control_random_effect:     ΔL_T under random-direction ablation.
    control_probe_effect:      ΔL_T under linear-probe-direction ablation.
    non_target_effects:        [ΔL_j for j ≠ T] under targeted ablation.
    """
    max_control = max(control_unrelated_effect, control_random_effect, control_probe_effect)
    E_T = target_effect - max_control
    S_spec = abs(target_effect) / (eps + sum(abs(e) for e in non_target_effects))

    return {
        "causal_strength": E_T,
        "specificity": S_spec,
        "target_effect": target_effect,
        "max_control_effect": max_control,
        "control_unrelated_effect": control_unrelated_effect,
        "control_random_effect": control_random_effect,
        "control_probe_effect": control_probe_effect,
    }


def compute_bootstrap_ci(data: List[float], n_boot: int = 1000, ci: float = 0.95) -> Dict[str, float]:
    """Compute mean, std, and bootstrap percentile confidence interval for a list of values."""
    arr = np.array(data, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    if len(arr) == 1:
        return {"mean": float(arr[0]), "std": 0.0, "ci_lower": float(arr[0]), "ci_upper": float(arr[0])}

    rng = np.random.default_rng(42)
    boot_means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, 100 * alpha))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha)))

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci_lower": lower,
        "ci_upper": upper,
    }


# ---------------------------------------------------------------------------
# Multiple-comparison corrections for feature-level causal testing (v2.1)
# ---------------------------------------------------------------------------

def benjamini_hochberg(pvals: List[float], q: float = 0.05) -> Dict[str, object]:
    """Benjamini–Hochberg FDR procedure.

    Returns per-test adjusted p-values (step-up) and the indices of
    discoveries at FDR level q.
    """
    arr = np.asarray(pvals, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"adjusted_pvals": [], "discoveries": [], "q": q}
    order = np.argsort(arr)
    ranked = arr[order]
    adj = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    adjusted = np.empty(n)
    adjusted[order] = adj
    discoveries = [int(i) for i in range(n) if adjusted[i] <= q]
    return {
        "adjusted_pvals": adjusted.tolist(),
        "discoveries": discoveries,
        "q": q,
    }


def per_feature_causal_pvalues(feature_results: List[Dict]) -> Dict[str, object]:
    """Per-feature paired statistics for target > strongest control, with
    Bonferroni and BH-FDR multiple-comparison corrections.

    Evaluations are grouped by feature_idx; within each group, each run
    contributes one paired observation (|target effect|, |control effects|).
    Under the null (target no better than the strongest control), the
    per-run difference d = |target| - max|controls| is symmetric about 0, so
    a one-sided paired sign test across runs is exact.

    Corrections (m = number of FEATURES tested, not runs):
      - Bonferroni: alpha / m
      - BH-FDR step-up at q = 0.05
    """
    from collections import defaultdict
    groups: Dict[int, List[Dict]] = defaultdict(list)
    for r in feature_results:
        groups[int(r["feature_idx"])].append(r)

    feature_stats: List[Dict] = []
    pvals: List[float] = []
    for feat_idx, runs in sorted(groups.items()):
        diffs = []
        for r in runs:
            cs = r["causal_score"]
            t = abs(cs["target_effect"])
            controls = max(
                abs(cs["control_unrelated_effect"]),
                abs(cs["control_random_effect"]),
                abs(cs["control_probe_effect"]),
            )
            diffs.append(t - controls)
        diffs_arr = np.asarray(diffs, dtype=np.float64)
        n_pos = int((diffs_arr > 0).sum())
        n_neg = int((diffs_arr < 0).sum())
        n = n_pos + n_neg  # ties excluded
        # One-sided exact sign test: H1 target > controls.
        if n == 0:
            p = 1.0
        else:
            from math import comb
            p = sum(comb(n, k) for k in range(n_pos, n + 1)) / (2 ** n)
        feature_stats.append({
            "feature_idx": feat_idx,
            "n_runs": len(runs),
            "mean_diff": float(diffs_arr.mean()),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "sign_test_p": float(p),
        })
        pvals.append(p)

    n_tests = len(pvals)
    alpha = 0.05
    bonferroni_threshold = alpha / max(n_tests, 1)
    bonferroni_survivors = [i for i, p in enumerate(pvals)
                            if p <= bonferroni_threshold]
    bh = benjamini_hochberg(pvals, q=alpha)
    bh_survivors = bh["discoveries"]

    return {
        "n_features_tested": n_tests,
        "per_feature": feature_stats,
        "raw_pvals": pvals,
        "bonferroni": {
            "threshold": bonferroni_threshold,
            "survivors": [feature_stats[i]["feature_idx"]
                          for i in bonferroni_survivors],
            "n_survivors": len(bonferroni_survivors),
        },
        "bh_fdr": {
            "q": alpha,
            "adjusted_pvals": bh["adjusted_pvals"],
            "survivors": [feature_stats[i]["feature_idx"]
                          for i in bh_survivors],
            "n_survivors": len(bh_survivors),
        },
        "chance_expected_hits_at_alpha": alpha * n_tests,
    }


# ---------------------------------------------------------------------------
# Batch inference-time experiments across candidate features
# ---------------------------------------------------------------------------

def run_inference_interventions(
    model: MLP,
    sae: SparseAutoencoder,
    pde: BasePDE,
    device: torch.device,
    dtype: torch.dtype,
    candidate_features: List[int],
    target_loss: str = "pde",
    layer_index: int = 1,
    n_interior: int = 256,
    alpha: float = 1.5,
    lambda_pde: float = 1.0,
    lambda_bc: float = 1.0,
    out_dir: Optional[Path] = None,
    probe_directions: Optional[Dict[int, torch.Tensor]] = None,
) -> List[Dict]:
    """Run targeted and 3-control interventions for each candidate feature.

    The probe-direction control requires `probe_directions` — a dict mapping
    feature_idx -> direction tensor in the SAE latent space, typically from
    `train_failure_probe()` (supervised logistic probe on run-type labels).
    If absent, the probe control falls back to a random direction (and the
    entry is flagged `probe_control_valid=False`).
    """
    results: List[Dict] = []

    for feat_idx in candidate_features:
        print(f"  Evaluating feature {feat_idx}...")

        # 1. Targeted ablation
        targeted = measure_intervention_effect(
            model, sae, pde, device, dtype,
            feature_idx=feat_idx, mode="ablate",
            n_interior=n_interior, alpha=alpha,
            layer_index=layer_index,
            lambda_pde=lambda_pde, lambda_bc=lambda_bc,
        )

        # 2. Targeted amplification
        amplified = measure_intervention_effect(
            model, sae, pde, device, dtype,
            feature_idx=feat_idx, mode="amplify",
            n_interior=n_interior, alpha=alpha,
            layer_index=layer_index,
        )

        # 3. Control 1: Unrelated feature
        ctrl_unrelated = measure_intervention_effect(
            model, sae, pde, device, dtype,
            feature_idx=feat_idx, mode="unrelated_control",
            n_interior=n_interior,
            layer_index=layer_index,
        )

        # 4. Control 2: Random direction
        ctrl_random = measure_intervention_effect(
            model, sae, pde, device, dtype,
            feature_idx=feat_idx, mode="random_direction",
            n_interior=n_interior,
            layer_index=layer_index,
        )

        # 5. Control 3: Linear Probe direction (real supervised probe, not a proxy)
        ctrl_probe = measure_probe_direction_effect(
            model, sae, pde, device, dtype,
            feature_idx=feat_idx,
            probe_direction=probe_directions.get(feat_idx) if probe_directions else None,
            n_interior=n_interior,
            layer_index=layer_index,
            alpha=alpha,
        )

        # Target vs non-target delta selection
        if target_loss == "pde":
            t_eff = targeted["delta_pde"]
            c_unrelated = ctrl_unrelated["delta_pde"]
            c_random = ctrl_random["delta_pde"]
            c_probe = ctrl_probe["delta_pde"]
            nontarget = [targeted["delta_bc"]]
        else:
            t_eff = targeted["delta_bc"]
            c_unrelated = ctrl_unrelated["delta_bc"]
            c_random = ctrl_random["delta_bc"]
            c_probe = ctrl_probe["delta_bc"]
            nontarget = [targeted["delta_pde"]]

        score = compute_causal_score(
            target_effect=t_eff,
            control_unrelated_effect=c_unrelated,
            control_random_effect=c_random,
            control_probe_effect=c_probe,
            non_target_effects=nontarget,
        )

        entry = {
            "feature_idx": feat_idx,
            "target_loss": target_loss,
            "targeted_ablate": targeted,
            "targeted_amplify": amplified,
            "ctrl_unrelated": ctrl_unrelated,
            "ctrl_random": ctrl_random,
            "ctrl_probe": ctrl_probe,
            "probe_control_valid": probe_directions is not None and feat_idx in (probe_directions or {}),
            "causal_score": score,
        }
        results.append(entry)

        if out_dir is not None:
            append_jsonl(Path(out_dir) / "inference_interventions.jsonl", entry)

    return results


# ---------------------------------------------------------------------------
# Multi-Seed Benchmark Causal Scoring
# ---------------------------------------------------------------------------

def aggregate_causal_benchmarks(
    feature_results: List[Dict],
) -> Dict[str, Dict[str, float]]:
    """Compute aggregate causal strength and specificity statistics with CIs across features."""
    strengths = [r["causal_score"]["causal_strength"] for r in feature_results]
    specificities = [r["causal_score"]["specificity"] for r in feature_results]

    return {
        "causal_strength_ci": compute_bootstrap_ci(strengths),
        "specificity_ci": compute_bootstrap_ci(specificities),
    }


# ---------------------------------------------------------------------------
# Linear probe direction control (v2 plan §1.3)
# ---------------------------------------------------------------------------

def train_failure_probe(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    labels: np.ndarray,
    device: torch.device,
    C: float = 1.0,
) -> torch.Tensor:
    """Train a logistic probe on SAE latents -> failure label; return its direction.

    The returned unit-norm direction lives in the SAE latent space and
    represents the strongest *supervised linear* signal for predicting failure.
    Intervening along it is the 'simple baseline' control from Wu et al. 2025
    (AXBench): if a plain probe direction produces comparable loss shifts to a
    discovered SAE feature, the feature adds no value over linear supervision.
    """
    from sklearn.linear_model import LogisticRegression

    sae.eval()
    with torch.no_grad():
        z, _ = sae(activations.to(device))
    z_np = z.cpu().numpy()

    probe = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")
    probe.fit(z_np, labels)

    direction = torch.tensor(probe.coef_.flatten(), dtype=torch.float32, device=device)
    direction = direction / direction.norm().clamp(min=1e-8)
    return direction


def make_probe_directions_for_features(
    probe_direction: torch.Tensor,
    candidate_features: List[int],
) -> Dict[int, torch.Tensor]:
    """Assign the (shared) probe direction to each candidate feature slot.

    A single supervised direction is the correct control — it does not depend
    on the candidate feature.  We store it per feature_idx only for interface
    convenience in run_inference_interventions.
    """
    return {feat_idx: probe_direction for feat_idx in candidate_features}


# ---------------------------------------------------------------------------
# Training-time intervention (v2 plan Phase 5, §9.2)
# ---------------------------------------------------------------------------

def run_training_intervention(
    cfg,
    pde: BasePDE,
    sae: SparseAutoencoder,
    device: torch.device,
    dtype: torch.dtype,
    feature_idx: int,
    layer_index: int = 1,
    mode: str = "ablate",
    apply_from_step: int = 0,
    apply_to_step: int = 500,
    total_steps: int = 1500,
    alpha: float = 1.5,
    eval_every: int = 50,
    out_dir: Optional[Path] = None,
) -> Dict:
    """Train with a feature intervention applied during [apply_from_step, apply_to_step),
    then continue unimpeded; compare against an intervention-free control run.

    Returns per-run final metrics plus the paired trajectory of
    (intervention, control) relative-L2 values for rescue/worsening analysis.
    """
    from pinn.model import MLP
    from pinn.reproducibility import set_seed

    def _run(with_intervention: bool) -> Dict:
        set_seed(cfg.run.seed, cfg.run.deterministic)
        model = MLP(
            cfg.model.input_dim, cfg.model.output_dim,
            cfg.model.hidden_layers, cfg.model.activation,
        ).to(device=device, dtype=dtype)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

        hook = None
        traj: List[Dict] = []
        for step in range(total_steps):
            if with_intervention and apply_from_step <= step < apply_to_step:
                if hook is None:
                    hook = SAEInterventionHook(
                        sae, mode=mode, feature_idx=feature_idx, alpha=alpha,
                    )
                    hook.register(model, layer_index)
            elif with_intervention and hook is not None:
                hook.remove()
                hook = None

            opt.zero_grad()
            x = pde.sample_interior(cfg.training.interior_points, device, dtype)
            r = pde.residual(model, x)
            lp = (r ** 2).mean()
            bc = pde.boundary_residual(model)
            lb = bc.mean()
            loss = cfg.training.lambda_pde * lp + cfg.training.lambda_bc * lb
            loss.backward()
            opt.step()

            if step % eval_every == 0 or step == total_steps - 1:
                with torch.no_grad():
                    xv = pde.validation_grid(cfg.pde.validation_points, device, dtype)
                    pred = model(xv)
                    exact = pde.exact(xv)
                    rel_l2 = float(
                        torch.linalg.vector_norm(pred - exact)
                        / torch.linalg.vector_norm(exact).clamp(min=1e-12)
                    )
                traj.append({
                    "step": step,
                    "loss": float(loss),
                    "loss_pde": float(lp),
                    "loss_bc": float(lb),
                    "relative_l2": rel_l2,
                })

        if hook is not None:
            hook.remove()

        return {"final": traj[-1], "trajectory": traj}

    intervened = _run(with_intervention=True)
    control = _run(with_intervention=False)

    result = {
        "feature_idx": feature_idx,
        "mode": mode,
        "layer_index": layer_index,
        "window": [apply_from_step, apply_to_step],
        "intervened_final_rel_l2": intervened["final"]["relative_l2"],
        "control_final_rel_l2": control["final"]["relative_l2"],
        "intervention_effect_on_final_l2":
            control["final"]["relative_l2"] - intervened["final"]["relative_l2"],
        "intervened_trajectory": intervened["trajectory"],
        "control_trajectory": control["trajectory"],
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"training_intervention_feat{feature_idx}.json", "w") as f:
            json.dump(result, f, indent=2)

    return result
