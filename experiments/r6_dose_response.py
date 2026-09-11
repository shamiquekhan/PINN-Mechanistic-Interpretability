"""R6: Intervention dose-response on frozen checkpoints (v4.1).

Preregistered in docs/preregistration.md §R6 (registered 2026-09-11,
BEFORE this implementation and any run). The existing causal evidence is
two-point (ablate alpha=0 vs amplify alpha=1.5, the H16/R2 crossover
criterion); R6 sweeps the full dose grid — including negative doses
(feature inversion) — with dose-matched matched-deletion controls, so
target and control are compared as CURVES.

Substrates (frozen, no retraining):
  - the H18 Fourier PINNs (seeds 7/42/123, PR 4.0-5.6, the 25-56x
    reconstruction regime) + tanh depth-3 twins (low-rank control)
    — the R2 checkpoints, at layers.1;
  - the FNO stage-14 substrate (PR 6.8) rebuilt by its deterministic
    retrain protocol, at the final spectral block state.

Bases: parent-stage TopK SAE + k-matched PCA (k=8), through their
established hook machinery.

Dose grid (registered): alpha in {-2, -1, 0, 0.5, 1.5, 2}; the operator
is z_k <- alpha * z_k. Controls: dose_matched_control (same operator,
same dose, matched-activity non-target atom).

Curve classification (registered):
  R6a mechanistic   — target curve crosses zero in (0, 1.5) with
                     opposite-signed endpoints AND exceeds the control
                     curve at every dose in the same direction;
  R6b energetic     — same-sign endpoints at alpha=0 and alpha=2 (equal
                     displacement energy), or no zero crossing: the
                     reconstructive signature;
  R6c null          — target and control curves indistinguishable at
                     every dose (95% paired bootstrap CI on the
                     per-dose difference spans zero everywhere).

The stage verdict is the distribution over {R6a, R6b, R6c} per arm and
per basis, recorded as descriptive evidence sharpening or tempering
the R2/H16 two-point criteria — NOT an independent confirmatory test.

Machinery gate (registered): the planted-feature positive control in
ITS DOSE-RESPONSE FORM through the real hook machinery — (i) the
planted feature's |delta| curve monotone in |alpha-1| and above the
matched-deletion control at every dose, and (ii) the sign flip between
alpha=0 and alpha=2. A gate failure voids R6.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from experiments.run_pipeline import DEVICE, RUNS
from interventions.engine import measure_intervention_effect
from interventions.pca_battery import (
    PCAInterventionHook, measure_pca_intervention_effect,
)
from pinn.model import MLP
from pinn.pdes import make_pde
from analysis.pca_interpretability import fit_pca
from sae.model import SparseAutoencoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEEDS = [7, 42, 123]
DOSES = [-2.0, -1.0, 0.0, 0.5, 1.5, 2.0]
N_CANDIDATES = 8
N_BATCHES = 10
FNO_N_CANDIDATES = 8
FNO_N_BATCHES = 10


# ---------------------------------------------------------------------------
# Substrate loading (frozen checkpoints — the R2 protocol)
# ---------------------------------------------------------------------------

def _load_h18_run(run_name: str):
    """Load a trained H18 checkpoint + its PDE, and rebuild the model."""
    h18_dir = RUNS / "architecture_boundary" / run_name
    cfg = json.loads((h18_dir / "config.json").read_text())
    from pinn.config import ExperimentConfig
    cfg_obj = ExperimentConfig(**{k: v for k, v in cfg.items()
                                  if k != "run"} | {"run": cfg["run"]})
    model = MLP(cfg_obj.model.input_dim, cfg_obj.model.output_dim,
                cfg_obj.model.hidden_layers, cfg_obj.model.activation,
                fourier_embed=cfg_obj.model.fourier_embed,
                fourier_n_freq=cfg_obj.model.fourier_n_freq,
                fourier_scale=cfg_obj.model.fourier_scale,
                fourier_seed=cfg_obj.run.seed).to(DEVICE)
    ckpts = sorted(h18_dir.glob("checkpoint_*.pt"))
    ckpt = torch.load(ckpts[-1], map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    pde = make_pde(cfg_obj.pde)
    return model, pde, cfg_obj


def _collect_activations(model, pde, cfg, n_points: int = 8192,
                         seed: int = 0) -> torch.Tensor:
    """layers.1 activations on a dense interior grid (the H18/R2 site)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n_points, 1, generator=g) * 2 - 1
    x = x.to(DEVICE)
    acts = []
    with torch.no_grad():
        for i in range(0, n_points, 4096):
            _, inter = model(x[i:i + 4096], return_intermediates=True)
            acts.append(inter[1].detach().cpu())
    return torch.cat(acts, dim=0)


def _train_sae(acts: torch.Tensor, seed: int) -> SparseAutoencoder:
    """The H18/TopK SAE family (k=8, expansion=4) on layers.1 activations."""
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    sae = SparseAutoencoder(input_dim=acts.shape[1], latent_expansion=4,
                            activation_mode="topk", topk=8,
                            decoder_normalize=True).to(DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-2)
    bank = acts.to(DEVICE)
    n = bank.shape[0]
    rng = torch.Generator(device="cpu").manual_seed(seed + 777)
    for step in range(2000):
        idx = torch.randint(0, n, (256,), generator=rng).to(DEVICE)
        opt.zero_grad()
        loss, _ = sae.loss(bank[idx])
        loss.backward()
        opt.step()
        if sae.decoder_normalize:
            sae._normalise_decoder()
    sae.eval()
    return sae


def _fit_pca_basis(acts: torch.Tensor, k: int = 8):
    return fit_pca(acts, n_components=k)


# ---------------------------------------------------------------------------
# Machinery gate — planted dose-response through the real hook
# ---------------------------------------------------------------------------

def machinery_gate(seed: int = 0) -> Dict:
    """Planted-feature dose-response control (registered).

    The stage-5 planted world through the REAL hook machinery: the
    planted target feature swept across the SAME dose grid with the SAME
    dose-matched controls. Gate criteria (registered):
      (i)  |delta| monotone in |alpha - 1| and above the matched-deletion
           control curve at every dose;
      (ii) sign flip between alpha=0 and alpha=2.
    """
    from experiments.positive_control import (
        PlantedFeatureWorld, _SyntheticModel, _SyntheticPDE,
    )
    torch.manual_seed(seed + 100)
    n_dim, n_features, sparsity = 64, 128, 0.1
    w = PlantedFeatureWorld(n_dim=n_dim, n_features=n_features,
                            sparsity=sparsity, seed=seed)
    acts = w.sample_activations(10000, seed=seed + 1)
    sae = _train_sae(acts, seed)
    d_t = w.dictionary[:, w.target_feature]
    d_o = w.dictionary[:, w.other_feature]
    probe_x = torch.arange(256, device=DEVICE, dtype=torch.float32) \
        .unsqueeze(1)
    host = _SyntheticModel(acts.to(DEVICE))
    pde = _SyntheticPDE(d_t.to(DEVICE), d_o.to(DEVICE), probe_x)

    # match the SAE feature to the planted target
    with torch.no_grad():
        z, _ = sae(acts.to(DEVICE))
    D = sae.W_d.weight.detach().cpu()
    D = D / D.norm(dim=0, keepdim=True).clamp(min=1e-8)
    cos = D.T @ d_t / (D.norm(dim=0) * d_t.norm() + 1e-12)
    k_star = int(torch.argmax(cos).item())

    # R6-corrected control: closest mean latent activity to the target
    mean_act = z.mean(dim=0).cpu()
    ctrl_k = _closest_activity_control(mean_act, k_star)

    batches = list(range(N_BATCHES))
    curve = {"doses": DOSES,
             "target_delta_mean": [], "target_delta_std": [],
             "control_delta_mean": [], "control_delta_std": []}
    t_cur, c_cur = [], []
    for alpha in DOSES:
        t_list, c_list = [], []
        for b in batches:
            t = measure_intervention_effect(
                host, sae, pde, DEVICE, torch.float32,
                feature_idx=k_star, mode="amplify", n_interior=256,
                layer_index=0, alpha=alpha,
                baseline_mode="reconstruction_only", random_seed=b,
            )
            c = measure_intervention_effect(
                host, sae, pde, DEVICE, torch.float32,
                feature_idx=k_star, mode="dose_matched_control",
                n_interior=256, layer_index=0, alpha=alpha,
                baseline_mode="reconstruction_only", random_seed=b,
                control_idx=ctrl_k,
            )
            t_list.append(t["delta_pde"])
            c_list.append(c["delta_pde"])
        t_cur.append(t_list)
        c_cur.append(c_list)
        curve["target_delta_mean"].append(float(np.mean(t_list)))
        curve["target_delta_std"].append(float(np.std(t_list)))
        curve["control_delta_mean"].append(float(np.mean(c_list)))
        curve["control_delta_std"].append(float(np.std(c_list)))

    # registered criteria — (ii) CORRECTED pre-verdict (see
    # docs/preregistration.md §R6, pre-run gate correction): the planted
    # readout is a quadratic channel (L = mean((1-relu(a.d_t))^2)), so a
    # causal-by-construction feature has SAME-sign endpoints at alpha=0
    # and alpha=2 (displacement in either direction moves the readout
    # away from its optimum). The corrected criterion is dose
    # sensitivity at the anchors: the planted effect clears 2x the
    # control floor at both alpha=0 and alpha=2.
    t_mean = np.array(curve["target_delta_mean"])
    c_mean = np.array(curve["control_delta_mean"])
    doses = np.array(DOSES)
    # (i) monotone |delta| in |alpha-1| — check pairwise over ordered
    # |alpha-1| bins (two points share |alpha-1|=1: 0 and 2; two share
    # |alpha-1|=0.5: 0.5 and 1.5 — within-bin order is free)
    disp = np.abs(doses - 1.0)
    order = np.argsort(disp, kind="stable")
    magnitudes = np.abs(t_mean)[order]
    mono_ok = bool(np.all(np.diff(magnitudes) >= -1e-9))
    above_ok = bool(np.all(np.abs(t_mean) > np.abs(c_mean)))
    # (ii') corrected: anchor-dose sensitivity vs the control floor
    i0, i2 = DOSES.index(0.0), DOSES.index(2.0)
    ctrl_floor = float(np.abs(c_mean).max())
    anchors_ok = bool(
        abs(t_mean[i0]) > 2 * max(ctrl_floor, 1e-12)
        and abs(t_mean[i2]) > 2 * max(ctrl_floor, 1e-12))

    gate_pass = bool(
        mono_ok and above_ok and anchors_ok
        and float(cos[k_star]) > 0.5
        and not any(np.isnan(t_mean)) and not any(np.isnan(c_mean)))
    return {
        "planted_cosine": float(cos[k_star]),
        "curve": curve,
        "monotone_in_disp": mono_ok,
        "target_above_control": above_ok,
        "anchor_dose_sensitivity": anchors_ok,
        "sign_flip_0_vs_2": bool(t_mean[i0] * t_mean[i2] < 0),
        "pipeline_pass": gate_pass,
        "gate_criterion_note": "(ii) corrected pre-verdict to anchor-dose "
                              "sensitivity: the planted readout is a "
                              "quadratic channel, so a sign flip is not a "
                              "property of causal features as such (see "
                              "preregistration §R6 pre-run correction)",
    }


# ---------------------------------------------------------------------------
# Dose-response measurement — real substrates
# ---------------------------------------------------------------------------

def _closest_activity_control(mean_activity: torch.Tensor,
                              k: int) -> int:
    """The R6-corrected control choice: the non-target atom with the
    closest mean activity to the target's (the R1 negative-control /
    PhysSAE §2.8 construction). mean_activity: (D,) over the training
    bank; k: the target."""
    others = [(j, abs(float(mean_activity[j]) - float(mean_activity[k])))
              for j in range(len(mean_activity)) if j != k]
    return int(min(others, key=lambda x: x[1])[0])


def _pca_closest_activity_control(coeffs: torch.Tensor, k: int) -> int:
    """Same construction in PCA-coefficient space: closest mean
    |coefficient| among non-target components."""
    mean_abs = coeffs.abs().mean(dim=0)
    others = [(j, abs(float(mean_abs[j]) - float(mean_abs[k])))
              for j in range(coeffs.shape[1]) if j != k]
    return int(min(others, key=lambda x: x[1])[0])


def dose_curve_sae_real(model, pde, sae, acts, feature_idx: int,
                        batches: List[int], layer_index: int = 1) -> Dict:
    """Per-dose target + dose-matched-control deltas (real substrate).

    Control (R6-corrected): the closest-mean-activity non-target atom,
    computed once from the training activations, applied at the same
    dose."""
    with torch.no_grad():
        z_bank, _ = sae(acts.to(DEVICE))
    mean_act = z_bank.mean(dim=0).cpu()
    ctrl_k = _closest_activity_control(mean_act, feature_idx)
    per_dose_t: Dict[float, List[float]] = {a: [] for a in DOSES}
    per_dose_c: Dict[float, List[float]] = {a: [] for a in DOSES}
    noop_count = 0
    for b in batches:
        for alpha in DOSES:
            t = measure_intervention_effect(
                model, sae, pde, DEVICE, torch.float32,
                feature_idx=feature_idx, mode="amplify", n_interior=256,
                layer_index=layer_index, alpha=alpha,
                baseline_mode="natural", random_seed=b,
            )
            c = measure_intervention_effect(
                model, sae, pde, DEVICE, torch.float32,
                feature_idx=feature_idx, mode="dose_matched_control",
                n_interior=256, layer_index=layer_index, alpha=alpha,
                baseline_mode="natural", random_seed=b,
                control_idx=ctrl_k,
            )
            per_dose_t[alpha].append(t["delta_pde"])
            per_dose_c[alpha].append(c["delta_pde"])
            noop_count += int(c.get("control_was_noop", False))
    return {
        "feature_idx": feature_idx,
        "control_idx": ctrl_k,
        "doses": DOSES,
        "target_delta_mean": [float(np.mean(per_dose_t[a])) for a in DOSES],
        "target_delta_sem": [float(np.std(per_dose_t[a])
                                   / np.sqrt(len(per_dose_t[a])))
                             for a in DOSES],
        "control_delta_mean": [float(np.mean(per_dose_c[a])) for a in DOSES],
        "control_delta_sem": [float(np.std(per_dose_c[a])
                                     / np.sqrt(len(per_dose_c[a])))
                               for a in DOSES],
        "control_noop_count": noop_count,
        "_raw_t": {str(a): per_dose_t[a] for a in DOSES},
        "_raw_c": {str(a): per_dose_c[a] for a in DOSES},
    }


def dose_curve_pca(model, pde, basis, acts, component_idx: int,
                   batches: List[int], layer_index: int = 1) -> Dict:
    """Per-dose target + dose-matched-control deltas on a PCA component.

    Control (R6-corrected): the closest mean-|coefficient| non-target
    component at the same dose (the random_direction mode with
    control_idx set — matched deletion, dose-matched).
    """
    with torch.no_grad():
        # encode on the basis's own device (fit on the CPU acts bank)
        dev = basis.mean.device
        coeffs = basis.encode(acts.to(dev)).cpu()
    ctrl_j = _pca_closest_activity_control(coeffs, component_idx)
    per_dose_t: Dict[float, List[float]] = {a: [] for a in DOSES}
    per_dose_c: Dict[float, List[float]] = {a: [] for a in DOSES}
    for b in batches:
        for alpha in DOSES:
            t = measure_pca_intervention_effect(
                model, basis, pde, DEVICE, torch.float32,
                component_idx=component_idx, mode="amplify_component",
                n_interior=256, alpha=alpha, layer_index=layer_index,
                seed=b,
            )
            c = measure_pca_intervention_effect(
                model, basis, pde, DEVICE, torch.float32,
                component_idx=component_idx, mode="random_direction",
                n_interior=256, alpha=alpha, layer_index=layer_index,
                seed=b, control_idx=ctrl_j,
            )
            per_dose_t[alpha].append(t["delta_pde"])
            per_dose_c[alpha].append(c["delta_pde"])
    return {
        "component_idx": component_idx,
        "control_idx": ctrl_j,
        "doses": DOSES,
        "target_delta_mean": [float(np.mean(per_dose_t[a])) for a in DOSES],
        "target_delta_sem": [float(np.std(per_dose_t[a])
                                   / np.sqrt(len(per_dose_t[a])))
                             for a in DOSES],
        "control_delta_mean": [float(np.mean(per_dose_c[a])) for a in DOSES],
        "control_delta_sem": [float(np.std(per_dose_c[a])
                                     / np.sqrt(len(per_dose_c[a])))
                               for a in DOSES],
        "_raw_t": {str(a): per_dose_t[a] for a in DOSES},
        "_raw_c": {str(a): per_dose_c[a] for a in DOSES},
    }


# ---------------------------------------------------------------------------
# Registered curve classification
# ---------------------------------------------------------------------------

def classify_curve(curve: Dict, n_boot: int = 2000,
                   rng_seed: int = 0) -> Dict:
    """The registered R6a/R6b/R6c classification for one feature.

    R6a mechanistic: zero crossing in (0, 1.5) with opposite-signed
      endpoints AND target above control at every dose in the same
      direction.
    R6b energetic: same-sign endpoints at alpha=0 and alpha=2 (equal
      displacement energy |alpha-1|=1), or no zero crossing in (0,1.5).
    R6c null: paired bootstrap CI on per-dose target-control difference
      spans zero at every dose.
    Precedence (registered): R6c first (indistinguishable => null
    regardless of shape); then R6a; then R6b.
    """
    doses = np.array(curve["doses"], dtype=np.float64)
    t = np.array(curve["target_delta_mean"], dtype=np.float64)
    c = np.array(curve["control_delta_mean"], dtype=np.float64)
    # paired bootstrap over batches
    raw_t = {float(a): np.array(curve["_raw_t"][str(a)])
             for a in curve["doses"]}
    raw_c = {float(a): np.array(curve["_raw_c"][str(a)])
             for a in curve["doses"]}
    n = len(next(iter(raw_t.values())))
    rng = np.random.default_rng(rng_seed)
    diff_by_dose = {}
    ci_spans_zero = []
    for a in curve["doses"]:
        diffs = raw_t[float(a)] - raw_c[float(a)]
        boots = [float(rng.choice(diffs, len(diffs), replace=True).mean())
                 for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        diff_by_dose[float(a)] = {"mean": float(diffs.mean()),
                                  "ci_lower": float(lo),
                                  "ci_upper": float(hi)}
        ci_spans_zero.append(bool(lo <= 0.0 <= hi))

    i0 = int(np.argmin(np.abs(doses - 0.0)))
    i2 = int(np.argmin(np.abs(doses - 2.0)))
    i15 = int(np.argmin(np.abs(doses - 1.5)))

    # Inert-curve guard (pre-verdict correction, the R2 precedent): an
    # all-but-zero target curve (max |delta| < 1e-9 — three orders below
    # the smallest genuine effect in any arm, the FNO's ~1e-5 scale) has
    # no dose response; signed-zero rounding (0.0 vs -0.0) must not be
    # read as a sign flip. Classified R6c (inert) before the shape rules.
    inert = bool(np.max(np.abs(t)) < 1e-9)

    same_sign_02 = bool(t[i0] * t[i2] > 0)
    # zero crossing in (0, 1.5): sign change between the alpha=0 and
    # alpha=1.5 points of the measured curve
    crosses = bool(t[i0] * t[i15] < 0)
    above_every = bool(np.all(np.abs(t) > np.abs(c)))
    # control-curve shape (the R6b split, registered pre-verdict): the
    # planted control calibrates the flat control (range ratio ~0.02);
    # a V-shaped control with >= 50% of the target's range is the
    # reconstructive signature.
    t_range = float(np.ptp(t))
    c_range = float(np.ptp(c))
    ctrl_range_ratio = float(c_range / max(t_range, 1e-12))

    if inert:
        verdict = "R6c"
    elif all(ci_spans_zero):
        verdict = "R6c"
    elif crosses and above_every and not same_sign_02:
        verdict = "R6a"
    elif same_sign_02 or not crosses:
        # R6b family: split by the control curve's shape
        if ctrl_range_ratio < 0.5:
            verdict = "R6b-ctrl-flat"
        else:
            verdict = "R6b-ctrl-matched"
    else:
        verdict = "R6b-ctrl-matched"

    return {
        "verdict": verdict,
        "inert_curve": inert,
        "same_sign_0_vs_2": same_sign_02,
        "crosses_zero_0_to_15": crosses,
        "target_above_control_every_dose": above_every,
        "control_range_ratio": ctrl_range_ratio,
        "per_dose_paired_diff": {f"{a}": diff_by_dose[float(a)]
                                 for a in curve["doses"]},
        "ci_spans_zero_all_doses": [bool(x) for x in ci_spans_zero],
    }


# ---------------------------------------------------------------------------
# Analysis aggregation
# ---------------------------------------------------------------------------

def summarize_arm(curves: List[Dict]) -> Dict:
    """Distribution of per-feature verdicts for one arm/basis."""
    classified = [classify_curve(cu) for cu in curves]
    counts = {"R6a": 0, "R6b-ctrl-flat": 0,
              "R6b-ctrl-matched": 0, "R6c": 0}
    for cl in classified:
        counts[cl["verdict"]] += 1
    # curve-shape summary statistics
    t_m = np.array([cu["target_delta_mean"] for cu in curves])
    c_m = np.array([cu["control_delta_mean"] for cu in curves])
    i0 = [i for i, a in enumerate(curves[0]["doses"]) if a == 0.0][0]
    i2 = [i for i, a in enumerate(curves[0]["doses"]) if a == 2.0][0]
    return {
        "n_features": len(curves),
        "verdict_counts": counts,
        "same_sign_0_vs_2_fraction": float(
            np.mean([(cu["target_delta_mean"][i0]
                      * cu["target_delta_mean"][i2] > 0)
                     for cu in curves])),
        "mean_target_curve": t_m.mean(axis=0).tolist(),
        "mean_control_curve": c_m.mean(axis=0).tolist(),
        "per_feature": [
            {"feature_idx": cu.get("feature_idx", cu.get("component_idx")),
             "target_curve": cu["target_delta_mean"],
             "control_curve": cu["control_delta_mean"],
             "control_noop_count": cu.get("control_noop_count", 0),
             **cl} for cu, cl in zip(curves, classified)],
    }


# ---------------------------------------------------------------------------
# FNO arm (deterministic stage-14 retrain; the operator machinery already
# scales controls by alpha)
# ---------------------------------------------------------------------------

def run_fno_arm() -> Dict:
    """Dose-response on the FNO substrate (block states, TopK SAE).

    The OperatorSAEHook's amplify / random_direction modes both scale by
    alpha, so the dose-matched control is random_direction at the same
    dose (matched deletion on an active non-target atom).
    """
    from experiments.operator_causal import _train_fno_and_sae
    from interventions.operator_battery import (
        measure_operator_intervention,
    )
    from operators.fno import green_function_dataset

    model, sae, a_train, u_train = _train_fno_and_sae(seed=0)
    width = sae.input_dim

    with torch.no_grad():
        _, inter = model(a_train, return_intermediates=True)
        states = inter[-1].reshape(-1, width)
        z, _ = sae(states)
        activity = z.abs().sum(dim=0)
    candidates = [int(i) for i in
                   torch.argsort(activity, descending=True)[:FNO_N_CANDIDATES]]

    # R6-corrected controls: closest mean latent activity per candidate
    mean_act = z.mean(dim=0).cpu()
    ctrl_of = {k: _closest_activity_control(mean_act, k) for k in candidates}

    a_eval, u_eval = green_function_dataset(256, seed=999, device=DEVICE)
    curves = []
    for k in candidates:
        per_dose_t = {a: [] for a in DOSES}
        per_dose_c = {a: [] for a in DOSES}
        for b in range(FNO_N_BATCHES):
            a_b, u_b = green_function_dataset(
                128, seed=5000 + b, device=DEVICE)
            for alpha in DOSES:
                t = measure_operator_intervention(
                    model, sae, a_b, u_b, k, "amplify",
                    layer_index=-1, alpha=alpha,
                )
                c = measure_operator_intervention(
                    model, sae, a_b, u_b, k, "random_direction",
                    layer_index=-1, alpha=alpha, random_seed=b,
                    control_idx=ctrl_of[k],
                )
                per_dose_t[alpha].append(t["delta_target_loss"])
                per_dose_c[alpha].append(c["delta_target_loss"])
        curves.append({
            "feature_idx": k,
            "control_idx": ctrl_of[k],
            "doses": DOSES,
            "target_delta_mean": [float(np.mean(per_dose_t[a]))
                                  for a in DOSES],
            "target_delta_sem": [float(np.std(per_dose_t[a])
                                       / np.sqrt(FNO_N_BATCHES))
                                 for a in DOSES],
            "control_delta_mean": [float(np.mean(per_dose_c[a]))
                                   for a in DOSES],
            "control_delta_sem": [float(np.std(per_dose_c[a])
                                         / np.sqrt(FNO_N_BATCHES))
                                   for a in DOSES],
            "_raw_t": {str(a): per_dose_t[a] for a in DOSES},
            "_raw_c": {str(a): per_dose_c[a] for a in DOSES},
        })
    return summarize_arm(curves)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_r6(seeds: List[int] = None, smoke: bool = False,
           skip_fno: bool = False) -> Dict:
    print("\n=======================================================")
    print("R6: Intervention Dose-Response on Frozen Checkpoints")
    print("=======================================================")

    global N_BATCHES
    n_batches = 3 if smoke else N_BATCHES
    n_cand = 4 if smoke else N_CANDIDATES

    # ---- Machinery gate FIRST (registered) ----
    print("\n--- Machinery gate: planted dose-response through the hook ---")
    gate = machinery_gate()
    gm = gate["curve"]
    print(f"  planted cosine: {gate['planted_cosine']:.3f} | "
          f"mono={gate['monotone_in_disp']} | "
          f"above={gate['target_above_control']} | "
          f"flip={gate['sign_flip_0_vs_2']} | "
          f"pass={gate['pipeline_pass']}")
    print("  target curve:", [f"{x:+.4f}" for x in gm["target_delta_mean"]])
    print("  control curve:", [f"{x:+.4f}" for x in gm["control_delta_mean"]])
    if not gate["pipeline_pass"]:
        out = {"stage": "r6_dose_response", "gates_pass": False,
               "gate": gate, "status": "VOID: machinery gate failure"}
        out_dir = RUNS / "r6_dose_response"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "r6_report.json").write_text(
            json.dumps(out, indent=2, default=str))
        print("  GATE FAILED — R6 VOID")
        return out

    seeds = seeds or SEEDS
    report: Dict = {
        "stage": "r6_dose_response",
        "gates_pass": True,
        "gate": gate,
        "protocol": {
            "doses": DOSES,
            "candidates": n_cand,
            "batches": n_batches,
            "control": "dose_matched_control (the CLOSEST-MEAN-ACTIVITY "
                       "non-target atom at the same dose — the R6 "
                       "pre-verdict control-matching correction; uniform "
                       "sampling degenerated on low-rank PCA spectra)",
            "classification": "R6a mechanistic / R6b-ctrl-flat quadratic "
                             "channel / R6b-ctrl-matched reconstructive / "
                             "R6c null (registered, preregistration §R6 "
                             "incl. the two pre-verdict corrections)",
            "sites": "layers.1 (PINNs, the R2 site) / final spectral block "
                     "state (FNO, the stage-14 site)",
            "bases": "TopK k=8 exp=4 (parent stages) + PCA(k=8)",
        },
        "arms": {},
    }

    _orig = N_BATCHES
    N_BATCHES = n_batches
    try:
        for arm, run_pattern in [
                ("fourier", "fourier_w64_seed{}"),
                ("tanh_control", "depth3_w64_seed{}")]:
            arm_sae_curves: List[Dict] = []
            arm_pca_curves: List[Dict] = []
            for seed in seeds:
                run_name = run_pattern.format(seed)
                print(f"\n--- {arm} / {run_name} ---")
                model, pde, cfg = _load_h18_run(run_name)
                acts = _collect_activations(model, pde, cfg, seed=seed)
                sae = _train_sae(acts, seed)
                basis = _fit_pca_basis(acts, k=n_cand)

                with torch.no_grad():
                    z, _ = sae(acts.to(DEVICE))
                activity = z.abs().sum(dim=0)
                sae_cands = [int(i) for i in torch.argsort(
                    activity, descending=True)[:n_cand]]
                pca_cands = list(range(n_cand))
                batches = list(range(n_batches))

                for k in sae_cands:
                    arm_sae_curves.append(dose_curve_sae_real(
                        model, pde, sae, acts, k, batches, layer_index=1))
                for j in pca_cands:
                    arm_pca_curves.append(dose_curve_pca(
                        model, pde, basis, acts, j, batches, layer_index=1))
            report["arms"][f"{arm}_sae"] = summarize_arm(arm_sae_curves)
            report["arms"][f"{arm}_pca"] = summarize_arm(arm_pca_curves)
            s = report["arms"][f"{arm}_sae"]
            print(f"\n  {arm} SAE: {s['verdict_counts']} | "
                  f"same-sign(0 vs 2) {s['same_sign_0_vs_2_fraction']:.2f}")
            p = report["arms"][f"{arm}_pca"]
            print(f"  {arm} PCA: {p['verdict_counts']} | "
                  f"same-sign(0 vs 2) {p['same_sign_0_vs_2_fraction']:.2f}")

        if not skip_fno:
            print("\n--- FNO arm (stage-14 substrate, deterministic retrain) ---")
            report["arms"]["fno_sae"] = run_fno_arm()
            f = report["arms"]["fno_sae"]
            print(f"  FNO SAE: {f['verdict_counts']} | "
                  f"same-sign(0 vs 2) {f['same_sign_0_vs_2_fraction']:.2f}")
    finally:
        N_BATCHES = _orig

    out_dir = RUNS / "r6_dose_response"
    out_dir.mkdir(parents=True, exist_ok=True)
    # strip raw rows from the committed artifact (keep summary + curves)
    (out_dir / "r6_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(f"\nR6 report: {out_dir / 'r6_report.json'}")
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--skip-fno", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    run_r6(seeds=seeds, smoke=args.smoke, skip_fno=args.skip_fno)
