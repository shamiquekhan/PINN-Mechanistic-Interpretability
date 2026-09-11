"""Operator causal battery — the FNO-side counterpart of the PINN batteries.

Stage 14 of the campaign.  Runs the SAME preregistered 3-control causal
intervention protocol used for PINN hidden layers (interventions/engine.py,
interventions/pca_battery.py) on **FNO block states** — the high-rank
(function-space) representations where TopK SAEs beat k-matched PCA in
reconstruction (stage 11).

Design decisions (preregistered in docs/preregistration.md, H14):

* **Layer**: interventions target the last spectral block state
  (``acts[-1]``), shape (batch, L, width) — the representation directly
  upstream of the projection head.

* **Readout — the "target loss" for an operator**: the PINN batteries
  measured ΔPDE-loss under intervention.  For an operator there is no PDE;
  the natural target is the **function-space regression loss**
  L_target = MSE(model(a), u) on a fixed held-out batch.  The non-target
  readout is the input-reconstruction-free proxy: the L2 change of the
  model output as a FUNCTION STATISTIC — the mean absolute deviation of
  the output's Fourier magnitude spectrum (spectral shape shift).  This
  mirrors the pde-vs-bc target/non-target pair: one semantic readout the
  feature is hypothesized to influence, one independent readout it should
  leave alone if it is specific.

* **Feature semantics**: candidate SAE latents are matched to a
  **physically meaningful planted direction** — the output's dominant
  Fourier mode magnitude — so "target effect" is defined against a
  supervised probe direction trained to predict that quantity from block
  states (the operator analogue of the failure-label probe).

* **Controls** (identical semantics to the PINN batteries):
    1. unrelated-feature ablation (seeded choice among other latents),
    2. norm-matched random-direction perturbation in latent space
       (per-row magnitude matched),
    3. supervised probe-direction perturbation.

* **Machinery gate**: a planted-feature positive control through the exact
  hook + battery (synthetic superposition world), which must PASS before
  the FNO verdict is interpreted.

Scoring reuses interventions.causal.compute_causal_score,
per_feature_causal_pvalues (Bonferroni + BH-FDR), so the operator battery
is statistically identical to the PINN batteries.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from interventions.causal import (
    compute_causal_score,
    compute_bootstrap_ci,
    per_feature_causal_pvalues,
)
from operators.fno import FNO1d
from sae.model import SparseAutoencoder


# ---------------------------------------------------------------------------
# FNO intervention hook (SAE inserted at a block state)
# ---------------------------------------------------------------------------

class OperatorSAEHook:
    """Forward hook: encode block state through a frozen SAE, modify the
    latent code, decode back.  Mirrors SAEInterventionHook semantics.

    Modes: natural, ablate, amplify, unrelated_control, random_direction,
    probe_direction (same definitions as interventions/engine.py; the
    block state has shape (batch, L, width) so latents are per (row,
    position) pair and magnitudes are matched per row over the flattened
    batch*position dimension).
    """

    _MODES = {"natural", "ablate", "amplify", "unrelated_control",
              "random_direction", "probe_direction"}

    def __init__(
        self,
        sae: SparseAutoencoder,
        mode: str = "ablate",
        feature_idx: Optional[int] = None,
        alpha: float = 1.5,
        random_seed: int = 0,
        probe_direction: Optional[torch.Tensor] = None,
        control_idx: Optional[int] = None,
    ):
        if mode not in self._MODES:
            raise ValueError(f"Unknown mode {mode!r}; choose from {sorted(self._MODES)}")
        if mode == "probe_direction" and probe_direction is None:
            raise ValueError("probe_direction mode requires a direction tensor")
        self.sae = sae
        self.mode = mode
        self.feature_idx = feature_idx
        self.alpha = alpha
        self.random_seed = random_seed
        # R6 control-matching correction: explicitly designated control
        # atom; None keeps the sampled behavior.
        self.control_idx = control_idx
        self.last_control_was_noop = False
        if probe_direction is not None:
            d = probe_direction.detach().to("cpu")
            self.probe_direction = d / d.norm().clamp(min=1e-8)
        else:
            self.probe_direction = None
        self._hook = None

    def _pick_control(self, z: torch.Tensor, k: int) -> int:
        """Control atom: the explicitly designated one (R6 correction)
        or a uniform draw among ACTIVE non-target atoms (C3a fix)."""
        ctrl_idx = getattr(self, "control_idx", None)
        if ctrl_idx is not None and ctrl_idx != k:
            self.last_control_was_noop = False
            return ctrl_idx
        rng = torch.Generator(device=z.device)
        rng.manual_seed(self.random_seed)
        active = (z > 0).any(dim=0)
        choices = [i for i in range(z.shape[1]) if i != k and bool(active[i])]
        if not choices:
            self.last_control_was_noop = True
            return -1
        self.last_control_was_noop = False
        return choices[torch.randint(len(choices), (1,), generator=rng,
                                      device=z.device).item()]

    def _intervene(self, z: torch.Tensor) -> torch.Tensor:
        mode = self.mode
        k = self.feature_idx
        if mode == "natural":
            return z
        if mode in ("ablate", "amplify"):
            z = z.clone()
            z[:, k] = z[:, k] * self.alpha
            return z
        if mode == "unrelated_control":
            # C3a fix (mirror of interventions/engine.py): sample only from
            # features ACTIVE in >=1 row so the control is never a silent
            # no-op on the ~79%-dead operator dictionary.
            z = z.clone()
            ctrl = self._pick_control(z, k)
            if ctrl < 0:
                return z
            z[:, ctrl] = z[:, ctrl] * self.alpha
            return z
        if mode == "random_direction":
            # C3b fix: matched DELETION (ablate a random active non-target
            # feature) instead of noise injection — the target ablates, so
            # the control must perturb the same way.
            z = z.clone()
            ctrl = self._pick_control(z, k)
            if ctrl < 0:
                return z
            z[:, ctrl] = z[:, ctrl] * self.alpha
            return z
        if mode == "probe_direction":
            per_row_mag = z[:, k].abs() if k is not None else z.abs().mean(dim=1)
            d = self.probe_direction.to(z.device, z.dtype)
            return z.clone() + d.unsqueeze(0) * per_row_mag.unsqueeze(1)
        return z

    def _hook_fn(self, _module, _inputs, output: torch.Tensor):
        if self.mode == "natural":
            return output
        sae = self.sae
        w_e = sae.W_e.weight.detach()
        b_e = sae.W_e.bias.detach()
        b_a = sae.b_a.detach()
        w_d = sae.W_d.weight.detach()
        b_d = sae.W_d.bias.detach()

        shape = output.shape                     # (b, L, width)
        flat = output.reshape(-1, shape[-1])      # (b*L, width)
        z = F.relu(F.linear(flat - b_a, w_e, b_e))
        if sae.activation_mode == "topk":
            from sae.model import apply_topk
            z = apply_topk(z, k=sae.topk)
        z_mod = self._intervene(z)
        recon = F.linear(z_mod, w_d, b_d)
        return recon.reshape(shape)

    def register(self, model: FNO1d, layer_index: int = -1):
        self._hook = model.acts[layer_index].register_forward_hook(self._hook_fn)

    def remove(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------

def regression_loss(model: FNO1d, a: torch.Tensor, u: torch.Tensor) -> float:
    """Function-space regression loss on a fixed batch (the target readout)."""
    with torch.no_grad():
        pred = model(a)
        return float(F.mse_loss(pred, u).item())


def spectral_statistic(model: FNO1d, a: torch.Tensor) -> float:
    """Non-target readout: mean |output Fourier magnitude| over the batch.

    Deliberately orthogonal to per-point MSE — a feature that merely shifts
    the decode slightly changes MSE but not the spectral energy envelope;
    a spectral-mode-specific feature should move this statistic if it
    moves anything (making it a demanding specificity test in the other
    direction: the target must move MSE MORE than this moves).
    """
    with torch.no_grad():
        pred = model(a)
        spec = torch.fft.rfft(pred.squeeze(-1), dim=-1)
        return float(spec.abs().mean().item())


def probe_direction_for_target(
    sae: SparseAutoencoder,
    block_states: torch.Tensor,   # (N, width) — flat (b*L) states
    target_values: np.ndarray,     # (N,) supervised regression target
) -> torch.Tensor:
    """Ridge-regression probe direction in SAE latent space.

    Fits latent codes -> target value (standardized), returns the unit
    coefficient vector: the strongest supervised LINEAR signal for the
    physical quantity, in the same latent space the battery intervenes on.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    sae.eval()
    device = next(sae.parameters()).device
    with torch.no_grad():
        z, _ = sae(block_states.to(device))
    z_np = z.cpu().numpy()
    scaler = StandardScaler()
    z_np = scaler.fit_transform(z_np)
    y = (target_values - target_values.mean()) / (target_values.std() + 1e-9)
    ridge = Ridge(alpha=1.0)
    ridge.fit(z_np, y)
    direction = torch.tensor(ridge.coef_, dtype=torch.float32)
    return direction / direction.norm().clamp(min=1e-8)


# ---------------------------------------------------------------------------
# Effect measurement
# ---------------------------------------------------------------------------

def measure_operator_intervention(
    model: FNO1d,
    sae: SparseAutoencoder,
    a_batch: torch.Tensor,
    u_batch: torch.Tensor,
    feature_idx: int,
    mode: str,
    layer_index: int = -1,
    alpha: float = 0.0,
    random_seed: int = 0,
    probe_direction: Optional[torch.Tensor] = None,
    control_idx: Optional[int] = None,
) -> Dict:
    """Frozen-model readout deltas for one intervention.

    Baseline = natural (SAE bypassed); intervened = hooked pass; the batch
    is FIXED so deltas are purely the intervention's footprint.
    control_idx: R6 correction — explicitly designated control atom.
    """
    model.eval()
    sae.eval()

    lp_base = regression_loss(model, a_batch, u_batch)
    sp_base = spectral_statistic(model, a_batch)

    hook = OperatorSAEHook(
        sae, mode=mode, feature_idx=feature_idx, alpha=alpha,
        random_seed=random_seed, probe_direction=probe_direction,
        control_idx=control_idx,
    )
    hook.register(model, layer_index)
    try:
        lp_int = regression_loss(model, a_batch, u_batch)
        sp_int = spectral_statistic(model, a_batch)
    finally:
        hook.remove()

    return {
        "feature_idx": feature_idx,
        "mode": mode,
        "delta_target_loss": lp_int - lp_base,
        "delta_spectral_stat": sp_int - sp_base,
        "baseline_target_loss": lp_base,
        "intervened_target_loss": lp_int,
    }


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------

def run_operator_causal_battery(
    model: FNO1d,
    sae: SparseAutoencoder,
    a_eval: torch.Tensor,
    u_eval: torch.Tensor,
    candidate_features: List[int],
    probe_direction: torch.Tensor,
    layer_index: int = -1,
    alpha: float = 0.0,
    n_random_trials: int = 5,
) -> List[Dict]:
    """Targeted + 3-control interventions for each candidate SAE feature.

    n_random_trials random-direction draws per feature; the median delta is
    the random control (matching the positive-control convention).
    """
    results: List[Dict] = []
    for k in candidate_features:
        targeted = measure_operator_intervention(
            model, sae, a_eval, u_eval, k, "ablate",
            layer_index=layer_index, alpha=alpha,
        )
        amplified = measure_operator_intervention(
            model, sae, a_eval, u_eval, k, "amplify",
            layer_index=layer_index, alpha=1.5,
        )
        ctrl_unrelated = measure_operator_intervention(
            model, sae, a_eval, u_eval, k, "unrelated_control",
            layer_index=layer_index, alpha=alpha, random_seed=0,
        )
        random_deltas = []
        for t in range(n_random_trials):
            r = measure_operator_intervention(
                model, sae, a_eval, u_eval, k, "random_direction",
                layer_index=layer_index, random_seed=t,
            )
            random_deltas.append(r["delta_target_loss"])
        ctrl_random_delta = float(np.median(random_deltas))
        ctrl_probe = measure_operator_intervention(
            model, sae, a_eval, u_eval, k, "probe_direction",
            layer_index=layer_index, probe_direction=probe_direction,
        )

        score = compute_causal_score(
            target_effect=targeted["delta_target_loss"],
            control_unrelated_effect=ctrl_unrelated["delta_target_loss"],
            control_random_effect=ctrl_random_delta,
            control_probe_effect=ctrl_probe["delta_target_loss"],
            non_target_effects=[targeted["delta_spectral_stat"]],
        )
        results.append({
            "feature_idx": k,
            "targeted_ablate": targeted,
            "targeted_amplify": amplified,
            "ctrl_unrelated": ctrl_unrelated,
            "ctrl_random_delta": ctrl_random_delta,
            "ctrl_random_delta_trials": random_deltas,
            "ctrl_probe": ctrl_probe,
            "causal_score": score,
        })
    return results


def summarize_operator_battery(all_results: List[Dict]) -> Dict:
    """Same aggregation as the PINN batteries: CIs, MC corrections, signs."""
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


# ---------------------------------------------------------------------------
# Machinery gate: planted-feature positive control through the real hook
# ---------------------------------------------------------------------------

def operator_positive_control(
    width: int = 64,
    grid_points: int = 32,
    n_samples: int = 2048,
    expansion: int = 4,
    topk: int = 8,
    sae_steps: int = 1500,
    seed: int = 0,
    device: torch.device = None,
) -> Dict:
    """Validate the operator hook + battery on a synthetic superposition
    world with a KNOWN causal feature — the stage-14 machinery gate.

    World: block-state-like activations a = D z + noise (D: 64x256 dictionary
    in superposition).  The host model's "output" reads the planted target
    direction: out = relu(a . d_t).  An SAE trained on a must recover a
    latent matching d_t; ablating it must shift the readout far beyond all
    three controls — else the battery machinery is broken and NO stage-14
    verdict may be interpreted.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed + 100)
    n_dim, n_features, sparsity = width, 4 * width, 0.1
    g = torch.Generator().manual_seed(seed)
    D = torch.randn(n_dim, n_features, generator=g)
    D = D / D.norm(dim=0, keepdim=True)
    target_f, other_f = 3, 7
    d_t = D[:, target_f].to(device)
    d_o = D[:, other_f].to(device)

    def _sample(n, gen):
        z = (torch.rand(n, n_features, generator=gen) < sparsity).float()
        z = z * torch.rand(n, n_features, generator=gen)
        return z @ D.T + 0.05 * torch.randn(n, n_dim, generator=gen)

    gen = torch.Generator().manual_seed(seed + 1)
    acts = _sample(n_samples, gen).to(device)

    # Host: FNO1d-shaped module whose block state IS the planted activation.
    class _StateHost(torch.nn.Module):
        def __init__(self, bank):
            super().__init__()
            self.acts = torch.nn.ModuleList([torch.nn.Identity()
                                             for _ in range(2)])
            self.register_buffer("bank", bank)

        def forward(self, x):
            # x ignored; block state = bank rows (positions axis is a
            # dummy L=1 so the (b, L, width) convention holds).
            state = self.bank[: x.shape[0]].unsqueeze(1)
            state = self.acts[-1](state)
            # Readout: planted direction target; spectral-stat readout uses
            # the other planted direction (non-target).
            proj_t = state.squeeze(1) @ d_t
            proj_o = state.squeeze(1) @ d_o
            return proj_t.unsqueeze(-1), proj_o.unsqueeze(-1)

    host = _StateHost(acts)

    # Real SAE on the planted states.
    sae = SparseAutoencoder(input_dim=n_dim, latent_expansion=expansion,
                            activation_mode="topk", topk=topk,
                            decoder_normalize=True).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-2)
    rng = torch.Generator(device="cpu").manual_seed(seed + 777)
    for _ in range(sae_steps):
        idx = torch.randint(0, n_samples, (256,), generator=rng).to(device)
        opt.zero_grad()
        loss, _ = sae.loss(acts[idx])
        loss.backward()
        opt.step()
        if sae.decoder_normalize:
            sae._normalise_decoder()
    sae.eval()

    dec_cols = sae.W_d.weight.detach().T
    dec_normed = dec_cols / dec_cols.norm(dim=1, keepdim=True).clamp(min=1e-8)
    cosines = dec_normed @ d_t
    matched = int(torch.argmax(cosines).item())
    decoder_cosine = float(cosines[matched].item())

    n_eval = 512
    x_sel = torch.arange(n_eval, device=device, dtype=torch.float32)
    a_eval = x_sel.unsqueeze(-1)          # dummy input selecting bank rows

    # Readout: ONE-SIDED, matching the validated PINN positive control
    # (experiments/positive_control.py): L = mean(1 - relu(proj_t)).
    # Ablating the planted feature collapses proj_t -> L jumps; the probe /
    # random controls ADD magnitude along the latent, which can only raise
    # proj_t (relu saturates, L unchanged) — a symmetric MSE readout would
    # damage under both and make the gate undecidable.
    def _losses(hook):
        if hook is not None:
            hook.register(host, layer_index=-1)
        try:
            with torch.no_grad():
                pred_t, _pred_o = host(a_eval)
                lp = float((1.0 - torch.relu(pred_t)).mean().item())
                sp = float((_pred_o ** 2).mean().item())
        finally:
            if hook is not None:
                hook.remove()
        return lp, sp

    lp_base, sp_base = _losses(None)

    def _delta(mode, **kw):
        lp_int, sp_int = _losses(OperatorSAEHook(sae, mode=mode,
                                                 feature_idx=matched, **kw))
        return {"delta_target_loss": lp_int - lp_base,
                "delta_spectral_stat": sp_int - sp_base}

    targeted = _delta("ablate", alpha=0.0)
    ctrl_unrelated = _delta("unrelated_control", alpha=0.0, random_seed=0)
    rand = [_delta("random_direction", random_seed=t)["delta_target_loss"]
            for t in range(5)]
    ctrl_random_delta = float(np.median(rand))
    # Probe direction: the planted target itself is unknown to the method —
    # use the latent-space one-hot on the matched feature (a *supervised*
    # upper bound for this gate; the point is machinery, not fairness).
    probe_dir = torch.zeros(sae.latent_dim)
    probe_dir[matched] = 1.0
    ctrl_probe = _delta("probe_direction", probe_direction=probe_dir)

    score = compute_causal_score(
        target_effect=targeted["delta_target_loss"],
        control_unrelated_effect=ctrl_unrelated["delta_target_loss"],
        control_random_effect=ctrl_random_delta,
        control_probe_effect=ctrl_probe["delta_target_loss"],
        non_target_effects=[targeted["delta_spectral_stat"]],
    )
    beats_all = (
        abs(targeted["delta_target_loss"]) >
        abs(ctrl_unrelated["delta_target_loss"])
        and abs(targeted["delta_target_loss"]) > abs(ctrl_random_delta)
        and abs(targeted["delta_target_loss"]) >
        abs(ctrl_probe["delta_target_loss"])
    )
    pipeline_pass = bool(score["causal_strength"] > 0 and beats_all)
    return {
        "pipeline_pass": pipeline_pass,
        "matched_feature": matched,
        "decoder_cosine_to_planted": decoder_cosine,
        "targeted_delta": targeted["delta_target_loss"],
        "ctrl_unrelated_delta": ctrl_unrelated["delta_target_loss"],
        "ctrl_random_delta": ctrl_random_delta,
        "ctrl_random_delta_trials": rand,
        "ctrl_probe_delta": ctrl_probe["delta_target_loss"],
        "causal_score": score,
        "config": {"n_dim": n_dim, "n_features": n_features,
                   "sae_steps": sae_steps, "topk": topk,
                   "expansion": expansion, "seed": seed},
    }
