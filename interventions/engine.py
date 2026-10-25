"""
Causal intervention engine — inserts a frozen SAE at a hidden layer
and supports targeted, reconstruction, matched-control, random-direction,
and probe-direction interventions.

Modes:
  natural              — original z (baseline, no modification)
  ablate               — z_k = 0 for target feature k
  amplify              — z_k *= alpha for target feature k
  replace              — z_k = reference_value from control run
  unrelated_control    — ablate an activity-matched non-target feature
  matched_deletion     — ablate a random active non-target feature
  random_direction     — perturb a random latent direction with matched norm
  reconstruction_only  — run SAE encode→decode without any intervention
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pinn.model import MLP
from sae.model import SparseAutoencoder


# ---------------------------------------------------------------------------
# SAE forward-hook insertion
# ---------------------------------------------------------------------------

class SAEInterventionHook:
    """Registers a forward hook on a specific hidden layer that:
       1. Encodes the activation through the frozen SAE
       2. Optionally modifies the latent code
       3. Decodes back and replaces the layer output
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        mode: str = "reconstruction_only",
        feature_idx: Optional[int] = None,
        alpha: float = 1.5,
        reference_value: Optional[float] = None,
        random_seed: int = 0,
        probe_direction: Optional[torch.Tensor] = None,
        control_idx: Optional[int] = None,
    ):
        self._validate_mode(mode)
        self.sae = sae
        self.mode = mode
        self.feature_idx = feature_idx
        self.alpha = alpha
        self.reference_value = reference_value
        self.random_seed = random_seed
        self.last_control_was_noop = False
        self.last_control_idx = None
        # R6 control-matching correction: explicitly designated control
        # target (the closest-mean-activity non-target atom, chosen by
        # the caller); when set, the control modes apply the operator to
        # THIS atom instead of sampling.
        self.control_idx = control_idx
        if mode == "probe_direction":
            if probe_direction is None:
                raise ValueError("probe_direction mode requires a probe_direction tensor")
            d = probe_direction.detach().to(next(sae.parameters()).device)
            self.probe_direction = d / d.norm().clamp(min=1e-8)
        else:
            self.probe_direction = None
        self._hook = None

    def _validate_mode(self, mode: str):
        valid = {
            "natural", "ablate", "amplify", "replace",
            "unrelated_control", "matched_deletion", "random_direction",
            "reconstruction_only",
            "probe_direction", "dose_matched_control",
        }
        if mode not in valid:
            raise ValueError(f"Unknown mode '{mode}'. Choose from {valid}")

    def _sample_active_non_target(self, z: torch.Tensor, k) -> int:
        """Sample an ACTIVE non-target feature for matched deletion.

        Uniform sampling is a no-op ~1-k/D of the time on TopK codes, so the
        selector restricts the draw to features active in at least one row.
        Shared by matched_deletion and dose_matched_control. If control_idx is
        set (the R6
        closest-mean-activity correction), it is used directly."""
        if self.control_idx is not None and self.control_idx != k:
            self.last_control_was_noop = False
            self.last_control_idx = self.control_idx
            return self.control_idx
        rng = torch.Generator(device=z.device)
        rng.manual_seed(self.random_seed)
        active = (z > 0).any(dim=0)
        choices = [i for i in range(z.shape[1]) if i != k and bool(active[i])]
        if not choices:
            # Degenerate fallback: nothing but the target is active;
            # record it so the caller can report control_was_noop.
            self.last_control_was_noop = True
            self.last_control_idx = None
            return -1
        self.last_control_was_noop = False
        self.last_control_idx = choices[torch.randint(
            len(choices), (1,), generator=rng, device=z.device
        ).item()]
        return self.last_control_idx

    def _closest_active_non_target(self, z: torch.Tensor, k) -> int:
        """Choose the active non-target with the closest mean activity.

        This is the deterministic activity-matched control.  It is kept
        separate from ``matched_deletion`` so the battery does not count the
        same random deletion twice under two names.
        """
        if self.control_idx is not None and self.control_idx != k:
            self.last_control_was_noop = False
            self.last_control_idx = self.control_idx
            return self.control_idx
        activity = z.abs().mean(dim=0)
        active = (z > 0).any(dim=0)
        choices = [i for i in range(z.shape[1]) if i != k and bool(active[i])]
        if not choices:
            self.last_control_was_noop = True
            self.last_control_idx = None
            return -1
        self.last_control_was_noop = False
        self.last_control_idx = min(
            choices, key=lambda i: (float((activity[i] - activity[k]).abs()), i)
        )
        return self.last_control_idx

    def _intervene(self, z: torch.Tensor) -> torch.Tensor:
        """Modify latent code z according to the current mode."""
        mode = self.mode
        k = self.feature_idx

        if mode in ("natural", "reconstruction_only"):
            return z

        if mode == "ablate":
            z = z.clone()
            z[:, k] = 0.0
            return z

        if mode == "amplify":
            z = z.clone()
            z[:, k] = z[:, k] * self.alpha
            return z

        if mode == "replace":
            z = z.clone()
            val = self.reference_value if self.reference_value is not None else 0.0
            z[:, k] = val
            return z

        if mode == "unrelated_control":
            # Activity-matched non-target ablation.  This is deliberately
            # deterministic; the random deletion control below is a separate
            # procedure with a different null hypothesis.
            z = z.clone()
            ctrl_feat = self._closest_active_non_target(z, k)
            if ctrl_feat < 0:
                return z
            z[:, ctrl_feat] = 0.0
            return z

        if mode == "matched_deletion":
            # Ablate a random active non-target feature.
            z = z.clone()
            ctrl_feat = self._sample_active_non_target(z, k)
            if ctrl_feat < 0:
                return z
            z[:, ctrl_feat] = 0.0
            return z

        if mode == "random_direction":
            # Draw one random latent-space direction per intervention and
            # match its per-row L2 dose to the target feature's activity.
            # This is intentionally an injection/rotation control, not a
            # second feature ablation.
            generator = torch.Generator(device=z.device)
            generator.manual_seed(self.random_seed)
            direction = torch.randn(
                z.shape[1], device=z.device, dtype=z.dtype, generator=generator
            )
            if k is not None:
                direction[k] = 0.0
            norm = direction.norm()
            if float(norm) <= 1e-8:
                self.last_control_was_noop = True
                return z
            direction = direction / norm
            per_row_mag = z[:, k].abs() if k is not None else z.abs().mean(dim=1)
            self.last_control_was_noop = bool(float(per_row_mag.max()) <= 1e-8)
            return z + self.alpha * direction.unsqueeze(0) * per_row_mag.unsqueeze(1)

        if mode == "dose_matched_control":
            # R6 dose-response control: the same operator as the target's
            # amplify-family intervention (z_j <- alpha * z_j) applied to a
            # matched-activity ACTIVE non-target atom, at the SAME dose
            # alpha — dose- and kind-matched (the registered R6 protocol,
            # docs/preregistration.md §R6).  Mirrors the alpha-scaled
            # control modes of the operator battery.
            z = z.clone()
            ctrl_feat = self._sample_active_non_target(z, k)
            if ctrl_feat < 0:
                return z
            z[:, ctrl_feat] = z[:, ctrl_feat] * self.alpha
            return z

        if mode == "probe_direction":
            # Supervised linear-probe direction, per-row magnitude matched to
            # the target feature's activity (same scaling as random control).
            per_row_mag = z[:, k].abs() if k is not None else \
                z.abs().mean(dim=1)
            out = z.clone() + self.probe_direction.unsqueeze(0) * per_row_mag.unsqueeze(1)
            return out

        return z

    def _hook_fn(self, module: nn.Module, input: tuple, output: torch.Tensor):
        if self.mode == "natural":
            return output

        # Use detached weights so autograd flows back to x, but frozen SAE
        # weights don't accumulate grads.
        w_e = self.sae.W_e.weight.detach()
        b_e = self.sae.W_e.bias.detach()
        b_a = self.sae.b_a.detach()
        w_d = self.sae.W_d.weight.detach()
        b_d = self.sae.W_d.bias.detach()

        # Encode exactly as the SAE was trained (ReLU, then TopK if topk mode).
        # The previous implementation skipped the TopK step, so intervened
        # latents did not match the SAE's true latent space.
        z = F.relu(F.linear(output - b_a, w_e, b_e))
        if self.sae.activation_mode == "topk":
            from sae.model import apply_topk
            z = apply_topk(z, k=self.sae.topk)
        z_mod = self._intervene(z)
        # Decode: a_hat = W_d z + b_d
        reconstructed = F.linear(z_mod, w_d, b_d)
        return reconstructed

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


# ---------------------------------------------------------------------------
# Inference-time counterfactual measurement
# ---------------------------------------------------------------------------

def measure_intervention_effect(
    model: MLP,
    sae: SparseAutoencoder,
    pde,
    device: torch.device,
    dtype: torch.dtype,
    feature_idx: int,
    mode: str,
    n_interior: int = 256,
    lambda_pde: float = 1.0,
    lambda_bc: float = 1.0,
    alpha: float = 1.5,
    reference_value: Optional[float] = None,
    layer_index: int = 1,
    baseline_mode: str = "natural",
    random_seed: int = 0,
    control_idx: Optional[int] = None,
    evaluation_points: Optional[torch.Tensor] = None,
    sample_seed: Optional[int] = None,
    probe_direction: Optional[torch.Tensor] = None,
) -> Dict:
    """Freeze model weights and measure how an intervention changes each loss.

    baseline_mode:
      "natural"              — raw activations, SAE bypassed (original v2 protocol).
      "reconstruction_only" — SAE inserted, no intervention.  This isolates the
                                intervention effect from SAE-insertion error and is
                                the correct baseline for the planted positive control
                                (and for any insertion-sensitive PINN).
    random_seed: seed for the random_direction / matched_deletion draws.
    control_idx: explicitly designated non-target control atom.  If omitted,
      each control mode uses its own selector.
    evaluation_points: optional shared collocation tensor.  Supplying this
      lets SAE and PCA measurements use exactly the same physical points.
    sample_seed: seed used only when evaluation_points is not supplied.
    probe_direction: optional supervised direction used by probe_direction.
    """
    if baseline_mode not in {"natural", "reconstruction_only"}:
        raise ValueError(
            "baseline_mode must be 'natural' or 'reconstruction_only'"
        )

    model_was_training = model.training
    sae_was_training = sae.training
    model.eval()
    sae.eval()

    if evaluation_points is None:
        x = pde.sample_interior(
            n_interior, device, dtype, seed=sample_seed
        )
    else:
        x = evaluation_points.detach().clone().to(device=device, dtype=dtype)

    def _losses(hook: Optional[SAEInterventionHook]):
        if hook is not None:
            hook.register(model, layer_index)
        try:
            with torch.enable_grad():
                residual = pde.residual(model, x)
                lp = float((residual ** 2).mean())
                boundary = pde.boundary_residual(model)
                lb = float(boundary.mean())
        finally:
            if hook is not None:
                hook.remove()
        return lp, lb

    try:
        lp_natural, lb_natural = _losses(None)
        lp_reconstruction, lb_reconstruction = _losses(
            SAEInterventionHook(sae, mode="reconstruction_only")
        )
        intervention_hook = SAEInterventionHook(
            sae, mode=mode, feature_idx=feature_idx,
            alpha=alpha, reference_value=reference_value,
            random_seed=random_seed, probe_direction=probe_direction,
            control_idx=control_idx,
        )
        lp_int, lb_int = _losses(intervention_hook)
        control_was_noop = bool(intervention_hook.last_control_was_noop)
    finally:
        model.train(model_was_training)
        sae.train(sae_was_training)

    lp_base = lp_reconstruction if baseline_mode == "reconstruction_only" else lp_natural
    lb_base = lb_reconstruction if baseline_mode == "reconstruction_only" else lb_natural

    return {
        "feature_idx":   feature_idx,
        "mode":          mode,
        "layer_index":   layer_index,
        "delta_pde":     lp_int - lp_base,
        "delta_bc":      lb_int - lb_base,
        "delta_total_pde": lp_int - lp_natural,
        "delta_total_bc": lb_int - lb_natural,
        "delta_projection_pde": lp_reconstruction - lp_natural,
        "delta_projection_bc": lb_reconstruction - lb_natural,
        "delta_intervention_pde": lp_int - lp_reconstruction,
        "delta_intervention_bc": lb_int - lb_reconstruction,
        "baseline_pde":  lp_base,
        "baseline_bc":   lb_base,
        "natural_pde": lp_natural,
        "natural_bc": lb_natural,
        "reconstruction_pde": lp_reconstruction,
        "reconstruction_bc": lb_reconstruction,
        "baseline_mode": baseline_mode,
        "intervened_pde": lp_int,
        "intervened_bc": lb_int,
        "control_was_noop": control_was_noop,
        "control_idx": intervention_hook.last_control_idx,
    }


def measure_probe_direction_effect(
    model: MLP,
    sae: SparseAutoencoder,
    pde,
    device: torch.device,
    dtype: torch.dtype,
    feature_idx: int,
    probe_direction: Optional[torch.Tensor],
    n_interior: int = 256,
    alpha: float = 1.5,
    layer_index: int = 1,
    baseline_mode: str = "natural",
    evaluation_points: Optional[torch.Tensor] = None,
    sample_seed: Optional[int] = None,
) -> Dict:
    """Measure loss shift when perturbing along a supervised linear-probe direction.

    This is the AXBench-style 'simple baseline' control: if the probe direction
    shifts the target loss as much as a discovered SAE feature, the feature
    adds no value over plain linear supervision.
    """
    if probe_direction is None:
        # Fall back to a random direction with the same protocol (flagged
        # by the caller via probe_control_valid=False).
        return measure_intervention_effect(
            model, sae, pde, device, dtype,
            feature_idx=feature_idx, mode="random_direction",
            n_interior=n_interior, alpha=1.0, layer_index=layer_index,
            baseline_mode=baseline_mode,
            evaluation_points=evaluation_points, sample_seed=sample_seed,
        )
    return measure_intervention_effect(
        model, sae, pde, device, dtype,
        feature_idx=feature_idx, mode="probe_direction",
        n_interior=n_interior, alpha=alpha, layer_index=layer_index,
        baseline_mode=baseline_mode, probe_direction=probe_direction,
        evaluation_points=evaluation_points, sample_seed=sample_seed,
    )
