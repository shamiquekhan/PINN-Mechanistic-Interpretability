"""
Causal intervention engine — inserts a frozen SAE at a hidden layer
and supports 8 intervention modes (7 from the v2 plan plus the
probe_direction control added in v2.1) as specified in the
implementation plan.

Modes:
  natural              — original z (baseline, no modification)
  ablate               — z_k = 0 for target feature k
  amplify              — z_k *= alpha for target feature k
  replace              — z_k = reference_value from control run
  unrelated_control    — same intervention applied to an unrelated feature
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
    ):
        self._validate_mode(mode)
        self.sae = sae
        self.mode = mode
        self.feature_idx = feature_idx
        self.alpha = alpha
        self.reference_value = reference_value
        self.random_seed = random_seed
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
            "unrelated_control", "random_direction", "reconstruction_only",
            "probe_direction",
        }
        if mode not in valid:
            raise ValueError(f"Unknown mode '{mode}'. Choose from {valid}")

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
            # External-review C3a fix: on a TopK SAE (L0=k of latent_dim),
            # a uniformly sampled 'unrelated' feature is already-zero in
            # ~1-k/latent_dim of rows, so the old uniform choice was
            # frequently a literal no-op control.  Sample only from
            # features that are ACTIVE in at least one row (excluding the
            # target), so the control is a real perturbation.
            z = z.clone()
            rng = torch.Generator(device=z.device)
            rng.manual_seed(self.random_seed)
            active = (z > 0).any(dim=0)
            choices = [i for i in range(z.shape[1]) if i != k and bool(active[i])]
            if not choices:
                # Degenerate fallback: nothing but the target is active;
                # record it so the caller can report control_was_noop.
                self.last_control_was_noop = True
                return z
            self.last_control_was_noop = False
            ctrl_feat = choices[torch.randint(len(choices), (1,), generator=rng, device=z.device).item()]
            z[:, ctrl_feat] = 0.0
            return z

        if mode == "random_direction":
            # External-review C3b fix: the target intervention ABLATES
            # (removes a feature's per-row magnitude).  The old control
            # ADDED a random unit direction scaled by the same magnitude —
            # a different KIND of perturbation (injection vs deletion),
            # biased toward large loss effects.  The matched-deletion
            # control ablates a random ACTIVE non-target feature instead,
            # so both arms perturb the same way.
            z = z.clone()
            rng = torch.Generator(device=z.device)
            rng.manual_seed(self.random_seed)
            active = (z > 0).any(dim=0)
            choices = [i for i in range(z.shape[1]) if i != k and bool(active[i])]
            if not choices:
                self.last_control_was_noop = True
                return z
            self.last_control_was_noop = False
            ctrl_feat = choices[torch.randint(len(choices), (1,), generator=rng, device=z.device).item()]
            z[:, ctrl_feat] = 0.0
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
        # C3 diagnostic: set by _intervene for the control modes when the
        # sampled control degenerated (nothing but the target active).
        # measure_intervention_effect copies this into each result row.
        self.last_control_was_noop = False

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
) -> Dict:
    """Freeze model weights and measure how an intervention changes each loss.

    baseline_mode:
      "natural"              — raw activations, SAE bypassed (original v2 protocol).
      "reconstruction_only" — SAE inserted, no intervention.  This isolates the
                               intervention effect from SAE-insertion error and is
                               the correct baseline for the planted positive control
                               (and for any insertion-sensitive PINN).
    random_seed: seed for the random_direction / unrelated_control draws.
    """
    model.eval()
    sae.eval()

    x = pde.sample_interior(n_interior, device, dtype)

    # ---- Baseline ----
    hook_base = SAEInterventionHook(sae, mode=baseline_mode)
    hook_base.register(model, layer_index)
    with torch.enable_grad():
        r_base = pde.residual(model, x)
        lp_base = float((r_base ** 2).mean())
        bc_res_base = pde.boundary_residual(model)
        lb_base = float(bc_res_base.mean())
    hook_base.remove()

    # ---- Intervened ----
    hook_int = SAEInterventionHook(
        sae, mode=mode, feature_idx=feature_idx,
        alpha=alpha, reference_value=reference_value,
        random_seed=random_seed,
    )
    hook_int.register(model, layer_index)
    with torch.enable_grad():
        r_int = pde.residual(model, x)
        lp_int = float((r_int ** 2).mean())
        bc_res_int = pde.boundary_residual(model)
        lb_int = float(bc_res_int.mean())
    hook_int.remove()

    model.train()

    return {
        "feature_idx":   feature_idx,
        "mode":          mode,
        "layer_index":   layer_index,
        "delta_pde":     lp_int - lp_base,
        "delta_bc":      lb_int - lb_base,
        "baseline_pde":  lp_base,
        "baseline_bc":   lb_base,
        "intervened_pde": lp_int,
        "intervened_bc": lb_int,
        # C3 diagnostic: whether the sampled control degenerated (nothing
        # but the target active).  Non-control modes always False.
        "control_was_noop": bool(getattr(hook_int, "last_control_was_noop", False)),
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
            n_interior=n_interior, alpha=alpha, layer_index=layer_index,
            baseline_mode=baseline_mode,
        )

    model.eval()
    sae.eval()

    x = pde.sample_interior(n_interior, device, dtype)

    # ---- Baseline ----
    hook_base = SAEInterventionHook(sae, mode=baseline_mode)
    hook_base.register(model, layer_index)
    with torch.enable_grad():
        r_base = pde.residual(model, x)
        lp_base = float((r_base ** 2).mean())
        bc_res_base = pde.boundary_residual(model)
        lb_base = float(bc_res_base.mean())
    hook_base.remove()

    # ---- Probe-direction intervention ----
    hook_int = SAEInterventionHook(
        sae, mode="probe_direction", feature_idx=feature_idx,
        alpha=alpha, probe_direction=probe_direction,
    )
    hook_int.register(model, layer_index)
    with torch.enable_grad():
        r_int = pde.residual(model, x)
        lp_int = float((r_int ** 2).mean())
        bc_res_int = pde.boundary_residual(model)
        lb_int = float(bc_res_int.mean())
    hook_int.remove()

    model.train()

    return {
        "feature_idx":   feature_idx,
        "mode":          "probe_direction",
        "layer_index":   layer_index,
        "delta_pde":     lp_int - lp_base,
        "delta_bc":      lb_int - lb_base,
        "baseline_pde":  lp_base,
        "baseline_bc":   lb_base,
        "intervened_pde": lp_int,
        "intervened_bc": lb_int,
    }
