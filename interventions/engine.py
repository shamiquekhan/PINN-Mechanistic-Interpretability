"""
Causal intervention engine — inserts a frozen SAE at a hidden layer
and supports 7 intervention modes as specified in the implementation plan.

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
    ):
        self._validate_mode(mode)
        self.sae = sae
        self.mode = mode
        self.feature_idx = feature_idx
        self.alpha = alpha
        self.reference_value = reference_value
        self.random_seed = random_seed
        self._hook = None

    def _validate_mode(self, mode: str):
        valid = {
            "natural", "ablate", "amplify", "replace",
            "unrelated_control", "random_direction", "reconstruction_only",
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
            z = z.clone()
            rng = torch.Generator(device=z.device)
            rng.manual_seed(self.random_seed)
            choices = [i for i in range(z.shape[1]) if i != k]
            ctrl_feat = choices[torch.randint(len(choices), (1,), generator=rng, device=z.device).item()]
            z[:, ctrl_feat] = 0.0
            return z

        if mode == "random_direction":
            z = z.clone()
            rng = torch.Generator(device=z.device)
            rng.manual_seed(self.random_seed)
            target_norm = z[:, k].norm()
            rand_dir = torch.randn(z.shape[1], device=z.device, dtype=z.dtype, generator=rng)
            rand_dir = rand_dir / rand_dir.norm().clamp(min=1e-8) * target_norm
            z = z + rand_dir.unsqueeze(0)
            return z

        return z

    def _hook_fn(self, module: nn.Module, input: tuple, output: torch.Tensor):
        if self.mode == "natural":
            return output

        # Use detached weights so autograd flows back to x, but frozen SAE weights don't accumulate grads
        w_e = self.sae.W_e.weight.detach()
        b_e = self.sae.W_e.bias.detach()
        b_a = self.sae.b_a.detach()
        w_d = self.sae.W_d.weight.detach()
        b_d = self.sae.W_d.bias.detach()

        # Encode: z = ReLU(W_e(a - b_a) + b_e)
        z = F.relu(F.linear(output - b_a, w_e, b_e))
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
) -> Dict:
    """Freeze model weights and measure how an intervention changes each loss."""
    model.eval()
    sae.eval()

    x = pde.sample_interior(n_interior, device, dtype)

    # ---- Baseline (natural) ----
    hook_base = SAEInterventionHook(sae, mode="natural")
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
    }
