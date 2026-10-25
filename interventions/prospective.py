"""Standardization-aware direct hidden-state interventions.

The ordinary :class:`SAEInterventionHook` decodes an intervened latent back
through the SAE.  The prospective protocol also needs the distinct direct
operation used by PhysSAE: modify the original hidden state along one frozen
decoder direction, without replacing the whole activation by an SAE
reconstruction.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from analysis.activation_protocol import ActivationStandardizer
from analysis.intervention_audit import make_intervention_audit
from pinn.model import MLP
from sae.model import SparseAutoencoder, apply_topk


def direct_hidden_state_counterfactual(
    hidden: torch.Tensor,
    sae: SparseAutoencoder,
    standardizer: ActivationStandardizer,
    *,
    feature_idx: int,
    latent_scale: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply ``h_cf = h + (s - 1) z_k d_k`` in raw hidden coordinates.

    ``z`` is encoded from train-split-standardized activations.  The decoder
    direction is converted back to raw hidden-state coordinates before the
    perturbation is applied.  The returned tuple is ``(hidden_cf, z, delta)``;
    ``delta`` is the raw-coordinate addition to the original hidden state.
    """
    if hidden.ndim < 2:
        raise ValueError("hidden must have shape (batch, width)")
    if not 0 <= feature_idx < sae.latent_dim:
        raise IndexError("feature_idx is outside the SAE latent dimension")
    if hidden.shape[-1] != standardizer.width or hidden.shape[-1] != sae.input_dim:
        raise ValueError("hidden, standardizer, and SAE widths must agree")

    hidden_std = standardizer.transform(hidden)
    # Detach SAE parameters while retaining the derivative of the
    # counterfactual with respect to the model hidden state.
    raw = F.linear(
        hidden_std - sae.b_a.detach(),
        sae.W_e.weight.detach(),
        sae.W_e.bias.detach(),
    )
    z = F.relu(raw)
    if sae.activation_mode == "topk":
        z = apply_topk(z, k=sae.topk)
    direction_std = sae.W_d.weight[:, feature_idx].detach()
    direction_raw = standardizer.direction_to_raw(direction_std)
    coefficient = z[..., feature_idx].unsqueeze(-1)
    delta = (float(latent_scale) - 1.0) * coefficient * direction_raw
    return hidden + delta, z, delta


class DirectHiddenStateInterventionHook:
    """Forward hook for the direct, decoder-direction hidden-state operator."""

    def __init__(
        self,
        sae: SparseAutoencoder,
        standardizer: ActivationStandardizer,
        *,
        feature_idx: int,
        latent_scale: float = 0.0,
        random_seed: int = 0,
    ) -> None:
        if not 0 <= feature_idx < sae.latent_dim:
            raise IndexError("feature_idx is outside the SAE latent dimension")
        if standardizer.width != sae.input_dim:
            raise ValueError("standardizer width must match SAE input width")
        self.sae = sae
        self.standardizer = standardizer
        self.feature_idx = int(feature_idx)
        self.latent_scale = float(latent_scale)
        self.random_seed = int(random_seed)
        self.last_audit = None
        self._hook: Optional[torch.utils.hooks.RemovableHandle] = None

    def _hook_fn(self, _module: nn.Module, _inputs: tuple, output: torch.Tensor):
        counterfactual, _z, delta = direct_hidden_state_counterfactual(
            output,
            self.sae,
            self.standardizer,
            feature_idx=self.feature_idx,
            latent_scale=self.latent_scale,
        )
        direction_std = self.sae.W_d.weight[:, self.feature_idx].detach()
        direction_raw = self.standardizer.direction_to_raw(direction_std)
        self.last_audit = make_intervention_audit(
            mode="direct_hidden_state",
            operator_kind="standardized_direct_component_scale",
            random_seed=self.random_seed,
            target_index=self.feature_idx,
            selected_control_index=None,
            before=output,
            after=counterfactual,
            control_was_noop=False,
            direction=direction_raw,
            direction_kind="decoder_column_raw",
        )
        # ``delta`` is intentionally computed above and retained in the audit
        # path; returning the tensor preserves gradients to the model input.
        del delta
        return counterfactual

    def register(self, model: MLP, layer_index: int) -> None:
        if not 0 <= layer_index < len(model.acts):
            raise IndexError("layer_index is outside the model hidden layers")
        if self._hook is not None:
            raise RuntimeError("direct intervention hook is already registered")
        self._hook = model.acts[layer_index].register_forward_hook(self._hook_fn)

    def remove(self) -> None:
        if self._hook is not None:
            self._hook.remove()
        self._hook = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.remove()
