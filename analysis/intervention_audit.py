"""Machine-readable provenance for causal intervention operators.

The frozen campaign predates this schema.  New prospective audit runs must
record the operator that actually ran rather than inferring semantics from a
control name in a result file.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

import numpy as np
import torch


AUDIT_SCHEMA_VERSION = 1


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    array = tensor.detach().to(device="cpu").contiguous().numpy()
    return np.asarray(array).tobytes()


def direction_digest(
    direction: Optional[torch.Tensor],
    *,
    kind: str,
    index: Optional[int] = None,
) -> Dict[str, Any]:
    """Return a compact, reproducible identifier for a selected direction."""
    if direction is None:
        result: Dict[str, Any] = {"kind": kind}
        if index is not None:
            result["index"] = int(index)
        return result

    values = direction.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    return {
        "kind": kind,
        "dimension": int(values.numel()),
        "l2": float(torch.linalg.vector_norm(values).item()),
        "sha256": hashlib.sha256(_tensor_bytes(values)).hexdigest(),
    }


def perturbation_summary(before: torch.Tensor, after: torch.Tensor) -> Dict[str, Any]:
    """Summarize per-row intervention magnitudes without storing activations."""
    if before.shape != after.shape:
        raise ValueError(
            f"cannot compare intervention shapes {tuple(before.shape)} and "
            f"{tuple(after.shape)}"
        )
    if before.ndim == 0:
        delta = (after - before).reshape(1, 1)
    else:
        delta = (after - before).reshape(after.shape[0], -1)
    norms = torch.linalg.vector_norm(delta.detach(), dim=1).to(device="cpu")
    nonzero = norms > 1e-8
    return {
        "n_rows": int(norms.numel()),
        "l2_min": float(norms.min().item()) if norms.numel() else 0.0,
        "l2_mean": float(norms.mean().item()) if norms.numel() else 0.0,
        "l2_max": float(norms.max().item()) if norms.numel() else 0.0,
        "n_nonzero_rows": int(nonzero.sum().item()),
    }


def make_intervention_audit(
    *,
    mode: str,
    operator_kind: str,
    random_seed: int,
    target_index: Optional[int],
    selected_control_index: Optional[int],
    before: torch.Tensor,
    after: torch.Tensor,
    control_was_noop: bool,
    direction: Optional[torch.Tensor] = None,
    direction_kind: str = "none",
) -> Dict[str, Any]:
    """Build the shared provenance record emitted by intervention hooks."""
    perturbation = perturbation_summary(before, after)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "requested_mode": mode,
        "operator_kind": operator_kind,
        "random_seed": int(random_seed),
        "target_index": None if target_index is None else int(target_index),
        "selected_control_index": (
            None
            if selected_control_index is None
            else int(selected_control_index)
        ),
        "selected_direction": direction_digest(
            direction,
            kind=direction_kind,
            index=selected_control_index,
        ),
        "control_was_noop": bool(control_was_noop),
        "operator_was_noop": perturbation["n_nonzero_rows"] == 0,
        "perturbation": perturbation,
    }
