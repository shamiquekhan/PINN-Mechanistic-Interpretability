"""Validate and aggregate raw rows for the prospective pilot.

This module is deliberately an aggregation boundary, not a model runner.  A
model-evaluation driver must produce the explicit candidate/batch schema below
before this code will compute a checkpoint-level pilot association.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from analysis.prospective_protocol import (
    causal_differences,
    checkpoint_association,
    signed_causal_strength,
)


REQUIRED_CONTROLS = ("matched", "random", "probe")
PRIMARY_INTERVENTION_ARM = "direct_hidden_state"


def _finite_scalar(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_candidate(
    candidate: Mapping[str, Any],
    *,
    expected_batch_count: int | None,
) -> Dict[str, Any]:
    if "candidate_idx" not in candidate:
        raise KeyError("candidate rows require candidate_idx")
    if "alignment" not in candidate or "batches" not in candidate:
        raise KeyError("candidate rows require alignment and batches")
    candidate_idx = int(candidate["candidate_idx"])
    if candidate_idx < 0:
        raise ValueError("candidate_idx must be non-negative")
    alignment = _finite_scalar(candidate["alignment"], "alignment")
    batches = candidate["batches"]
    if not isinstance(batches, Sequence) or isinstance(batches, (str, bytes)):
        raise TypeError("batches must be a sequence of mappings")
    if not batches:
        raise ValueError("each candidate requires at least one causal batch")
    if expected_batch_count is not None and len(batches) != expected_batch_count:
        raise ValueError("all candidates must have the registered batch count")

    normalized_batches = []
    seen_batch_ids = set()
    for ordinal, batch in enumerate(batches):
        if not isinstance(batch, Mapping):
            raise TypeError("each causal batch must be a mapping")
        batch_id = str(batch.get("batch_id", ordinal))
        if batch_id in seen_batch_ids:
            raise ValueError(f"duplicate batch_id within candidate: {batch_id}")
        seen_batch_ids.add(batch_id)
        if "target_effect" not in batch or "control_effects" not in batch:
            raise KeyError("causal batches require target_effect and control_effects")
        controls = batch["control_effects"]
        if not isinstance(controls, Mapping):
            raise TypeError("control_effects must be a named mapping")
        missing = [name for name in REQUIRED_CONTROLS if name not in controls]
        if missing:
            raise KeyError(f"missing registered controls: {missing}")
        normalized_batches.append({
            "batch_id": batch_id,
            "target_effect": _finite_scalar(
                batch["target_effect"], "target_effect"
            ),
            "control_effects": {
                name: _finite_scalar(controls[name], f"control_{name}")
                for name in REQUIRED_CONTROLS
            },
        })
    return {
        "candidate_idx": candidate_idx,
        "alignment": alignment,
        "batches": normalized_batches,
    }


def aggregate_checkpoint_pilot_row(
    row: Mapping[str, Any],
    *,
    expected_candidates: int = 16,
    expected_batch_count: int | None = None,
) -> Dict[str, Any]:
    """Validate one checkpoint and compute its primary pilot association.

    The input must contain all fixed candidate slots.  No candidate selection,
    dose selection, control selection, or checkpoint selection occurs here.
    """
    required = ("checkpoint_id", "pde_family", "candidates")
    missing = [key for key in required if key not in row]
    if missing:
        raise KeyError(f"pilot checkpoint row missing keys: {missing}")
    arm = str(row.get("intervention_arm", PRIMARY_INTERVENTION_ARM))
    if arm != PRIMARY_INTERVENTION_ARM:
        raise ValueError(
            f"primary pilot aggregation requires {PRIMARY_INTERVENTION_ARM!r}"
        )
    candidates = row["candidates"]
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise TypeError("candidates must be a sequence of mappings")
    normalized = [
        _validate_candidate(candidate, expected_batch_count=expected_batch_count)
        for candidate in candidates
    ]
    indices = [candidate["candidate_idx"] for candidate in normalized]
    if len(normalized) != expected_candidates:
        raise ValueError(
            f"expected exactly {expected_candidates} fixed candidate slots"
        )
    if sorted(indices) != list(range(expected_candidates)):
        raise ValueError("candidate slots must be exactly 0..expected_candidates-1")

    primary_summaries = []
    for candidate in normalized:
        target = [batch["target_effect"] for batch in candidate["batches"]]
        controls = np.asarray([
            [batch["control_effects"][name] for name in REQUIRED_CONTROLS]
            for batch in candidate["batches"]
        ])
        d_values = causal_differences(target, controls)
        signed_values = signed_causal_strength(target, controls)
        primary_summaries.append({
            "candidate_idx": candidate["candidate_idx"],
            "alignment": candidate["alignment"],
            "d_bar": float(np.mean(d_values)),
            "e_t_signed": float(np.mean(signed_values)),
            "n_batches": len(candidate["batches"]),
            "d_values": d_values.tolist(),
            "batches": candidate["batches"],
        })

    association = checkpoint_association(
        [summary["alignment"] for summary in primary_summaries],
        [summary["d_bar"] for summary in primary_summaries],
    )
    return {
        "checkpoint_id": str(row["checkpoint_id"]),
        "pde_family": str(row["pde_family"]),
        "intervention_arm": arm,
        "rho": association["rho"],
        "association": association,
        "candidate_summaries": primary_summaries,
        "n_candidates": expected_candidates,
        "n_batches_per_candidate": sorted({
            summary["n_batches"] for summary in primary_summaries
        }),
        "provenance": dict(row.get("provenance", {})),
    }


def aggregate_pilot_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_candidates: int = 16,
    expected_batch_count: int | None = None,
) -> list[Dict[str, Any]]:
    """Aggregate and deterministically order multiple pilot checkpoints."""
    if not rows:
        raise ValueError("at least one pilot checkpoint row is required")
    output = [
        aggregate_checkpoint_pilot_row(
            row,
            expected_candidates=expected_candidates,
            expected_batch_count=expected_batch_count,
        )
        for row in rows
    ]
    ids = [row["checkpoint_id"] for row in output]
    if len(set(ids)) != len(ids):
        raise ValueError("pilot checkpoint_id values must be unique")
    return sorted(output, key=lambda row: (row["pde_family"], row["checkpoint_id"]))
