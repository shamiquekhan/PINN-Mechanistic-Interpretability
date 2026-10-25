"""Checkpoint-level analysis and power planning for the prospective protocol.

These functions deliberately keep the trained-checkpoint as the inferential
unit.  Candidate features and evaluation batches are repeated measurements
nested inside that unit and are never silently promoted to independent samples.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata


def alignment_scores(
    codes: np.ndarray,
    concept_fields: np.ndarray | Mapping[str, np.ndarray],
) -> np.ndarray:
    """Return max absolute concept correlation for every candidate column.

    Parameters
    ----------
    codes:
        Matrix with shape ``(n_rows, n_candidates)``.
    concept_fields:
        Matrix with shape ``(n_rows, n_concepts)`` or a mapping whose values
        are one-dimensional fields. Constant columns have correlation zero.
    """
    z = np.asarray(codes, dtype=float)
    if z.ndim != 2 or z.shape[0] < 2:
        raise ValueError("codes must have shape (rows, candidates) with >=2 rows")
    if isinstance(concept_fields, Mapping):
        if not concept_fields:
            raise ValueError("concept_fields must not be empty")
        fields = np.column_stack([np.asarray(v, dtype=float)
                                  for v in concept_fields.values()])
    else:
        fields = np.asarray(concept_fields, dtype=float)
    if fields.ndim == 1:
        fields = fields[:, None]
    if fields.ndim != 2 or fields.shape[0] != z.shape[0]:
        raise ValueError("concept fields must have the same row count as codes")
    if not np.isfinite(z).all() or not np.isfinite(fields).all():
        raise ValueError("codes and concept fields must be finite")

    z_centered = z - z.mean(axis=0, keepdims=True)
    c_centered = fields - fields.mean(axis=0, keepdims=True)
    z_norm = np.linalg.norm(z_centered, axis=0)
    c_norm = np.linalg.norm(c_centered, axis=0)
    numer = z_centered.T @ c_centered
    denom = z_norm[:, None] * c_norm[None, :]
    correlations = np.divide(
        numer,
        denom,
        out=np.zeros_like(numer, dtype=float),
        where=denom > 0,
    )
    return np.max(np.abs(correlations), axis=1)


def causal_differences(
    target_effect: Sequence[float],
    control_effects: np.ndarray,
) -> np.ndarray:
    """Compute the audited absolute target-versus-maximum-control contrast."""
    target = np.asarray(target_effect, dtype=float)
    controls = np.asarray(control_effects, dtype=float)
    if target.ndim != 1 or controls.ndim != 2:
        raise ValueError("target must be 1-D and controls must be 2-D")
    if controls.shape[0] != target.shape[0] or controls.shape[1] < 1:
        raise ValueError("target and controls have incompatible shapes")
    if not np.isfinite(target).all() or not np.isfinite(controls).all():
        raise ValueError("effects must be finite")
    return np.abs(target) - np.max(np.abs(controls), axis=1)


def signed_causal_strength(
    target_effect: Sequence[float],
    control_effects: np.ndarray,
) -> np.ndarray:
    """Compute the signed project diagnostic without changing control labels."""
    target = np.asarray(target_effect, dtype=float)
    controls = np.asarray(control_effects, dtype=float)
    if target.ndim != 1 or controls.ndim != 2:
        raise ValueError("target must be 1-D and controls must be 2-D")
    if controls.shape[0] != target.shape[0] or controls.shape[1] < 1:
        raise ValueError("target and controls have incompatible shapes")
    return target - np.max(controls, axis=1)


def checkpoint_association(
    alignment: Sequence[float],
    causal_summary: Sequence[float],
) -> Dict[str, Any]:
    """Compute a candidate-level Spearman association for one checkpoint."""
    a = np.asarray(alignment, dtype=float)
    c = np.asarray(causal_summary, dtype=float)
    if a.ndim != 1 or c.ndim != 1 or len(a) != len(c):
        raise ValueError("alignment and causal_summary must be equal-length vectors")
    if len(a) < 2:
        raise ValueError("at least two candidates are required")
    if not np.isfinite(a).all() or not np.isfinite(c).all():
        raise ValueError("alignment and causal_summary must be finite")
    a_rank = rankdata(a, method="average")
    c_rank = rankdata(c, method="average")
    a_centered = a_rank - a_rank.mean()
    c_centered = c_rank - c_rank.mean()
    denom = np.linalg.norm(a_centered) * np.linalg.norm(c_centered)
    rho = float(a_centered @ c_centered / denom) if denom > 0 else float("nan")
    return {
        "rho": rho,
        "n_candidates": int(len(a)),
        "n_alignment_ties": int(len(a) - len(np.unique(a))),
        "n_causal_ties": int(len(c) - len(np.unique(c))),
        "association_valid": bool(np.isfinite(rho)),
    }


def _validate_checkpoint_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("at least one checkpoint result is required")
    values = []
    groups = []
    for row in rows:
        if "rho" not in row or "pde_family" not in row:
            raise KeyError("checkpoint rows require rho and pde_family")
        rho = float(row["rho"])
        if not np.isfinite(rho) or abs(rho) > 1.0:
            raise ValueError("checkpoint rho must be finite and inside [-1, 1]")
        # Perfect rank agreement is valid; clip only to make Fisher-z finite.
        values.append(np.arctanh(np.clip(rho, -1.0 + 1e-12, 1.0 - 1e-12)))
        groups.append(str(row["pde_family"]))
    return np.asarray(values, dtype=float), np.asarray(groups, dtype=object)


def _blocked_mean(values: np.ndarray, groups: np.ndarray) -> float:
    unique_groups = sorted(set(groups.tolist()))
    if not unique_groups:
        raise ValueError("at least one blocking group is required")
    return float(np.mean([
        values[groups == group].mean() for group in unique_groups
    ]))


def _sign_flip_pvalue(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    observed = _blocked_mean(values, groups)
    exceed = 0
    for _ in range(n_resamples):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(values))
        # Sign flips remain at the checkpoint level, while the statistic gives
        # each PDE-family block equal weight.
        shuffled = values * signs
        if _blocked_mean(shuffled, groups) >= observed - 1e-15:
            exceed += 1
    return float((exceed + 1) / (n_resamples + 1))


def checkpoint_level_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_boot: int = 2_000,
    n_resamples: int = 10_000,
    seed: int = 0,
    meaningful_rho: float = 0.30,
) -> Dict[str, Any]:
    """Summarize checkpoint-level association with cluster uncertainty."""
    if not 0.0 < meaningful_rho < 1.0:
        raise ValueError("meaningful_rho must be in (0, 1)")
    values, groups = _validate_checkpoint_rows(rows)
    if n_boot < 1 or n_resamples < 1:
        raise ValueError("bootstrap and randomization counts must be positive")
    rng = np.random.default_rng(seed)
    observed_z = _blocked_mean(values, groups)
    group_names = sorted(set(groups.tolist()))
    boot_means = []
    for _ in range(n_boot):
        sample = []
        for group in group_names:
            idx = np.flatnonzero(groups == group)
            sample.extend(values[rng.choice(idx, size=len(idx), replace=True)])
        boot_means.append(float(np.mean(sample)))
    lo, hi = np.percentile(np.asarray(boot_means), [2.5, 97.5])
    return {
        "n_checkpoints": int(len(values)),
        "pde_families": group_names,
        "mean_fisher_z": observed_z,
        "mean_rho": float(np.tanh(observed_z)),
        "meaningful_rho": float(meaningful_rho),
        "meaningful_fisher_z": float(np.arctanh(meaningful_rho)),
        "bootstrap_ci_fisher_z": [float(lo), float(hi)],
        "bootstrap_ci_rho": [float(np.tanh(lo)), float(np.tanh(hi))],
        "randomization_pvalue": _sign_flip_pvalue(
            values, groups, n_resamples=n_resamples, seed=seed + 1
        ),
        "checkpoint_rows": [dict(row) for row in rows],
    }


def _simulated_randomization_pvalue(
    values: np.ndarray,
    *,
    n_resamples: int,
    rng: np.random.Generator,
) -> float:
    observed = float(values.mean())
    signs = rng.choice(
        np.array([-1.0, 1.0]), size=(n_resamples, len(values))
    )
    null_means = (signs * values[None, :]).mean(axis=1)
    return float((1 + np.count_nonzero(null_means >= observed - 1e-15)) /
                 (n_resamples + 1))


def choose_confirmatory_sample_size(
    pilot_rhos: Sequence[float],
    *,
    candidates: Iterable[int] = (6, 8, 10, 12),
    target_rho: float = 0.30,
    alpha: float = 0.05,
    target_power: float = 0.80,
    n_simulations: int = 2_000,
    n_resamples: int = 2_000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Apply the registered pilot-to-confirmatory sample-size rule.

    ``n`` is the number of independent PINN seeds per PDE family.  The power
    simulation uses two equally sized PDE-family blocks and the same
    checkpoint-level sign-randomization test used by the primary analysis.
    """
    pilot = np.asarray(pilot_rhos, dtype=float)
    if pilot.ndim != 1 or pilot.size < 2:
        raise ValueError("pilot_rhos must contain at least two values")
    if not np.isfinite(pilot).all() or np.any(np.abs(pilot) > 1.0):
        raise ValueError("pilot_rhos must be finite and inside [-1, 1]")
    if not 0.0 < target_rho < 1.0:
        raise ValueError("target_rho must be in (0, 1)")
    if not 0.0 < alpha < 1.0 or not 0.0 < target_power <= 1.0:
        raise ValueError("alpha and target_power must be valid probabilities")
    candidates_tuple = tuple(int(n) for n in candidates)
    if not candidates_tuple or any(n < 1 for n in candidates_tuple):
        raise ValueError("candidates must contain positive sample sizes")
    if n_simulations < 1 or n_resamples < 1:
        raise ValueError("simulation counts must be positive")

    pilot_z = np.arctanh(np.clip(pilot, -1.0 + 1e-12, 1.0 - 1e-12))
    sigma = float(np.std(pilot_z, ddof=1))
    sigma = max(sigma, 1e-6)
    true_z = float(np.arctanh(target_rho))
    rng = np.random.default_rng(seed)
    power_by_n: Dict[str, float] = {}
    for n_per_family in candidates_tuple:
        n_total = 2 * n_per_family
        successes = 0
        for _ in range(n_simulations):
            simulated = rng.normal(true_z, sigma, size=n_total)
            p = _simulated_randomization_pvalue(
                simulated, n_resamples=n_resamples, rng=rng
            )
            successes += int(p <= alpha)
        power_by_n[str(n_per_family)] = float(successes / n_simulations)

    selected = next(
        (n for n in candidates_tuple if power_by_n[str(n)] >= target_power),
        max(candidates_tuple),
    )
    return {
        "pilot_n": int(pilot.size),
        "pilot_rhos": pilot.tolist(),
        "pilot_fisher_z_sd": sigma,
        "target_rho": target_rho,
        "alpha": alpha,
        "target_power": target_power,
        "power_by_n_per_pde_family": power_by_n,
        "selected_n_per_pde_family": int(selected),
        "hit_target_power": bool(power_by_n[str(selected)] >= target_power),
        "n_simulations": int(n_simulations),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def pilot_power_plan_report(
    pilot_rows: Sequence[Mapping[str, Any]],
    *,
    protocol_version: str = "1.0",
    expected_pilot_per_family: int = 3,
    candidates: Iterable[int] = (6, 8, 10, 12),
    target_rho: float = 0.30,
    alpha: float = 0.05,
    target_power: float = 0.80,
    n_simulations: int = 2_000,
    n_resamples: int = 2_000,
    simulation_seed: int = 0,
) -> Dict[str, Any]:
    """Build the preregistered pilot-to-confirmatory power-plan artifact.

    ``pilot_rows`` must contain one row per independent pilot checkpoint with
    ``checkpoint_id``, ``pde_family``, and checkpoint-level ``rho``.  The
    default requires the registered three pilot checkpoints per PDE family.
    Confirmatory outcomes are intentionally not accepted by this function.
    """
    if expected_pilot_per_family < 1:
        raise ValueError("expected_pilot_per_family must be positive")
    if not pilot_rows:
        raise ValueError("pilot_rows must not be empty")
    candidates_tuple = tuple(int(n) for n in candidates)

    normalized = []
    seen_ids = set()
    for row in pilot_rows:
        if "checkpoint_id" not in row:
            raise KeyError("pilot rows require checkpoint_id")
        checkpoint_id = str(row["checkpoint_id"])
        if checkpoint_id in seen_ids:
            raise ValueError(f"duplicate pilot checkpoint_id: {checkpoint_id}")
        seen_ids.add(checkpoint_id)
        if "pde_family" not in row or "rho" not in row:
            raise KeyError("pilot rows require pde_family and rho")
        rho = float(row["rho"])
        if not np.isfinite(rho) or abs(rho) > 1.0:
            raise ValueError("pilot rho must be finite and inside [-1, 1]")
        normalized.append({
            **dict(row),
            "checkpoint_id": checkpoint_id,
            "pde_family": str(row["pde_family"]),
            "rho": rho,
        })

    normalized.sort(key=lambda row: (row["pde_family"], row["checkpoint_id"]))
    family_counts: Dict[str, int] = {}
    for row in normalized:
        family = row["pde_family"]
        family_counts[family] = family_counts.get(family, 0) + 1
    if len(family_counts) != 2:
        raise ValueError("the registered power plan requires exactly two PDE families")
    if any(count != expected_pilot_per_family for count in family_counts.values()):
        raise ValueError(
            "pilot checkpoint count must match expected_pilot_per_family for "
            "each PDE family"
        )

    power_plan = choose_confirmatory_sample_size(
        [row["rho"] for row in normalized],
        candidates=candidates_tuple,
        target_rho=target_rho,
        alpha=alpha,
        target_power=target_power,
        n_simulations=n_simulations,
        n_resamples=n_resamples,
        seed=simulation_seed,
    )
    return {
        "schema_version": 1,
        "artifact_kind": "prospective_pilot_power_plan",
        "protocol_version": str(protocol_version),
        "pilot": {
            "checkpoint_rows": normalized,
            "n_checkpoints": len(normalized),
            "pde_families": sorted(family_counts),
            "n_checkpoints_by_pde_family": family_counts,
            "expected_per_pde_family": expected_pilot_per_family,
            "confirmatory_outcomes_used": False,
        },
        "power_plan": power_plan,
        "decision_rule": {
            "independent_unit": "trained PINN checkpoint",
            "blocking_factor": "PDE family",
            "effect_scale": "Fisher-z checkpoint-level Spearman rho",
            "target_rho": target_rho,
            "one_sided_alpha": alpha,
            "target_power": target_power,
            "candidate_n_per_pde_family": list(candidates_tuple),
            "fallback": "use maximum candidate and report achieved power",
        },
    }
