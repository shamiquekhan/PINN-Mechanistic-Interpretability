"""Tests for the PCA causal battery (interventions/pca_battery.py)."""
import numpy as np
import pytest
import torch

from analysis.pca_interpretability import fit_pca
from interventions.pca_battery import (
    PCAInterventionHook, measure_pca_intervention_effect,
    measure_pca_probe_effect, run_pca_causal_battery, summarize_battery,
    train_pca_probe_direction, compare_pca_vs_sae, pca_basis_to_device,
)
from pinn.model import MLP
from pinn.pdes import Poisson1D

DEVICE = torch.device("cpu")
DTYPE = torch.float32


def _setup():
    torch.manual_seed(0)
    pde = Poisson1D()
    model = MLP(1, 1, [16, 16], activation="tanh").to(device=DEVICE, dtype=DTYPE)
    acts = torch.randn(200, 16)
    basis = fit_pca(acts, n_components=4)
    return pde, model, basis


def test_natural_mode_is_identity():
    pde, model, basis = _setup()
    x = pde.validation_grid(16, DEVICE, DTYPE)
    with torch.no_grad():
        out_orig = model(x).clone()
    hook = PCAInterventionHook(basis, mode="natural", component_idx=0)
    hook.register(model, layer_index=1)
    with torch.no_grad():
        out_hooked = model(x)
    hook.remove()
    assert torch.allclose(out_orig, out_hooked, atol=1e-6)


def test_ablate_component_changes_output():
    pde, model, basis = _setup()
    x = pde.validation_grid(16, DEVICE, DTYPE)
    with torch.no_grad():
        out_orig = model(x).clone()
    hook = PCAInterventionHook(basis, mode="ablate_component",
                               component_idx=0, alpha=0.0)
    hook.register(model, layer_index=1)
    with torch.no_grad():
        out_hooked = model(x)
    hook.remove()
    assert not torch.allclose(out_orig, out_hooked, atol=1e-6)


def test_probe_direction_mode_requires_direction():
    pde, model, basis = _setup()
    try:
        PCAInterventionHook(basis, mode="probe_direction", component_idx=0)
        assert False, "must raise"
    except ValueError:
        pass


def test_probe_effect_perturbs():
    pde, model, basis = _setup()
    probe = torch.randn(basis.n_components)
    x = pde.validation_grid(16, DEVICE, DTYPE)
    with torch.no_grad():
        out_orig = model(x).clone()
    hook = PCAInterventionHook(basis, mode="probe_direction",
                               component_idx=0, probe_direction=probe)
    hook.register(model, layer_index=1)
    with torch.no_grad():
        out_hooked = model(x)
    hook.remove()
    assert not torch.allclose(out_orig, out_hooked, atol=1e-5)


def test_battery_runs_and_scores():
    pde, model, basis = _setup()
    labels = np.random.default_rng(0).integers(0, 2, 200)
    probe = train_pca_probe_direction(basis, torch.randn(200, 16),
                                      labels, DEVICE)
    rows = run_pca_causal_battery(
        model, basis, pde, DEVICE, DTYPE,
        candidate_components=[0, 1, 2, 3],
        probe_direction=probe, n_interior=32,
    )
    assert len(rows) == 4
    for r in rows:
        assert set(r["causal_score"].keys()) >= {
            "causal_strength", "specificity", "target_effect",
            "max_control_effect"}
    summary = summarize_battery(rows)
    assert summary["n_evaluations"] == 4
    assert "multiple_comparisons" in summary


def test_measure_pca_intervention_effect_runs():
    pde, model, basis = _setup()
    out = measure_pca_intervention_effect(
        model, basis, pde, DEVICE, DTYPE,
        component_idx=0, mode="ablate_component",
        n_interior=16, alpha=0.0, layer_index=1,
    )
    assert "delta_pde" in out and "delta_bc" in out
    # Ablating a nonzero-variance component must change the PDE loss.
    assert abs(out["delta_pde"]) > 0
    assert out["delta_total_pde"] == pytest.approx(
        out["delta_projection_pde"] + out["delta_intervention_pde"]
    )


def test_pca_reconstruction_only_is_a_valid_baseline():
    pde, model, basis = _setup()
    points = pde.sample_interior(16, DEVICE, DTYPE, seed=23)
    out = measure_pca_intervention_effect(
        model, basis, pde, DEVICE, DTYPE,
        component_idx=0, mode="ablate_component", n_interior=16,
        alpha=0.0, layer_index=1, seed=23,
        baseline_mode="reconstruction_only", evaluation_points=points,
    )
    assert out["baseline_mode"] == "reconstruction_only"
    assert out["delta_pde"] == pytest.approx(out["delta_intervention_pde"])


def test_measure_pca_probe_effect_runs():
    pde, model, basis = _setup()
    probe = torch.randn(basis.n_components)
    out = measure_pca_probe_effect(
        model, basis, probe, pde, DEVICE, DTYPE,
        component_idx=0, n_interior=16, layer_index=1,
    )
    assert out["mode"] == "probe_direction"
    assert abs(out["delta_pde"]) > 0


def test_compare_pca_vs_sae_pairs():
    pde, model, basis = _setup()
    labels = np.random.default_rng(0).integers(0, 2, 200)
    probe = train_pca_probe_direction(basis, torch.randn(200, 16),
                                      labels, DEVICE)
    rows = run_pca_causal_battery(
        model, basis, pde, DEVICE, DTYPE,
        candidate_components=[0, 1], probe_direction=probe, n_interior=16,
    )
    for r in rows:
        r["run"] = "run_a"
    cmp = compare_pca_vs_sae(rows, [dict(r) for r in rows])
    assert cmp["n_pairs"] == 2
    assert "mean_pca_causal_strength" in cmp


def test_compare_pca_vs_sae_joins_by_explicit_keys_not_order():
    pde, model, basis = _setup()
    labels = np.random.default_rng(0).integers(0, 2, 200)
    probe = train_pca_probe_direction(basis, torch.randn(200, 16), labels, DEVICE)
    pca_rows = run_pca_causal_battery(
        model, basis, pde, DEVICE, DTYPE,
        candidate_components=[0, 1], probe_direction=probe, n_interior=16,
    )
    sae_rows = [dict(row) for row in pca_rows]
    for row in pca_rows + sae_rows:
        row["run"] = "run_a"
    pca_rows.reverse()
    result = compare_pca_vs_sae(pca_rows, sae_rows)
    assert [pair["candidate_slot"] for pair in result["pairs"]] == [0, 1]


def test_compare_pca_vs_sae_rejects_missing_or_duplicate_keys():
    pde, model, basis = _setup()
    labels = np.random.default_rng(0).integers(0, 2, 200)
    probe = train_pca_probe_direction(basis, torch.randn(200, 16), labels, DEVICE)
    rows = run_pca_causal_battery(
        model, basis, pde, DEVICE, DTYPE,
        candidate_components=[0, 1], probe_direction=probe, n_interior=8,
    )
    for row in rows:
        row["run"] = "run_a"
    missing = [dict(rows[0])]
    missing[0].pop("candidate_slot")
    with pytest.raises(ValueError, match="candidate_slot"):
        compare_pca_vs_sae(rows[:1], missing)
    duplicate = [dict(rows[0]), dict(rows[0])]
    with pytest.raises(ValueError, match="duplicate"):
        compare_pca_vs_sae(duplicate, rows[:2])


def test_pca_basis_to_device():
    pde, model, basis = _setup()
    moved = pca_basis_to_device(basis, torch.device("cpu"))
    assert moved.n_components == basis.n_components
    assert torch.allclose(moved.components, basis.components)
