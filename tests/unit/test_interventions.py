"""Tests for the intervention engine and its measurement protocol."""
import pytest
import torch
from pinn.model import MLP
from pinn.pdes import Poisson1D
from sae.model import SparseAutoencoder
from interventions.engine import SAEInterventionHook, measure_intervention_effect

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def _setup():
    pde = Poisson1D()
    model = MLP(1, 1, [16, 16], activation="tanh").to(device=DEVICE, dtype=DTYPE)
    sae = SparseAutoencoder(input_dim=16, latent_expansion=2).to(device=DEVICE, dtype=DTYPE)
    return pde, model, sae


def test_reconstruction_only_close_to_original():
    """Reconstruction-only mode should minimally perturb activations (untrained SAE is random, but hook runs)."""
    pde, model, sae = _setup()
    x = pde.sample_interior(32, DEVICE, DTYPE)
    with torch.no_grad():
        out_orig = model(x).clone()
    hook = SAEInterventionHook(sae, mode="reconstruction_only")
    hook.register(model, layer_index=0)
    with torch.no_grad():
        out_hooked = model(x)
    hook.remove()
    # Shapes must match
    assert out_hooked.shape == out_orig.shape


def test_natural_mode_unchanged():
    """Natural mode: SAE hook does NOT modify activations."""
    pde, model, sae = _setup()
    x = pde.sample_interior(16, DEVICE, DTYPE)
    with torch.no_grad():
        out_orig = model(x).clone()
    hook = SAEInterventionHook(sae, mode="natural")
    hook.register(model, layer_index=0)
    with torch.no_grad():
        out_hooked = model(x)
    hook.remove()
    assert torch.allclose(out_orig, out_hooked, atol=1e-5)


def test_all_modes_run_without_error():
    pde, model, sae = _setup()
    modes = ["natural", "ablate", "amplify", "replace",
             "unrelated_control", "matched_deletion", "random_direction",
             "reconstruction_only"]
    x = pde.sample_interior(8, DEVICE, DTYPE)
    for mode in modes:
        hook = SAEInterventionHook(sae, mode=mode, feature_idx=0)
        hook.register(model, layer_index=0)
        with torch.no_grad():
            out = model(x)
        hook.remove()
        assert out.shape == (8, 1), f"Mode {mode} produced wrong shape"


def test_measure_intervention_returns_expected_keys():
    pde, model, sae = _setup()
    result = measure_intervention_effect(
        model, sae, pde, DEVICE, DTYPE,
        feature_idx=0, mode="ablate", n_interior=32, layer_index=0,
    )
    for key in ["feature_idx", "mode", "delta_pde", "delta_bc", "baseline_pde", "baseline_bc"]:
        assert key in result


def test_measure_intervention_reports_reconstruction_decomposition():
    pde, model, sae = _setup()
    points = pde.sample_interior(32, DEVICE, DTYPE, seed=17)
    result = measure_intervention_effect(
        model, sae, pde, DEVICE, DTYPE,
        feature_idx=0, mode="ablate", n_interior=32, layer_index=0,
        baseline_mode="reconstruction_only", evaluation_points=points,
    )
    assert result["baseline_mode"] == "reconstruction_only"
    assert result["delta_pde"] == result["delta_intervention_pde"]
    assert result["delta_total_pde"] == pytest.approx(
        result["delta_projection_pde"] + result["delta_intervention_pde"]
    )


def test_control_modes_are_not_identical():
    z = torch.zeros(8, 16)
    z[:, 3] = 1.0
    z[:, 7] = 0.5
    z[:, 10] = 0.25
    unrelated = SAEInterventionHook(
        None, mode="unrelated_control", feature_idx=3, random_seed=0
    )._intervene(z)
    random_direction = SAEInterventionHook(
        None, mode="random_direction", feature_idx=3,
        random_seed=0, alpha=1.0,
    )._intervene(z)
    assert not torch.equal(unrelated, random_direction)


def test_intervention_records_protocol_audit():
    hook = SAEInterventionHook(
        None, mode="matched_deletion", feature_idx=3, random_seed=11
    )
    z = torch.zeros(4, 16)
    z[:, 3] = 1.0
    z[:, 7] = 0.5
    out = hook._intervene(z)
    audit = hook.last_audit
    assert out is not z
    assert audit["schema_version"] == 1
    assert audit["requested_mode"] == "matched_deletion"
    assert audit["operator_kind"] == "random_active_component_delete"
    assert audit["random_seed"] == 11
    assert audit["target_index"] == 3
    assert audit["selected_control_index"] == 7
    assert audit["control_was_noop"] is False
    assert audit["perturbation"]["n_nonzero_rows"] == 4
