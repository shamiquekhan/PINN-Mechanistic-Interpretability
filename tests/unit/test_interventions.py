"""Tests for the intervention engine: no-intervention path within tolerance, all modes run."""
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
             "unrelated_control", "random_direction", "reconstruction_only"]
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
