"""Tests for the operator causal battery (interventions/operator_battery.py)."""
import numpy as np
import torch
import torch.nn.functional as F

from interventions.operator_battery import (
    OperatorSAEHook, measure_operator_intervention,
    operator_positive_control, probe_direction_for_target,
    run_operator_causal_battery, summarize_operator_battery,
)
from operators.fno import FNO1d, green_function_dataset
from sae.model import SparseAutoencoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _trained_small_operator(seed=0):
    torch.manual_seed(seed)
    a, u = green_function_dataset(128, grid_points=32, seed=seed,
                                  device=DEVICE, nonlinear=False)
    model = FNO1d(width=16, modes=6, n_layers=2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(80):
        loss = F.mse_loss(model(a), u)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        states = model(a, return_intermediates=True)[1][-1].reshape(-1, 16)
    sae = SparseAutoencoder(input_dim=16, latent_expansion=4,
                            activation_mode="topk", topk=4,
                            decoder_normalize=True).to(DEVICE)
    opt_s = torch.optim.Adam(sae.parameters(), lr=1e-2)
    g = torch.Generator().manual_seed(seed + 5)
    for _ in range(150):
        idx = torch.randint(0, states.shape[0], (128,), generator=g).to(DEVICE)
        opt_s.zero_grad()
        loss, _ = sae.loss(states[idx])
        loss.backward(); opt_s.step()
        sae._normalise_decoder()
    sae.eval()
    return model, sae, a, u


def test_hook_natural_is_identity():
    model, sae, a, u = _trained_small_operator()
    with torch.no_grad():
        out0 = model(a).clone()
    h = OperatorSAEHook(sae, mode="natural")
    h.register(model, layer_index=-1)
    with torch.no_grad():
        out1 = model(a)
    h.remove()
    assert torch.allclose(out0, out1, atol=1e-6)


def test_hook_ablate_changes_output():
    model, sae, a, u = _trained_small_operator()
    with torch.no_grad():
        out0 = model(a).clone()
    h = OperatorSAEHook(sae, mode="ablate", feature_idx=0, alpha=0.0)
    h.register(model, layer_index=-1)
    with torch.no_grad():
        out1 = model(a)
    h.remove()
    assert not torch.allclose(out0, out1, atol=1e-6)


def test_hook_probe_requires_direction():
    model, sae, a, u = _trained_small_operator()
    try:
        OperatorSAEHook(sae, mode="probe_direction", feature_idx=0)
        assert False
    except ValueError:
        pass


def test_probe_direction_unit_norm():
    model, sae, a, u = _trained_small_operator()
    with torch.no_grad():
        states = model(a, return_intermediates=True)[1][-1]
    flat = states.reshape(-1, 16)
    target = np.abs(np.random.default_rng(0).normal(size=flat.shape[0]))
    d = probe_direction_for_target(sae, flat, target)
    assert d.shape == (sae.latent_dim,)
    assert abs(d.norm().item() - 1.0) < 1e-5


def test_measure_operator_intervention_runs():
    model, sae, a, u = _trained_small_operator()
    out = measure_operator_intervention(
        model, sae, a, u, feature_idx=0, mode="ablate", alpha=0.0,
    )
    assert "delta_target_loss" in out and "delta_spectral_stat" in out
    # Ablating an ACTIVE latent through the decoder must change the loss.
    assert out["delta_target_loss"] != 0.0


def test_battery_and_summary_shape():
    model, sae, a, u = _trained_small_operator()
    probe = torch.ones(sae.latent_dim) / (sae.latent_dim ** 0.5)
    rows = run_operator_causal_battery(
        model, sae, a, u, [0, 1, 2, 3], probe,
        layer_index=-1, alpha=0.0, n_random_trials=2,
    )
    assert len(rows) == 4
    for r in rows:
        assert {"causal_strength", "specificity"} <= set(r["causal_score"])
    s = summarize_operator_battery(rows)
    assert s["n_evaluations"] == 4
    assert "multiple_comparisons" in s
    assert s["multiple_comparisons"]["n_features_tested"] == 4


def test_positive_control_gate_small():
    """Reduced-scale machinery gate: must PASS through the real hook.

    Note: at reduced scale (width 32, topk 4 over 128 latents) the TopK SAE
    splits a planted direction across several latents, so the matched
    decoder cosine is ~0.4-0.5 (not ~1 as in the full-scale PINN control
    with denser dictionaries).  The gate criterion is causal, not
    representational: targeted ablation must beat all three controls.
    """
    pc = operator_positive_control(width=32, grid_points=16, n_samples=512,
                                   sae_steps=300, seed=0, device=DEVICE)
    assert pc["pipeline_pass"], (
        f"operator machinery gate failed: cosine="
        f"{pc['decoder_cosine_to_planted']:.3f}, targeted="
        f"{pc['targeted_delta']:.4f} vs random={pc['ctrl_random_delta']:.4f}"
    )
    assert abs(pc["targeted_delta"]) > 3 * abs(pc["ctrl_random_delta"])
    assert pc["decoder_cosine_to_planted"] > 0.3
