"""Tests for the prospective alignment/specificity protocol utilities."""
import numpy as np
import pytest
import torch

from analysis.activation_protocol import (
    ActivationStandardizer,
    load_activation_protocol_manifest,
    make_activation_split,
    write_activation_protocol_manifest,
)
from analysis.prospective_protocol import (
    alignment_scores,
    causal_differences,
    checkpoint_association,
    checkpoint_level_summary,
    choose_confirmatory_sample_size,
    pilot_power_plan_report,
)
from analysis.prospective_pilot import aggregate_checkpoint_pilot_row
from interventions.causal import run_inference_interventions
from interventions.engine import SAEInterventionHook
from interventions.prospective import (
    DirectHiddenStateInterventionHook,
    direct_hidden_state_counterfactual,
)
from pinn.model import MLP
from pinn.pdes import Poisson1D
from sae.model import SparseAutoencoder


def _identity_sae(width: int) -> SparseAutoencoder:
    sae = SparseAutoencoder(
        input_dim=width,
        latent_expansion=1,
        activation_mode="relul1",
        decoder_normalize=False,
    )
    with torch.no_grad():
        sae.b_a.zero_()
        sae.W_e.weight.copy_(torch.eye(width))
        sae.W_e.bias.zero_()
        sae.W_d.weight.copy_(torch.eye(width))
        sae.W_d.bias.zero_()
    return sae


def test_activation_split_is_deterministic_disjoint_and_manifested():
    first = make_activation_split(100, split_key="burgers:seed7:grid")
    second = make_activation_split(100, split_key="burgers:seed7:grid")
    assert np.array_equal(first.train_idx, second.train_idx)
    assert np.array_equal(first.selection_idx, second.selection_idx)
    assert np.array_equal(first.confirmatory_idx, second.confirmatory_idx)
    assert len(first.train_idx) == 60
    assert len(first.selection_idx) == 20
    assert len(first.confirmatory_idx) == 20
    assert len(np.unique(np.concatenate([
        first.train_idx, first.selection_idx, first.confirmatory_idx
    ]))) == 100
    assert first.manifest() == second.manifest()
    assert first.manifest()["assignment_sha256"]


def test_activation_split_rejects_invalid_fraction_or_small_bank():
    with pytest.raises(ValueError, match="sum to 1"):
        make_activation_split(10, split_key="x", fractions=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="too small"):
        make_activation_split(3, split_key="x")


def test_activation_protocol_manifest_round_trips_split_and_standardizer(tmp_path):
    activations = torch.arange(40, dtype=torch.float32).reshape(20, 2)
    split = make_activation_split(20, split_key="burgers:seed7:grid")
    stats = ActivationStandardizer.fit(activations[split.train_idx])
    path = tmp_path / "manifest.json"
    written = write_activation_protocol_manifest(
        path,
        split,
        standardizer=stats,
        activation_bank=activations,
        metadata={"pde_family": "burgers", "pinn_seed": 7},
    )
    payload, restored_split, restored_stats = load_activation_protocol_manifest(path)
    assert payload == written
    assert np.array_equal(restored_split.assignment, split.assignment)
    assert restored_stats is not None
    assert torch.equal(restored_stats.mean, stats.mean)
    assert torch.equal(restored_stats.std, stats.std)
    assert payload["activation_bank"]["shape"] == [20, 2]


def test_standardizer_is_fit_from_training_data_and_maps_directions():
    train = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    stats = ActivationStandardizer.fit(train)
    standardized = stats.transform(train)
    assert torch.allclose(standardized.mean(dim=0), torch.zeros(2))
    assert torch.allclose(stats.inverse(standardized), train)
    raw = stats.direction_to_raw(torch.tensor([1.0, -1.0]))
    assert torch.allclose(raw, stats.std * torch.tensor([1.0, -1.0]))
    assert stats.manifest()["width"] == 2


def test_direct_hidden_state_operator_converts_standardized_direction_to_raw():
    sae = _identity_sae(2)
    stats = ActivationStandardizer(
        mean=torch.tensor([1.0, 2.0]),
        std=torch.tensor([2.0, 3.0]),
    )
    hidden = torch.tensor([[3.0, 5.0]])  # standardized code is [1, 1]
    counterfactual, codes, delta = direct_hidden_state_counterfactual(
        hidden, sae, stats, feature_idx=0, latent_scale=0.0
    )
    assert torch.allclose(codes, torch.tensor([[1.0, 1.0]]))
    assert torch.allclose(delta, torch.tensor([[-2.0, 0.0]]))
    assert torch.allclose(counterfactual, torch.tensor([[1.0, 5.0]]))


def test_direct_hook_emits_provenance_and_preserves_model_shape():
    model = MLP(1, 1, [2], activation="tanh")
    sae = _identity_sae(2)
    stats = ActivationStandardizer(
        mean=torch.zeros(2),
        std=torch.ones(2),
    )
    hook = DirectHiddenStateInterventionHook(
        sae, stats, feature_idx=0, latent_scale=0.0, random_seed=7
    )
    hook.register(model, layer_index=0)
    output = model(torch.tensor([[0.25], [0.5]]))
    hook.remove()
    assert output.shape == (2, 1)
    assert hook.last_audit["operator_kind"] == "standardized_direct_component_scale"
    assert hook.last_audit["target_index"] == 0


def test_decoder_mediated_hook_round_trips_standardized_identity():
    sae = _identity_sae(2)
    stats = ActivationStandardizer(
        mean=torch.tensor([1.0, 2.0]),
        std=torch.tensor([2.0, 3.0]),
    )
    hook = SAEInterventionHook(
        sae,
        mode="reconstruction_only",
        activation_standardizer=stats,
    )
    output = torch.tensor([[3.0, 5.0]])
    reconstructed = hook._hook_fn(None, (), output)
    assert torch.allclose(reconstructed, output)


def test_batch_intervention_api_accepts_train_split_standardizer():
    device = torch.device("cpu")
    model = MLP(1, 1, [4, 4], activation="tanh").to(device)
    sae = SparseAutoencoder(
        input_dim=4,
        latent_expansion=1,
        activation_mode="relul1",
        decoder_normalize=False,
    ).to(device)
    stats = ActivationStandardizer.fit(torch.randn(32, 4))
    rows = run_inference_interventions(
        model=model,
        sae=sae,
        pde=Poisson1D(),
        device=device,
        dtype=torch.float32,
        candidate_features=[0],
        layer_index=0,
        n_interior=8,
        baseline_mode="reconstruction_only",
        sample_seed=4,
        activation_standardizer=stats,
    )
    assert len(rows) == 1
    assert rows[0]["targeted_ablate"]["baseline_mode"] == "reconstruction_only"


def test_alignment_and_causal_contrast_keep_controls_separate():
    codes = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, -1.0]])
    concepts = np.column_stack([codes[:, 0], np.array([0.0, 1.0, 0.0, -1.0])])
    scores = alignment_scores(codes, concepts)
    assert scores.shape == (2,)
    assert scores[0] == pytest.approx(1.0)
    diffs = causal_differences(
        [2.0, -1.0],
        np.array([[1.0, -0.5, 0.25], [0.25, 0.5, -0.25]]),
    )
    assert np.allclose(diffs, [1.0, 0.5])


def test_checkpoint_summary_uses_checkpoint_rows_not_grid_rows():
    assoc = checkpoint_association(
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5],
    )
    assert assoc["rho"] == pytest.approx(1.0)
    rows = [
        {"checkpoint_id": "b7", "pde_family": "burgers", "rho": 0.8},
        {"checkpoint_id": "b8", "pde_family": "burgers", "rho": 0.6},
        {"checkpoint_id": "a7", "pde_family": "allen_cahn", "rho": 0.7},
        {"checkpoint_id": "a8", "pde_family": "allen_cahn", "rho": 0.5},
    ]
    summary = checkpoint_level_summary(
        rows, n_boot=100, n_resamples=100, seed=3
    )
    assert summary["n_checkpoints"] == 4
    assert summary["pde_families"] == ["allen_cahn", "burgers"]
    assert summary["mean_rho"] > 0.0
    assert 0.0 <= summary["randomization_pvalue"] <= 1.0


def test_power_planner_returns_registered_sample_size_rule():
    plan = choose_confirmatory_sample_size(
        [0.15, 0.25, 0.35],
        n_simulations=20,
        n_resamples=20,
        seed=9,
    )
    assert set(plan["power_by_n_per_pde_family"]) == {"6", "8", "10", "12"}
    assert plan["selected_n_per_pde_family"] in {6, 8, 10, 12}
    assert plan["pilot_fisher_z_sd"] > 0.0


def test_pilot_power_plan_requires_registered_family_counts_and_is_serializable():
    rows = [
        {"checkpoint_id": f"b{i}", "pde_family": "burgers", "rho": 0.15 + 0.03 * i}
        for i in range(3)
    ] + [
        {"checkpoint_id": f"a{i}", "pde_family": "allen_cahn", "rho": 0.20 + 0.02 * i}
        for i in range(3)
    ]
    report = pilot_power_plan_report(
        rows,
        n_simulations=10,
        n_resamples=10,
        simulation_seed=4,
    )
    assert report["artifact_kind"] == "prospective_pilot_power_plan"
    assert report["pilot"]["confirmatory_outcomes_used"] is False
    assert report["pilot"]["n_checkpoints_by_pde_family"] == {
        "allen_cahn": 3,
        "burgers": 3,
    }
    assert report["power_plan"]["selected_n_per_pde_family"] in {6, 8, 10, 12}
    with pytest.raises(ValueError, match="pilot checkpoint count"):
        pilot_power_plan_report(rows[:-1], n_simulations=2, n_resamples=2)


def test_pilot_aggregation_requires_all_fixed_slots_and_keeps_controls_named():
    raw = {
        "checkpoint_id": "burgers_seed_7",
        "pde_family": "burgers",
        "intervention_arm": "direct_hidden_state",
        "candidates": [
            {
                "candidate_idx": idx,
                "alignment": float(idx),
                "batches": [
                    {
                        "batch_id": f"b{batch}",
                        "target_effect": float(idx + 1 + 0.01 * batch),
                        "control_effects": {
                            "matched": 0.10,
                            "random": -0.05,
                            "probe": 0.02,
                        },
                    }
                    for batch in range(3)
                ],
            }
            for idx in range(16)
        ],
    }
    result = aggregate_checkpoint_pilot_row(raw, expected_batch_count=3)
    assert result["intervention_arm"] == "direct_hidden_state"
    assert result["n_candidates"] == 16
    assert result["n_batches_per_candidate"] == [3]
    assert result["rho"] == pytest.approx(1.0)
    assert result["candidate_summaries"][0]["d_bar"] > 0.0
    raw["candidates"][0]["batches"][0]["control_effects"].pop("probe")
    with pytest.raises(KeyError, match="missing registered controls"):
        aggregate_checkpoint_pilot_row(raw, expected_batch_count=3)
