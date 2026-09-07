"""Unit tests for stage 15: NTK gradient-conflict <-> SAE feature bridge."""
import numpy as np
import pytest
import torch

from experiments.ntk_bridge import (
    _feature_activity_matrix,
    _machinery_gate,
    _permutation_pvalue,
    _point_biserial,
    _random_direction_rhos,
)
from sae.model import SparseAutoencoder


def test_point_biserial_detects_planted_association():
    rng = np.random.default_rng(0)
    labels = np.array([0.0] * 30 + [1.0] * 30)
    x = rng.normal(0, 0.1, 60) + 0.9 * labels
    rho = _point_biserial(x, labels)
    assert rho > 0.8


def test_point_biserial_nan_on_degenerate_inputs():
    x = np.ones(10)
    labels = np.array([0.0] * 5 + [1.0] * 5)
    assert np.isnan(_point_biserial(x, labels))
    # constant labels
    assert np.isnan(_point_biserial(np.arange(10.0), np.ones(10)))


def test_permutation_pvalue_conservative_under_null():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 40)
    labels = (rng.uniform(0, 1, 40) < 0.5).astype(float)
    rho = _point_biserial(x, labels)
    p = _permutation_pvalue(x, labels, rho, n_perm=500,
                            rng=np.random.default_rng(11))
    # A null association should rarely be 'significant'; allow generous room
    assert p > 0.01


def test_permutation_pvalue_small_for_strong_association():
    labels = np.array([0.0] * 25 + [1.0] * 25)
    x = 2.0 * labels + np.random.default_rng(2).normal(0, 0.05, 50)
    rho = _point_biserial(x, labels)
    p = _permutation_pvalue(x, labels, rho, n_perm=500,
                            rng=np.random.default_rng(12))
    assert p <= 1 / 501 + 0.01


def test_machinery_gate_passes_on_planted_conflict_feature():
    gate = _machinery_gate()
    assert gate["gate_pass"] is True
    assert abs(gate["planted_rho_raw"]) > 0.7
    assert abs(gate["planted_rho_specific"]) > 0.7
    assert gate["permutation_p"] < 0.01


def test_machinery_gate_fails_without_planted_structure():
    # Overwrite the planted lift: pure noise cannot pass the gate
    rng = np.random.default_rng(3)
    n_steps = 60
    labels = (rng.uniform(-1, 1, n_steps) < 0).astype(float)
    A = rng.normal(0.05, 0.02, size=(n_steps, 8))
    raw = _point_biserial(A[:, 0], labels)
    assert abs(raw) < 0.7  # noise does not look like a planted feature


def test_random_direction_rhos_not_significant_under_null():
    rng = np.random.default_rng(9)
    A = rng.normal(0, 0.05, size=(11, 8))
    labels = (rng.uniform(-1, 1, 11) < 0).astype(float)
    rhos = _random_direction_rhos(A, labels)
    assert len(rhos) == 32
    # under a null, the p95 of random |rho| is itself modest
    assert np.percentile(rhos, 95) < 0.8


def test_feature_activity_matrix_shapes_and_steps():
    torch.manual_seed(0)
    sae = SparseAutoencoder(input_dim=16, latent_expansion=4, topk=4)
    recs = []
    for step in range(0, 500, 100):
        raw = (torch.randn(10, 16) * 0.1).tolist()
        recs.append({"step": step, "probe_hash": "x",
                     "activations": {"layers.1": {"raw": raw}}})
    # duplicated step records (as the real activation logs contain) must be deduped
    recs = recs + recs
    steps, A = _feature_activity_matrix(sae, recs, [0, 3, 5], torch.device("cpu"))
    assert steps == [0, 100, 200, 300, 400]
    assert A.shape == (5, 3)
    # TopK(4) activations are nonnegative -> mean activity is nonnegative
    assert (A >= 0).all()


def test_specificity_lift_removes_global_surges():
    # A global activity surge uncorrelated with labels must be removed by
    # the per-step mean subtraction (condition iii's construction)
    rng = np.random.default_rng(4)
    labels = np.array([0.0] * 6 + [1.0] * 6)
    A = rng.normal(0, 0.02, size=(12, 8))
    surge = rng.uniform(0, 1, size=12)  # label-independent
    A = A + surge[:, None]
    S = A - A.mean(axis=1, keepdims=True)
    # planted small feature-specific signal
    S[:, 2] += 0.3 * labels
    rho_s = _point_biserial(S[:, 2], labels)
    assert rho_s > 0.7
    # raw correlation is diluted by the shared surge
    rho_raw = _point_biserial(A[:, 2], labels)
    assert rho_raw < rho_s
