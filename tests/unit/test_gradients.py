"""Tests for gradient logging: norms, cosine similarities, conflict score."""
import torch
from pinn.gradients import (
    compute_per_loss_gradients,
    compute_gradient_norms,
    compute_gradient_cosine_similarities,
    compute_gradient_conflict_score,
    log_gradient_stats,
)
from pinn.model import MLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def _model():
    return MLP(1, 1, [16, 16], activation="tanh").to(device=DEVICE, dtype=DTYPE)


def test_per_loss_gradient_keys():
    m = _model()
    x = torch.rand(8, 1, device=DEVICE, dtype=DTYPE)
    grads = compute_per_loss_gradients(m, {"pde": (m(x)**2).mean(), "bc": m(x).abs().mean()})
    assert "pde" in grads and "bc" in grads


def test_norms_positive():
    m = _model()
    x = torch.rand(8, 1, device=DEVICE, dtype=DTYPE)
    grads = compute_per_loss_gradients(m, {"a": (m(x)**2).mean()})
    assert compute_gradient_norms(grads)["a"] >= 0


def test_cosine_identical_losses():
    m = _model()
    x = torch.rand(8, 1, device=DEVICE, dtype=DTYPE)
    loss = (m(x)**2).mean()
    grads = compute_per_loss_gradients(m, {"a": loss, "b": loss.clone()})
    cosines = compute_gradient_cosine_similarities(grads)
    assert abs(cosines.get("a_vs_b", 0) - 1.0) < 1e-4


def test_conflict_score_no_conflict():
    m = _model()
    x = torch.rand(8, 1, device=DEVICE, dtype=DTYPE)
    loss = (m(x)**2).mean()
    grads = compute_per_loss_gradients(m, {"a": loss, "b": loss.clone()})
    cosines = compute_gradient_cosine_similarities(grads)
    assert compute_gradient_conflict_score(cosines) == 0.0


def test_log_gradient_stats_keys():
    m = _model()
    x = torch.rand(8, 1, device=DEVICE, dtype=DTYPE)
    grads = compute_per_loss_gradients(m, {"pde": (m(x)**2).mean()})
    stats = log_gradient_stats(grads)
    assert all(k in stats for k in ["gradient_norms", "gradient_cosines", "gradient_conflict_score"])
