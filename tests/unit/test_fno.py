"""Tests for the FNO operator module (operators/fno.py)."""
import torch

from operators.fno import FNO1d, SpectralConv1d, green_function_dataset

DEVICE = torch.device("cpu")
DTYPE = torch.float32


def test_spectral_conv_shapes():
    layer = SpectralConv1d(8, 8, modes=4)
    x = torch.randn(2, 8, 32)
    y = layer(x)
    assert y.shape == x.shape


def test_fno_forward_shape():
    model = FNO1d(in_channels=1, out_channels=1, width=8, modes=4, n_layers=2)
    x = torch.randn(3, 32, 1)
    y = model(x)
    assert y.shape == (3, 32, 1)


def test_fno_intermediates():
    model = FNO1d(width=8, modes=4, n_layers=3)
    x = torch.randn(2, 32, 1)
    y, intermediates = model(x, return_intermediates=True)
    assert len(intermediates) == 3
    for state in intermediates:
        assert state.shape == (2, 32, 8)


def test_fno_backward():
    model = FNO1d(width=8, modes=4, n_layers=2)
    x = torch.randn(2, 32, 1)
    y = model(x)
    loss = y.square().mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_green_dataset_shapes_and_determinism():
    a1, u1 = green_function_dataset(16, grid_points=64, seed=0)
    a2, u2 = green_function_dataset(16, grid_points=64, seed=0)
    assert a1.shape == (16, 64, 1)
    assert u1.shape == (16, 64, 1)
    assert torch.allclose(a1, a2)
    assert torch.allclose(u1, u2)


def test_green_dataset_linear_variant():
    a, u = green_function_dataset(8, grid_points=64, seed=1, nonlinear=False)
    assert torch.isfinite(u).all()


def test_fno_learns_small_task():
    """A few hundred steps should reduce MSE on the (linear) task."""
    torch.manual_seed(0)
    a_train, u_train = green_function_dataset(64, grid_points=32,
                                               seed=0, nonlinear=False)
    model = FNO1d(width=8, modes=6, n_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(60):
        pred = model(a_train)
        loss = torch.nn.functional.mse_loss(pred, u_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], "FNO must reduce training loss"
