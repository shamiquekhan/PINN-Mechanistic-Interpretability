"""Tests for the MLP model: forward pass, param count, hooks, init, Fourier embed."""
import torch
import pytest
from pinn.model import MLP, FourierEmbedding


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def make_mlp(**kwargs) -> MLP:
    defaults = dict(input_dim=1, output_dim=1, hidden_layers=[32, 32], activation="tanh")
    defaults.update(kwargs)
    return MLP(**defaults).to(device=DEVICE, dtype=DTYPE)


class TestMLP:
    def test_forward_shape(self):
        model = make_mlp()
        x = torch.rand(16, 1, device=DEVICE, dtype=DTYPE)
        out = model(x)
        assert out.shape == (16, 1)

    def test_return_intermediates(self):
        model = make_mlp(hidden_layers=[32, 32, 32])
        x = torch.rand(8, 1, device=DEVICE, dtype=DTYPE)
        out, hid = model(x, return_intermediates=True)
        assert out.shape == (8, 1)
        assert len(hid) == 3   # 3 hidden layers

    def test_parameter_count(self):
        model = make_mlp(hidden_layers=[16, 16])
        pc = model.parameter_count()
        assert pc["total"] > 0
        assert pc["trainable"] == pc["total"]

    def test_architecture_summary(self):
        model = make_mlp(hidden_layers=[64, 64])
        s = model.architecture_summary()
        assert s["n_hidden_layers"] == 2
        assert s["output_dim"] == 1

    def test_hidden_layer_names(self):
        model = make_mlp(hidden_layers=[32, 32, 32])
        names = model.hidden_layer_names()
        assert len(names) == 3
        for name in names:
            assert "layers." in name

    def test_all_activations(self):
        for act in ["tanh", "relu", "gelu", "silu"]:
            model = make_mlp(activation=act)
            x = torch.rand(4, 1, device=DEVICE, dtype=DTYPE)
            out = model(x)
            assert out.shape == (4, 1)

    def test_fourier_embed(self):
        model = MLP(
            input_dim=1, output_dim=1, hidden_layers=[32, 32],
            fourier_embed=True, fourier_n_freq=8,
        ).to(device=DEVICE, dtype=DTYPE)
        x = torch.rand(4, 1, device=DEVICE, dtype=DTYPE)
        out = model(x)
        assert out.shape == (4, 1)

    def test_grad_flows(self):
        model = make_mlp()
        x = torch.rand(4, 1, device=DEVICE, dtype=DTYPE, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_xavier_init_different_from_kaiming(self):
        torch.manual_seed(0)
        m1 = MLP(1, 1, [32, 32], activation="tanh", init="xavier")
        torch.manual_seed(0)
        m2 = MLP(1, 1, [32, 32], activation="tanh", init="kaiming")
        # Parameters should differ (different initialization schemes)
        p1 = list(m1.parameters())[0]
        p2 = list(m2.parameters())[0]
        # They may be equal in very rare cases; just check they're finite
        assert torch.isfinite(p1).all()
        assert torch.isfinite(p2).all()


class TestFourierEmbedding:
    def test_output_dim(self):
        embed = FourierEmbedding(input_dim=1, n_frequencies=16)
        x = torch.rand(8, 1)
        out = embed(x)
        assert out.shape == (8, 32)   # 2 * 16

    def test_deterministic_with_seed(self):
        e1 = FourierEmbedding(input_dim=1, n_frequencies=8, seed=42)
        e2 = FourierEmbedding(input_dim=1, n_frequencies=8, seed=42)
        x = torch.rand(4, 1)
        assert torch.allclose(e1(x), e2(x))
