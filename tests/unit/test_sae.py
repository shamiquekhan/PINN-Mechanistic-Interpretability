"""Tests for the Sparse Autoencoder: reconstruction, sparsity, dead features."""
import torch
import pytest
from sae.model import SparseAutoencoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def make_sae(input_dim=16, expansion=2, sparsity=1e-3) -> SparseAutoencoder:
    return SparseAutoencoder(
        input_dim=input_dim,
        latent_expansion=expansion,
        sparsity_coeff=sparsity,
        decoder_normalize=True,
    ).to(device=DEVICE, dtype=DTYPE)


def make_data(n=100, d=16) -> torch.Tensor:
    return torch.randn(n, d, device=DEVICE, dtype=DTYPE)


class TestSparseAutoencoder:
    def test_forward_shapes(self):
        sae = make_sae(input_dim=16, expansion=4)
        data = make_data(32, 16)
        z, a_hat = sae(data)
        assert z.shape == (32, 64)      # expansion=4 → latent_dim=64
        assert a_hat.shape == (32, 16)

    def test_encode_nonnegative(self):
        sae = make_sae()
        data = make_data()
        z = sae.encode(data)
        assert (z >= 0).all(), "Encoder output should be non-negative (ReLU)"

    def test_loss_returns_positive(self):
        sae = make_sae()
        data = make_data()
        total, info = sae.loss(data)
        assert total.item() >= 0
        assert info["recon_loss"] >= 0
        assert info["sparsity_loss"] >= 0

    def test_dead_feature_tracking(self):
        sae = make_sae(input_dim=8, expansion=2)
        data = make_data(50, 8)
        z = sae.encode(data)
        sae.update_feature_use(z)
        frac = sae.dead_feature_fraction()
        assert 0.0 <= frac <= 1.0

    def test_decoder_normalisation(self):
        sae = make_sae()
        sae._normalise_decoder()
        norms = sae.W_d.weight.norm(dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_save_load(self, tmp_path):
        sae = make_sae(input_dim=8, expansion=2)
        path = tmp_path / "sae.pt"
        sae.save(path)
        sae2 = SparseAutoencoder.load(path, DEVICE)
        data = make_data(10, 8)
        z1, _ = sae(data)
        z2, _ = sae2(data)
        assert torch.allclose(z1, z2, atol=1e-5)

    def test_feature_summary(self):
        sae = make_sae()
        s = sae.feature_summary()
        assert "latent_dim" in s
        assert "dead_feature_fraction" in s

    def test_gradient_flows_through_loss(self):
        sae = make_sae()
        data = make_data(16, 16)
        total, _ = sae.loss(data)
        total.backward()
        for p in sae.parameters():
            assert p.grad is not None, f"No grad for {p.shape}"

    def test_reconstruction_better_than_mean(self):
        """After training for a few steps, reconstruction should beat the trivial mean predictor."""
        sae = make_sae(input_dim=16, expansion=4, sparsity=0)
        data = torch.randn(200, 16, device=DEVICE, dtype=DTYPE) * 2   # non-trivial variance
        opt = torch.optim.Adam(sae.parameters(), lr=1e-2)
        for _ in range(200):
            opt.zero_grad()
            total, _ = sae.loss(data)
            total.backward()
            opt.step()
            sae._normalise_decoder()

        sae.eval()
        with torch.no_grad():
            _, a_hat = sae(data)
        recon_mse = float(((data - a_hat) ** 2).mean())
        mean_pred = data.mean(dim=0, keepdim=True).expand_as(data)
        mean_mse  = float(((data - mean_pred) ** 2).mean())
        assert recon_mse < mean_mse, \
            f"SAE recon ({recon_mse:.4f}) should beat mean ({mean_mse:.4f})"
