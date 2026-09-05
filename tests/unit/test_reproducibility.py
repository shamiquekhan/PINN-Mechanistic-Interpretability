"""Tests for reproducibility: same seed → same output, different seed → different output."""
import torch
import pytest
from pinn.reproducibility import set_seed
from pinn.model import MLP
from pinn.pdes import Poisson1D


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def _init_and_run(seed: int):
    set_seed(seed, deterministic=True)
    model = MLP(1, 1, [32, 32], activation="tanh").to(device=DEVICE, dtype=DTYPE)
    x = torch.rand(8, 1, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        out = model(x)
    return out.cpu()


class TestReproducibility:
    def test_same_seed_same_output(self):
        out1 = _init_and_run(seed=42)
        out2 = _init_and_run(seed=42)
        assert torch.allclose(out1, out2, atol=1e-6), \
            "Same seed produced different outputs — determinism broken"

    def test_different_seed_different_output(self):
        out1 = _init_and_run(seed=1)
        out2 = _init_and_run(seed=2)
        assert not torch.allclose(out1, out2, atol=1e-4), \
            "Different seeds produced identical outputs — seeding not applied"

    def test_set_seed_covers_numpy_and_random(self):
        import random, numpy as np
        set_seed(99)
        v1 = random.random()
        n1 = np.random.rand()
        set_seed(99)
        v2 = random.random()
        n2 = np.random.rand()
        assert abs(v1 - v2) < 1e-12
        assert abs(n1 - n2) < 1e-12
