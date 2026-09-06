"""Tests for the SOTA baseline trainers (small-scale smoke)."""
from pathlib import Path

import numpy as np

from experiments.sota_baselines import (
    train_gradnorm, train_ntk_adaptive, train_rba,
)
from pinn.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG = PROJECT_ROOT / "configs" / "failure_boundary_starvation.yaml"


def _cfg():
    return load_config(CONFIG)


def test_train_gradnorm_smoke(monkeypatch=None):
    cfg = _cfg()
    out = train_gradnorm(cfg, total_steps=30)
    assert len(out["traj"]) >= 1
    assert np.isfinite(out["final_rel_l2"])
    # Weights must stay positive and bounded.
    lam = out["traj"][-1]
    assert lam["lambda_pde"] > 0 and lam["lambda_bc"] > 0
    assert lam["lambda_pde"] < 1e6 and lam["lambda_bc"] < 1e6


def test_train_ntk_adaptive_smoke():
    cfg = _cfg()
    out = train_ntk_adaptive(cfg, total_steps=30, update_every=5)
    assert np.isfinite(out["final_rel_l2"])


def test_train_rba_smoke():
    cfg = _cfg()
    out = train_rba(cfg, total_steps=30, resample_every=10)
    assert np.isfinite(out["final_rel_l2"])


import numpy as np
from pathlib import Path
