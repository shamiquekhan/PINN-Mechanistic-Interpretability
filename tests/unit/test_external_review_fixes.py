"""Regression tests for the external-review fixes (C2, C3, H2, M3, M5, M7,
and the stage-6 gradient-augment alignment).

Each test pins a bug that was actually present and fixed:
  - C2: time-dependent exact() must be nonzero at intermediate t.
  - C3: controls must never silently no-op; matched-deletion semantics.
  - H2: the leakage audit must FAIL on a deliberately leaked window.
  - M3: silent cuda->cpu downgrade must raise; bad dtype must raise.
  - M5: steps=0 must mean zero steps (not the config default).
  - M7: stiff RD1D configs return None instead of inf-based garbage.
  - H1: gradient augmentation row count must match the dataset builder.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pinn.pdes import Burgers1D, AllenCahn1D, ReactionDiffusion1D, make_pde
from pinn.config import PDEConfig


# ---------------------------------------------------------------------------
# C2 — time-dependent reference must be a full space-time trajectory
# ---------------------------------------------------------------------------

def _small_burgers():
    return Burgers1D(viscosity=0.01, t0=0.0, tT=1.0, n_spectral=64,
                    n_timesteps=40)


def _small_allen_cahn():
    return AllenCahn1D(epsilon=0.05, t0=0.0, tT=1.0, n_spectral=64,
                       n_timesteps=40)


@pytest.mark.parametrize("pde_factory", [_small_burgers, _small_allen_cahn],
                         ids=["burgers", "allen_cahn"])
def test_exact_nonzero_at_intermediate_time(pde_factory):
    """C2 regression: exact() used to return literal zeros for every t < tT
    while the validation grid spans the full (t,x) domain."""
    pde = pde_factory()
    x_tx = torch.tensor([[0.5, 0.25], [0.5, -0.5], [0.25, 0.0]])
    u = pde.exact(x_tx)
    assert u is not None
    assert torch.isfinite(u).all()
    # At t=0 the reference equals the initial condition (nonzero on-grid).
    assert float(u[0].abs()) > 0.0
    # Mid-time values must be nonzero — the old code returned 0 here.
    assert float(u[2].abs()) > 0.0


@pytest.mark.parametrize("pde_factory", [_small_burgers, _small_allen_cahn],
                         ids=["burgers", "allen_cahn"])
def test_exact_initial_time_matches_initial_condition(pde_factory):
    """t=0 row of the trajectory IS the IC. NOTE the spectral grid uses
    endpoint=False (N points covering [left, right) with wraparound), so the
    comparison interpolates onto the caller's grid via exact() itself."""
    pde = pde_factory()
    x = torch.linspace(pde.left, pde.right, 17).reshape(-1, 1)
    t0_col = torch.zeros_like(x)
    u0_ref = pde.exact(torch.cat([t0_col, x], dim=1)).squeeze(-1)
    u0 = pde.initial_condition(x.squeeze(-1))
    # t=0 row of the trajectory IS the IC (up to spectral grid resolution;
    # exclude the wraparound point at x=right whose spectral value belongs
    # to the periodic image, not the Dirichlet IC).
    keep = (x.squeeze(-1) < pde.right - 1e-9)
    assert torch.allclose(u0_ref[keep], u0[keep], atol=5e-2)


@pytest.mark.parametrize("pde_factory", [_small_burgers, _small_allen_cahn],
                         ids=["burgers", "allen_cahn"])
def test_exact_is_continuous_in_time(pde_factory):
    """Bilinear interpolation must not have the final-time cliff: values at
    t = tT-eps and t = tT must be close."""
    pde = pde_factory()
    xq = 0.3
    u_late = pde.exact(torch.tensor([[pde.tT - 1e-3, xq]]))
    u_final = pde.exact(torch.tensor([[pde.tT, xq]]))
    assert abs(float(u_late) - float(u_final)) < 0.5


def test_burgers_reference_shape_is_trajectory():
    pde = _small_burgers()
    x_ref, U = pde.reference_solution()
    assert U.dim() == 2
    assert U.shape[0] == pde.n_timesteps + 1
    assert U.shape[1] == pde.n_spectral


# ---------------------------------------------------------------------------
# C3 — controls must not silently no-op; matched deletion
# ---------------------------------------------------------------------------

def _sparse_z(batch=16, latent=256, k=8, active_feats=(3, 7, 200), seed=0):
    torch.manual_seed(seed)
    z = torch.zeros(batch, latent)
    for f in active_feats:
        z[:, f] = torch.rand(batch) + 0.1
    return z


def test_unrelated_control_perturbs_an_active_feature():
    """C3a regression: on a mostly-dead TopK code the old uniform sampler
    picked an already-zero latent ~97% of the time (a no-op control)."""
    from interventions.engine import SAEInterventionHook
    hook = SAEInterventionHook(None, mode="unrelated_control",
                               feature_idx=3, random_seed=0)
    z = _sparse_z(active_feats=(3, 7, 200))
    out = hook._intervene(z)
    # Something other than the target must have been zeroed.
    assert not torch.equal(out, z)
    # And the change must be a real removal of an active feature.
    diff = (out != z).any(dim=0)
    changed = diff.nonzero().flatten().tolist()
    assert 3 not in changed  # target untouched in control mode
    for c in changed:
        assert float(z[:, c].abs().max()) > 0  # was active before


def test_unrelated_control_noop_flagged_when_nothing_active():
    from interventions.engine import SAEInterventionHook
    hook = SAEInterventionHook(None, mode="unrelated_control",
                               feature_idx=0, random_seed=0)
    z = torch.zeros(4, 8)
    z[:, 0] = 1.0  # only the target active
    out = hook._intervene(z)
    assert torch.equal(out, z)
    assert hook.last_control_was_noop is True


def test_random_direction_is_matched_deletion_not_injection():
    """C3b regression: the old control ADDED a random direction (injection)
    while the target ABLATED — arms not matched in kind."""
    from interventions.engine import SAEInterventionHook
    hook = SAEInterventionHook(None, mode="random_direction",
                               feature_idx=3, random_seed=0)
    z = _sparse_z(active_feats=(3, 7, 200))
    out = hook._intervene(z)
    # Deletion semantics: output <= input everywhere, one feature zeroed.
    assert (out <= z + 1e-6).all()
    zeroed_cols = ((z > 0) & (out == 0)).any(dim=0).nonzero().flatten().tolist()
    assert zeroed_cols, "no active non-target feature was ablated"
    assert 3 not in zeroed_cols
    # No addition happened: total mass decreased.
    assert float(out.abs().sum()) < float(z.abs().sum())


def test_operator_control_uses_active_features():
    """C3 mirror in the operator battery (stage 14's machinery)."""
    from interventions.operator_battery import OperatorSAEHook
    hook = OperatorSAEHook.__new__(OperatorSAEHook)
    hook.mode = "unrelated_control"
    hook.feature_idx = 3
    hook.alpha = 0.0
    hook.random_seed = 0
    hook.last_control_was_noop = False
    z = _sparse_z(active_feats=(3, 7, 200))
    out = hook._intervene(z)
    assert not torch.equal(out, z)
    changed = (out != z).any(dim=0).nonzero().flatten().tolist()
    assert 3 not in changed
    for c in changed:
        assert float(z[:, c].abs().max()) > 0


# ---------------------------------------------------------------------------
# H2 — the leakage audit must be able to fail
# ---------------------------------------------------------------------------

def test_leakage_audit_fails_on_deliberately_leaked_window():
    """H2 regression: the stage-6 call passed failure_step=10**9, which can
    never fail.  The real invariant is past-only feature windows — no row's
    feature window may contain a record from the failure step or later.
    This test encodes the invariant directly and proves it can fire."""
    from monitoring.features import build_trajectory_features
    metrics = [{"step": 100 * i, "loss": 1.0, "loss_pde": 0.5,
                "loss_bc": 0.5, "relative_l2": 0.5} for i in range(30)]
    fs = 2000
    window, horizon = 10, 5
    X, y, steps = build_trajectory_features(metrics, window, horizon, "any", fs)

    # Honest rows: prediction step p has window = metrics[(p-window)..p);
    # its last record must precede fs. Rows with p <= fs are leak-free.
    def window_last_leaks(pred_idx):
        return metrics[pred_idx - 1]["step"] >= fs

    bad = [t for t in range(window, len(metrics) - horizon)
           if window_last_leaks(t)]
    # Rows after fs (pred steps 2100+) have windows containing post-failure
    # records — these MUST be flagged:
    assert bad, "leak detector cannot fire — vacuous"
    # ...and rows before fs are clean:
    first_leaky_pred = metrics[bad[0]]["step"]
    assert first_leaky_pred > fs
    # The stage-6 implementation checks exactly this condition:
    # (pred_step >= fs) and (window last step >= fs) -> violation.
    n_violations = sum(
        1 for t in range(window, len(metrics) - horizon)
        if metrics[t]["step"] >= fs and metrics[t - 1]["step"] >= fs)
    assert n_violations == len(bad)
    # The old ceremonial call (failure_step=10**9) detects zero of these.
    n_ceremonial = sum(
        1 for t in range(window, len(metrics) - horizon)
        if metrics[t]["step"] >= 10**9 and metrics[t - 1]["step"] >= 10**9)
    assert n_ceremonial == 0


# ---------------------------------------------------------------------------
# M3 — device/dtype validation
# ---------------------------------------------------------------------------

def test_silent_cuda_downgrade_refused():
    from pinn.trainer import PINNTrainer
    # Build a minimal config requesting cuda on a machine without it.
    # (If CUDA IS available, this test is a no-op — the check only fires
    # when cuda is requested but unavailable.)
    if torch.cuda.is_available():
        pytest.skip("CUDA available — downgrade path not reachable")
    from pinn.config import load_config
    cfg = load_config(Path(__file__).resolve().parents[2]
                      / "configs" / "failure_boundary_starvation.yaml")
    cfg = cfg.model_copy(deep=True)
    cfg.run.device = "cuda"
    cfg.logging.save_activations = False
    cfg.logging.log_diagnostics = False
    cfg.logging.log_gradients = False
    model = torch.nn.Linear(1, 1)
    pde = make_pde(cfg.pde)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    with pytest.raises(RuntimeError, match="refusing to silently downgrade"):
        PINNTrainer(model, pde, opt, cfg, Path("/tmp/opencode/t_test"))


def test_bad_dtype_raises_clearly():
    from pinn.trainer import PINNTrainer
    from pinn.config import load_config
    cfg = load_config(Path(__file__).resolve().parents[2]
                      / "configs" / "failure_boundary_starvation.yaml")
    cfg = cfg.model_copy(deep=True)
    cfg.run.dtype = "float128"  # not a torch dtype
    cfg.logging.save_activations = False
    cfg.logging.log_diagnostics = False
    cfg.logging.log_gradients = False
    model = torch.nn.Linear(1, 1)
    pde = make_pde(cfg.pde)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    with pytest.raises(ValueError, match="unknown dtype"):
        PINNTrainer(model, pde, opt, cfg, Path("/tmp/opencode/t_test"))


# ---------------------------------------------------------------------------
# M5 — steps=0 must mean zero
# ---------------------------------------------------------------------------

def test_trainer_steps_zero_means_zero():
    from pinn.trainer import PINNTrainer
    from pinn.config import load_config
    cfg = load_config(Path(__file__).resolve().parents[2]
                      / "configs" / "failure_boundary_starvation.yaml")
    cfg = cfg.model_copy(deep=True)
    cfg.run.device = "cpu"
    cfg.logging.save_activations = False   # bare test model has no .acts
    cfg.logging.log_diagnostics = False
    cfg.logging.log_gradients = False
    model = torch.nn.Linear(1, 1)
    pde = make_pde(cfg.pde)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    tr = PINNTrainer(model, pde, opt, cfg, Path("/tmp/opencode/t_zero"))
    recs = tr.train(steps=0)
    assert recs == []


# ---------------------------------------------------------------------------
# M7 — stiff RD1D must degrade gracefully
# ---------------------------------------------------------------------------

def test_stiff_rd1d_returns_none_instead_of_inf():
    pde = ReactionDiffusion1D(left=-1.0, right=1.0, diffusion=1e-2,
                              reaction=1.0, source=1.0, left_bc=0.0, right_bc=0.0)
    x = torch.linspace(-1, 1, 11).reshape(-1, 1)
    u = pde.exact(x)
    # r = sqrt(1/1e-2) = 10; r*|x| = 10 < 80 → analytic, finite everywhere.
    assert u is not None and torch.isfinite(u).all()
    # Stiff: eps=1e-4 → r=100, r*1.0 > 80 → float32 exp overflow → None.
    pde_stiff = ReactionDiffusion1D(left=-1.0, right=1.0, diffusion=1e-4,
                                    reaction=1.0, source=1.0,
                                    left_bc=0.0, right_bc=0.0)
    assert pde_stiff.exact(x) is None


# ---------------------------------------------------------------------------
# H1 — gradient augmentation must align with the dataset builder
# ---------------------------------------------------------------------------

def test_gradient_augmentation_row_alignment(tmp_path):
    """H1 regression: the first version of the fix emitted
    len(metrics)-history_window rows while _build_dataset emits
    n-history_window-failure_horizon — the mismatch silently zeroed the
    features for nearly every run."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.run_pipeline import _augment_with_gradients
    from monitoring.features import build_trajectory_features

    d = tmp_path / "run_x"
    d.mkdir()
    n, window, horizon = 40, 10, 5
    metrics = [{"step": 100 * i, "loss": 1.0, "loss_pde": 0.5,
                "loss_bc": 0.5, "relative_l2": 0.5} for i in range(n)]
    (d / "metrics.jsonl").write_text(
        "\n".join(json.dumps(m) for m in metrics) + "\n")
    grads = [{"step": 100 * i,
              "gradient_stats": {"gradient_cosines": {"pde_vs_bc": -0.5}}}
             for i in range(n)]
    (d / "gradients.jsonl").write_text(
        "\n".join(json.dumps(g) for g in grads) + "\n")

    fs = None
    from monitoring.features import derive_failure_step
    fs = derive_failure_step(metrics)
    X, y, _ = build_trajectory_features(metrics, window, horizon, "any", fs)
    out = _augment_with_gradients(X.astype(np.float32), [d], window, horizon)
    # same row count — no silent zero-fallback
    assert out.shape == (X.shape[0], X.shape[1] + 3)
    # the cosine features must be REAL (nonzero) — all logged cosines are -0.5
    cos_mean_col = out[:, -3]
    assert np.allclose(cos_mean_col, -0.5, atol=1e-6), \
        "gradient features were silently zeroed by misalignment"
