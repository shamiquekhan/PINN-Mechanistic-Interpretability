"""
Regression tests for the v2-plan bug fixes (Sep 2026 audit):
  1. Controller rollback must restore REAL pre-action lambdas.
  2. ControllerEvent must record rel_l2 (rescue claims must be auditable).
  3. SAE load must not silently flip legacy relul1 checkpoints to topk.
  4. Intervention modes must produce DIFFERENT loss deltas.
  5. probe_direction mode requires an explicit direction.
  6. derive_failure_step must label from the trajectory, not a constant.
  7. build_dataloaders must never silently use the train set as test set.
  8. Feature associations must beat random-direction controls.
"""
import json
import numpy as np
import pytest
import torch

from controller.state_machine import (
    PINNController, ControllerConfig, ControllerState,
)
from sae.model import SparseAutoencoder
from sae.dataset import make_run_splits
from interventions.engine import (
    SAEInterventionHook, measure_intervention_effect,
    measure_probe_direction_effect,
)
from interventions.causal import (
    compute_causal_score, train_failure_probe, run_training_intervention,
)
from monitoring.features import derive_failure_step
from pinn.model import MLP
from pinn.pdes import Poisson1D


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1 & 2. Controller rollback and rel_l2 tracking
# ---------------------------------------------------------------------------

class TestControllerRollback:
    def test_rollback_restores_pre_action_lambdas(self):
        """The 0.279 -> 0.579 regression happened because rollback restored
        the *current* (already-modified) lambdas instead of the pre-action
        values.  Rollback must restore the true pre-action snapshot."""
        cfg = ControllerConfig(
            alarm_threshold=0.5, confirmation_steps=1, cooldown_steps=5,
            rollback_on_degradation=True, degradation_patience=3,
            degradation_tolerance=1.10, bc_rebalance_factor=4.0,
            max_lambda_bc=200.0, verbose=False,
        )
        c = PINNController(cfg)

        lp, lb = 100.0, 0.01
        # Tick 0: IDLE -> WARNING (alarm).  Tick 1: confirmed -> action.
        lp, lb, _ = c.step(0, monitor_score=0.9, rel_l2=0.30, lambda_pde=lp,
                           lambda_bc=lb, failure_class="boundary_starvation")
        assert lb == pytest.approx(0.01), "no action before confirmation"
        lp, lb, _ = c.step(1, monitor_score=0.9, rel_l2=0.30, lambda_pde=lp,
                          lambda_bc=lb, failure_class="boundary_starvation")
        assert lb == pytest.approx(0.04)  # 0.01 * 4.0
        assert c._pre_action_lambdas == (100.0, 0.01), (
            "controller must snapshot the true pre-action lambdas"
        )

        # Degrading rel_l2 for enough ticks inside cooldown -> rollback.
        for i in range(2, 8):
            lp, lb, ev = c.step(i, monitor_score=0.0, rel_l2=0.30 * (1.2 ** i),
                                lambda_pde=lp, lambda_bc=lb,
                                failure_class="boundary_starvation")
        rollback_events = [e for e in c.event_log if e.rollback]
        assert len(rollback_events) == 1, "rollback must fire exactly once"
        assert (lp, lb) == (100.0, 0.01), (
            "rollback must restore the pre-action lambdas, not current values"
        )
        assert c.summary()["n_rollbacks"] == 1

    def test_events_track_rel_l2(self):
        """Every ControllerEvent must carry rel_l2 so rescue claims are
        verifiable from controller_events.jsonl alone."""
        c = PINNController(ControllerConfig(verbose=False))
        _, _, ev = c.step(0, monitor_score=0.0, rel_l2=0.123, lambda_pde=1.0,
                          lambda_bc=1.0)
        assert ev.rel_l2 == pytest.approx(0.123)
        assert hasattr(ev, "rel_l2")

    def test_no_degradation_no_rollback(self):
        cfg = ControllerConfig(
            alarm_threshold=0.5, confirmation_steps=1, cooldown_steps=3,
            rollback_on_degradation=True, degradation_patience=2,
            verbose=False,
        )
        c = PINNController(cfg)
        lp, lb = 100.0, 0.01
        lp, lb, _ = c.step(0, monitor_score=0.9, rel_l2=0.30, lambda_pde=lp,
                           lambda_bc=lb, failure_class="boundary_starvation")
        # Improving rel_l2 must never trigger rollback.
        for i in range(1, 4):
            lp, lb, ev = c.step(i, monitor_score=0.0, rel_l2=0.30 * 0.5 ** i,
                                lambda_pde=lp, lambda_bc=lb,
                                failure_class="boundary_starvation")
        assert all(not e.rollback for e in c.event_log)

    def test_summary_reports_best_and_final_rel_l2(self):
        c = PINNController(ControllerConfig(verbose=False))
        rels = [0.9, 0.5, 0.7, 0.3, 0.4]
        for i, r in enumerate(rels):
            c.step(i, 0.0, r, 1.0, 1.0)
        s = c.summary()
        assert s["best_rel_l2"] == pytest.approx(0.3)
        assert s["final_rel_l2"] == pytest.approx(0.4)
        assert s["first_rel_l2"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# 3. SAE checkpoint mode-skew
# ---------------------------------------------------------------------------

class TestSAECheckpointModes:
    def _legacy_ckpt(self, tmp_path, sparsity=1e-3):
        """Simulate a legacy (pre-topk) checkpoint that lacks activation_mode."""
        sae = SparseAutoencoder(input_dim=8, latent_expansion=4,
                                activation_mode="relul1",
                                sparsity_coeff=sparsity)
        p = tmp_path / "legacy_sae.pt"
        torch.save({
            "state_dict": sae.state_dict(),
            "input_dim": 8, "latent_expansion": 4,
            "sparsity_coeff": sparsity,
            "decoder_normalize": True,
        }, p)
        return p

    def test_legacy_relul1_checkpoint_not_flipped_to_topk(self, tmp_path):
        p = self._legacy_ckpt(tmp_path)
        loaded = SparseAutoencoder.load(p, DEVICE)
        assert loaded.activation_mode == "relul1", (
            "Legacy checkpoints trained with ReLU+L1 must not silently load "
            "in topk mode — that retrains forward semantics on mismatched weights."
        )

    def test_explicit_mode_roundtrip(self, tmp_path):
        sae = SparseAutoencoder(input_dim=8, latent_expansion=2,
                                activation_mode="topk", topk=4)
        p = tmp_path / "sae.pt"
        sae.save(p)
        loaded = SparseAutoencoder.load(p, DEVICE)
        assert loaded.activation_mode == "topk"
        assert loaded.topk == 4


# ---------------------------------------------------------------------------
# 4 & 5. Intervention engine correctness
# ---------------------------------------------------------------------------

class TestInterventionModes:
    def _setup(self):
        torch.manual_seed(0)
        model = MLP(1, 1, [64, 64, 64]).to(DEVICE)
        sae = SparseAutoencoder(input_dim=64, latent_expansion=4,
                                activation_mode="topk", topk=8).to(DEVICE)
        # Briefly fit the SAE to real activations so latent features are
        # actually active (an untrained TopK SAE leaves most latents dead,
        # making ablate == amplify trivially).
        with torch.no_grad():
            x = torch.linspace(-1, 1, 64, device=DEVICE).reshape(-1, 1)
            acts = model(x, return_intermediates=True)[1][1]  # (64, 64)
        opt = torch.optim.Adam(sae.parameters(), lr=1e-2)
        for _ in range(100):
            opt.zero_grad()
            z, a_hat = sae(acts)
            loss = torch.nn.functional.mse_loss(a_hat, acts)
            loss.backward()
            opt.step()
        sae.eval()
        pde = Poisson1D()
        return model, sae, pde

    def test_modes_produce_distinct_deltas(self):
        """Regression for the 'identical deltas across all modes' bug."""
        model, sae, pde = self._setup()
        x = pde.sample_interior(64, DEVICE, torch.float32)

        # Pick a feature that is genuinely active in the SAE latents.
        with torch.no_grad():
            xv = torch.linspace(-1, 1, 64, device=DEVICE).reshape(-1, 1)
            acts = model(xv, return_intermediates=True)[1][1]
            z, _ = sae(acts)
        active = (z > 0).float().sum(dim=0)
        assert active.sum() > 0, "SAE must have active features after fitting"
        feat = int(torch.argmax(active).item())

        losses = {}
        for mode in ["natural", "reconstruction_only", "ablate",
                     "amplify", "random_direction"]:
            hook = SAEInterventionHook(sae, mode=mode, feature_idx=feat, alpha=2.0)
            hook.register(model, 1)
            r = pde.residual(model, x)
            losses[mode] = float((r ** 2).mean())
            hook.remove()

        # The natural pass must differ from every intervention pass...
        assert losses["natural"] != pytest.approx(losses["ablate"])
        assert losses["natural"] != pytest.approx(losses["amplify"])
        # ...and ablate (z_k -> 0) must differ from amplify (z_k -> 2 z_k)
        # for an active feature.
        assert losses["ablate"] != pytest.approx(losses["amplify"])

    def test_measure_intervention_effect_returns_finite_deltas(self):
        model, sae, pde = self._setup()
        res = measure_intervention_effect(
            model, sae, pde, DEVICE, torch.float32,
            feature_idx=3, mode="ablate", n_interior=32, layer_index=1,
        )
        assert np.isfinite(res["delta_pde"])
        assert np.isfinite(res["delta_bc"])
        # Ablating one of 32 latents must change the PDE loss somehow.
        assert res["delta_pde"] != 0.0

    def test_probe_direction_requires_direction(self):
        model, sae, pde = self._setup()
        with pytest.raises(ValueError):
            SAEInterventionHook(sae, mode="probe_direction", feature_idx=0)

    def test_probe_direction_effect_runs(self):
        model, sae, pde = self._setup()
        direction = torch.randn(sae.latent_dim, device=DEVICE)
        res = measure_probe_direction_effect(
            model, sae, pde, DEVICE, torch.float32,
            feature_idx=0, probe_direction=direction,
            n_interior=32, layer_index=1,
        )
        assert res["mode"] == "probe_direction"
        assert np.isfinite(res["delta_pde"])

    def test_probe_fallback_when_direction_missing(self):
        model, sae, pde = self._setup()
        res = measure_probe_direction_effect(
            model, sae, pde, DEVICE, torch.float32,
            feature_idx=0, probe_direction=None,
            n_interior=32, layer_index=1,
        )
        # Falls back to a random-direction protocol.
        assert res["mode"] == "random_direction"


# ---------------------------------------------------------------------------
# 6. Trajectory-derived failure labels
# ---------------------------------------------------------------------------

class TestDerivedFailureSteps:
    def test_failure_step_from_trajectory(self):
        metrics = [
            {"step": 0,   "loss": 1.0, "loss_pde": 1.0, "loss_bc": 0.0, "relative_l2": 0.01},
            {"step": 100, "loss": 0.5, "loss_pde": 0.5, "loss_bc": 0.0, "relative_l2": 0.01},
            {"step": 200, "loss": 0.3, "loss_pde": 0.3, "loss_bc": 0.0, "relative_l2": 0.02},
            {"step": 300, "loss": 0.3, "loss_pde": 0.3, "loss_bc": 0.0, "relative_l2": 0.09},
            {"step": 400, "loss": 0.3, "loss_pde": 0.3, "loss_bc": 0.0, "relative_l2": 0.10},
            {"step": 500, "loss": 0.3, "loss_pde": 0.3, "loss_bc": 0.0, "relative_l2": 0.12},
        ]
        fs = derive_failure_step(metrics, fail_threshold=0.05, confirm_records=3)
        assert fs == 300, "failure step should be the first step of the crossing"

    def test_success_run_has_no_failure_step(self):
        metrics = [
            {"step": i * 100, "loss": 1.0 - i * 0.1, "loss_pde": 1.0,
             "loss_bc": 0.0, "relative_l2": 0.001}
            for i in range(10)
        ]
        assert derive_failure_step(metrics) is None

    def test_single_spike_not_labeled_failure(self):
        metrics = [
            {"step": i * 100, "loss": 0.5, "loss_pde": 0.5, "loss_bc": 0.0,
             "relative_l2": 0.06 if i == 4 else 0.01}
            for i in range(10)
        ]
        assert derive_failure_step(metrics, confirm_records=3) is None


# ---------------------------------------------------------------------------
# 7. Run-level splits must stay disjoint
# ---------------------------------------------------------------------------

class TestSplits:
    def test_make_run_splits_disjoint(self, tmp_path):
        dirs = [tmp_path / f"run_{i}" for i in range(10)]
        for d in dirs:
            d.mkdir(exist_ok=True)
        train, val, test = make_run_splits(dirs, seed=0)
        all_ids = [id(x) for x in train + val + test]
        assert len(all_ids) == len(set(all_ids)) == len(dirs), (
            "run-level splits must partition the runs with no overlap"
        )
        assert set(train) & set(test) == set()


# ---------------------------------------------------------------------------
# 8. Random-direction association control
# ---------------------------------------------------------------------------

class TestAssociationBaselines:
    def test_probe_training_and_direction(self):
        torch.manual_seed(0)
        sae = SparseAutoencoder(input_dim=16, latent_expansion=4,
                                activation_mode="topk", topk=4)
        acts = torch.randn(200, 16)
        labels = (acts[:, 0] > 0).long().numpy()  # linearly separable signal
        direction = train_failure_probe(sae, acts, labels, torch.device("cpu"))
        assert direction.shape == (sae.latent_dim,)
        assert abs(float(direction.norm()) - 1.0) < 1e-5

    def test_causal_score_flags_weak_features(self):
        """compute_causal_score must penalise features that do not exceed
        the strongest control (the v2 gate)."""
        weak = compute_causal_score(
            target_effect=0.01,
            control_unrelated_effect=0.008,
            control_random_effect=0.009,
            control_probe_effect=0.005,
            non_target_effects=[0.02],
        )
        strong = compute_causal_score(
            target_effect=0.5,
            control_unrelated_effect=0.01,
            control_random_effect=0.02,
            control_probe_effect=0.03,
            non_target_effects=[0.01],
        )
        assert weak["causal_strength"] < 0.01
        assert strong["causal_strength"] > 0.4
        assert strong["specificity"] > weak["specificity"]


# ---------------------------------------------------------------------------
# Training-time intervention plumbing
# ---------------------------------------------------------------------------

class TestTrainingIntervention:
    def _cfg(self, tmp_path):
        cfg_dict = {
            "run": {"name": "t", "output_dir": "runs", "seed": 0,
                    "deterministic": True, "device": "cpu", "dtype": "float32"},
            "pde": {"name": "poisson_1d", "domain": [-1.0, 1.0], "source": 1.0,
                    "boundary_values": [0.0, 0.0], "validation_points": 101},
            "model": {"input_dim": 1, "output_dim": 1,
                      "hidden_layers": [16, 16], "activation": "tanh"},
            "training": {"optimizer": "adam", "learning_rate": 1e-3,
                         "steps": 20, "interior_points": 32, "boundary_points": 2,
                         "log_every": 10, "checkpoint_every": 100,
                         "lambda_pde": 1.0, "lambda_bc": 1.0},
            "logging": {"save_activations": False, "activation_layers": [1],
                        "save_pointwise_residuals": True},
        }
        from pinn.config import ExperimentConfig
        cfg = ExperimentConfig.model_validate(cfg_dict)
        return cfg

    def test_training_intervention_runs_and_compares(self, tmp_path):
        cfg = self._cfg(tmp_path)
        pde = Poisson1D()
        sae = SparseAutoencoder(input_dim=16, latent_expansion=4,
                                activation_mode="topk", topk=4)
        res = run_training_intervention(
            cfg, pde, sae, torch.device("cpu"), torch.float32,
            feature_idx=1, layer_index=0, mode="ablate",
            apply_from_step=0, apply_to_step=10, total_steps=20,
            eval_every=5, out_dir=tmp_path,
        )
        assert "control_final_rel_l2" in res
        assert "intervened_final_rel_l2" in res
        assert len(res["intervened_trajectory"]) == len(res["control_trajectory"])
        assert (tmp_path / "training_intervention_feat1.json").exists()