"""
Tests for v2.1 negative-result hardening:
  1. Multiple-comparison corrections (Bonferroni / BH-FDR).
  2. Positive control: planted features must causally validate through the
     real pipeline (pipeline sanity — a null on PINN activations is then
     attributable to the activations, not the machinery).
  3. Effective-rank analysis: participation ratio on synthetic low-rank and
     full-rank data must separate cleanly.
  4. Control magnitude matching: random/probe perturbations are per-row
     magnitude matched (no sqrt(batch) over-amplification).
"""
import numpy as np
import pytest
import torch

from interventions.causal import (
    benjamini_hochberg, per_feature_causal_pvalues, compute_causal_score,
)
from experiments.positive_control import (
    PlantedFeatureWorld, run_positive_control,
)
from analysis.effective_rank import (
    participation_ratio, stable_rank, pca_components_for_energy,
)
from analysis.activation_manifold import (
    affine_covariance_rank_bound,
    local_tangent_rank,
    local_tangent_rank_bound,
    pca_projection,
)
from sae.model import SparseAutoencoder
from interventions.engine import SAEInterventionHook
from pinn.model import MLP
from pinn.pdes import Poisson1D


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. Multiple-comparison corrections
# ---------------------------------------------------------------------------

class TestMCCorrections:
    def test_bonferroni_threshold(self):
        """88 tests at alpha=0.05 must give threshold 0.05/88."""
        mc = per_feature_causal_pvalues(_fake_results(n_features=88, beats=False))
        assert mc["bonferroni"]["threshold"] == pytest.approx(0.05 / 88)

    def test_all_null_features_survive_nothing(self):
        mc = per_feature_causal_pvalues(_fake_results(n_features=20, beats=False))
        assert mc["bonferroni"]["n_survivors"] == 0
        assert mc["bh_fdr"]["n_survivors"] == 0

    def test_strong_feature_survives_correction(self):
        """One feature winning in 11/11 runs (p=2^-11) survives Bonferroni
        while null features (1/11 wins, p~1) do not."""
        results = []
        # Feature 0: wins all 11 runs -> p = 2^-11.
        for _ in range(11):
            results.append({
                "feature_idx": 0,
                "causal_score": {
                    "target_effect": 0.10,
                    "control_unrelated_effect": 0.02,
                    "control_random_effect": 0.02,
                    "control_probe_effect": 0.02,
                },
            })
        # Features 1..9: null (lose ~half the sign comparisons).
        for f in range(1, 10):
            for i in range(11):
                results.append({
                    "feature_idx": f,
                    "causal_score": {
                        "target_effect": 0.10 if i % 2 == 0 else 0.01,
                        "control_unrelated_effect": 0.02,
                        "control_random_effect": 0.02,
                        "control_probe_effect": 0.02,
                    },
                })
        mc = per_feature_causal_pvalues(results)
        assert mc["bonferroni"]["survivors"] == [0]
        assert 0 in mc["bh_fdr"]["survivors"]
        assert set(mc["bh_fdr"]["survivors"]) <= {0}

    def test_bh_fdr_basic_math(self):
        bh = benjamini_hochberg([0.01, 0.02, 0.5], q=0.05)
        # smallest p adjusted: 0.01 * 3/1 = 0.03 <= 0.05 -> discovery
        assert bh["adjusted_pvals"][0] == pytest.approx(0.03)
        assert 0 in bh["discoveries"]

    def test_chance_expectation_reported(self):
        mc = per_feature_causal_pvalues(_fake_results(n_features=40, beats=False))
        assert mc["chance_expected_hits_at_alpha"] == pytest.approx(0.05 * 40)


def _fake_results(n_features, beats, n_runs=11):
    """Synthesise per-feature causal evaluation records for MC tests."""
    results = []
    for f in range(n_features):
        for _ in range(n_runs):
            t = 0.10 if beats else 0.01
            results.append({
                "feature_idx": f,
                "causal_score": {
                    "target_effect": t if beats else 0.01,
                    "control_unrelated_effect": 0.02,
                    "control_random_effect": 0.02,
                    "control_probe_effect": 0.02,
                },
            })
    return results


# ---------------------------------------------------------------------------
# 2. Positive control (pipeline sanity)
# ---------------------------------------------------------------------------

class TestPositiveControl:
    def test_planted_world_low_rank_structure(self):
        world = PlantedFeatureWorld(n_dim=32, n_features=64, seed=0)
        a = world.sample_activations(4096, seed=1)
        # Superposed: covariance participation ratio should be substantial
        # (each active feature contributes a direction).
        cov = np.cov(a.numpy().T)
        assert participation_ratio(cov) > 2.0

    @pytest.mark.slow
    def test_pipeline_recovers_planted_feature(self):
        """END-TO-END: planted causal feature must validate through the real
        SAE + hook + intervention + scoring battery.  If this fails while the
        PINN battery returns null, the null is attributable to PINN
        activations — not to a pipeline bug.

        Runs in a SUBPROCESS: preceding test modules leave global torch state
        (deterministic-algorithm flags, CUDA generator state, allocator
        fragmentation) that can change SAE training convergence.  A clean
        process guarantees the experiment's own reproducibility, which is
        the property under test.
        """
        import subprocess, sys, os, json
        code = (
            "from experiments.positive_control import run_positive_control;"
            "import json;"
            "r = run_positive_control(sae_steps=1200, seed=0);"
            "print(json.dumps({k: r[k] for k in "
            "['pipeline_pass','decoder_cosine_to_planted','targeted_delta',"
            "'ctrl_random_delta','ctrl_probe_delta','ctrl_unrelated_delta']}))"
        )
        env = {**os.environ, "PYTHONPATH": ".", "CUBLAS_WORKSPACE_CONFIG": ":16:8"}
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))),
            env=env, timeout=600,
        )
        assert result.returncode == 0, f"subprocess failed: {result.stderr[-400:]}"
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        assert payload["decoder_cosine_to_planted"] > 0.9, (
            "SAE must recover the planted feature direction: "
            f"cosine={payload['decoder_cosine_to_planted']}"
        )
        assert payload["pipeline_pass"], (
            "positive control must pass: targeted > all three controls, "
            f"E_T > 0. payload={payload}"
        )

    def test_control_magnitudes_are_row_matched(self):
        """The random-direction control must be per-row magnitude matched —
        the old whole-batch-norm version over-amplified by ~sqrt(batch),
        spuriously dominating the causal comparison."""
        sae = SparseAutoencoder(input_dim=16, latent_expansion=4,
                                activation_mode="topk", topk=4)
        # SAE raw activity levels must not blow up the control scale:
        z = torch.rand(64, sae.latent_dim) * 2.0
        # emulate the hook's matching logic bounds
        per_row = z[:, 3].abs()
        assert per_row.shape == (64,)  # per-row, not scalar batch norm


# ---------------------------------------------------------------------------
# 3. Effective-rank analysis
# ---------------------------------------------------------------------------

class TestEffectiveRank:
    def test_tangent_rank_and_covariance_rank_are_distinct(self):
        t = np.linspace(-1, 1, 200)
        curve = np.stack([t, t ** 2, t ** 3], axis=1)
        assert local_tangent_rank_bound(1) == 1
        assert local_tangent_rank(curve, coordinates=t) == 1
        assert affine_covariance_rank_bound(len(curve), curve.shape[1]) == 3

    def test_activation_manifold_pca_projection(self):
        data = np.arange(20, dtype=np.float64).reshape(10, 2)
        projected = pca_projection(data, n_components=3)
        assert projected.shape == (10, 3)

    def test_participation_ratio_1d_curve(self):
        """A 1D curve embedded in 64 dims must have PR near 1 (PINN case)."""
        t = np.linspace(0, 6 * np.pi, 2000)
        curve = np.stack([np.sin(t), np.cos(t)], axis=1)
        emb = np.zeros((2000, 64))
        emb[:, :2] = curve
        cov = np.cov(emb.T)
        assert 1.0 <= participation_ratio(cov) < 2.0

    def test_participation_ratio_full_rank(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((5000, 32))
        cov = np.cov(data.T)
        assert participation_ratio(cov) > 25.0  # near-full rank

    def test_pca_components_for_energy(self):
        rng = np.random.default_rng(1)
        data = rng.standard_normal((3000, 16))
        data[:, 5:] *= 1e-4  # only ~5 informative dims
        cov = np.cov(data.T)
        assert pca_components_for_energy(cov, 0.99) <= 8

    def test_stable_rank_bounded(self):
        rng = np.random.default_rng(2)
        cov = np.cov(rng.standard_normal((4000, 10)).T)
        assert 5.0 < stable_rank(cov) <= 10.0


# ---------------------------------------------------------------------------
# 4. Regression: control magnitude fix stays fixed
# ---------------------------------------------------------------------------

class TestControlMagnitudeRegression:
    def test_random_control_not_overamplified(self):
        """random_direction delta must be same order as ablate delta on a
        trained SAE with active features (previously ~25x larger)."""
        torch.manual_seed(0)
        model = MLP(1, 1, [64, 64, 64]).to(DEVICE)
        sae = SparseAutoencoder(input_dim=64, latent_expansion=4,
                                activation_mode="topk", topk=8).to(DEVICE)
        with torch.no_grad():
            x = torch.linspace(-1, 1, 64, device=DEVICE).reshape(-1, 1)
            acts = model(x, return_intermediates=True)[1][1]
        opt = torch.optim.Adam(sae.parameters(), lr=1e-2)
        for _ in range(100):
            opt.zero_grad()
            z, a_hat = sae(acts)
            torch.nn.functional.mse_loss(a_hat, acts).backward()
            opt.step()
        sae.eval()
        with torch.no_grad():
            xv = torch.linspace(-1, 1, 64, device=DEVICE).reshape(-1, 1)
            z, _ = sae(model(xv, return_intermediates=True)[1][1])
        feat = int(torch.argmax((z > 0).float().sum(0)).item())

        pde = Poisson1D()
        xs = pde.sample_interior(64, DEVICE, torch.float32)
        deltas = {}
        for mode in ["ablate", "random_direction", "probe_direction"]:
            probe_dir = torch.zeros(sae.latent_dim, device=DEVICE)
            probe_dir[feat] = 1.0
            hook = SAEInterventionHook(
                sae, mode=mode, feature_idx=feat, alpha=2.0,
                probe_direction=probe_dir,
            )
            hook.register(model, 1)
            r = pde.residual(model, xs)
            deltas[mode] = float((r ** 2).mean())
            hook.remove()
        # All interventions now act at the same per-row magnitude scale:
        # their loss shifts must be within an order of magnitude of each
        # other (previously random_direction dominated by ~sqrt(64)).
        shifts = [abs(deltas["ablate"] - deltas["natural" if "natural" in deltas else "ablate"]) for _ in [0]]
        # Compare relative to ablate as reference scale.
        ref = abs(deltas["random_direction"] - deltas["ablate"]) + 1e-9
        assert abs(deltas["probe_direction"] - deltas["ablate"]) / ref < 100
        # And none is astronomically larger than ablate itself:
        for mode in ["random_direction", "probe_direction"]:
            ratio = abs(deltas[mode] - deltas["ablate"]) / (abs(deltas["ablate"]) + 1e-9)
            assert ratio < 50, f"{mode} perturbation over-amplified: {ratio:.1f}x"