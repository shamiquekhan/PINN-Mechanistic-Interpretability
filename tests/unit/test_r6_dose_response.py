"""Tests for R6: intervention dose-response (experiments/r6_dose_response.py
+ the dose_matched_control engine mode)."""
import numpy as np
import torch

from interventions.engine import SAEInterventionHook
from sae.model import SparseAutoencoder
from pinn.model import MLP
from pinn.pdes import Poisson1D

from experiments.r6_dose_response import (
    DOSES, classify_curve, summarize_arm,
    _closest_activity_control, _pca_closest_activity_control,
)

DEVICE = torch.device("cpu")
DTYPE = torch.float32


def _toy_sae_and_codes():
    torch.manual_seed(0)
    sae = SparseAutoencoder(input_dim=16, latent_expansion=4,
                             activation_mode="topk", topk=4,
                             decoder_normalize=True)
    sae.eval()
    codes = torch.zeros(64, sae.latent_dim)
    codes[:, 0] = 1.0
    codes[:, 1] = 0.5
    codes[:, 5] = 0.25
    return sae, codes


class _CodeHost(torch.nn.Module):
    """Hosts codes so the hook's _intervene path can be exercised."""

    def __init__(self, codes):
        super().__init__()
        self.register_buffer("codes", codes)

    def forward(self, x):
        return self.codes


def _intervene(mode, alpha, feature_idx, random_seed=0):
    sae, codes = _toy_sae_and_codes()
    hook = SAEInterventionHook(sae, mode=mode, feature_idx=feature_idx,
                               alpha=alpha, random_seed=random_seed)
    return hook._intervene(codes.clone()).numpy()


class TestDoseMatchedControlMode:
    def test_mode_is_accepted(self):
        sae, _ = _toy_sae_and_codes()
        SAEInterventionHook(sae, mode="dose_matched_control",
                             feature_idx=0, alpha=0.0)  # must not raise

    def test_unknown_mode_still_rejected(self):
        sae, _ = _toy_sae_and_codes()
        try:
            SAEInterventionHook(sae, mode="not_a_mode",
                                 feature_idx=0, alpha=0.0)
            assert False, "must raise"
        except ValueError:
            pass

    def test_control_scales_a_non_target_atom_by_alpha(self):
        # alpha=2: the control atom doubles; the target atom is untouched
        out = _intervene("dose_matched_control", 2.0, feature_idx=0)
        assert out[:, 0].max() == 1.0            # target untouched
        # at least one non-target active atom changed (scaled by alpha)
        changed = np.zeros(out.shape[1], dtype=bool)
        orig = np.zeros_like(out)
        orig[:, 0], orig[:, 1], orig[:, 5] = 1.0, 0.5, 0.25
        changed = np.any(~np.isclose(out, orig), axis=0)
        assert changed[0] is np.False_ or not changed[0]  # target untouched
        assert changed[1:].sum() >= 1                     # some control moved

    def test_negative_dose_flips_sign(self):
        out = _intervene("dose_matched_control", -1.0, feature_idx=0)
        # some non-target active atom is now <= 0
        assert (out[:, 1:] <= 0).any()

    def test_control_never_touches_target(self):
        for alpha in DOSES:
            out = _intervene("dose_matched_control", alpha, feature_idx=0)
            assert np.allclose(out[:, 0], 1.0), f"target moved at {alpha}"

    def test_noop_flag_when_no_other_active(self):
        sae, _ = _toy_sae_and_codes()
        codes = torch.zeros(8, sae.latent_dim)
        codes[:, 0] = 1.0                        # only the target active
        hook = SAEInterventionHook(sae, mode="dose_matched_control",
                                   feature_idx=0, alpha=2.0)
        out = hook._intervene(codes.clone())
        assert bool(hook.last_control_was_noop) is True
        assert torch.allclose(out, codes)          # untouched

    def test_existing_modes_unchanged(self):
        # random_direction still hard-deletes (z_j <- 0)
        out = _intervene("random_direction", 2.0, feature_idx=0)
        assert (out == 0).any()
        # amplify still scales the TARGET
        out = _intervene("amplify", 2.0, feature_idx=0)
        assert np.allclose(out[:, 0], 2.0)


class TestClosestActivityControl:
    """The R6 pre-verdict control-matching correction."""

    def test_closest_activity_sae(self):
        # activity profile: target (idx 0) has activity 1.0; idx 5
        # (0.25) is closer than idx 1 (0.5)? No: |0.5-1.0|=0.5 vs
        # |0.25-1.0|=0.75 -> control must be idx 1.
        act = torch.zeros(8)
        act[0], act[1], act[5] = 1.0, 0.5, 0.25
        assert _closest_activity_control(act, 0) == 1

    def test_closest_activity_pca_skips_target(self):
        coeffs = torch.zeros(64, 8)
        coeffs[:, 0] = 3.0
        coeffs[:, 2] = 2.5
        coeffs[:, 5] = 0.01     # near-inert component
        ctrl = _pca_closest_activity_control(coeffs, 0)
        assert ctrl == 2        # never the near-inert component

    def test_control_idx_designates_the_control_atom(self):
        sae, _ = _toy_sae_and_codes()
        codes = torch.zeros(16, sae.latent_dim)
        codes[:, 0] = 1.0     # target
        codes[:, 3] = 0.4    # designated control
        hook = SAEInterventionHook(sae, mode="dose_matched_control",
                                   feature_idx=0, alpha=3.0, control_idx=3)
        out = hook._intervene(codes.clone())
        assert torch.allclose(out[:, 0], torch.full((16,), 1.0))  # target kept
        assert torch.allclose(out[:, 3], torch.full((16,), 1.2))  # 0.4 * 3
        assert not hook.last_control_was_noop


class TestCurveClassification:
    """The registered R6a/R6b/R6c classifier."""

    @staticmethod
    def _curve(t_means, c_means=None, n=10, noise=0.0, seed=0):
        """Build a synthetic curve. noise is added to the TARGET arm only —
        modeling the real battery's paired design: collocation-point noise
        cancels in the difference; the residual diff variance is the
        intervention stochasticity (which control atom is sampled)."""
        rng = np.random.default_rng(seed)
        c_means = c_means if c_means is not None else [0.0] * len(DOSES)
        raw_t, raw_c = {}, {}
        for a, tm, cm in zip(DOSES, t_means, c_means):
            eps = noise * rng.standard_normal(n)     # diff noise
            raw_t[str(a)] = (tm + eps).tolist()
            raw_c[str(a)] = [float(cm)] * n
        return {
            "doses": list(DOSES),
            "target_delta_mean": list(t_means),
            "control_delta_mean": list(c_means),
            "_raw_t": raw_t, "_raw_c": raw_c,
        }

    def test_mechanistic_curve_classified_r6a(self):
        # monotone sign flip through alpha=1 (positive -> negative),
        # target above control at every dose
        t = [4.0, 2.0, 1.0, 0.5, -0.8, -2.0]  # at DOSES order
        c = [1.0, 0.5, 0.3, 0.1, -0.2, -0.5]
        cl = classify_curve(self._curve(t, c))
        assert cl["verdict"] == "R6a"
        assert cl["crosses_zero_0_to_15"] is True
        assert cl["same_sign_0_vs_2"] is False

    def test_energetic_curve_with_matched_control_is_r6b_matched(self):
        # dose-symmetric loss increase: delta(alpha) ~ c*(alpha-1)^2 >= 0 —
        # positive at alpha=0 AND alpha=2 (equal |alpha-1|=1 displacement),
        # no crossing in (0, 1.5) — AND the control shows the same V
        # shape (any atom's displacement damages the decode): the
        # reconstructive signature.
        t = [9.0, 4.0, 1.0, 0.25, 0.25, 1.0]
        c = [4.5, 2.0, 0.5, 0.1, 0.1, 0.5]
        cl = classify_curve(self._curve(t, c))
        assert cl["verdict"] == "R6b-ctrl-matched"
        assert cl["same_sign_0_vs_2"] is True
        assert cl["crosses_zero_0_to_15"] is False
        assert cl["control_range_ratio"] >= 0.5

    def test_planted_quadratic_channel_is_r6b_ctrl_flat(self):
        # the planted-control calibration: V-shaped target (causal
        # feature through a quadratic readout channel) with a FLAT
        # control curve (range ratio ~0.02 in the gate run)
        t = [0.045, 0.043, 0.020, 0.005, 0.005, 0.020]
        c = [0.0004, 0.0002, 0.0003, 0.0002, 0.0003, 0.0007]
        cl = classify_curve(self._curve(t, c))
        assert cl["verdict"] == "R6b-ctrl-flat"
        assert cl["control_range_ratio"] < 0.5

    def test_null_curve_classified_r6c(self):
        # target and control identical -> every paired-diff CI spans zero
        t = [4.0, 2.0, 1.0, 0.5, 1.2, 2.0]
        cl = classify_curve(self._curve(t, t, noise=0.0))
        assert cl["verdict"] == "R6c"
        assert all(cl["ci_spans_zero_all_doses"])

    def test_noisy_null_prefers_r6c(self):
        # small target edge at every dose, but noise >> difference
        t = [0.05, 0.03, 0.02, 0.01, 0.02, 0.04]
        c = [0.04, 0.02, 0.01, 0.0, 0.01, 0.03]
        cl = classify_curve(self._curve(t, c, noise=2.0, n=50))
        assert cl["verdict"] == "R6c"

    def test_inert_curve_is_r6c_not_r6a(self):
        # pre-verdict correction: all-zero target curves (inert components)
        # must NOT be read as R6a via signed-zero rounding (0.0 * -0.0 < 0)
        t = [0.0, 0.0, 0.0, 0.0, -0.0, -0.0]
        c = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        cl = classify_curve(self._curve(t, c))
        assert cl["verdict"] == "R6c"
        assert cl["inert_curve"] is True

    def test_precedence_null_over_shape(self):
        # energetic-shaped target but control IDENTICAL -> R6c wins
        t = [4.0, 2.0, 0.0, 1.0, 3.0, 4.0]
        cl = classify_curve(self._curve(t, t, noise=0.0))
        assert cl["verdict"] == "R6c"

    def test_summarize_arm_counts(self):
        curves = [
            self._curve([4.0, 2.0, 1.0, 0.5, -0.8, -2.0],
                        [1.0, 0.5, 0.3, 0.1, -0.2, -0.5]),   # R6a
            self._curve([9.0, 4.0, 1.0, 0.25, 0.25, 1.0],
                        [4.5, 2.0, 0.5, 0.1, 0.1, 0.5]),     # R6b-matched
            self._curve([0.045, 0.043, 0.020, 0.005, 0.005, 0.020],
                        [0.0004, 0.0002, 0.0003, 0.0002,
                         0.0003, 0.0007]),                  # R6b-flat
            self._curve([0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),      # R6c
        ]
        for i, cu in enumerate(curves):
            cu["feature_idx"] = i
        s = summarize_arm(curves)
        assert s["n_features"] == 4
        assert s["verdict_counts"] == {"R6a": 1, "R6b-ctrl-flat": 1,
                                       "R6b-ctrl-matched": 1, "R6c": 1}
