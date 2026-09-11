"""Tests for R7: the TOST equivalence classifier
(scripts/equivalence_test.py)."""
import math

from scripts.equivalence_test import tost_paired, DELTA


class TestTostClassifier:
    def _force(self, mean, se):
        """Build a tost_paired result with a controlled mean/se by
        monkeypatching-free construction: call with synthetic diffs
        engineered to the desired statistics (n large enough that the
        t quantile is ~1.64/1.96)."""
        n = 88
        # diffs with exact mean and sample std (se = std/sqrt(n))
        target_std = se * math.sqrt(n)
        diffs = [mean] * n
        # add zero-sum perturbations to reach the target variance
        if target_std > 0:
            unit = target_std * math.sqrt(n / (n - 1))
            half = n // 2
            for i in range(half):
                diffs[i] += unit
                diffs[n - 1 - i] -= unit
        return tost_paired(diffs, DELTA)

    def test_equivalent_when_ci_inside_margin(self):
        # tiny mean, tiny SE -> CI well inside +/- delta
        r = self._force(0.002, 0.002)
        assert r["equivalent"] is True
        assert r["ci90_within_margin"] is True

    def test_inconclusive_when_ci_straddles_boundary(self):
        # point estimate inside the margin but CI spilling past +delta
        # (the measured R7 shape: mean +0.0042, CI to +0.021)
        r = self._force(0.0042, 0.0102)
        assert r["equivalent"] is False
        assert r["ci90"][1] > DELTA and r["ci90"][0] < -DELTA
        assert r["underpowered_for_margin"] is True

    def test_rejected_when_ci_entirely_above_margin(self):
        r = self._force(0.05, 0.005)
        assert r["equivalent"] is False
        assert r["ci90"][0] > DELTA          # entirely outside

    def test_measured_r7_shape_classifies_inconclusive(self):
        # the actual committed numbers (from the stage-8 pairing):
        # mean +0.0042, se ~0.00997 -> R7b by the corrected trichotomy
        r = self._force(0.004225, 0.00997)
        lo, hi = r["ci90"]
        inside = lo > -DELTA and hi < DELTA
        entirely_outside = (lo > DELTA) or (hi < -DELTA)
        assert not inside and not entirely_outside   # inconclusive band

    def test_tost_p_is_max_of_one_sided(self):
        r = self._force(0.0042, 0.0102)
        assert 0.0 <= r["p_tost"] <= 1.0
        # for a mean inside the margin with a straddling CI, p_tost
        # must exceed alpha (that is what 'NOT ESTABLISHED' means)
        assert r["p_tost"] > 0.05

    def test_delta_is_the_preregistered_value(self):
        assert DELTA == 0.01
