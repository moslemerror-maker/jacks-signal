from __future__ import annotations

import math
import random
import statistics as pystats

import pytest

from backend.app.research.statistics import (
    MeanTestResult,
    naive_iid_mean_test,
    newey_west_mean_test,
    non_overlapping_subsample_mean_test,
)


class TestInputValidation:
    def test_newey_west_rejects_fewer_than_two_values(self):
        with pytest.raises(ValueError, match="at least 2"):
            newey_west_mean_test([1.0], max_lag=0)

    def test_newey_west_rejects_negative_max_lag(self):
        with pytest.raises(ValueError, match="max_lag"):
            newey_west_mean_test([1.0, 2.0, 3.0], max_lag=-1)

    def test_newey_west_rejects_max_lag_too_large(self):
        with pytest.raises(ValueError, match="max_lag"):
            newey_west_mean_test([1.0, 2.0, 3.0], max_lag=3)

    def test_non_overlapping_rejects_stride_below_one(self):
        with pytest.raises(ValueError, match="stride"):
            non_overlapping_subsample_mean_test([1.0, 2.0, 3.0], stride=0)

    def test_non_overlapping_rejects_too_few_resulting_observations(self):
        with pytest.raises(ValueError, match="not enough"):
            non_overlapping_subsample_mean_test([1.0, 2.0, 3.0], stride=5)

    def test_naive_rejects_fewer_than_two_values(self):
        with pytest.raises(ValueError, match="at least 2"):
            naive_iid_mean_test([1.0])

    def test_invalid_confidence_level_rejected(self):
        with pytest.raises(ValueError, match="confidence_level"):
            newey_west_mean_test([1.0, 2.0, 3.0], max_lag=0, confidence_level=1.5)


class TestNeweyWestDegeneratesToNaiveAtZeroLag:
    def test_zero_lag_matches_naive_exactly(self):
        values = [0.01, -0.02, 0.03, 0.015, -0.005, 0.02, -0.01, 0.0]
        hac = newey_west_mean_test(values, max_lag=0)
        naive = naive_iid_mean_test(values)
        assert hac.standard_error == pytest.approx(naive.standard_error)
        assert hac.test_statistic == pytest.approx(naive.test_statistic)
        assert hac.p_value == pytest.approx(naive.p_value)


class TestNonOverlappingStrideOneMatchesNaive:
    def test_stride_one_matches_naive_exactly(self):
        values = [0.01, -0.02, 0.03, 0.015, -0.005, 0.02, -0.01, 0.0]
        subsample = non_overlapping_subsample_mean_test(values, stride=1)
        naive = naive_iid_mean_test(values)
        assert subsample.standard_error == pytest.approx(naive.standard_error)
        assert subsample.n == naive.n


class TestNeweyWestHandComputedExample:
    def test_exact_value_max_lag_one(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = newey_west_mean_test(values, max_lag=1)
        expected_se = math.sqrt(2.8 / 5)
        assert result.mean == pytest.approx(3.0)
        assert result.standard_error == pytest.approx(expected_se)


class TestResultShape:
    def test_confidence_interval_contains_the_mean_and_is_ordered(self):
        values = [0.01, -0.02, 0.03, 0.015, -0.005, 0.02, -0.01, 0.0, 0.005, -0.015]
        for result in (
            newey_west_mean_test(values, max_lag=2),
            non_overlapping_subsample_mean_test(values, stride=2),
            naive_iid_mean_test(values),
        ):
            low, high = result.confidence_interval
            assert low < high
            assert low <= result.mean <= high

    def test_clearly_positive_constant_offset_series_rejects_zero_mean(self):
        values = [1.0 + 0.001 * i for i in range(200)]
        result = naive_iid_mean_test(values)
        assert result.p_value < 0.001

    def test_exactly_zero_mean_symmetric_series_does_not_reject(self):
        values = [1.0, -1.0] * 50
        result = naive_iid_mean_test(values)
        assert result.mean == pytest.approx(0.0)
        assert result.p_value == pytest.approx(1.0)


def _rolling_mean_of_iid_shocks(shocks: list[float], window: int) -> list[float]:
    return [
        pystats.fmean(shocks[i : i + window]) for i in range(len(shocks) - window + 1)
    ]


class TestAdversarialOverlappingData:

    WINDOW = 5
    N_SHOCKS = 2500

    @classmethod
    @pytest.fixture(scope="class")
    def overlapping_series(cls):
        rng = random.Random(20260916)
        shocks = [rng.uniform(-1.0, 1.0) for _ in range(cls.N_SHOCKS)]
        return _rolling_mean_of_iid_shocks(shocks, cls.WINDOW)

    def test_naive_standard_error_is_undersized_by_roughly_sqrt_window(
        self, overlapping_series
    ):
        naive = naive_iid_mean_test(overlapping_series)
        hac = newey_west_mean_test(overlapping_series, max_lag=self.WINDOW - 1)

        ratio = hac.standard_error / naive.standard_error
        expected_ratio = math.sqrt(self.WINDOW)

        assert ratio == pytest.approx(expected_ratio, rel=0.25), (
            f"HAC/naive standard-error ratio was {ratio:.3f}, expected close "
            f"to sqrt({self.WINDOW})={expected_ratio:.3f} - if this drifts, "
            "either the HAC implementation or the synthetic data construction "
            "has a bug."
        )
        assert naive.standard_error < hac.standard_error

    def test_non_overlapping_subsample_agrees_with_hac_not_with_naive(
        self, overlapping_series
    ):
        naive = naive_iid_mean_test(overlapping_series)
        hac = newey_west_mean_test(overlapping_series, max_lag=self.WINDOW - 1)
        subsample = non_overlapping_subsample_mean_test(overlapping_series, stride=self.WINDOW)

        assert subsample.standard_error == pytest.approx(hac.standard_error, rel=0.4)
        assert subsample.standard_error > naive.standard_error * 1.5

    def test_naive_p_value_is_smaller_than_hac_p_value_for_the_same_data(
        self, overlapping_series
    ):
        naive = naive_iid_mean_test(overlapping_series)
        hac = newey_west_mean_test(overlapping_series, max_lag=self.WINDOW - 1)
        assert naive.p_value < hac.p_value

    def test_method_labels_are_distinct_and_explicit(self, overlapping_series):
        naive = naive_iid_mean_test(overlapping_series)
        hac = newey_west_mean_test(overlapping_series, max_lag=self.WINDOW - 1)
        subsample = non_overlapping_subsample_mean_test(overlapping_series, stride=self.WINDOW)
        assert {naive.method, hac.method, subsample.method} == {
            "naive_iid",
            "newey_west",
            "non_overlapping_subsample",
        }
