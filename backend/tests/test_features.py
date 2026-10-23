from __future__ import annotations

from datetime import date, timedelta

import pytest

from backend.app.research.features.definitions import (
    STAGE_3_FEATURE_DEFINITIONS,
    validate_no_lagged_features,
)
from backend.app.research.features.price import compute_atr, compute_return_n_day
from backend.app.research.features.trend import compute_close_vs_sma, compute_sma
from backend.app.research.features.types import Bar


def _bars(closes: list[float], highs: list[float] | None = None, lows: list[float] | None = None) -> list[Bar]:
    highs = highs if highs is not None else closes
    lows = lows if lows is not None else closes
    start = date(2024, 1, 1)
    return [
        Bar(trade_date=start + timedelta(days=i), close=c, high=h, low=l)
        for i, (c, h, l) in enumerate(zip(closes, highs, lows))
    ]


class TestReturnNDay:
    def test_exact_value_with_sufficient_history(self):
        bars = _bars([100, 101, 102, 103, 104, 110])
        value = compute_return_n_day(bars, index=5, n=5)
        assert value == pytest.approx(0.10)

    def test_none_when_insufficient_history(self):
        bars = _bars([100, 101, 102])
        assert compute_return_n_day(bars, index=2, n=5) is None

    def test_exact_boundary_index_equals_n(self):
        bars = _bars([50, 51, 52, 53, 54, 55])
        value = compute_return_n_day(bars, index=5, n=5)
        assert value == pytest.approx(55 / 50 - 1)

    def test_one_short_of_boundary_is_none(self):
        bars = _bars([50, 51, 52, 53, 54])
        assert compute_return_n_day(bars, index=4, n=5) is None


class TestSmaAndCloseVsSma:
    def test_sma_exact_value(self):
        bars = _bars([1, 2, 3, 4, 5])
        assert compute_sma(bars, index=4, window=5) == pytest.approx(3.0)

    def test_sma_none_when_insufficient_history(self):
        bars = _bars([1, 2, 3, 4])
        assert compute_sma(bars, index=3, window=5) is None

    def test_close_vs_sma_exact_value(self):
        bars = _bars([1, 2, 3, 4, 5])
        value = compute_close_vs_sma(bars, index=4, window=5)
        assert value == pytest.approx(5 / 3 - 1)

    def test_close_vs_sma_none_when_sma_not_yet_computable(self):
        bars = _bars([1, 2, 3])
        assert compute_close_vs_sma(bars, index=2, window=5) is None


class TestAtr:
    def test_exact_value_hand_computed(self):
        bars = [
            Bar(trade_date=date(2024, 1, 1), close=10, high=10, low=10),
            Bar(trade_date=date(2024, 1, 2), close=11, high=12, low=8),
            Bar(trade_date=date(2024, 1, 3), close=12, high=13, low=11),
        ]
        assert compute_atr(bars, index=2, window=2) == pytest.approx(3.0)

    def test_none_when_index_less_than_window(self):
        bars = [
            Bar(trade_date=date(2024, 1, 1), close=10, high=10, low=10),
            Bar(trade_date=date(2024, 1, 2), close=11, high=12, low=8),
        ]
        assert compute_atr(bars, index=1, window=2) is None

    def test_exact_boundary_index_equals_window(self):
        bars = [
            Bar(trade_date=date(2024, 1, 1), close=10, high=10, low=10),
            Bar(trade_date=date(2024, 1, 2), close=11, high=12, low=8),
            Bar(trade_date=date(2024, 1, 3), close=12, high=13, low=11),
        ]
        assert compute_atr(bars, index=2, window=2) is not None


class TestNoFutureLeakage:
    def test_return_n_day_unaffected_by_appended_future_bars(self):
        closes = [100, 101, 102, 103, 104, 110]
        bars = _bars(closes)
        before = compute_return_n_day(bars, index=5, n=5)

        extended = _bars(closes + [999999.0])
        after = compute_return_n_day(extended, index=5, n=5)

        assert before == after == pytest.approx(0.10)

    def test_sma_and_close_vs_sma_unaffected_by_appended_future_bars(self):
        closes = [1, 2, 3, 4, 5]
        bars = _bars(closes)
        before_sma = compute_sma(bars, index=4, window=5)
        before_ratio = compute_close_vs_sma(bars, index=4, window=5)

        extended = _bars(closes + [-999999.0, 0.0001, 5_000_000.0])
        after_sma = compute_sma(extended, index=4, window=5)
        after_ratio = compute_close_vs_sma(extended, index=4, window=5)

        assert before_sma == after_sma == pytest.approx(3.0)
        assert before_ratio == after_ratio == pytest.approx(5 / 3 - 1)

    def test_atr_unaffected_by_appended_future_bars(self):
        bars = [
            Bar(trade_date=date(2024, 1, 1), close=10, high=10, low=10),
            Bar(trade_date=date(2024, 1, 2), close=11, high=12, low=8),
            Bar(trade_date=date(2024, 1, 3), close=12, high=13, low=11),
        ]
        before = compute_atr(bars, index=2, window=2)

        extended = bars + [
            Bar(trade_date=date(2024, 1, 4), close=999999, high=1000000, low=1)
        ]
        after = compute_atr(extended, index=2, window=2)

        assert before == after == pytest.approx(3.0)

    def test_a_future_outlier_bar_never_enters_a_full_series_recomputation(self):
        closes = [100 + i for i in range(30)]
        bars = _bars(closes)
        before = [compute_return_n_day(bars, i, n=5) for i in range(len(bars))]

        extended = _bars(closes + [-500000.0])
        after = [compute_return_n_day(extended, i, n=5) for i in range(len(bars))]

        assert before == after


class TestValidateNoLaggedFeatures:
    def test_stage_3_definitions_all_pass(self):
        validate_no_lagged_features(STAGE_3_FEATURE_DEFINITIONS)

    def test_rejects_a_definition_with_nonzero_lag(self):
        definitions = [
            dict(STAGE_3_FEATURE_DEFINITIONS[0], name="delayed_feature", availability_lag=2)
        ]
        with pytest.raises(ValueError, match="delayed_feature"):
            validate_no_lagged_features(definitions)

    def test_rejects_mixed_set_naming_only_the_offending_ones(self):
        good = dict(STAGE_3_FEATURE_DEFINITIONS[0])
        bad = dict(STAGE_3_FEATURE_DEFINITIONS[1], name="lagged_one", availability_lag=1)
        with pytest.raises(ValueError) as exc_info:
            validate_no_lagged_features([good, bad])
        assert "lagged_one" in str(exc_info.value)
        assert good["name"] not in str(exc_info.value)
