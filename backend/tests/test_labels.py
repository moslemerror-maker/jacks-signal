from __future__ import annotations

from datetime import date, timedelta

import pytest

from backend.app.research.features.types import Bar
from backend.app.research.labels.forward_return import compute_forward_return


def _bars(closes: list[float]) -> list[Bar]:
    start = date(2024, 1, 1)
    return [
        Bar(trade_date=start + timedelta(days=i), close=c, high=c, low=c)
        for i, c in enumerate(closes)
    ]


class TestComputeForwardReturn:
    def test_exact_value_with_sufficient_future_bars(self):
        bars = _bars([100, 101, 102, 103, 104, 110])
        horizon_end_date, realized_return = compute_forward_return(bars, index=0, horizon_days=5)
        assert horizon_end_date == bars[5].trade_date
        assert realized_return == pytest.approx(0.10)

    def test_none_when_fewer_than_horizon_days_of_future_bars_exist(self):
        bars = _bars([100, 101, 102, 103, 104])
        assert compute_forward_return(bars, index=0, horizon_days=5) is None

    def test_boundary_exactly_enough_future_bars(self):
        bars = _bars([100, 101, 102, 103, 104, 105])
        result = compute_forward_return(bars, index=0, horizon_days=5)
        assert result is not None

    def test_boundary_one_bar_short_is_none(self):
        bars = _bars([100, 101, 102, 103, 104])
        assert compute_forward_return(bars, index=0, horizon_days=5) is None

    def test_horizon_end_date_is_the_actual_calendar_trade_date_of_the_target_bar(self):
        bars = _bars([100, 101, 102, 103, 104, 110])
        horizon_end_date, _ = compute_forward_return(bars, index=0, horizon_days=5)
        assert horizon_end_date == date(2024, 1, 6)


class TestNoFutureLeakageBeyondHorizon:
    def test_appending_bars_past_the_horizon_does_not_change_the_result(self):
        closes = [100, 101, 102, 103, 104, 110]
        bars = _bars(closes)
        before = compute_forward_return(bars, index=0, horizon_days=5)

        extended = _bars(closes + [999999.0, -999999.0])
        after = compute_forward_return(extended, index=0, horizon_days=5)

        assert before == after

    def test_a_missing_label_becomes_computable_only_once_the_real_future_bar_arrives(self):
        closes = [100, 101, 102, 103, 104]
        bars = _bars(closes)
        assert compute_forward_return(bars, index=0, horizon_days=5) is None

        extended = _bars(closes + [120.0])
        result = compute_forward_return(extended, index=0, horizon_days=5)
        assert result is not None
        horizon_end_date, realized_return = result
        assert realized_return == pytest.approx(0.20)
