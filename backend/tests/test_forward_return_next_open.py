from __future__ import annotations

from datetime import date, timedelta

import pytest

from backend.app.research.labels.forward_return import compute_forward_return
from backend.app.research.labels.forward_return_next_open import (
    OpenBar,
    compute_forward_return_next_open,
)
from backend.app.research.features.types import Bar


def _open_bars(opens: list[float]) -> list[OpenBar]:
    start = date(2024, 1, 1)
    return [
        OpenBar(trade_date=start + timedelta(days=i), open=o) for i, o in enumerate(opens)
    ]


class TestExactIndexing:
    def test_entry_is_one_session_after_signal_exit_is_horizon_days_after_entry(self):
        bars = _open_bars([100, 101, 102, 103, 104, 105])
        result = compute_forward_return_next_open(bars, index=0, horizon_days=3)
        entry_date, exit_date, entry_price, exit_price, realized_return = result
        assert entry_date == bars[1].trade_date
        assert entry_price == bars[1].open
        assert exit_date == bars[4].trade_date
        assert exit_price == bars[4].open
        assert realized_return == pytest.approx(bars[4].open / bars[1].open - 1.0)

    def test_boundary_exactly_enough_bars(self):
        bars = _open_bars([100, 101, 102, 103, 104])
        assert compute_forward_return_next_open(bars, index=0, horizon_days=3) is not None

    def test_boundary_one_bar_short_is_none(self):
        bars = _open_bars([100, 101, 102, 103])
        assert compute_forward_return_next_open(bars, index=0, horizon_days=3) is None


class TestMissingDataBehavior:
    def test_no_entry_bar_at_all_is_none(self):
        bars = _open_bars([100])
        assert compute_forward_return_next_open(bars, index=0, horizon_days=3) is None

    def test_zero_open_price_is_none(self):
        bars = _open_bars([100, 0, 102, 103, 104])
        assert compute_forward_return_next_open(bars, index=0, horizon_days=3) is None


class TestNoFutureLeakage:
    def test_appending_future_bars_does_not_change_an_earlier_result(self):
        opens = [100, 101, 102, 103, 104, 105]
        bars = _open_bars(opens)
        before = compute_forward_return_next_open(bars, index=0, horizon_days=3)

        extended = _open_bars(opens + [999999.0, -999999.0])
        after = compute_forward_return_next_open(extended, index=0, horizon_days=3)
        assert before == after

    def test_a_missing_result_becomes_computable_only_once_the_real_future_bar_arrives(self):
        opens = [100, 101, 102, 103]
        bars = _open_bars(opens)
        assert compute_forward_return_next_open(bars, index=0, horizon_days=3) is None

        extended = _open_bars(opens + [120.0])
        result = compute_forward_return_next_open(extended, index=0, horizon_days=3)
        assert result is not None
        assert result[4] == pytest.approx(120.0 / 101.0 - 1.0)


class TestOvernightGapInclusion:
    def test_a_gap_between_signal_close_and_next_open_is_reflected(self):
        dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(7)]
        closes = [100, 109, 110, 111, 112, 113, 114]
        opens = [100, 110, 110, 111, 112, 113, 114]

        close_bars = [Bar(trade_date=d, close=c, high=c, low=c) for d, c in zip(dates, closes)]
        open_bars = [OpenBar(trade_date=d, open=o) for d, o in zip(dates, opens)]

        same_close_result = compute_forward_return(close_bars, index=0, horizon_days=5)
        next_open_result = compute_forward_return_next_open(open_bars, index=0, horizon_days=5)

        same_close_return = same_close_result[1]
        next_open_return = next_open_result[4]

        assert same_close_return == pytest.approx(113 / 100 - 1)
        assert next_open_return == pytest.approx(114 / 110 - 1)
        assert same_close_return != pytest.approx(next_open_return)


class TestSameCloseVsNextOpenAreIntentionallyDifferentQuantities:
    def test_identical_flat_series_with_no_gap_can_still_differ_due_to_the_index_shift(self):
        dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(8)]
        prices = [100 + i for i in range(8)]

        close_bars = [Bar(trade_date=d, close=p, high=p, low=p) for d, p in zip(dates, prices)]
        open_bars = [OpenBar(trade_date=d, open=p) for d, p in zip(dates, prices)]

        same_close = compute_forward_return(close_bars, index=0, horizon_days=5)[1]
        next_open = compute_forward_return_next_open(open_bars, index=0, horizon_days=5)[4]

        assert same_close == pytest.approx(0.05)
        assert next_open == pytest.approx(106 / 101 - 1)
        assert same_close != pytest.approx(next_open)
