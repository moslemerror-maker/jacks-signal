from __future__ import annotations

import datetime as dt

from backend.app.research.intraday_features.bar_math import Bar
from scripts.run_phase_h_compute_features import _day_blocks, mover_diagnostics

UTC = dt.timezone.utc


def _bar(day: dt.date, minute: int, close: float, high: float | None = None, low: float | None = None) -> Bar:
    ts = dt.datetime(day.year, day.month, day.day, 9, 15, tzinfo=UTC) + dt.timedelta(minutes=minute)
    h = high if high is not None else close
    l = low if low is not None else close
    return Bar(ts, day, close, h, l, close, 100.0)


class TestDayBlocks:
    def test_splits_a_multi_day_array_into_contiguous_blocks(self):
        d1, d2 = dt.date(2026, 1, 5), dt.date(2026, 1, 6)
        bars = [_bar(d1, i, 100.0) for i in range(3)] + [_bar(d2, i, 100.0) for i in range(5)]
        blocks = _day_blocks(bars)
        assert blocks == [(0, 3, d1), (3, 5, d2)]

    def test_single_day_array_is_one_block(self):
        d1 = dt.date(2026, 1, 5)
        bars = [_bar(d1, i, 100.0) for i in range(10)]
        assert _day_blocks(bars) == [(0, 10, d1)]

    def test_empty_array_returns_no_blocks(self):
        assert _day_blocks([]) == []

    def test_blocks_cover_every_bar_exactly_once(self):
        d1, d2, d3 = dt.date(2026, 1, 5), dt.date(2026, 1, 6), dt.date(2026, 1, 7)
        bars = [_bar(d1, i, 100.0) for i in range(7)] + [_bar(d2, i, 100.0) for i in range(4)] + [_bar(d3, i, 100.0) for i in range(9)]
        blocks = _day_blocks(bars)
        total = sum(length for _, length, _ in blocks)
        assert total == len(bars)
        assert len(blocks) == 3


class TestMoverDiagnostics:
    def test_computes_unsigned_move_and_total_range_within_the_window(self):
        d1 = dt.date(2026, 1, 5)
        future = [_bar(d1, 1, 101.0, high=102.0, low=99.5), _bar(d1, 2, 98.0, high=101.5, low=97.0)]
        result = mover_diagnostics(reference_price=100.0, reference_trading_date=d1, future_bars=future, horizon_bars=2)
        assert result["future_abs_return"] == abs(98.0 - 100.0) / 100.0
        assert result["future_range_pct"] == (102.0 - 97.0) / 100.0

    def test_undefined_when_future_window_crosses_a_session_boundary(self):
        d1, d2 = dt.date(2026, 1, 5), dt.date(2026, 1, 6)
        future = [_bar(d1, 1, 101.0), _bar(d2, 0, 99.0)]
        result = mover_diagnostics(reference_price=100.0, reference_trading_date=d1, future_bars=future, horizon_bars=2)
        assert result["future_abs_return"] is None
        assert result["future_range_pct"] is None

    def test_undefined_with_insufficient_future_bars(self):
        d1 = dt.date(2026, 1, 5)
        future = [_bar(d1, 1, 101.0)]
        result = mover_diagnostics(reference_price=100.0, reference_trading_date=d1, future_bars=future, horizon_bars=5)
        assert result["future_abs_return"] is None

    def test_reference_trading_date_not_futures_own_date_gates_the_window(self):
        d1, d2 = dt.date(2026, 1, 5), dt.date(2026, 1, 6)
        future = [_bar(d2, i, 100.0 + i) for i in range(3)]
        result = mover_diagnostics(reference_price=100.0, reference_trading_date=d1, future_bars=future, horizon_bars=3)
        assert result["future_abs_return"] is None
        assert result["future_range_pct"] is None
