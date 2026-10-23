from __future__ import annotations

import datetime as dt

from backend.app.research.intraday_features.bar_math import Bar
from scripts.run_phase_k_entry_and_path import entry_and_path

UTC = dt.timezone.utc


def _bar(day: dt.date, minute: int, open_: float, high: float, low: float, close: float) -> Bar:
    ts = dt.datetime(day.year, day.month, day.day, 9, 15, tzinfo=UTC) + dt.timedelta(minutes=minute)
    return Bar(ts, day, open_, high, low, close, 1000.0)


class TestEntryAndPath:
    def test_entry_price_is_the_open_of_the_bar_immediately_after_t(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, 0, 100, 101, 99, 100.5), _bar(d, 1, 105, 106, 104, 105.5), _bar(d, 2, 105.5, 107, 105, 106)]
        result = entry_and_path(bars, t_index=0, horizon_bars=2)
        assert result.entry_price == 105.0

    def test_gross_return_is_signed_entry_to_horizon_close(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, 0, 100, 100, 100, 100), _bar(d, 1, 100, 101, 99, 100), _bar(d, 2, 100, 102, 98, 110)]
        result = entry_and_path(bars, t_index=0, horizon_bars=2)
        assert result.gross_return == (110.0 / 100.0) - 1.0

    def test_mfe_is_the_max_high_relative_to_entry_and_mae_the_min_low(self):
        d = dt.date(2026, 1, 5)
        bars = [
            _bar(d, 0, 100, 100, 100, 100),
            _bar(d, 1, 100, 108, 100, 100),
            _bar(d, 2, 100, 103, 90, 100),
            _bar(d, 3, 100, 105, 95, 100),
        ]
        result = entry_and_path(bars, t_index=0, horizon_bars=3)
        assert result.entry_price == 100.0
        assert result.mfe == (108.0 / 100.0) - 1.0
        assert result.mae == (100.0 - 90.0) / 100.0
        assert result.bars_to_mfe == 1
        assert result.bars_to_mae == 2

    def test_favorable_move_before_adverse_move_is_distinguishable_via_offsets(self):
        d = dt.date(2026, 1, 5)
        bars = [
            _bar(d, 0, 100, 100, 100, 100),
            _bar(d, 1, 100, 120, 100, 100),
            _bar(d, 2, 100, 101, 80, 100),
        ]
        result = entry_and_path(bars, t_index=0, horizon_bars=2)
        assert result.bars_to_mfe < result.bars_to_mae

    def test_undefined_when_entry_bar_crosses_the_session_boundary(self):
        d1, d2 = dt.date(2026, 1, 5), dt.date(2026, 1, 6)
        bars = [_bar(d1, 0, 100, 100, 100, 100), _bar(d2, 0, 101, 101, 101, 101)]
        assert entry_and_path(bars, t_index=0, horizon_bars=1) is None

    def test_undefined_when_the_window_crosses_the_session_boundary(self):
        d1, d2 = dt.date(2026, 1, 5), dt.date(2026, 1, 6)
        bars = [_bar(d1, 0, 100, 100, 100, 100), _bar(d1, 1, 100, 100, 100, 100), _bar(d2, 0, 101, 101, 101, 101)]
        assert entry_and_path(bars, t_index=0, horizon_bars=2) is None

    def test_undefined_with_insufficient_bars_remaining(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, 0, 100, 100, 100, 100), _bar(d, 1, 100, 100, 100, 100)]
        assert entry_and_path(bars, t_index=0, horizon_bars=5) is None

    def test_mfe_and_mae_are_never_negative(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, 0, 100, 100, 100, 100), _bar(d, 1, 100, 100, 100, 100), _bar(d, 2, 100, 100, 100, 100)]
        result = entry_and_path(bars, t_index=0, horizon_bars=2)
        assert result.mfe >= 0
        assert result.mae >= 0

    def test_same_bar_close_is_never_used_as_the_entry_price(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, 0, 100, 100, 100, 999.0), _bar(d, 1, 555.0, 556, 554, 555), _bar(d, 2, 555, 556, 554, 555)]
        result = entry_and_path(bars, t_index=0, horizon_bars=1)
        assert result.entry_price != 999.0
        assert result.entry_price == 555.0
