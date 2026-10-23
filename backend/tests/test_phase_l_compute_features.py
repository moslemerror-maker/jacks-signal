from __future__ import annotations

import datetime as dt

from backend.app.research.intraday_features.bar_math import Bar
from scripts.run_phase_l_compute_features import compute_g5_g6_row

UTC = dt.timezone.utc


def _bar(day: dt.date, minute: int, close: float, *, open_: float | None = None, high: float | None = None, low: float | None = None, volume: float = 1000.0) -> Bar:
    ts = dt.datetime(day.year, day.month, day.day, 9, 15, tzinfo=UTC) + dt.timedelta(minutes=minute)
    o = open_ if open_ is not None else close
    h = high if high is not None else max(o, close)
    l = low if low is not None else min(o, close)
    return Bar(ts, day, o, h, l, close, volume)


class TestComputeG5G6Row:
    def test_returns_exactly_the_two_expected_keys(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, i, 100.0 + i * 0.1) for i in range(20)]
        result = compute_g5_g6_row(bars)
        assert set(result.keys()) == {"g5_directional_pressure_10", "g6_opening_range_position"}

    def test_both_none_with_minimal_history(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, 0, 100.0)]
        result = compute_g5_g6_row(bars)
        assert result["g5_directional_pressure_10"] is None
        assert result["g6_opening_range_position"] is None

    def test_both_defined_with_enough_history(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, i, 100.0 + i * 0.1) for i in range(20)]
        result = compute_g5_g6_row(bars)
        assert result["g5_directional_pressure_10"] is not None
        assert result["g6_opening_range_position"] is not None
