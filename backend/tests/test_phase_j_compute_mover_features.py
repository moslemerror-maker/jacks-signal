from __future__ import annotations

import datetime as dt

from backend.app.research.intraday_features.bar_math import Bar
from scripts.run_phase_j_compute_mover_features import _nearest_levels, compute_mover_mechanics_row

UTC = dt.timezone.utc


def _bar(day: dt.date, minute: int, close: float, high: float | None = None, low: float | None = None) -> Bar:
    ts = dt.datetime(day.year, day.month, day.day, 9, 15, tzinfo=UTC) + dt.timedelta(minutes=minute)
    h = high if high is not None else close
    l = low if low is not None else close
    return Bar(ts, day, close, h, l, close, 1000.0)


class TestNearestLevels:
    def test_picks_the_closest_level_above_and_below_current_price(self):
        d1, d2 = dt.date(2026, 1, 5), dt.date(2026, 1, 6)
        prior_day = [_bar(d1, i, 100.0, high=105.0, low=95.0) for i in range(20)]
        today = [_bar(d2, i, 100.0, high=100.5, low=99.5) for i in range(20)]
        bars = prior_day + today
        support, resistance = _nearest_levels(bars)
        assert resistance is not None and resistance <= 105.0
        assert support is not None and support >= 95.0

    def test_returns_none_when_no_levels_exist_yet(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, 0, 100.0)]
        support, resistance = _nearest_levels(bars)
        assert support is None
        assert resistance is None


class TestComputeMoverMechanicsRow:
    def test_returns_exactly_the_five_expected_keys(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, i, 100.0 + (i % 3) * 0.1, high=100.5 + i * 0.01, low=99.5 - i * 0.01) for i in range(30)]
        result = compute_mover_mechanics_row(bars)
        assert set(result.keys()) == {
            "compression_then_expansion", "recent_breakout_up_opening_range", "recent_breakout_down_opening_range",
            "level_touch_count_support", "level_touch_count_resistance",
        }

    def test_does_not_crash_with_minimal_history(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, 0, 100.0)]
        result = compute_mover_mechanics_row(bars)
        assert result["compression_then_expansion"] is None
