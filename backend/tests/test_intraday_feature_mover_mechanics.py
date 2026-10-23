from __future__ import annotations

import datetime as dt

import pytest

from backend.app.research.intraday_features import bar_math
from backend.app.research.intraday_features.bar_math import Bar
from backend.app.research.intraday_features.mover_mechanics import (
    compression_then_expansion,
    directional_pressure,
    level_touch_count,
    opening_range_position,
    recent_breakout_down,
    recent_breakout_up,
)

UTC = dt.timezone.utc


def _bar(day: dt.date, minute: int, close: float, *, open_: float | None = None, high: float | None = None, low: float | None = None, volume: float = 1000.0) -> Bar:
    ts = dt.datetime(day.year, day.month, day.day, 9, 15, tzinfo=UTC) + dt.timedelta(minutes=minute)
    o = open_ if open_ is not None else close
    h = high if high is not None else max(o, close)
    l = low if low is not None else min(o, close)
    return Bar(ts, day, o, h, l, close, volume)


def _flat_session(day: dt.date, n: int, price: float = 100.0) -> list[Bar]:
    return [_bar(day, i, price) for i in range(n)]


class TestCompressionThenExpansion:
    def test_true_when_genuinely_compressed_then_expanding(self):
        d = dt.date(2026, 1, 5)
        bars = []
        price = 100.0
        for i in range(60):
            price += 0.3 if i % 2 == 0 else -0.3
            bars.append(_bar(d, i, price, high=price + 0.01, low=price - 0.01))
        for i in range(60, 75):
            price += 0.01 if i % 2 == 0 else -0.01
            bars.append(_bar(d, i, price, high=price + 0.01, low=price - 0.01))
        for i in range(75, 90):
            price += 3.0 if i % 2 == 0 else -2.0
            bars.append(_bar(d, i, price, high=price + 0.01, low=price - 0.01))
        result = compression_then_expansion(bars[:85], lookback_bars=10, short_n=5, long_n=20)
        assert result is True

    def test_false_when_no_real_compression_happened(self):
        d = dt.date(2026, 1, 5)
        bars = []
        price = 100.0
        for i in range(80):
            drift = 0.3 if i % 2 == 0 else -0.3
            price += drift
            bars.append(_bar(d, i, price, high=price + 0.5, low=price - 0.5))
        result = compression_then_expansion(bars, lookback_bars=10, short_n=5, long_n=20)
        assert result is False

    def test_none_with_insufficient_history(self):
        d = dt.date(2026, 1, 5)
        bars = _flat_session(d, 5)
        assert compression_then_expansion(bars, lookback_bars=10) is None

    def test_uses_strictly_less_information_for_the_earlier_point_never_the_future(self):
        d = dt.date(2026, 1, 5)
        bars = []
        price = 100.0
        for i in range(75):
            price += 0.002 if i % 2 == 0 else -0.002
            bars.append(_bar(d, i, price, high=price + 0.003, low=price - 0.003))
        bars.append(_bar(d, 75, price + 20.0, high=price + 22.0, low=price - 2.0))
        result = compression_then_expansion(bars, lookback_bars=10, short_n=5, long_n=20)
        assert result in (True, False)


class TestRecentBreakoutUp:
    def test_true_when_crossed_within_the_window_and_still_holding(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, i, 99.0) for i in range(5)] + [_bar(d, 5, 101.0)]
        result = recent_breakout_up(bars, level=100.0, within_bars=5)
        assert result is True

    def test_false_when_already_above_level_long_before_the_window(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, i, 101.0) for i in range(10)]
        result = recent_breakout_up(bars, level=100.0, within_bars=5)
        assert result is False

    def test_false_when_not_currently_above_level(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, i, 99.0) for i in range(10)]
        assert recent_breakout_up(bars, level=100.0, within_bars=5) is False

    def test_none_with_no_level_or_insufficient_history(self):
        d = dt.date(2026, 1, 5)
        bars = _flat_session(d, 3)
        assert recent_breakout_up(bars, level=None, within_bars=5) is None
        assert recent_breakout_up(bars, level=100.0, within_bars=5) is None


class TestRecentBreakoutDown:
    def test_mirrors_breakout_up_for_the_downside(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, i, 101.0) for i in range(5)] + [_bar(d, 5, 99.0)]
        assert recent_breakout_down(bars, level=100.0, within_bars=5) is True


class TestLevelTouchCount:
    def test_counts_bars_within_tolerance_excluding_the_current_bar(self):
        d = dt.date(2026, 1, 5)
        bars = [
            _bar(d, 0, 100.0, high=100.05, low=99.95),
            _bar(d, 1, 105.0, high=105.0, low=104.0),
            _bar(d, 2, 100.05, high=100.1, low=99.9),
            _bar(d, 3, 200.0, high=200.0, low=200.0),
        ]
        result = level_touch_count(bars, level=100.0, tolerance_pct=0.002, lookback_bars=100)
        assert result == 2

    def test_current_bar_never_counted_even_if_it_touches(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, 0, 105.0, high=105.0, low=105.0), _bar(d, 1, 100.0, high=100.0, low=100.0)]
        result = level_touch_count(bars, level=100.0, tolerance_pct=0.001, lookback_bars=100)
        assert result == 0

    def test_none_with_no_level(self):
        d = dt.date(2026, 1, 5)
        bars = _flat_session(d, 5)
        assert level_touch_count(bars, level=None) is None

    def test_respects_lookback_window(self):
        d = dt.date(2026, 1, 5)
        far_touch = [_bar(d, 0, 100.0, high=100.0, low=100.0)]
        filler = [_bar(d, i, 200.0, high=200.0, low=200.0) for i in range(1, 20)]
        current = [_bar(d, 20, 200.0)]
        bars = far_touch + filler + current
        result = level_touch_count(bars, level=100.0, tolerance_pct=0.001, lookback_bars=5)
        assert result == 0


class TestAdversarialFutureBarIndependence:

    def _fixture(self) -> list[Bar]:
        d1, d2 = dt.date(2026, 1, 5), dt.date(2026, 1, 6)
        bars = []
        price = 100.0
        for i in range(90):
            price += 0.05 if i % 3 else -0.05
            bars.append(_bar(d1, i, price, high=price + 0.3, low=price - 0.3))
        for i in range(60):
            price += 0.05 if i % 2 else -0.05
            bars.append(_bar(d2, i, price, high=price + 0.3, low=price - 0.3))
        return bars

    @pytest.mark.parametrize("name,fn", [
        ("compression_then_expansion", lambda b: compression_then_expansion(b, lookback_bars=10, short_n=5, long_n=20)),
        ("recent_breakout_up", lambda b: recent_breakout_up(b, level=100.0, within_bars=5)),
        ("recent_breakout_down", lambda b: recent_breakout_down(b, level=100.0, within_bars=5)),
        ("level_touch_count", lambda b: level_touch_count(b, level=100.0, tolerance_pct=0.01, lookback_bars=50)),
        ("directional_pressure", lambda b: directional_pressure(b, lookback_bars=10)),
        ("opening_range_position", lambda b: opening_range_position(b, n_bars=15)),
    ])
    def test_value_unchanged_when_future_bars_appended(self, name, fn):
        all_bars = self._fixture()
        t_index = 70
        t = all_bars[t_index].bar_start_utc

        bars_at_t = bar_math.bars_as_of(all_bars[: t_index + 1], t)
        bars_with_future = bar_math.bars_as_of(all_bars, t)
        assert bars_at_t == bars_with_future

        value_at_t = fn(bars_at_t)
        value_with_future = fn(bars_with_future)
        assert value_at_t == value_with_future, f"{name} changed when future bars were appended: {value_at_t!r} != {value_with_future!r}"


class TestDirectionalPressure:
    def test_all_up_candles_gives_positive_sum_of_volumes(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, i, close=100.0 + i, open_=100.0 + i - 0.5, volume=1000.0) for i in range(10)]
        result = directional_pressure(bars, lookback_bars=10)
        assert result == pytest.approx(10 * 1000.0)

    def test_all_down_candles_gives_negative_sum_of_volumes(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, i, close=100.0 - i, open_=100.0 - i + 0.5, volume=1000.0) for i in range(10)]
        result = directional_pressure(bars, lookback_bars=10)
        assert result == pytest.approx(-10 * 1000.0)

    def test_mixed_directions_are_volume_weighted_not_just_counted(self):
        d = dt.date(2026, 1, 5)
        bars = [
            _bar(d, 0, close=101.0, open_=100.0, volume=5000.0),
            _bar(d, 1, close=99.0, open_=100.0, volume=100.0),
        ]
        result = directional_pressure(bars, lookback_bars=2)
        assert result == pytest.approx(5000.0 - 100.0)

    def test_none_with_insufficient_history(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, 0, 100.0)]
        assert directional_pressure(bars, lookback_bars=10) is None

    def test_none_when_any_bar_in_window_has_no_volume(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, i, 100.0 + i, volume=1000.0) for i in range(5)]
        bars[2] = bars[2]._replace(volume=None)
        assert directional_pressure(bars, lookback_bars=5) is None

    def test_only_the_trailing_window_is_used_not_the_whole_history(self):
        d = dt.date(2026, 1, 5)
        old_bars = [_bar(d, i, close=100.0 - i, open_=100.0 - i + 1.0, volume=9999.0) for i in range(20)]
        recent_up_bars = [_bar(d, 20 + i, close=100.0 + i, open_=100.0 + i - 1.0, volume=1000.0) for i in range(10)]
        result = directional_pressure(old_bars + recent_up_bars, lookback_bars=10)
        assert result == pytest.approx(10 * 1000.0)


class TestOpeningRangePosition:
    def test_at_the_opening_range_low_returns_zero(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, i, close=105.0, high=110.0, low=100.0) for i in range(15)]
        bars[-1] = bars[-1]._replace(close=100.0)
        result = opening_range_position(bars, n_bars=15)
        assert result == pytest.approx(0.0)

    def test_at_the_opening_range_high_returns_one(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, i, close=105.0, high=110.0, low=100.0) for i in range(15)]
        bars[-1] = bars[-1]._replace(close=110.0)
        result = opening_range_position(bars, n_bars=15)
        assert result == pytest.approx(1.0)

    def test_midpoint_returns_one_half(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, i, close=105.0, high=110.0, low=100.0) for i in range(15)]
        result = opening_range_position(bars, n_bars=15)
        assert result == pytest.approx(0.5)

    def test_none_before_the_opening_range_is_defined(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, i, 100.0) for i in range(5)]
        assert opening_range_position(bars, n_bars=15) is None

    def test_can_exceed_zero_one_bounds_when_price_moves_beyond_the_opening_range(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, i, close=105.0, high=110.0, low=100.0) for i in range(15)]
        bars.append(_bar(d, 15, close=120.0, high=120.0, low=118.0))
        result = opening_range_position(bars, n_bars=15)
        assert result > 1.0
