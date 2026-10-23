from __future__ import annotations

import datetime as dt

from backend.app.research.intraday_features.bar_math import Bar
from backend.app.research.intraday_features.momentum_oscillators import (
    ema,
    macd_histogram,
    relative_strength_index,
    rsi_slope,
)

UTC = dt.timezone.utc


def _bars(closes: list[float]) -> list[Bar]:
    base = dt.datetime(2026, 1, 5, 3, 45, tzinfo=UTC)
    out = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c
        h, l = max(o, c) + 0.01, min(o, c) - 0.01
        out.append(Bar(base + dt.timedelta(minutes=i), dt.date(2026, 1, 5), o, h, l, c, 1000.0))
    return out


class TestRelativeStrengthIndex:
    def test_none_with_insufficient_history(self):
        assert relative_strength_index(_bars([100, 101, 102]), period=14) is None

    def test_all_gains_is_rsi_100(self):
        closes = [100 + i for i in range(15)]
        assert relative_strength_index(_bars(closes), period=14) == 100.0

    def test_all_losses_is_rsi_0(self):
        closes = [100 - i for i in range(15)]
        assert relative_strength_index(_bars(closes), period=14) == 0.0

    def test_hand_computed_mixed_case(self):
        closes = [100, 102, 101, 104, 102]
        result = relative_strength_index(_bars(closes), period=4)
        assert abs(result - 62.5) < 1e-9

    def test_flat_window_with_zero_change_is_50(self):
        closes = [100.0] * 15
        assert relative_strength_index(_bars(closes), period=14) == 50.0

    def test_only_uses_the_trailing_window_not_earlier_history(self):
        tail = [100, 102, 101, 104, 102]
        short = relative_strength_index(_bars(tail), period=4)
        with_extra_history = relative_strength_index(_bars([50, 60, 45, 70, 55, 65] + tail), period=4)
        assert short == with_extra_history


class TestRsiSlope:
    def test_none_with_insufficient_history(self):
        assert rsi_slope(_bars([100, 101, 102]), period=4, slope_bars=2) is None

    def test_positive_when_rsi_rose(self):
        closes = [100, 99, 101, 100, 102, 104, 106, 108, 110]
        result = rsi_slope(_bars(closes), period=4, slope_bars=3)
        assert result is not None and result > 0


class TestMacdHistogram:
    def test_none_with_insufficient_history(self):
        assert macd_histogram(_bars([100] * 10), fast=12, slow=26, signal=9) is None

    def test_flat_price_series_has_zero_histogram(self):
        closes = [100.0] * 40
        result = macd_histogram(_bars(closes), fast=12, slow=26, signal=9)
        assert abs(result) < 1e-9

    def test_steadily_rising_price_produces_a_positive_histogram(self):
        closes = [100.0 + i * 0.5 for i in range(40)]
        result = macd_histogram(_bars(closes), fast=12, slow=26, signal=9)
        assert result > 0

    def test_steadily_falling_price_produces_a_negative_histogram(self):
        closes = [140.0 - i * 0.5 for i in range(40)]
        result = macd_histogram(_bars(closes), fast=12, slow=26, signal=9)
        assert result < 0

    def test_recomputing_from_the_full_history_matches_a_manual_ema_reference(self):
        closes = [100.0 + (i % 5) - 2 for i in range(50)]

        def _manual_ema(vals, span):
            alpha = 2.0 / (span + 1.0)
            out = [vals[0]]
            for v in vals[1:]:
                out.append(alpha * v + (1 - alpha) * out[-1])
            return out

        ema12, ema26 = _manual_ema(closes, 12), _manual_ema(closes, 26)
        macd_line = [f - s for f, s in zip(ema12, ema26)]
        signal_line = _manual_ema(macd_line, 9)
        expected = macd_line[-1] - signal_line[-1]

        result = macd_histogram(_bars(closes), fast=12, slow=26, signal=9)
        assert abs(result - expected) < 1e-9


class TestEma:
    def test_none_with_insufficient_history(self):
        assert ema(_bars([100.0] * 5), span=20) is None

    def test_flat_price_series_equals_the_flat_price(self):
        result = ema(_bars([100.0] * 30), span=9)
        assert abs(result - 100.0) < 1e-9

    def test_matches_a_manual_recursive_ema_reference(self):
        closes = [100.0 + (i % 7) - 3 for i in range(40)]

        def _manual_ema(vals, span):
            alpha = 2.0 / (span + 1.0)
            out = [vals[0]]
            for v in vals[1:]:
                out.append(alpha * v + (1 - alpha) * out[-1])
            return out

        expected = _manual_ema(closes, 20)[-1]
        result = ema(_bars(closes), span=20)
        assert abs(result - expected) < 1e-9

    def test_a_rising_series_has_ema_below_the_latest_price(self):
        closes = [100.0 + i * 0.5 for i in range(30)]
        result = ema(_bars(closes), span=9)
        assert result < closes[-1]

    def test_only_depends_on_bars_up_to_t_not_earlier_history(self):
        tail = [100.0, 102.0, 101.0, 104.0, 103.0, 105.0, 104.0, 106.0, 105.0, 107.0]
        short = ema(_bars(tail), span=5)
        with_different_history = ema(_bars([50.0, 55.0, 45.0, 60.0] + tail), span=5)
        assert short != with_different_history
