from __future__ import annotations

from typing import Sequence

from .bar_math import Bar, today_bars
from .price_action import candle_direction
from .volatility_volume import volatility_expansion_ratio
from .vwap_levels import opening_range_high_low


def compression_then_expansion(
    bars_up_to_t: Sequence[Bar], *, lookback_bars: int = 10, short_n: int = 10, long_n: int = 60,
    compression_threshold: float = 0.7, expansion_threshold: float = 1.3,
) -> bool | None:
    if len(bars_up_to_t) <= lookback_bars:
        return None
    ratio_now = volatility_expansion_ratio(bars_up_to_t, short_n, long_n)
    ratio_then = volatility_expansion_ratio(bars_up_to_t[:-lookback_bars], short_n, long_n)
    if ratio_now is None or ratio_then is None:
        return None
    return ratio_then <= compression_threshold and ratio_now >= expansion_threshold


def recent_breakout_up(bars_up_to_t: Sequence[Bar], level: float | None, *, within_bars: int = 5) -> bool | None:
    if level is None or len(bars_up_to_t) <= within_bars:
        return None
    now_close = bars_up_to_t[-1].close
    then_close = bars_up_to_t[-1 - within_bars].close
    return now_close > level and then_close <= level


def recent_breakout_down(bars_up_to_t: Sequence[Bar], level: float | None, *, within_bars: int = 5) -> bool | None:
    if level is None or len(bars_up_to_t) <= within_bars:
        return None
    now_close = bars_up_to_t[-1].close
    then_close = bars_up_to_t[-1 - within_bars].close
    return now_close < level and then_close >= level


def level_touch_count(bars_up_to_t: Sequence[Bar], level: float | None, *, tolerance_pct: float = 0.002, lookback_bars: int = 100) -> int | None:
    if level is None or len(bars_up_to_t) < 2:
        return None
    history = bars_up_to_t[:-1]
    window = history[-lookback_bars:] if len(history) > lookback_bars else history
    tolerance = level * tolerance_pct
    return sum(1 for b in window if abs(b.high - level) <= tolerance or abs(b.low - level) <= tolerance)


def directional_pressure(bars_up_to_t: Sequence[Bar], *, lookback_bars: int = 10) -> float | None:
    if len(bars_up_to_t) < lookback_bars:
        return None
    window = bars_up_to_t[-lookback_bars:]
    if any(b.volume is None for b in window):
        return None
    return sum(candle_direction(b) * b.volume for b in window)


def opening_range_position(bars_up_to_t: Sequence[Bar], *, n_bars: int = 15) -> float | None:
    today = today_bars(bars_up_to_t)
    if not today:
        return None
    or_hl = opening_range_high_low(bars_up_to_t, n_bars)
    if or_hl is None:
        return None
    or_high, or_low = or_hl
    if or_high == or_low:
        return 0.5
    return (bars_up_to_t[-1].close - or_low) / (or_high - or_low)
