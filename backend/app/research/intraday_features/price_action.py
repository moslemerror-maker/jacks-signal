from __future__ import annotations

from typing import Sequence

from ...data.calendar.equity_trading_calendar import IST, SESSION_OPEN_IST
from .bar_math import Bar, mean, prior_session_bars, simple_return, today_bars


def return_over_bars(bars_up_to_t: Sequence[Bar], n_bars: int) -> float | None:
    if len(bars_up_to_t) <= n_bars:
        return None
    return simple_return(bars_up_to_t[-1 - n_bars].close, bars_up_to_t[-1].close)


def distance_from_session_open(bars_up_to_t: Sequence[Bar]) -> float | None:
    today = today_bars(bars_up_to_t)
    if not today:
        return None
    return simple_return(today[0].open, bars_up_to_t[-1].close)


def distance_from_prev_close(bars_up_to_t: Sequence[Bar]) -> float | None:
    prior = prior_session_bars(bars_up_to_t)
    if not prior:
        return None
    return simple_return(prior[-1].close, bars_up_to_t[-1].close)


def range_position(bars_up_to_t: Sequence[Bar], n_bars: int) -> float | None:
    if len(bars_up_to_t) < n_bars:
        return None
    window = bars_up_to_t[-n_bars:]
    hi, lo = max(b.high for b in window), min(b.low for b in window)
    if hi == lo:
        return 0.5
    return (bars_up_to_t[-1].close - lo) / (hi - lo)


def range_expansion(bars_up_to_t: Sequence[Bar], n_bars: int) -> float | None:
    if len(bars_up_to_t) <= n_bars:
        return None
    current_range = bars_up_to_t[-1].high - bars_up_to_t[-1].low
    prior_window = bars_up_to_t[-1 - n_bars : -1]
    avg_prior_range = mean([b.high - b.low for b in prior_window])
    if not avg_prior_range:
        return None
    return current_range / avg_prior_range


def momentum_acceleration(bars_up_to_t: Sequence[Bar], n_bars: int) -> float | None:
    if len(bars_up_to_t) <= 2 * n_bars:
        return None
    r_recent = simple_return(bars_up_to_t[-1 - n_bars].close, bars_up_to_t[-1].close)
    r_prior = simple_return(bars_up_to_t[-1 - 2 * n_bars].close, bars_up_to_t[-1 - n_bars].close)
    return r_recent - r_prior


def candle_body_ratio(bar: Bar) -> float:
    rng = bar.high - bar.low
    return abs(bar.close - bar.open) / rng if rng else 0.0


def candle_upper_wick_ratio(bar: Bar) -> float:
    rng = bar.high - bar.low
    return (bar.high - max(bar.open, bar.close)) / rng if rng else 0.0


def candle_lower_wick_ratio(bar: Bar) -> float:
    rng = bar.high - bar.low
    return (min(bar.open, bar.close) - bar.low) / rng if rng else 0.0


def candle_direction(bar: Bar) -> int:
    if bar.close > bar.open:
        return 1
    if bar.close < bar.open:
        return -1
    return 0


def consecutive_same_direction_count(bars_up_to_t: Sequence[Bar]) -> int:
    if not bars_up_to_t:
        return 0
    d = candle_direction(bars_up_to_t[-1])
    if d == 0:
        return 0
    count = 0
    for b in reversed(bars_up_to_t):
        if candle_direction(b) == d:
            count += 1
        else:
            break
    return count


def is_bullish_engulfing(bars_up_to_t: Sequence[Bar]) -> bool | None:
    if len(bars_up_to_t) < 2:
        return None
    prev, cur = bars_up_to_t[-2], bars_up_to_t[-1]
    return candle_direction(prev) < 0 and candle_direction(cur) > 0 and cur.open <= prev.close and cur.close >= prev.open


def is_bearish_engulfing(bars_up_to_t: Sequence[Bar]) -> bool | None:
    if len(bars_up_to_t) < 2:
        return None
    prev, cur = bars_up_to_t[-2], bars_up_to_t[-1]
    return candle_direction(prev) > 0 and candle_direction(cur) < 0 and cur.open >= prev.close and cur.close <= prev.open


def minutes_since_open(bars_up_to_t: Sequence[Bar]) -> int | None:
    if not bars_up_to_t:
        return None
    t_ist = bars_up_to_t[-1].bar_start_utc.astimezone(IST)
    open_dt = t_ist.replace(hour=SESSION_OPEN_IST.hour, minute=SESSION_OPEN_IST.minute, second=0, microsecond=0)
    return int((t_ist - open_dt).total_seconds() // 60)


def time_bucket(bars_up_to_t: Sequence[Bar]) -> str | None:
    m = minutes_since_open(bars_up_to_t)
    if m is None:
        return None
    if m < 30:
        return "OPENING"
    if m < 300:
        return "MIDDAY"
    return "CLOSING"


def overnight_gap(bars_up_to_t: Sequence[Bar]) -> float | None:
    today = today_bars(bars_up_to_t)
    prior = prior_session_bars(bars_up_to_t)
    if not today or not prior:
        return None
    return simple_return(prior[-1].close, today[0].open)
