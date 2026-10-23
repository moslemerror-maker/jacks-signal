from __future__ import annotations

from typing import Sequence

from .bar_math import Bar, prior_session_bars, simple_return, today_bars
from .volatility_volume import realized_volatility


def _typical_price(bar: Bar) -> float:
    return (bar.high + bar.low + bar.close) / 3.0


def session_vwap(bars_up_to_t: Sequence[Bar]) -> float | None:
    today = today_bars(bars_up_to_t)
    if not today:
        return None
    if all(b.volume for b in today):
        num = sum(_typical_price(b) * b.volume for b in today)
        den = sum(b.volume for b in today)
        return num / den if den else None
    return sum(_typical_price(b) for b in today) / len(today)


def distance_from_vwap(bars_up_to_t: Sequence[Bar]) -> float | None:
    vwap = session_vwap(bars_up_to_t)
    if not vwap:
        return None
    return simple_return(vwap, bars_up_to_t[-1].close)


def vwap_slope(bars_up_to_t: Sequence[Bar], n_bars: int) -> float | None:
    today = today_bars(bars_up_to_t)
    if len(today) <= n_bars:
        return None
    vwap_now = session_vwap(bars_up_to_t)
    vwap_then = session_vwap(bars_up_to_t[: len(bars_up_to_t) - n_bars])
    if not vwap_now or not vwap_then:
        return None
    return simple_return(vwap_then, vwap_now)


def vwap_cross(bars_up_to_t: Sequence[Bar]) -> str | None:
    today = today_bars(bars_up_to_t)
    if len(today) < 2:
        return None
    vwap_prev = session_vwap(bars_up_to_t[:-1])
    vwap_now = session_vwap(bars_up_to_t)
    if vwap_prev is None or vwap_now is None:
        return None
    prev_close, cur_close = today[-2].close, today[-1].close
    if prev_close < vwap_prev and cur_close >= vwap_now:
        return "RECLAIM"
    if prev_close > vwap_prev and cur_close <= vwap_now:
        return "REJECTION"
    return "NONE"


def vwap_deviation_normalized(bars_up_to_t: Sequence[Bar], vol_n_bars: int) -> float | None:
    dist = distance_from_vwap(bars_up_to_t)
    vol = realized_volatility(bars_up_to_t, vol_n_bars)
    if dist is None or not vol:
        return None
    return dist / vol


def previous_day_high_low_close(bars_up_to_t: Sequence[Bar]) -> tuple[float, float, float] | None:
    prior = prior_session_bars(bars_up_to_t)
    if not prior:
        return None
    return max(b.high for b in prior), min(b.low for b in prior), prior[-1].close


def session_high_low_so_far(bars_up_to_t: Sequence[Bar]) -> tuple[float, float] | None:
    today = today_bars(bars_up_to_t)
    if not today:
        return None
    return max(b.high for b in today), min(b.low for b in today)


def opening_range_high_low(bars_up_to_t: Sequence[Bar], n_bars: int = 15) -> tuple[float, float] | None:
    today = today_bars(bars_up_to_t)
    if len(today) < n_bars:
        return None
    window = today[:n_bars]
    return max(b.high for b in window), min(b.low for b in window)


def rolling_swing_points(bars_up_to_t: Sequence[Bar], left: int = 5, right: int = 5) -> tuple[list[float], list[float]]:
    highs: list[float] = []
    lows: list[float] = []
    n = len(bars_up_to_t)
    for i in range(left, n - right):
        window = bars_up_to_t[i - left : i + right + 1]
        if bars_up_to_t[i].high == max(b.high for b in window):
            highs.append(bars_up_to_t[i].high)
        if bars_up_to_t[i].low == min(b.low for b in window):
            lows.append(bars_up_to_t[i].low)
    return highs, lows


def nearest_level_distance(current_price: float, levels: Sequence[float], *, above: bool) -> float | None:
    candidates = [lv for lv in levels if (lv > current_price if above else lv < current_price)]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda lv: abs(lv - current_price))
    return (nearest / current_price) - 1.0
