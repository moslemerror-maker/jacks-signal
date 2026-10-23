from __future__ import annotations

import math
from typing import Sequence

from .bar_math import Bar, mean, simple_return, stdev, today_bars, true_range
from .price_action import return_over_bars


def realized_volatility(bars_up_to_t: Sequence[Bar], n_bars: int) -> float | None:
    if len(bars_up_to_t) <= n_bars:
        return None
    window = bars_up_to_t[-n_bars - 1 :]
    rets = [simple_return(window[i].close, window[i + 1].close) for i in range(len(window) - 1)]
    return stdev(rets)


def average_true_range(bars_up_to_t: Sequence[Bar], n_bars: int) -> float | None:
    if len(bars_up_to_t) <= n_bars:
        return None
    start = len(bars_up_to_t) - n_bars
    trs = [true_range(bars_up_to_t[i], bars_up_to_t[i - 1].close if i > 0 else None) for i in range(start, len(bars_up_to_t))]
    return mean(trs)


def volatility_expansion_ratio(bars_up_to_t: Sequence[Bar], short_n: int = 10, long_n: int = 60) -> float | None:
    short_vol = realized_volatility(bars_up_to_t, short_n)
    long_vol = realized_volatility(bars_up_to_t, long_n)
    if not short_vol or not long_vol:
        return None
    return short_vol / long_vol


def volatility_normalized_return(bars_up_to_t: Sequence[Bar], return_n_bars: int, vol_n_bars: int) -> float | None:
    r = return_over_bars(bars_up_to_t, return_n_bars)
    vol = realized_volatility(bars_up_to_t, vol_n_bars)
    if r is None or not vol:
        return None
    return r / vol


def _same_bar_index_windows(bars_up_to_t: Sequence[Bar], lookback_sessions: int) -> list[list[Bar]]:
    if not bars_up_to_t:
        return []
    current_date = bars_up_to_t[-1].trading_date
    today = today_bars(bars_up_to_t)
    bar_index_today = len(today) - 1

    by_date: dict = {}
    for b in bars_up_to_t:
        if b.trading_date < current_date:
            by_date.setdefault(b.trading_date, []).append(b)
    sessions = sorted(by_date.keys())[-lookback_sessions:]

    out = []
    for d in sessions:
        day_bars = by_date[d]
        if len(day_bars) > bar_index_today:
            out.append(day_bars[: bar_index_today + 1])
    return out


def realized_vol_vs_time_of_day_baseline(bars_up_to_t: Sequence[Bar], n_bars: int = 10, lookback_sessions: int = 10) -> float | None:
    current_vol = realized_volatility(bars_up_to_t, n_bars)
    if not current_vol:
        return None
    baseline_vols = []
    for window in _same_bar_index_windows(bars_up_to_t, lookback_sessions):
        v = realized_volatility(window, n_bars)
        if v is not None:
            baseline_vols.append(v)
    baseline = mean(baseline_vols)
    if not baseline:
        return None
    return current_vol / baseline


def relative_volume(bars_up_to_t: Sequence[Bar], n_bars: int) -> float | None:
    if len(bars_up_to_t) <= n_bars:
        return None
    cur_vol = bars_up_to_t[-1].volume
    if cur_vol is None:
        return None
    window = bars_up_to_t[-1 - n_bars : -1]
    avg = mean([b.volume for b in window if b.volume is not None])
    if not avg:
        return None
    return cur_vol / avg


def volume_acceleration(bars_up_to_t: Sequence[Bar], n_bars: int) -> float | None:
    if len(bars_up_to_t) < 2 * n_bars:
        return None
    recent_sum = sum(b.volume or 0 for b in bars_up_to_t[-n_bars:])
    prior_sum = sum(b.volume or 0 for b in bars_up_to_t[-2 * n_bars : -n_bars])
    if not prior_sum:
        return None
    return recent_sum / prior_sum


def volume_vs_time_of_day_baseline(bars_up_to_t: Sequence[Bar], lookback_sessions: int = 10) -> float | None:
    if not bars_up_to_t or bars_up_to_t[-1].volume is None:
        return None
    baseline_vols = []
    for window in _same_bar_index_windows(bars_up_to_t, lookback_sessions):
        if window[-1].volume is not None:
            baseline_vols.append(window[-1].volume)
    baseline = mean(baseline_vols)
    if not baseline:
        return None
    return bars_up_to_t[-1].volume / baseline


def price_volume_correlation(bars_up_to_t: Sequence[Bar], n_bars: int) -> float | None:
    if len(bars_up_to_t) <= n_bars:
        return None
    window = bars_up_to_t[-n_bars - 1 :]
    rets = [simple_return(window[i].close, window[i + 1].close) for i in range(len(window) - 1)]
    vols = [window[i + 1].volume or 0.0 for i in range(len(window) - 1)]
    if len(rets) < 2:
        return None
    mr, mv = mean(rets), mean(vols)
    num = sum((r - mr) * (v - mv) for r, v in zip(rets, vols))
    den_r = math.sqrt(sum((r - mr) ** 2 for r in rets))
    den_v = math.sqrt(sum((v - mv) ** 2 for v in vols))
    if den_r == 0 or den_v == 0:
        return None
    return num / (den_r * den_v)
