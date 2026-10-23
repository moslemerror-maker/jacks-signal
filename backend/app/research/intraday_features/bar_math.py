from __future__ import annotations

import dataclasses
import datetime as dt
import math
from typing import NamedTuple, Sequence


class Bar(NamedTuple):
    bar_start_utc: dt.datetime
    trading_date: dt.date
    open: float
    high: float
    low: float
    close: float
    volume: float | None


def from_canonical(bar) -> Bar:
    return Bar(bar.bar_start_utc, bar.trading_date, bar.open, bar.high, bar.low, bar.close, bar.volume)


def bars_as_of(all_bars: Sequence[Bar], t: dt.datetime) -> list[Bar]:
    return [b for b in all_bars if b.bar_start_utc <= t]


def today_bars(bars_up_to_t: Sequence[Bar]) -> list[Bar]:
    if not bars_up_to_t:
        return []
    d = bars_up_to_t[-1].trading_date
    out = []
    for b in reversed(bars_up_to_t):
        if b.trading_date != d:
            break
        out.append(b)
    out.reverse()
    return out


def prior_session_bars(bars_up_to_t: Sequence[Bar]) -> list[Bar]:
    if not bars_up_to_t:
        return []
    current_date = bars_up_to_t[-1].trading_date
    prior_date = None
    for b in reversed(bars_up_to_t):
        if b.trading_date < current_date:
            prior_date = b.trading_date
            break
    if prior_date is None:
        return []
    return [b for b in bars_up_to_t if b.trading_date == prior_date]


def simple_return(price_from: float, price_to: float) -> float:
    return (price_to / price_from) - 1.0 if price_from else 0.0


def true_range(bar: Bar, prev_close: float | None) -> float:
    if prev_close is None:
        return bar.high - bar.low
    return max(bar.high - bar.low, abs(bar.high - prev_close), abs(bar.low - prev_close))


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def stdev(values: Sequence[float]) -> float | None:
    n = len(values)
    if n < 2:
        return None
    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))


def percentile_rank(values: Sequence[float], x: float) -> float | None:
    if not values:
        return None
    return sum(1 for v in values if v < x) / len(values)
