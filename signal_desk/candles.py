from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Iterable, Sequence

from .clock import IST, from_epoch


@dataclasses.dataclass(frozen=True)
class Candle:
    start: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    minutes: int = 15

    @property
    def end(self) -> dt.datetime:
        return self.start + dt.timedelta(minutes=self.minutes)

    @property
    def green(self) -> bool:
        return self.close >= self.open


def from_arrays(d: dict, minutes: int) -> list[Candle]:
    out = []
    n = len(d["close"])
    vols = d.get("volume") or [0] * n
    for i in range(n):
        try:
            out.append(Candle(from_epoch(d["timestamp"][i]), float(d["open"][i]), float(d["high"][i]),
                              float(d["low"][i]), float(d["close"][i]), float(vols[i] or 0), minutes))
        except (TypeError, ValueError, IndexError):
            continue
    out.sort(key=lambda c: c.start)
    return out


def aggregate(candles: Sequence[Candle], minutes: int, anchor: dt.time = dt.time(9, 15)) -> list[Candle]:
    out: list[Candle] = []
    group: list[Candle] = []
    key = None

    def flush():
        if group and group[-1].end == group[0].start.replace(hour=key[1] // 60, minute=key[1] % 60) + dt.timedelta(minutes=minutes):
            out.append(Candle(group[0].start.replace(hour=key[1] // 60, minute=key[1] % 60), group[0].open,
                              max(c.high for c in group), min(c.low for c in group), group[-1].close,
                              sum(c.volume for c in group), minutes))
    base = anchor.hour * 60 + anchor.minute
    for c in sorted(candles, key=lambda x: x.start):
        m = c.start.hour * 60 + c.start.minute
        k = (c.start.date(), base + (m - base) // minutes * minutes)
        if k != key:
            flush()
            group, key = [], k
        group.append(c)
    flush()
    return out


def for_day(candles: Iterable[Candle], day: dt.date) -> list[Candle]:
    return [c for c in candles if c.start.date() == day]


def completed(candles: Iterable[Candle], now: dt.datetime, grace_s: int = 3) -> list[Candle]:
    return [c for c in candles if c.end <= now - dt.timedelta(seconds=grace_s)]


def prior_days(candles: Iterable[Candle], day: dt.date) -> list[dt.date]:
    return sorted({c.start.date() for c in candles if c.start.date() < day})


def daily_hlc(candles: Sequence[Candle]) -> tuple[float, float, float] | None:
    if not candles:
        return None
    return max(c.high for c in candles), min(c.low for c in candles), candles[-1].close


def atr(candles: Sequence[Candle], n: int = 14) -> float:
    if len(candles) < 2:
        return (candles[0].high - candles[0].low) if candles else 0.0
    trs = []
    for prev, cur in zip(candles, candles[1:]):
        trs.append(max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close)))
    trs = trs[-n:]
    return sum(trs) / len(trs)


def vwap(candles: Sequence[Candle]) -> float | None:
    vol = sum(c.volume for c in candles)
    if vol <= 0:
        return None
    return sum(((c.high + c.low + c.close) / 3) * c.volume for c in candles) / vol


def slot_avg_volume(history: Sequence[Candle], slot: dt.time, days: int = 5) -> float:
    vols = [c.volume for c in history if c.start.time() == slot]
    vols = vols[-days:]
    return sum(vols) / len(vols) if vols else 0.0


def today_ist(now: dt.datetime) -> dt.date:
    return now.astimezone(IST).date()
