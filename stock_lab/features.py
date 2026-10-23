from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Sequence

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
T0915, T0920, T0930, T1030, T1500 = (dt.time(9, 15), dt.time(9, 20), dt.time(9, 30), dt.time(10, 30), dt.time(15, 0))
MIN_STOP, MAX_STOP = 0.005, 0.02


@dataclasses.dataclass(slots=True)
class Trade:
    filled: bool
    entry: float = 0.0
    risk_pct: float = 0.0
    hold_pct: float = 0.0
    r2_pct: float = 0.0
    hold_out: str = ""
    r2_out: str = ""


@dataclasses.dataclass(slots=True)
class Day:
    date: dt.date
    symbol: str
    eligible: bool
    prev_close: float
    chg920: float
    chg930: float
    rv5: float
    rv15: float
    dir5: int
    p920: float
    p930: float
    p1500: float
    atr_pct: float
    turnover_cr: float
    long: Trade | None = None
    short: Trade | None = None


def _simulate(side: str, level: float, stop: float, bars: Sequence[tuple]) -> Trade:
    long = side == "LONG"
    if level <= 0:
        return Trade(False)
    if abs(level - stop) / level < MIN_STOP:
        stop = level * (1 - MIN_STOP) if long else level * (1 + MIN_STOP)
    risk = abs(level - stop)
    if risk / level > MAX_STOP:
        return Trade(False)
    entry = None
    i0 = 0
    for i, b in enumerate(bars):
        t = b[6]
        if t >= T1030:
            break
        if (long and b[2] >= level) or (not long and b[3] <= level):
            entry = max(level, b[1]) if long else min(level, b[1])
            i0 = i
            break
    if entry is None:
        return Trade(False)
    tgt = entry + 2 * risk if long else entry - 2 * risk
    sgn = 1 if long else -1
    res_hold = res_r2 = None
    last = entry
    for j, b in enumerate(bars[i0:]):
        t = b[6]
        if t >= T1500:
            break
        last = b[4]
        hit_stop = (b[3] <= stop) if long else (b[2] >= stop)
        hit_tgt = (b[2] >= tgt) if long else (b[3] <= tgt)
        if hit_stop:
            if res_hold is None:
                res_hold = ("SL", stop)
            if res_r2 is None:
                res_r2 = ("SL", stop)
            break
        if res_r2 is None and hit_tgt and j > 0:
            res_r2 = ("T2R", tgt)
        if res_hold and res_r2:
            break
    if res_hold is None:
        res_hold = ("15:00", last)
    if res_r2 is None:
        res_r2 = ("15:00", last)
    pct = lambda px: sgn * (px - entry) / entry * 100
    return Trade(True, entry, risk / entry * 100, pct(res_hold[1]), pct(res_r2[1]), res_hold[0], res_r2[0])


def day_records(symbol: str, rows: Sequence[tuple], sessions: set[dt.date] | None = None) -> list[Day]:
    by: dict[dt.date, list[tuple]] = {}
    for r in sorted(rows):
        t = dt.datetime.fromtimestamp(r[0], IST)
        if not (T0915 <= t.time() < dt.time(15, 30)):
            continue
        by.setdefault(t.date(), []).append((r[0], r[1], r[2], r[3], r[4], r[5], t.time()))
    days = sorted(by)
    out: list[Day] = []
    daily = []
    for k, d in enumerate(days):
        bars = by[d]
        hi, lo, cl = max(b[2] for b in bars), min(b[3] for b in bars), bars[-1][4]
        turn = sum(b[4] * b[5] for b in bars)
        f5 = next((b for b in bars if b[6] == T0915), None)
        v5 = f5[5] if f5 else 0.0
        v15 = sum(b[5] for b in bars if b[6] < T0930)
        if k >= 15 and (sessions is None or d in sessions) and f5 and len(bars) >= 60:
            prev = daily[-1]
            pc = prev[2]
            hist = daily[-14:]
            trs = [max(h - l, abs(h - pcc), abs(l - pcc)) for (h, l, _, _, _, _), (_, _, pcc, _, _, _) in zip(daily[-14:], daily[-15:-1])]
            atr = sum(trs) / len(trs) if trs else 0
            avg5 = sum(x[4] for x in hist) / len(hist)
            avg15 = sum(x[5] for x in hist) / len(hist)
            turnover = sum(x[3] for x in daily[-20:]) / min(20, len(daily))
            b930 = next((b for b in bars if b[6] == dt.time(9, 25)), None)
            b1500 = next((b for b in reversed(bars) if b[6] < T1500), None)
            if pc > 0 and b930 and b1500:
                rec = Day(d, symbol,
                          eligible=pc >= 100 and turnover >= 50e7 and atr / pc >= 0.01,
                          prev_close=pc, chg920=(f5[4] - pc) / pc * 100, chg930=(b930[4] - pc) / pc * 100,
                          rv5=(v5 / avg5) if avg5 else 0.0, rv15=(v15 / avg15) if avg15 else 0.0,
                          dir5=(1 if f5[4] > f5[1] else -1 if f5[4] < f5[1] else 0),
                          p920=f5[4], p930=b930[4], p1500=b1500[4], atr_pct=atr / pc * 100, turnover_cr=turnover / 1e7)
                after = [b for b in bars if b[6] >= T0920]
                rec.long = _simulate("LONG", f5[2], f5[3], after)
                rec.short = _simulate("SHORT", f5[3], f5[2], after)
                out.append(rec)
        daily.append((hi, lo, cl, turn, v5, v15))
    return out
