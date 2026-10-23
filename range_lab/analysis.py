from __future__ import annotations

import dataclasses
import datetime as dt
import random
import statistics as st
from typing import Sequence

from signal_desk import candles as C
from signal_desk.research import VARIANTS, replay, stats

LOOK = 10
BUCKETS = ("narrow", "normal", "wide")
MIN_N = 15
MIN_GAP_R = 0.10
MIN_PF = 1.20
MAX_P = 0.10
SHUFFLES = 2000


@dataclasses.dataclass
class Day:
    date: dt.date
    open: float
    high: float
    low: float
    close: float
    ratio: float | None = None
    nr7: bool = False
    inside: bool = False
    bucket: str | None = None

    @property
    def rng(self) -> float:
        return self.high - self.low


def days_from(n_all: Sequence[C.Candle], min_candles: int = 20) -> list[Day]:
    by: dict[dt.date, list[C.Candle]] = {}
    for c in n_all:
        by.setdefault(c.start.date(), []).append(c)
    out = []
    for d in sorted(by):
        cs = by[d]
        if len(cs) >= min_candles:
            out.append(Day(d, cs[0].open, max(c.high for c in cs), min(c.low for c in cs), cs[-1].close))
    for i, x in enumerate(out):
        if i >= LOOK + 1:
            prev, base = out[i - 1], out[i - 1 - LOOK:i - 1]
            avg = sum(b.rng for b in base) / LOOK
            x.ratio = prev.rng / avg if avg > 0 else None
            x.nr7 = prev.rng <= min(b.rng for b in out[i - 7:i])
            x.inside = prev.high <= out[i - 2].high and prev.low >= out[i - 2].low
    return out


def terciles(vals: Sequence[float]) -> tuple[float, float]:
    v = sorted(vals)
    return v[len(v) // 3], v[2 * len(v) // 3]


def assign(days: list[Day], older: set[dt.date]) -> tuple[float, float]:
    q1, q2 = terciles([d.ratio for d in days if d.ratio is not None and d.date in older])
    for d in days:
        if d.ratio is not None:
            d.bucket = "narrow" if d.ratio < q1 else ("wide" if d.ratio >= q2 else "normal")
    return q1, q2


def gate0(days: Sequence[Day]) -> dict:
    out = {}
    idx = {d.date: i for i, d in enumerate(days)}
    for b in BUCKETS:
        sel = [d for d in days if d.bucket == b]
        exp, trend, tdays = [], [], 0
        for d in sel:
            i = idx[d.date]
            avg = sum(x.rng for x in days[i - LOOK:i]) / LOOK
            if avg > 0 and d.rng > 0:
                exp.append(d.rng / avg)
                ts = abs(d.close - d.open) / d.rng
                trend.append(ts)
                tdays += ts >= 0.6
        out[b] = {"days": len(sel), "expansion": round(st.mean(exp), 3) if exp else None,
                  "trend_share": round(st.mean(trend), 3) if trend else None,
                  "trend_days_pct": round(100 * tdays / len(exp), 1) if exp else None}
    return out


def _avg_r(rs) -> float | None:
    return sum(x.r for x in rs) / len(rs) if rs else None


def shuffle_p(trades, bucket_of: dict[dt.date, str], seed: int = 7) -> float | None:
    days = sorted({t.signal.time.date() for t in trades if bucket_of.get(t.signal.time.date())})
    if not days:
        return None

    def gap(lab):
        n = [t for t in trades if lab.get(t.signal.time.date()) == "narrow"]
        w = [t for t in trades if lab.get(t.signal.time.date()) == "wide"]
        a, b = _avg_r(n), _avg_r(w)
        return None if a is None or b is None else a - b
    real = gap(bucket_of)
    if real is None:
        return None
    labels = [bucket_of[d] for d in days]
    rnd = random.Random(seed)
    hits = 0
    for _ in range(SHUFFLES):
        rnd.shuffle(labels)
        g = gap(dict(zip(days, labels)))
        hits += g is not None and g >= real - 1e-12
    return round(hits / SHUFFLES, 3)


def gate1(data: dict, days: list[Day], tested: list[dt.date], older: set[dt.date], lot: int = 65) -> dict:
    live = VARIANTS[0]
    assert live.key == "live"
    rs = replay(live, data["nifty"], data["banknifty"], data["vix"], tested, lot=lot)
    take = [x for x in rs if x.signal.action == "TAKE"]
    bucket_of = {d.date: d.bucket for d in days if d.bucket}
    out: dict = {"halves": {}}
    for half in ("older", "newer"):
        in_half = (lambda d: d in older) if half == "older" else (lambda d: d not in older)
        part = [x for x in take if in_half(x.signal.time.date())]
        n_days = sum(1 for d in tested if in_half(d))
        row = {"all": stats(part, n_days)}
        for b in BUCKETS:
            bd = sum(1 for d in tested if in_half(d) and bucket_of.get(d) == b)
            sel = [x for x in part if bucket_of.get(x.signal.time.date()) == b]
            row[b] = stats(sel, bd)
            row[b]["puts"] = stats([x for x in sel if x.signal.side == "SHORT"], bd)["n"]
        for flag in ("nr7", "inside"):
            fd = {d.date for d in days if getattr(d, flag)}
            row[flag] = stats([x for x in part if x.signal.time.date() in fd], sum(1 for d in tested if in_half(d) and d in fd))
        row["shuffle_p"] = shuffle_p(part, bucket_of)
        out["halves"][half] = row
    out["shuffle_p_pooled"] = shuffle_p(take, bucket_of)
    out["verdict"], out["why"] = verdict(out)
    return out


def verdict(g: dict) -> tuple[str, list[str]]:
    why, ok, direction = [], True, True
    for half in ("older", "newer"):
        h = g["halves"][half]
        n, w = h["narrow"], h["wide"]
        gap = None if n["avg_r"] is None or w["avg_r"] is None else round(n["avg_r"] - w["avg_r"], 3)
        why.append(f"{half}: narrow n={n['n']} avgR={n['avg_r']} PF={n['pf']} | wide n={w['n']} avgR={w['avg_r']} "
                   f"PF={w['pf']} | gap={gap}")
        if gap is None or gap <= 0:
            direction = False
        if n["n"] < MIN_N or gap is None or gap < MIN_GAP_R or (n["pf"] or 0) < MIN_PF:
            ok = False
    p = g["shuffle_p_pooled"]
    why.append(f"day-shuffle p (pooled) = {p}  (pass needs <= {MAX_P})")
    if ok and p is not None and p <= MAX_P:
        return "PASS - narrow-day signals clearly better in both halves; worth using as a grade booster", why
    if direction:
        return "PROMISING BUT UNPROVEN - right direction in both halves, below the pass mark; keep as paper idea", why
    return "FAIL - not consistently better after a narrow day; do not use it", why
