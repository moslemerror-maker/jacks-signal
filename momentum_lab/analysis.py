from __future__ import annotations

import dataclasses
import datetime as dt
import random
import statistics as st
from typing import Sequence

from signal_desk.candles import Candle

LOT, DELTA, COST = 65, 0.5, 60.0
MIN_N, MIN_PF, MIN_HIT, MAX_P, SHUFFLES = 40, 1.15, 50.0, 0.10, 2000


@dataclasses.dataclass
class Day:
    date: dt.date
    prev_close: float
    c945: float
    c1430: float
    c1500: float
    c1515: float
    c1530: float
    vix1500: float | None = None

    @property
    def r1(self) -> float:
        return self.c945 / self.prev_close - 1

    @property
    def rpen(self) -> float:
        return self.c1500 / self.c1430 - 1

    @property
    def move(self) -> float:
        return self.c1530 - self.c1500


def _close_at(cs: dict[dt.time, Candle], end: dt.time) -> float | None:
    start = (dt.datetime.combine(dt.date.min, end) - dt.timedelta(minutes=15)).time()
    c = cs.get(start)
    return c.close if c else None


def build_days(nifty: Sequence[Candle], vix: Sequence[Candle] = ()) -> list[Day]:
    by: dict[dt.date, dict[dt.time, Candle]] = {}
    for c in nifty:
        by.setdefault(c.start.date(), {})[c.start.time()] = c
    vby: dict[dt.date, dict[dt.time, Candle]] = {}
    for c in vix:
        vby.setdefault(c.start.date(), {})[c.start.time()] = c
    out, prev_close = [], None
    for d in sorted(by):
        cs = by[d]
        full = len(cs) >= 25
        vals = [_close_at(cs, t) for t in (dt.time(9, 45), dt.time(14, 30), dt.time(15, 0), dt.time(15, 15), dt.time(15, 30))]
        if full and prev_close and all(v is not None for v in vals):
            out.append(Day(d, prev_close, *vals, vix1500=_close_at(vby.get(d, {}), dt.time(15, 0))))
        prev_close = cs[max(cs)].close
    return out


def rupees(points: float) -> float:
    return points * DELTA * LOT - COST


def stats(pts: Sequence[float]) -> dict:
    n = len(pts)
    if not n:
        return {"n": 0, "hit": None, "avg_pts": None, "avg_rs": None, "pf": None, "rupees": 0, "max_dd": 0}
    rs = [rupees(p) for p in pts]
    gw, gl = sum(x for x in rs if x > 0), -sum(x for x in rs if x < 0)
    eq = peak = dd = 0.0
    for x in rs:
        eq += x
        peak = max(peak, eq)
        dd = min(dd, eq - peak)
    return {"n": n, "hit": round(100 * sum(1 for x in rs if x > 0) / n, 1), "avg_pts": round(st.mean(pts), 2),
            "avg_rs": round(st.mean(rs)), "pf": round(gw / gl, 2) if gl else None, "rupees": round(sum(rs)), "max_dd": round(dd)}


def primary_trades(days: Sequence[Day]) -> list[tuple[Day, int]]:
    out = []
    for d in days:
        s1, s2 = (d.r1 > 0) - (d.r1 < 0), (d.rpen > 0) - (d.rpen < 0)
        if s1 and s1 == s2:
            out.append((d, s1))
    return out


def first_only_trades(days: Sequence[Day]) -> list[tuple[Day, int]]:
    return [(d, (d.r1 > 0) - (d.r1 < 0)) for d in days if d.r1 != 0]


def flip_p(trades: Sequence[tuple[Day, int]], seed: int = 11) -> float | None:
    if not trades:
        return None
    moves = [d.move for d, _ in trades]
    real = st.mean(m * s for (d, s), m in zip(trades, moves))
    rnd = random.Random(seed)
    hits = sum(st.mean(m * rnd.choice((1, -1)) for m in moves) >= real - 1e-12 for _ in range(SHUFFLES))
    return round(hits / SHUFFLES, 3)


def approx_expiry(d: dt.date) -> bool:
    return d.weekday() == (3 if d < dt.date(2025, 9, 1) else 1)


def run(days: list[Day], older: set[dt.date]) -> dict:
    vix_cut = st.median([d.vix1500 for d in days if d.date in older and d.vix1500]) if any(d.vix1500 for d in days) else None
    res: dict = {"vix_cut": vix_cut, "halves": {}}
    prim_all = primary_trades(days)
    for half in ("older", "newer"):
        inh = (lambda d: d.date in older) if half == "older" else (lambda d: d.date not in older)
        hd = [d for d in days if inh(d)]
        prim = [(d, s) for d, s in prim_all if inh(d)]
        first = first_only_trades(hd)
        row = {"days": len(hd), "primary": stats([d.move * s for d, s in prim]),
               "first_only": stats([d.move * s for d, s in first]),
               "primary_exit_1515": stats([(d.c1515 - d.c1500) * s for d, s in prim]),
               "primary_flip_p": flip_p(prim), "first_only_flip_p": flip_p(first)}
        if vix_cut:
            row["primary_high_vix"] = stats([d.move * s for d, s in prim if d.vix1500 and d.vix1500 >= vix_cut])
            row["primary_low_vix"] = stats([d.move * s for d, s in prim if d.vix1500 and d.vix1500 < vix_cut])
        row["primary_expiry"] = stats([d.move * s for d, s in prim if approx_expiry(d.date)])
        row["primary_non_expiry"] = stats([d.move * s for d, s in prim if not approx_expiry(d.date)])
        row["avg_abs_last30_pts"] = round(st.mean(abs(d.move) for d in hd), 1) if hd else None
        res["halves"][half] = row
    res["primary_flip_p_pooled"] = flip_p(prim_all)
    res["verdict"], res["why"] = verdict(res)
    return res


def verdict(r: dict) -> tuple[str, list[str]]:
    why, ok, positive = [], True, True
    for half in ("older", "newer"):
        p = r["halves"][half]["primary"]
        why.append(f"{half}: n={p['n']} hit={p['hit']}% avg={p['avg_pts']} pts / ₹{p['avg_rs']} PF={p['pf']}")
        if (p["avg_rs"] or -1) <= 0:
            positive = False
        if p["n"] < MIN_N or (p["avg_rs"] or -1) <= 0 or (p["pf"] or 0) < MIN_PF or (p["hit"] or 0) <= MIN_HIT:
            ok = False
    pp = r["primary_flip_p_pooled"]
    why.append(f"random-direction p (pooled) = {pp}  (pass needs <= {MAX_P})")
    if ok and pp is not None and pp <= MAX_P:
        return "PASS - intraday momentum holds on Nifty in both years after costs", why
    if positive:
        return "PROMISING BUT UNPROVEN - profitable in both years but below the pass mark", why
    return "FAIL - no reliable last-half-hour momentum on Nifty after costs; do not use it", why
