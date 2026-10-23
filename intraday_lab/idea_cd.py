from __future__ import annotations

import csv
import datetime as dt
import gzip
import random
import statistics as st
from pathlib import Path
from typing import Sequence

from . import idea_b as B

IST = B.IST
M5 = Path(__file__).resolve().parent.parent / "stock_lab" / "cache" / "m5"
VERSION = 1
TOP_C, RV_MIN, ATR_FRAC = 20, 1.0, 0.10
TOP_D, NIFTY_MOVE, D_STOP = 3, 0.003, 0.01
MIN_PF, MAX_P = 1.2, 0.05
T = dt.time


def read_m5(sid: int, folder: Path | None = None) -> dict[dt.date, list[tuple]]:
    p = (folder or M5) / f"{sid}.csv.gz"
    by: dict[dt.date, list[tuple]] = {}
    if not p.exists():
        return by
    with gzip.open(p, "rt", newline="") as f:
        for r in csv.reader(f):
            if len(r) < 6:
                continue
            try:
                ts = int(r[0])
                t = dt.datetime.fromtimestamp(ts, IST)
                by.setdefault(t.date(), []).append((t.time(), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])))
            except ValueError:
                continue
    for v in by.values():
        v.sort()
    return by


def charges_fut(entry: float, exit_: float, notional: float = 500_000.0, side: str = "LONG") -> float:
    buy, sell = (entry, exit_) if side == "LONG" else (exit_, entry)
    q = notional / entry
    bv, sv = buy * q, sell * q
    brok = 40.0
    stt = 0.0002 * sv
    exch = 0.0000173 * (bv + sv)
    sebi = 0.000001 * (bv + sv)
    stamp = 0.00002 * bv
    gst = 0.18 * (brok + exch + sebi)
    return 100 * (brok + stt + exch + sebi + stamp + gst + 0.0005 * (bv + sv)) / notional


def _res(side: int, entry: float, exit_: float) -> tuple[float, float, float]:
    g = B.gross_pct(side, entry, exit_)
    return (g, B.net_pct(side, entry, exit_), g - charges_fut(entry, exit_, side="LONG" if side > 0 else "SHORT"))


def _orb(side: int, level: float, atr: float, bars: Sequence[tuple], two_r: bool):
    entry = None
    for i, b in enumerate(bars):
        tm, o, h, l, c = b[0], b[1], b[2], b[3], b[4]
        if tm > T(15, 10):
            break
        if entry is None:
            if (side > 0 and h >= level) or (side < 0 and l <= level):
                entry = max(level, o) if side > 0 else min(level, o)
                risk = ATR_FRAC * atr
                stop, tgt = entry - side * risk, entry + side * 2 * risk
            else:
                continue
        if (side > 0 and l <= stop) or (side < 0 and h >= stop):
            return _res(side, entry, stop)
        if two_r and ((side > 0 and h >= tgt) or (side < 0 and l <= tgt)):
            return _res(side, entry, tgt)
        if tm == T(15, 10):
            return _res(side, entry, c)
    if entry is None:
        return None
    return _res(side, entry, bars[-1][4])


def _hold(side: int, entry: float, bars: Sequence[tuple], until: dt.time):
    stop = entry * (1 - side * D_STOP)
    last = None
    for b in bars:
        tm, h, l, c = b[0], b[2], b[3], b[4]
        if tm > until:
            break
        if (side > 0 and l <= stop) or (side < 0 and h >= stop):
            return _res(side, entry, stop)
        last = c
        if tm == until:
            return _res(side, entry, c)
    return _res(side, entry, last) if last is not None else None


def stock_days(sid: int, symbol: str, by: dict[dt.date, list[tuple]]) -> list[dict]:
    days = sorted(by)
    daily = {}
    for d in days:
        bs = by[d]
        daily[d] = (max(b[2] for b in bs), min(b[3] for b in bs), bs[-1][4], sum(b[4] * b[5] for b in bs))
    out = []
    for i in range(15, len(days)):
        d, pd = days[i], days[i - 1]
        bars = by[d]
        if bars[0][0] != T(9, 15) or len(bars) < 60:
            continue
        late = [b for b in by[pd] if b[0] >= T(15, 0)]
        vol = sum(b[5] for b in late)
        if not late or vol <= 0:
            continue
        prev = sum(b[4] * b[5] for b in late) / vol
        hist = days[max(0, i - 20):i]
        turn = st.median(daily[x][3] for x in hist)
        trs = []
        for j in range(max(1, i - 14), i):
            h, l, _, _ = daily[days[j]]
            pc = daily[days[j - 1]][2]
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        atr = sum(trs) / len(trs)
        fv = [by[x][0][5] for x in days[max(0, i - 14):i] if by[x] and by[x][0][0] == T(9, 15)]
        f = bars[0]
        rv = f[5] / (sum(fv) / len(fv)) if fv and sum(fv) > 0 else 0.0
        rest = [b for b in bars if b[0] >= T(9, 20)]
        rec = {"date": d, "sid": sid, "symbol": symbol, "eligible": turn >= B.MIN_TURNOVER and prev >= B.MIN_PRICE,
               "rv": rv, "color": (f[4] > f[1]) - (f[4] < f[1]), "atr": atr}
        rec["c"] = {(s, k): _orb(s, f[2] if s > 0 else f[3], atr, rest, k == "2R") for s in (1, -1) for k in ("EOD", "2R")}
        b945 = next((b for b in bars if b[0] == T(9, 45)), None)
        b940 = next((b for b in bars if b[0] == T(9, 40)), None)
        if b945 and b940:
            after = [b for b in bars if b[0] >= T(9, 45)]
            rec["chg945"] = b940[4] / prev - 1
            rec["d"] = {(s, k): _hold(s, b945[1], after, u) for s in (1, -1)
                        for k, u in (("EOD", T(15, 10)), ("1130", T(11, 25)), ("1300", T(12, 55)))}
        out.append(rec)
    return out


def nifty_945(nifty15: Sequence) -> dict[dt.date, float]:
    by: dict[dt.date, list] = {}
    for c in nifty15:
        by.setdefault(c.start.date(), []).append(c)
    days = sorted(by)
    out = {}
    for i in range(1, len(days)):
        c930 = next((c for c in by[days[i]] if c.start.time() == T(9, 30)), None)
        if c930:
            out[days[i]] = c930.close / by[days[i - 1]][-1].close - 1
    return out


def _stats(vals: list[tuple]) -> dict:
    return B._stats([v[1] for v in vals], [v[0] for v in vals]) | {
        "fut_net": round(st.mean(v[2] for v in vals), 3) if vals else None}


def _flip_p(pairs: list[tuple], real: float, seed: int = 5) -> float | None:
    pairs = [p for p in pairs if p[0] is not None or p[1] is not None]
    if len(pairs) < 20:
        return None
    rnd = random.Random(seed)
    hits = 0
    for _ in range(B.SHUFFLES):
        vals = [x for x in ((p[0] if rnd.random() < 0.5 else p[1]) for p in pairs) if x is not None]
        hits += bool(vals) and st.mean(vals) >= real - 1e-12
    return round(hits / B.SHUFFLES, 4)


def trades_c(records, days_sel, key="EOD", seed=11):
    by: dict[dt.date, list[dict]] = {}
    for r in records:
        if r["eligible"] and r["date"] in days_sel:
            by.setdefault(r["date"], []).append(r)
    rnd = random.Random(seed)
    real, rand, pairs = [], [], []
    for d in sorted(by):
        lst = by[d]
        pick = sorted((r for r in lst if r["rv"] >= RV_MIN), key=lambda r: -r["rv"])[:TOP_C]
        for r in pick:
            if r["color"]:
                v = r["c"][(r["color"], key)]
                if v:
                    real.append(v)
                pairs.append(tuple((r["c"][(s, key)] or (None, None))[1] for s in (1, -1)))
        for r in rnd.sample(lst, min(TOP_C, len(lst))):
            v = r["c"][(rnd.choice((1, -1)), key)]
            if v:
                rand.append(v)
    return real, rand, pairs


def trades_d(records, days_sel, nifty: dict, key="EOD", seed=12):
    by: dict[dt.date, list[dict]] = {}
    for r in records:
        if r["eligible"] and r["date"] in days_sel and "d" in r:
            by.setdefault(r["date"], []).append(r)
    rnd = random.Random(seed)
    real, rand, pairs = [], [], []
    for d in sorted(by):
        n = nifty.get(d)
        if n is None or abs(n) < NIFTY_MOVE:
            continue
        side = 1 if n < 0 else -1
        lst = by[d]
        cand = [r for r in lst if (r["chg945"] > 0 if side > 0 else r["chg945"] < 0)]
        pick = sorted(cand, key=lambda r: -(r["chg945"] - n) * side)[:TOP_D]
        for r in pick:
            v = r["d"][(side, key)]
            if v:
                real.append(v)
            pairs.append(tuple((r["d"][(s, key)] or (None, None))[1] for s in (1, -1)))
        for r in rnd.sample(lst, min(TOP_D, len(lst))):
            v = r["d"][(side, key)]
            if v:
                rand.append(v)
    return real, rand, pairs


def explore(records, explore_days: list[dt.date], nifty: dict) -> dict:
    half = len(explore_days) // 2
    halves = {"older": set(explore_days[:half]), "newer": set(explore_days[half:])}
    rows = []
    for idea, fn, keys in (("C", trades_c, ("EOD", "2R")), ("D", trades_d, ("EOD", "1130", "1300"))):
        for key in keys:
            primary = key == "EOD"
            st_, rn = {}, {}
            for h, s in halves.items():
                args = (records, s, nifty, key) if idea == "D" else (records, s, key)
                real, rand, _ = fn(*args)
                st_[h], rn[h] = _stats(real), _stats(rand)["net"]
            args = (records, set(explore_days), nifty, key) if idea == "D" else (records, set(explore_days), key)
            real, _, pairs = fn(*args)
            p = _flip_p(pairs, st.mean(v[1] for v in real)) if real else None
            ok = primary and p is not None and p <= MAX_P and all(
                (st_[h]["net"] or -9) > 0 and (st_[h]["pf"] or 0) >= MIN_PF and (st_[h]["net"] or -9) > (rn[h] or -9)
                for h in halves)
            rows.append({"exit": key, "mode": f"idea {idea}", "older": st_["older"], "newer": st_["newer"], "random": rn,
                         "p": p, "verdict": ("VAULT CANDIDATE" if ok else "fail") if primary else "info"})
    return {"rows": rows, "halves": {h: [str(min(s)), str(max(s)), len(s)] for h, s in halves.items()},
            "candidates": [f"{r['mode']}/{r['exit']}" for r in rows if r["verdict"] == "VAULT CANDIDATE"]}
