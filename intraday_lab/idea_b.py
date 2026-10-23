from __future__ import annotations

import bisect
import datetime as dt
import random
import statistics as st
from typing import Sequence

from stock_lab.costs import charges

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
VERSION = 1
NOTIONAL = 100_000.0
MIN_TURNOVER, MIN_PRICE, TOP = 50e7, 100.0, 3
EXITS = ("T03", "T05", "T10", "X930", "X945", "X1015")
TARGETS = {"T03": 0.003, "T05": 0.005, "T10": 0.010}
TIMES = {"X930": dt.time(9, 29), "X945": dt.time(9, 44), "X1015": dt.time(10, 14)}
CAP = dt.time(10, 14)
MIN_PF, MAX_P, SHUFFLES = 1.2, 0.01, 2000


def _t(ts: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(ts, IST)


def _walk(side: int, entry: float, bars: Sequence[tuple], exit_key: str) -> float | None:
    if exit_key in TIMES:
        end = TIMES[exit_key]
        for b in bars:
            if _t(b[0]).time() == end:
                return b[4]
        return None
    x = TARGETS[exit_key]
    tgt, stp = entry * (1 + side * x), entry * (1 - side * x)
    last = None
    for b in bars:
        tm = _t(b[0]).time()
        if tm > CAP:
            break
        hi, lo = b[2], b[3]
        if (side > 0 and lo <= stp) or (side < 0 and hi >= stp):
            return stp
        if (side > 0 and hi >= tgt) or (side < 0 and lo <= tgt):
            return tgt
        last = b[4]
        if tm == CAP:
            return last
    return last


def net_pct(side: int, entry: float, exit_: float) -> float:
    qty = max(int(NOTIONAL // entry), 1)
    gross = (exit_ - entry) * side * qty
    cost = charges(entry, exit_, qty, "LONG" if side > 0 else "SHORT")
    return 100 * (gross - cost) / (qty * entry)


def gross_pct(side: int, entry: float, exit_: float) -> float:
    return 100 * (exit_ - entry) * side / entry


def stock_days(sid: int, bars: dict[int, tuple], turn: dict[dt.date, float]) -> list[dict]:
    by: dict[dt.date, list[tuple]] = {}
    for k in sorted(bars):
        by.setdefault(_t(k).date(), []).append(bars[k])
    tdays = sorted(turn)
    out = []
    days = sorted(by)
    for i in range(1, len(days)):
        d, pd = days[i], days[i - 1]
        late = [b for b in by[pd] if _t(b[0]).time() >= dt.time(15, 0)]
        vol = sum(b[5] for b in late)
        if not late or vol <= 0:
            continue
        prev = sum(b[4] * b[5] for b in late) / vol
        j = bisect.bisect_left(tdays, d)
        hist = [turn[x] for x in tdays[max(0, j - 20):j]]
        eligible = len(hist) >= 15 and st.median(hist) >= MIN_TURNOVER and prev >= MIN_PRICE
        today = by[d]
        b916 = next((b for b in today if _t(b[0]).time() == dt.time(9, 16)), None)
        rest = [b for b in today if _t(b[0]).time() >= dt.time(9, 17)]
        if b916 is None or not rest or _t(rest[0][0]).time() != dt.time(9, 17):
            continue
        entry = rest[0][1]
        res = {}
        for side in (1, -1):
            for ek in EXITS:
                x = _walk(side, entry, rest, ek)
                if x is not None:
                    res[(side, ek)] = (gross_pct(side, entry, x), net_pct(side, entry, x))
        out.append({"date": d, "sid": sid, "eligible": eligible, "chg": b916[4] / prev - 1, "res": res})
    return out


def picks(records: Sequence[dict]) -> dict[dt.date, list[tuple[dict, int]]]:
    by: dict[dt.date, list[dict]] = {}
    for r in records:
        if r["eligible"]:
            by.setdefault(r["date"], []).append(r)
    out = {}
    for d, lst in by.items():
        g = sorted((r for r in lst if r["chg"] > 0), key=lambda r: -r["chg"])[:TOP]
        l_ = sorted((r for r in lst if r["chg"] < 0), key=lambda r: r["chg"])[:TOP]
        out[d] = [(r, 1) for r in g] + [(r, -1) for r in l_]
    return out


def _stats(vals_net: list[float], vals_gross: list[float]) -> dict:
    n = len(vals_net)
    if not n:
        return {"n": 0, "hit": None, "gross": None, "net": None, "pf": None, "rs_per_trade": None, "t": None}
    gw, gl = sum(v for v in vals_net if v > 0), -sum(v for v in vals_net if v < 0)
    m = st.mean(vals_net)
    sd = st.pstdev(vals_net) if n > 1 else 0
    return {"n": n, "hit": round(100 * sum(1 for v in vals_net if v > 0) / n, 1), "gross": round(st.mean(vals_gross), 3),
            "net": round(m, 3), "pf": round(gw / gl, 2) if gl else None, "rs_per_trade": round(m / 100 * NOTIONAL),
            "t": round(m / (sd / n ** 0.5), 2) if sd else None}


def evaluate(records: Sequence[dict], days_sel: set, seed: int = 9) -> dict:
    pk = picks([r for r in records if r["date"] in days_sel])
    by_day: dict[dt.date, list[dict]] = {}
    for r in records:
        if r["eligible"] and r["date"] in days_sel:
            by_day.setdefault(r["date"], []).append(r)
    rnd = random.Random(seed)
    out = {}
    for ek in EXITS:
        for mode, sgn in (("follow", 1), ("fade", -1)):
            net, gross, moves = [], [], []
            for d in sorted(pk):
                for r, g in pk[d]:
                    v = r["res"].get((g * sgn, ek))
                    if v:
                        gross.append(v[0]); net.append(v[1])
                        moves.append(((r["res"].get((1, ek)) or (0, 0))[1], (r["res"].get((-1, ek)) or (0, 0))[1]))
            out[(ek, mode)] = {"stats": _stats(net, gross), "moves": moves}
        rn, rg = [], []
        for d in sorted(by_day):
            lst = by_day[d]
            for r in rnd.sample(lst, min(2 * TOP, len(lst))):
                v = r["res"].get((rnd.choice((1, -1)), ek))
                if v:
                    rg.append(v[0]); rn.append(v[1])
        out[(ek, "random")] = {"stats": _stats(rn, rg), "moves": []}
    return out


def flip_p(moves: list[tuple[float, float]], real: float, seed: int = 4) -> float | None:
    if len(moves) < 20:
        return None
    rnd = random.Random(seed)
    hits = 0
    for _ in range(SHUFFLES):
        hits += st.mean(m[0] if rnd.random() < 0.5 else m[1] for m in moves) >= real - 1e-12
    return round(hits / SHUFFLES, 4)


def explore(records: Sequence[dict], explore_days: list[dt.date]) -> dict:
    half = len(explore_days) // 2
    halves = {"older": set(explore_days[:half]), "newer": set(explore_days[half:])}
    ev = {h: evaluate(records, s) for h, s in halves.items()}
    pooled = evaluate(records, set(explore_days))
    rows = []
    for ek in EXITS:
        rnd_net = {h: ev[h][(ek, "random")]["stats"]["net"] for h in halves}
        for mode in ("follow", "fade"):
            o, n = ev["older"][(ek, mode)]["stats"], ev["newer"][(ek, mode)]["stats"]
            pl = pooled[(ek, mode)]
            p = flip_p(pl["moves"], pl["stats"]["net"]) if pl["stats"]["n"] else None
            ok = all((x["net"] or -9) > 0 and (x["pf"] or 0) >= MIN_PF and (x["net"] or -9) > (rnd_net[h] or -9)
                     for x, h in ((o, "older"), (n, "newer"))) and p is not None and p <= MAX_P
            rows.append({"exit": ek, "mode": mode, "older": o, "newer": n, "random": rnd_net, "p": p,
                         "verdict": "VAULT CANDIDATE" if ok else "fail"})
    return {"rows": rows, "halves": {h: [str(min(s)), str(max(s)), len(s)] for h, s in halves.items()},
            "candidates": [f"{r['exit']}/{r['mode']}" for r in rows if r["verdict"] == "VAULT CANDIDATE"]}
