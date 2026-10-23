from __future__ import annotations

import bisect
import datetime as dt
from typing import Sequence

from . import idea_b as B

POOL = 10
LEGS = (("news", "follow"), ("nonews", "fade"))
INFO = (("news", "fade"), ("nonews", "follow"), ("anynews", "follow"))


def news_index(events, material_only: bool) -> dict[str, list[dt.datetime]]:
    idx: dict[str, list[dt.datetime]] = {}
    for sym, t, mat in events:
        if mat or not material_only:
            idx.setdefault(sym, []).append(t)
    for v in idx.values():
        v.sort()
    return idx


def had_news(idx: dict, sym: str, prev_day: dt.date, day: dt.date) -> bool:
    ts = idx.get(sym)
    if not ts:
        return False
    a = dt.datetime.combine(prev_day, dt.time(15, 30), B.IST)
    b = dt.datetime.combine(day, dt.time(9, 15), B.IST)
    i = bisect.bisect_right(ts, a)
    return i < len(ts) and ts[i] <= b


def classify(records: Sequence[dict], events) -> dict[dt.date, list[tuple[dict, int, dict]]]:
    mat, anyn = news_index(events, True), news_index(events, False)
    days = sorted({r["date"] for r in records})
    prev = {d: days[i - 1] for i, d in enumerate(days) if i}
    by: dict[dt.date, list[dict]] = {}
    for r in records:
        if r["eligible"] and r["date"] in prev:
            by.setdefault(r["date"], []).append(r)
    out = {}
    for d, lst in by.items():
        g = sorted((r for r in lst if r["chg"] > 0), key=lambda r: -r["chg"])[:POOL]
        l_ = sorted((r for r in lst if r["chg"] < 0), key=lambda r: r["chg"])[:POOL]
        out[d] = [(r, s, {"news": had_news(mat, r["symbol"], prev[d], d), "anynews": had_news(anyn, r["symbol"], prev[d], d)})
                  for r, s in [(x, 1) for x in g] + [(x, -1) for x in l_]]
    return out


def leg_trades(cls: dict, days_sel: set, leg: str, mode: str, ek: str):
    net, gross, moves = [], [], []
    for d in sorted(cls):
        if d not in days_sel:
            continue
        for r, g, f in cls[d]:
            hit = f["anynews"] if leg == "anynews" else (f["news"] if leg == "news" else not f["anynews"])
            if not hit:
                continue
            side = g if mode == "follow" else -g
            v = r["res"].get((side, ek))
            if v:
                gross.append(v[0]); net.append(v[1])
                moves.append(((r["res"].get((1, ek)) or (0, 0))[1], (r["res"].get((-1, ek)) or (0, 0))[1]))
    return net, gross, moves


def explore(records: Sequence[dict], events, explore_days: list[dt.date]) -> dict:
    half = len(explore_days) // 2
    halves = {"older": set(explore_days[:half]), "newer": set(explore_days[half:])}
    cls = classify(records, events)
    rnd = {h: B.evaluate(records, s) for h, s in halves.items()}
    rows = []
    for leg, mode in LEGS + INFO:
        for ek in B.EXITS:
            st = {}
            for h, s in halves.items():
                n, g, _ = leg_trades(cls, s, leg, mode, ek)
                st[h] = B._stats(n, g)
            n_all, _, mv = leg_trades(cls, set(explore_days), leg, mode, ek)
            p = B.flip_p(mv, sum(n_all) / len(n_all)) if n_all else None
            rn = {h: rnd[h][(ek, "random")]["stats"]["net"] for h in halves}
            primary = (leg, mode) in LEGS
            ok = primary and p is not None and p <= B.MAX_P and all(
                (st[h]["net"] or -9) > 0 and (st[h]["pf"] or 0) >= B.MIN_PF and (st[h]["net"] or -9) > (rn[h] or -9)
                for h in halves)
            rows.append({"exit": ek, "mode": f"{leg}-{mode}", "older": st["older"], "newer": st["newer"], "random": rn,
                         "p": p, "verdict": ("VAULT CANDIDATE" if ok else "fail") if primary else "info"})
    per_day = [sum(1 for _, _, f in v if f["news"]) for d, v in cls.items() if d in halves["older"] | halves["newer"]]
    return {"rows": rows, "halves": {h: [str(min(s)), str(max(s)), len(s)] for h, s in halves.items()},
            "news_movers_per_day": round(sum(per_day) / max(len(per_day), 1), 2),
            "candidates": [f"{r['exit']}/{r['mode']}" for r in rows if r["verdict"] == "VAULT CANDIDATE"]}
