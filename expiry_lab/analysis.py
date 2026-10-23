from __future__ import annotations

import datetime as dt
import random
from typing import Sequence

from signal_desk.research import VARIANTS, replay, stats

SWITCH = dt.date(2025, 9, 1)
MIN_N, MIN_GAP, MAX_P, SHUFFLES = 15, 0.15, 0.10, 2000
GROUPS = ("expiry", "day_before", "other")


def expiry_days(trading_days: Sequence[dt.date]) -> set[dt.date]:
    tset = sorted(set(trading_days))
    have = set(tset)
    out = set()
    if not tset:
        return out
    d = tset[0] - dt.timedelta(days=tset[0].weekday())
    while d <= tset[-1]:
        sched = d + dt.timedelta(days=3 if d < SWITCH else 1)
        x = sched
        while x >= d and x not in have:
            x -= dt.timedelta(days=1)
        if x >= d and x in have:
            out.add(x)
        d += dt.timedelta(days=7)
    return out


def groups(trading_days: Sequence[dt.date]) -> dict[dt.date, str]:
    tds = sorted(set(trading_days))
    exp = expiry_days(tds)
    g = {}
    for i, d in enumerate(tds):
        if d in exp:
            g[d] = "expiry"
        elif i + 1 < len(tds) and tds[i + 1] in exp:
            g[d] = "day_before"
        else:
            g[d] = "other"
    return g


def _avg_r(rs):
    return sum(x.r for x in rs) / len(rs) if rs else None


def _gap(trades, lab):
    e = [t for t in trades if lab.get(t.signal.time.date()) == "expiry"]
    o = [t for t in trades if lab.get(t.signal.time.date()) == "other"]
    a, b = _avg_r(e), _avg_r(o)
    return None if a is None or b is None else a - b


def shuffle_p(trades, lab: dict[dt.date, str], seed: int = 5) -> float | None:
    real = _gap(trades, lab)
    if real is None:
        return None
    days = sorted({t.signal.time.date() for t in trades})
    labels = [lab[d] for d in days]
    rnd = random.Random(seed)
    hits = 0
    for _ in range(SHUFFLES):
        rnd.shuffle(labels)
        g = _gap(trades, dict(zip(days, labels)))
        hits += g is not None and abs(g) >= abs(real) - 1e-12
    return round(hits / SHUFFLES, 3)


def run(data: dict, tested: list[dt.date], older: set[dt.date], all_days: Sequence[dt.date], lot: int = 65) -> dict:
    live = VARIANTS[0]
    assert live.key == "live"
    lab = groups(all_days)
    rs = replay(live, data["nifty"], data["banknifty"], data["vix"], tested, lot=lot)
    take = [x for x in rs if x.signal.action == "TAKE"]
    out: dict = {"halves": {}, "expiry_days_tested": sum(1 for d in tested if lab.get(d) == "expiry")}
    for half in ("older", "newer"):
        inh = (lambda d: d in older) if half == "older" else (lambda d: d not in older)
        part = [x for x in take if inh(x.signal.time.date())]
        row = {"all": stats(part, sum(1 for d in tested if inh(d)))}
        for g in GROUPS:
            sel = [x for x in part if lab.get(x.signal.time.date()) == g]
            nd = sum(1 for d in tested if inh(d) and lab.get(d) == g)
            row[g] = stats(sel, nd)
            row[g + "_calls"] = stats([x for x in sel if x.signal.side == "LONG"], nd)
            row[g + "_puts"] = stats([x for x in sel if x.signal.side == "SHORT"], nd)
        row["gap"] = None if _gap(part, lab) is None else round(_gap(part, lab), 3)
        row["shuffle_p"] = shuffle_p(part, lab)
        out["halves"][half] = row
    for era, cond in (("thursday_era", lambda d: d < SWITCH), ("tuesday_era", lambda d: d >= SWITCH)):
        part = [x for x in take if cond(x.signal.time.date())]
        out[era] = {g: stats([x for x in part if lab.get(x.signal.time.date()) == g], 1) for g in ("expiry", "other")}
    out["shuffle_p_pooled"] = shuffle_p(take, lab)
    out["verdict"], out["why"] = verdict(out)
    return out


def verdict(r: dict) -> tuple[str, list[str]]:
    why, signs, ok = [], set(), True
    for half in ("older", "newer"):
        h = r["halves"][half]
        e, o, gap = h["expiry"], h["other"], h["gap"]
        why.append(f"{half}: expiry n={e['n']} avgR={e['avg_r']} PF={e['pf']} | other n={o['n']} avgR={o['avg_r']} "
                   f"PF={o['pf']} | gap={gap}")
        if gap is None or gap == 0:
            ok = False
            continue
        signs.add(gap > 0)
        if e["n"] < MIN_N or abs(gap) < MIN_GAP:
            ok = False
    p = r["shuffle_p_pooled"]
    why.append(f"two-sided day-shuffle p (pooled) = {p}  (pass needs <= {MAX_P})")
    if len(signs) != 1:
        return "FAIL - expiry day is not consistently different (direction flips between years); no adjustment", why
    better = signs.pop()
    word = "BETTER" if better else "WORSE"
    if ok and p is not None and p <= MAX_P:
        return f"PASS - expiry-day signals are consistently {word}; worth a grade adjustment", why
    return f"PROMISING BUT UNPROVEN - expiry day {word} in both years but below the pass mark; no change", why
