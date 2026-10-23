from __future__ import annotations

import dataclasses
import datetime as dt
import math
import random
from typing import Callable, Sequence

from .costs import charges
from .features import Day

NOTIONAL = 100000.0


def _t(xs: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return round(m / (sd / math.sqrt(n)), 2) if sd else None


def _by_day(days: Sequence[Day]) -> dict[dt.date, list[Day]]:
    out: dict[dt.date, list[Day]] = {}
    for r in days:
        if r.eligible:
            out.setdefault(r.date, []).append(r)
    return out


def split_sessions(days: Sequence[Day], recent: int = 120) -> tuple[set, set]:
    ds = sorted({r.date for r in days})
    return set(ds[:-recent]), set(ds[-recent:])


G0_RULES: list[tuple[str, str, Callable[[Day], float], Callable[[Day], int], str]] = [
    ("chg930", "Top 3 by % change at 9:30 (your idea)", lambda r: abs(r.chg930), lambda r: (r.chg930 > 0) - (r.chg930 < 0), "930"),
    ("rv15", "Top 3 by 15-min relative volume at 9:30", lambda r: r.rv15, lambda r: (r.chg930 > 0) - (r.chg930 < 0), "930"),
    ("both930", "Top 3 by % change AND rel. volume at 9:30", None, lambda r: (r.chg930 > 0) - (r.chg930 < 0), "930"),
    ("rv5", "Top 3 by 5-min relative volume at 9:20", lambda r: r.rv5, lambda r: r.dir5, "920"),
    ("chg920", "Top 3 by % change at 9:20", lambda r: abs(r.chg920), lambda r: (r.chg920 > 0) - (r.chg920 < 0), "920"),
    ("random", "Random 3 stocks at 9:30 (benchmark)", None, lambda r: (r.chg930 > 0) - (r.chg930 < 0), "930"),
]


def _move(r: Day, d: int, when: str) -> float:
    p0 = r.p930 if when == "930" else r.p920
    return d * (r.p1500 - p0) / p0 * 100


def _stats0(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    return {"n": len(xs), "follow_avg": round(sum(xs) / len(xs), 3), "fade_avg": round(-sum(xs) / len(xs), 3),
            "median": round(s[len(s) // 2], 3), "follow_win": round(100 * sum(1 for x in xs if x > 0) / len(xs), 1),
            "abs_move": round(sum(abs(x) for x in xs) / len(xs), 3), "t": _t(xs)}


def gate0(days: Sequence[Day], top: int = 3, recent: int = 120, seed: int = 7) -> dict:
    by = _by_day(days)
    older, newer = split_sessions(days, recent)
    rnd = random.Random(seed)
    rows = []
    for key, label, rank, direction, when in G0_RULES:
        res = {"older": [], "recent": []}
        for d, lst in by.items():
            if key == "random":
                pick = rnd.sample(lst, min(top, len(lst)))
            elif key == "both930":
                a = {id(r): i for i, r in enumerate(sorted(lst, key=lambda r: -abs(r.chg930)))}
                b = {id(r): i for i, r in enumerate(sorted(lst, key=lambda r: -r.rv15))}
                pick = sorted(lst, key=lambda r: a[id(r)] + b[id(r)])[:top]
            else:
                pick = sorted(lst, key=lambda r: -rank(r))[:top]
            for r in pick:
                dd = direction(r)
                if dd:
                    res["older" if d in older else "recent"].append(_move(r, dd, when))
        row = {"key": key, "label": label, "older": _stats0(res["older"]), "recent": _stats0(res["recent"])}
        o, n = row["older"], row["recent"]
        best = max(abs(o.get("follow_avg") or 0), 0)
        side = "follow" if (o.get("follow_avg") or 0) >= 0 else "fade"
        same_side = ((o.get("follow_avg") or 0) >= 0) == ((n.get("follow_avg") or 0) >= 0)
        ok = (best >= 0.25 and abs(n.get("follow_avg") or 0) >= 0.25 and same_side and abs(o.get("t") or 0) >= 2
              and key != "random")
        row["verdict"] = (f"PASS - {side} the move, about {best:.2f}% raw per trade" if ok else
                          f"fail - raw move too small or not consistent ({side} {best:.2f}%)")
        rows.append(row)
    allr = [r for lst in by.values() for r in lst if r.chg930]
    allr.sort(key=lambda r: r.rv15)
    buckets = []
    if allr:
        cuts = [0, .5, .8, .9, .95, .99, 1.0]
        for a, b in zip(cuts, cuts[1:]):
            part = allr[int(a * len(allr)):int(b * len(allr))]
            xs = [_move(r, (r.chg930 > 0) - (r.chg930 < 0), "930") for r in part]
            if part:
                buckets.append({"bucket": f"{int(a * 100)}-{int(b * 100)}%", "rv15_from": round(part[0].rv15, 2),
                                "rv15_to": round(part[-1].rv15, 2), **_stats0(xs)})
    passed = [r["label"] for r in rows if r["verdict"].startswith("PASS")]
    return {"gate": 0, "rows": rows, "rv_buckets": buckets, "passed": passed,
            "sessions": {"older": len(older), "recent": len(newer)},
            "summary": ("Gate 0 PASSED for: " + "; ".join(passed) + ". Go to Gate 1.") if passed else
                       "Gate 0 FAILED: no selection rule gives a raw move of at least 0.25% per trade, consistently. Stop here.",
            "notes": ["Raw = before any costs. 'Follow' = trade in the direction of the move so far; 'fade' = the opposite.",
                      "Entry at the 9:20 or 9:30 close, exit at 15:00 - no stops yet (Gate 1 adds them).",
                      "Eligible = price >= Rs 100, 20-day turnover >= Rs 50 cr, daily ATR >= 1%, all from previous days only."]}


@dataclasses.dataclass(frozen=True)
class G1:
    key: str
    label: str
    rank: str = "rv5"
    top: int = 3
    min_rv: float = 2.0
    direction: str = "follow"
    exit: str = "hold"


G1_VARIANTS = [
    G1("main", "Rel. volume top 3 (RV>=2), follow, hold to 15:00"),
    G1("r2", "Same, but take profit at 2R", exit="r2"),
    G1("chg", "% change top 3, follow, hold", rank="chg920", min_rv=0),
    G1("both", "% change + rel. volume top 3, follow, hold", rank="both"),
    G1("top1", "Rel. volume top 1 only", top=1),
    G1("norv", "Rel. volume top 3, no RV minimum", min_rv=0),
    G1("fade", "Rel. volume top 3, FADE the first candle", direction="fade"),
    G1("random", "Random 3 stocks (benchmark)", rank="random", min_rv=0),
]


@dataclasses.dataclass
class Fill:
    day: dt.date
    symbol: str
    side: str
    entry: float
    gross_pct: float
    risk_pct: float
    outcome: str
    qty: int
    net: float


def _fills(v: G1, days: Sequence[Day], seed: int = 11) -> tuple[list[Fill], int]:
    by = _by_day(days)
    rnd = random.Random(seed)
    fills, no_fill = [], 0
    for d in sorted(by):
        lst = [r for r in by[d] if r.dir5 != 0 and r.rv5 >= v.min_rv]
        if v.rank == "random":
            pick = rnd.sample(lst, min(v.top, len(lst)))
        elif v.rank == "both":
            a = {id(r): i for i, r in enumerate(sorted(lst, key=lambda r: -abs(r.chg920)))}
            b = {id(r): i for i, r in enumerate(sorted(lst, key=lambda r: -r.rv5))}
            pick = sorted(lst, key=lambda r: a[id(r)] + b[id(r)])[:v.top]
        else:
            key = (lambda r: r.rv5) if v.rank == "rv5" else (lambda r: abs(r.chg920))
            pick = sorted(lst, key=lambda r: -key(r))[:v.top]
        for r in pick:
            up = r.dir5 > 0 if v.direction == "follow" else r.dir5 < 0
            tr = r.long if up else r.short
            if not tr or not tr.filled:
                no_fill += 1
                continue
            side = "LONG" if up else "SHORT"
            pct = tr.hold_pct if v.exit == "hold" else tr.r2_pct
            out = tr.hold_out if v.exit == "hold" else tr.r2_out
            qty = int(NOTIONAL // tr.entry)
            if qty < 1:
                continue
            exit_px = tr.entry * (1 + pct / 100) if side == "LONG" else tr.entry * (1 - pct / 100)
            net = pct / 100 * tr.entry * qty - charges(tr.entry, exit_px, qty, side)
            fills.append(Fill(d, r.symbol, side, tr.entry, pct, tr.risk_pct, out, qty, net))
    return fills, no_fill


def _stats1(fs: list[Fill], sessions: int) -> dict:
    n = len(fs)
    if not n:
        return {"n": 0}
    gw = sum(f.net for f in fs if f.net > 0)
    gl = -sum(f.net for f in fs if f.net < 0)
    eq = pk = dd = 0.0
    for f in sorted(fs, key=lambda f: f.day):
        eq += f.net
        pk = max(pk, eq)
        dd = min(dd, eq - pk)
    return {"n": n, "per_day": round(n / max(sessions, 1), 2), "win": round(100 * sum(1 for f in fs if f.net > 0) / n, 1),
            "gross_pct": round(sum(f.gross_pct for f in fs) / n, 3), "avg_net": round(sum(f.net for f in fs) / n),
            "pf": round(gw / gl, 2) if gl else None, "net": round(sum(f.net for f in fs)), "max_dd": round(dd),
            "t": _t([f.net for f in fs]), "avg_risk_pct": round(sum(f.risk_pct for f in fs) / n, 2)}


def gate1(days: Sequence[Day], recent: int = 120, variants: Sequence[G1] = G1_VARIANTS) -> dict:
    older, newer = split_sessions(days, recent)
    rows = []
    for v in variants:
        fs, nf = _fills(v, days)
        o = _stats1([f for f in fs if f.day in older], len(older))
        n = _stats1([f for f in fs if f.day in newer], len(newer))
        if (o.get("n") or 0) < 200:
            verdict = "too few trades to judge"
        elif (o.get("pf") or 0) >= 1.2 and (o.get("t") or 0) >= 3 and (n.get("pf") or 0) >= 1.0 and v.key != "random":
            verdict = "HOLDS UP - profitable after costs in both periods; next: 60 days of paper trading"
        elif (o.get("pf") or 0) >= 1.0 and (n.get("pf") or 0) >= 1.0:
            verdict = "profitable but not convincing (needs PF >= 1.2 and t >= 3)"
        else:
            verdict = "loses after costs"
        rows.append({"key": v.key, "label": v.label, "older": o, "recent": n, "no_fill": nf, "verdict": verdict,
                     "last_trades": [dataclasses.asdict(f) | {"day": str(f.day)} for f in fs[-30:]] if v.key == "main" else []})
    winners = [r["label"] for r in rows if r["verdict"].startswith("HOLDS UP")]
    return {"gate": 1, "rows": rows, "winners": winners, "sessions": {"older": len(older), "recent": len(newer)},
            "summary": ("Gate 1 PASSED: " + "; ".join(winners)) if winners else
                       "Gate 1: no rule set is convincingly profitable after costs - do not trade it.",
            "notes": [f"Each trade = Rs {NOTIONAL:,.0f} position (about Rs 20,000 margin at 5x). Costs: Dhan charges + 0.05% slippage per side.",
                      "Entry = stop order beyond the first 5-min candle (9:15-9:20) between 9:20 and 10:30; stop at its other side "
                      "(stops under 0.5% of price are widened to 0.5%; over 2% are skipped). Exit at 15:00 unless stopped (or 2R).",
                      "If a stop and target fall in the same 5-min bar, the stop is assumed. 'no_fill' = picked but never triggered.",
                      "Pass needs >= 200 older trades, PF >= 1.2 and t >= 3 (several variants tested), and PF >= 1 on the recent period."]}


def fmt0(res: dict) -> str:
    L = [f"GATE 0 - raw move from 9:20/9:30 to 15:00, before costs (older {res['sessions']['older']} / recent {res['sessions']['recent']} sessions)", ""]
    L.append(f"{'rule':44} | {'OLDER n':>7} {'follow%':>8} {'win%':>5} {'t':>6} | {'RECENT n':>8} {'follow%':>8} {'t':>6} | verdict")
    for r in res["rows"]:
        o, n = r["older"], r["recent"]
        L.append(f"{r['label'][:44]:44} | {o.get('n', 0):>7} {o.get('follow_avg', 0):>8.3f} {o.get('follow_win', 0):>5.1f} {str(o.get('t')):>6} | "
                 f"{n.get('n', 0):>8} {n.get('follow_avg', 0):>8.3f} {str(n.get('t')):>6} | {r['verdict']}")
    L += ["", "Relative-volume buckets (all eligible stock-days, follow the 9:30 direction):"]
    for b in res["rv_buckets"]:
        L.append(f"  RV15 {b['bucket']:>8} ({b['rv15_from']}-{b['rv15_to']}x): n {b['n']:>6}  follow {b['follow_avg']:>7.3f}%  "
                 f"avg abs move {b['abs_move']:.3f}%  t {b['t']}")
    L += ["", res["summary"]] + ["note: " + x for x in res["notes"]]
    return "\n".join(L)


def fmt1(res: dict) -> str:
    L = [f"GATE 1 - full rules with costs (older {res['sessions']['older']} / recent {res['sessions']['recent']} sessions)", ""]
    L.append(f"{'variant':44} | {'OLDER n':>7} {'/day':>5} {'win%':>5} {'gross%':>7} {'avg net':>8} {'PF':>5} {'net Rs':>9} {'t':>6} | "
             f"{'RECENT n':>8} {'PF':>5} {'net Rs':>9} | verdict")
    for r in res["rows"]:
        o, n = r["older"], r["recent"]
        g = lambda s, k, d=None: s.get(k) if s.get(k) is not None else "-"
        L.append(f"{r['label'][:44]:44} | {o.get('n', 0):>7} {g(o, 'per_day'):>5} {g(o, 'win'):>5} {g(o, 'gross_pct'):>7} {g(o, 'avg_net'):>8} "
                 f"{g(o, 'pf'):>5} {g(o, 'net'):>9} {g(o, 't'):>6} | {n.get('n', 0):>8} {g(n, 'pf'):>5} {g(n, 'net'):>9} | {r['verdict']}")
    L += ["", res["summary"]] + ["note: " + x for x in res["notes"]]
    return "\n".join(L)
