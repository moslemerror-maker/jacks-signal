from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from . import candles as C
from .candles import Candle
from .clock import at
from .config import CACHE_DIR

CAPITAL = 30000.0
SLIPPAGE = 0.0005
CHUNK_DAYS = 85
CACHE = CACHE_DIR / "stocks15"


def charges(entry: float, exit_: float, qty: int, side: str) -> float:
    buy, sell = (entry, exit_) if side == "LONG" else (exit_, entry)
    bv, sv = buy * qty, sell * qty
    brokerage = min(20.0, 0.0003 * bv) + min(20.0, 0.0003 * sv)
    stt = 0.00025 * sv
    exch = 0.0000297 * (bv + sv)
    sebi = 0.000001 * (bv + sv)
    stamp = 0.00003 * bv
    gst = 0.18 * (brokerage + exch + sebi)
    slip = SLIPPAGE * (bv + sv)
    return brokerage + stt + exch + sebi + stamp + gst + slip


def _cache_path(sid: int) -> Path:
    return CACHE / f"{sid}.json"


def load_stock(api: Any, sid: int, start: dt.date, end: dt.date, use_cache: bool = True) -> list[Candle]:
    rows: dict[int, list] = {}
    p = _cache_path(sid)
    if use_cache and p.exists():
        try:
            for r in json.loads(p.read_text()):
                rows[int(r[0])] = r
        except (OSError, ValueError):
            rows = {}
    have = sorted(rows)
    first_have = dt.datetime.fromtimestamp(have[0]).date() if have else None
    last_have = dt.datetime.fromtimestamp(have[-1]).date() if have else None
    ranges = []
    if not have or first_have > start + dt.timedelta(days=5):
        ranges.append((start, (first_have - dt.timedelta(days=1)) if have else end))
    if have and last_have < end:
        ranges.append((last_have, end))
    for a0, b0 in ranges:
        a = a0
        while a <= b0:
            b = min(a + dt.timedelta(days=CHUNK_DAYS), b0)
            try:
                d = api.intraday(sid, "NSE_EQ", "EQUITY", a, b + dt.timedelta(days=1), 15)
            except Exception as exc:
                if "auth" in type(exc).__name__.lower():
                    raise
                d = None
            if d and d.get("close"):
                vols = d.get("volume") or [0] * len(d["close"])
                for i in range(len(d["close"])):
                    try:
                        ts = int(d["timestamp"][i])
                        rows[ts] = [ts, float(d["open"][i]), float(d["high"][i]), float(d["low"][i]),
                                    float(d["close"][i]), float(vols[i] or 0)]
                    except (TypeError, ValueError, IndexError):
                        continue
            a = b + dt.timedelta(days=1)
    if use_cache and rows:
        CACHE.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps([rows[k] for k in sorted(rows)], separators=(",", ":")))
    arr = [rows[k] for k in sorted(rows)]
    out = C.from_arrays({"timestamp": [r[0] for r in arr], "open": [r[1] for r in arr], "high": [r[2] for r in arr],
                         "low": [r[3] for r in arr], "close": [r[4] for r in arr], "volume": [r[5] for r in arr]}, 15)
    return [c for c in out if start <= c.start.date() <= end]


@dataclasses.dataclass
class DayStat:
    o: float
    h: float
    l: float
    c: float
    turnover: float


def daily_stats(cs: Sequence[Candle]) -> dict[dt.date, DayStat]:
    out: dict[dt.date, DayStat] = {}
    by: dict[dt.date, list[Candle]] = {}
    for c in cs:
        by.setdefault(c.start.date(), []).append(c)
    for d, xs in by.items():
        out[d] = DayStat(xs[0].open, max(x.high for x in xs), min(x.low for x in xs), xs[-1].close,
                         sum(x.close * x.volume for x in xs))
    return out


def premarket_score(days: Sequence[DayStat]) -> tuple[float, str, dict] | None:
    if len(days) < 21:
        return None
    p = days[-1]
    trs = [max(d.h - d.l, abs(d.h - q.c), abs(d.l - q.c)) for q, d in zip(days[-15:-1], days[-14:])]
    atr = sum(trs) / len(trs)
    turnover = sum(d.turnover for d in days[-20:]) / 20
    info = {"atr_pct": atr / p.c * 100, "turnover_cr": turnover / 1e7, "price": p.c}
    if p.c < 100 or turnover < 50e7 or atr / p.c < 0.015:
        return None
    hi20, lo20 = max(d.h for d in days[-20:]), min(d.l for d in days[-20:])
    rng = max(p.h - p.l, 1e-9)
    pos = (p.c - p.l) / rng
    nr7 = (p.h - p.l) <= min(d.h - d.l for d in days[-7:]) + 1e-9
    inside = p.h <= days[-2].h and p.l >= days[-2].l
    vr = days[-1].turnover / turnover if turnover else 0
    if hi20 - p.c <= 0.5 * atr and pos >= 0.7:
        return 2 + min(vr, 3) + atr / p.c * 20, "LONG", info
    if p.c - lo20 <= 0.5 * atr and pos <= 0.3:
        return 2 + min(vr, 3) + atr / p.c * 20, "SHORT", info
    if nr7 or inside:
        return 1.5 + min(vr, 2) / 2 + atr / p.c * 20, "BOTH", info
    return None


@dataclasses.dataclass(slots=True)
class Event:
    day: dt.date
    time: dt.datetime
    symbol: str
    side: str
    entry: float
    rvol: float
    rs: float
    vwap_ok: bool
    pos: float
    gate: bool
    risk: float
    out_15: tuple[str, float]
    out_10: tuple[str, float]
    score: float = 0.0


def _exit(side: str, entry: float, risk: float, k: float, after: Sequence[Candle], sq: dt.datetime) -> tuple[str, float]:
    long = side == "LONG"
    stop = entry - risk if long else entry + risk
    tgt = entry + k * risk if long else entry - k * risk
    last = entry
    for c in after:
        if c.start >= sq:
            break
        last = c.close
        if long:
            if c.low <= stop:
                return "SL", stop
            if c.high >= tgt:
                return "T1", tgt
        else:
            if c.high >= stop:
                return "SL", stop
            if c.low <= tgt:
                return "T1", tgt
    return "timeout", last


def stock_events(symbol: str, cs: Sequence[Candle], nifty_by_day: dict[dt.date, list[Candle]],
                 nifty_prev_close: dict[dt.date, float], days: Iterable[dt.date], start: str = "09:45",
                 end: str = "13:00", squareoff: str = "15:15") -> list[Event]:
    by: dict[dt.date, list[Candle]] = {}
    for c in cs:
        by.setdefault(c.start.date(), []).append(c)
    all_days = sorted(by)
    idx = {d: i for i, d in enumerate(all_days)}
    out: list[Event] = []
    for d in days:
        today = by.get(d)
        if not today or len(today) < 20 or d not in idx or idx[d] == 0:
            continue
        prev_days = all_days[max(0, idx[d] - 10):idx[d]]
        prev_close = by[all_days[idx[d] - 1]][-1].close
        n_today = nifty_by_day.get(d, [])
        npc = nifty_prev_close.get(d)
        slot_vol: dict[dt.time, list[float]] = {}
        for pd_ in prev_days:
            for c in by[pd_]:
                slot_vol.setdefault(c.start.time(), []).append(c.volume)
        t0, t1, sq = at(d, start), at(d, end), at(d, squareoff)
        pv = vol = 0.0
        for i, cur in enumerate(today):
            pv += (cur.high + cur.low + cur.close) / 3 * cur.volume
            vol += cur.volume
            if i == 0 or cur.end < t0 or cur.end > t1:
                continue
            prior = today[:i]
            hi, lo = max(x.high for x in prior), min(x.low for x in prior)
            side = "LONG" if cur.close > hi else "SHORT" if cur.close < lo else None
            if side is None:
                continue
            long = side == "LONG"
            base = slot_vol.get(cur.start.time()) or []
            rvol = (cur.volume / (sum(base) / len(base))) if base and sum(base) > 0 else 0.0
            ncl = next((x.close for x in reversed(n_today) if x.end <= cur.end), None)
            schg = (cur.close - prev_close) / prev_close * 100
            nchg = ((ncl - npc) / npc * 100) if ncl and npc else 0.0
            rs = (schg - nchg) if long else (nchg - schg)
            vw = pv / vol if vol else cur.close
            vwap_ok = cur.close > vw if long else cur.close < vw
            rng = max(cur.high - cur.low, 1e-9)
            pos = (cur.close - cur.low) / rng if long else (cur.high - cur.close) / rng
            nopen = n_today[0].open if n_today else None
            gate = bool(ncl and nopen and npc and ((ncl > nopen and ncl > npc) if long else (ncl < nopen and ncl < npc)))
            if pos < 0.6 or not vwap_ok or rvol < 1.0:
                continue
            raw = (cur.close - cur.low if long else cur.high - cur.close) + 0.0005 * cur.close
            risk = min(max(raw, 0.004 * cur.close), 0.012 * cur.close)
            after = today[i + 1:]
            out.append(Event(d, cur.end, symbol, side, cur.close, rvol, rs, vwap_ok, pos, gate, risk,
                             _exit(side, cur.close, risk, 1.5, after, sq), _exit(side, cur.close, risk, 1.0, after, sq)))
    return out


@dataclasses.dataclass(frozen=True)
class SVariant:
    key: str
    label: str
    watch_only: bool = True
    top_n: int = 10
    start: str = "10:00"
    end: str = "13:00"
    rvol: float = 2.0
    rs: float = 1.0
    vwap: bool = True
    pos: float = 0.6
    gate: bool = True
    k: float = 1.5
    per_day: int = 2
    per_candle: int = 1


SVARIANTS: list[SVariant] = [
    SVariant("funnel", "Funnel (all filters)"),
    SVariant("all", "All liquid F&O stocks (no pre-market list)", watch_only=False),
    SVariant("early", "Start 9:45 (not 10:00)", start="09:45"),
    SVariant("rvol15", "Volume 1.5x (not 2x)", rvol=1.5),
    SVariant("nogate", "No Nifty market gate", gate=False),
    SVariant("nors", "No relative-strength filter", rs=-99.0),
    SVariant("t1r", "Exit at 1R (not 1.5R)", k=1.0),
    SVariant("one", "Max 1 stock trade a day", per_day=1),
    SVariant("monday", "Monday's old rules (for reference)", watch_only=False, start="09:45", rvol=1.0, rs=-99.0,
             vwap=True, pos=0.6, gate=False, per_day=99, per_candle=99),
]


@dataclasses.dataclass
class STrade:
    ev: Event
    outcome: str
    exit: float
    qty: int
    gross: float
    cost: float

    @property
    def net(self) -> float:
        return self.gross - self.cost

    @property
    def r(self) -> float:
        return self.net / (self.ev.risk * self.qty) if self.qty and self.ev.risk else 0.0


def pick(v: SVariant, events: Sequence[Event], watch: dict[dt.date, dict[str, str]], capital: float = CAPITAL) -> list[STrade]:
    t_start = dt.time(*map(int, v.start.split(":")))
    t_end = dt.time(*map(int, v.end.split(":")))
    ok = []
    for e in events:
        tt = e.time.time()
        if tt < t_start or tt > t_end:
            continue
        if v.watch_only:
            bias = watch.get(e.day, {}).get(e.symbol)
            if bias is None or (bias != "BOTH" and bias != e.side):
                continue
        if e.rvol < v.rvol or e.rs < v.rs or (v.vwap and not e.vwap_ok) or e.pos < v.pos or (v.gate and not e.gate):
            continue
        e.score = min(e.rvol, 5) + max(min(e.rs, 4), -4) + e.pos
        ok.append(e)
    ok.sort(key=lambda e: (e.time, -e.score))
    trades: list[STrade] = []
    per_day: dict[dt.date, int] = {}
    per_slot: dict[dt.datetime, int] = {}
    done: set[tuple[dt.date, str]] = set()
    for e in ok:
        if (e.day, e.symbol) in done or per_day.get(e.day, 0) >= v.per_day or per_slot.get(e.time, 0) >= v.per_candle:
            continue
        outcome, px = e.out_15 if v.k >= 1.5 else e.out_10
        qty = int(capital // e.entry)
        if qty < 1:
            continue
        gross = (px - e.entry) * qty if e.side == "LONG" else (e.entry - px) * qty
        trades.append(STrade(e, outcome, px, qty, gross, charges(e.entry, px, qty, e.side)))
        done.add((e.day, e.symbol))
        per_day[e.day] = per_day.get(e.day, 0) + 1
        per_slot[e.time] = per_slot.get(e.time, 0) + 1
    return trades


def sstats(ts: Sequence[STrade], sessions: int) -> dict:
    n = len(ts)
    if not n:
        return {"n": 0, "per_day": 0.0, "days_traded": 0, "win_rate": None, "avg_r": None, "pf": None, "rupees": 0,
                "avg_net": None, "costs": 0, "max_dd": 0}
    wins = [t for t in ts if t.net > 0]
    gw, gl = sum(t.net for t in ts if t.net > 0), -sum(t.net for t in ts if t.net < 0)
    eq = peak = dd = 0.0
    for t in sorted(ts, key=lambda t: t.ev.time):
        eq += t.net
        peak = max(peak, eq)
        dd = min(dd, eq - peak)
    return {"n": n, "per_day": round(n / max(sessions, 1), 2), "days_traded": len({t.ev.day for t in ts}),
            "win_rate": round(100 * len(wins) / n, 1), "avg_r": round(sum(t.r for t in ts) / n, 3),
            "pf": round(gw / gl, 2) if gl else None, "rupees": round(sum(t.net for t in ts)),
            "avg_net": round(sum(t.net for t in ts) / n), "costs": round(sum(t.cost for t in ts)), "max_dd": round(dd),
            "avg_gross": round(sum(t.gross for t in ts) / n), "avg_cost": round(sum(t.cost for t in ts) / n),
            "gross_pct": round(100 * sum(t.gross / (t.ev.entry * t.qty) for t in ts) / n, 3)}


def _verdict(row: dict, min_n: int) -> str:
    u, s = row["unseen"], row["seen"]
    if u["n"] < min_n:
        return "too few trades to judge"
    good_u = (u["pf"] or 0) >= 1.2 and (u["avg_net"] or 0) > 0
    good_s = (s["pf"] or 0) >= 1.0 and (s["rupees"] or 0) > 0
    if good_u and good_s:
        return "HOLDS UP - profitable after costs in both periods"
    if good_u:
        return "profitable on unseen only - not enough"
    if good_s:
        return "profitable on recent only - likely luck"
    return "loses after costs"


def run(api: Any, stocks: Sequence[Any], nifty_id: int, today: dt.date, days: int = 250, seen: int = 60,
        variants: Sequence[SVariant] = SVARIANTS, progress: Callable[[str], None] | None = None,
        use_cache: bool = True, capital: float = CAPITAL) -> dict:
    from .research import fetch_range
    say = progress or (lambda _m: None)
    start = today - dt.timedelta(days=int(days * 1.47) + 40)
    end = today - dt.timedelta(days=1)
    say("fetching Nifty history")
    nifty = fetch_range(api, nifty_id, start, end)
    n_by: dict[dt.date, list[Candle]] = {}
    for c in nifty:
        n_by.setdefault(c.start.date(), []).append(c)
    n_days = sorted(d for d in n_by if len(n_by[d]) >= 20)
    n_prev = {d: n_by[n_days[i - 1]][-1].close for i, d in enumerate(n_days) if i}
    sessions = n_days[-days:]
    seen_days, unseen_days = set(sessions[-seen:]), set(sessions[:-seen])

    events: list[Event] = []
    scores: dict[dt.date, list[tuple[float, str, str]]] = {}
    for k, st in enumerate(stocks):
        if k % 10 == 0:
            say(f"stocks {k}/{len(stocks)} ({st.symbol})")
        cs = load_stock(api, st.security_id, start, end, use_cache=use_cache)
        if len(cs) < 200:
            continue
        ds = daily_stats(cs)
        sd = sorted(ds)
        for i, d in enumerate(sd):
            if d not in seen_days and d not in unseen_days:
                continue
            sc = premarket_score([ds[x] for x in sd[max(0, i - 40):i]])
            if sc:
                scores.setdefault(d, []).append((sc[0], st.symbol, sc[1]))
        events.extend(stock_events(st.symbol, cs, n_by, n_prev, sessions))
    watch: dict[dt.date, dict[str, str]] = {}
    for d, lst in scores.items():
        watch[d] = {sym: bias for _, sym, bias in sorted(lst, reverse=True)[:10]}

    say("testing variants")
    min_n = max(15, len(unseen_days) // 8)
    rows = []
    for v in variants:
        ts = pick(v, events, watch, capital)
        row = {"key": v.key, "label": v.label,
               "unseen": sstats([t for t in ts if t.ev.day in unseen_days], len(unseen_days)),
               "seen": sstats([t for t in ts if t.ev.day in seen_days], len(seen_days))}
        row["verdict"] = _verdict(row, min_n)
        if v.key == "funnel":
            row["trades"] = [{"date": str(t.ev.day), "time": f"{t.ev.time:%H:%M}", "symbol": t.ev.symbol,
                              "side": t.ev.side, "entry": round(t.ev.entry, 2), "qty": t.qty, "outcome": t.outcome,
                              "net": round(t.net), "rvol": round(t.ev.rvol, 1), "rs": round(t.ev.rs, 2)}
                             for t in sorted(ts, key=lambda t: t.ev.time)[-80:]]
        rows.append(row)
    winners = [r["label"] for r in rows if r["verdict"].startswith("HOLDS UP") and r["key"] != "monday"]
    return {
        "status": "done", "sessions": len(sessions), "stocks": len(stocks), "events": len(events),
        "seen": {"n": len(seen_days), "from": str(min(seen_days)) if seen_days else None, "to": str(max(seen_days)) if seen_days else None},
        "unseen": {"n": len(unseen_days), "from": str(min(unseen_days)) if unseen_days else None,
                   "to": str(max(unseen_days)) if unseen_days else None},
        "rows": rows, "winners": winners,
        "summary": (("Holds up: " + ", ".join(winners) + ". Next step: 2-4 weeks of PAPER stock alerts.") if winners else
                    "No stock rule set makes money after costs in both periods - keep stock alerts OFF."),
        "capital": capital,
        "notes": [f"Position = Rs {capital:,.0f} per trade, exit all at T1 / stop / 15:15. Costs + 0.05% slippage per side included.",
                  "Pre-market list and all filters use only data available at that moment (no look-ahead).",
                  "Same-candle stop/target ties count as stops. Stock news/results are not known to the test.",
                  "Today's F&O list is used for the whole year (slight survivorship bias - real results a bit worse)."],
    }


def format_table(res: dict) -> str:
    lines = [f"Stocks {res['stocks']} | sessions {res['sessions']} | unseen {res['unseen']['from']}..{res['unseen']['to']} "
             f"| seen {res['seen']['from']}..{res['seen']['to']}", ""]
    hdr = f"{'variant':44} {'UNSEEN n':>8} {'/day':>5} {'win%':>5} {'PF':>5} {'net Rs':>8} | {'SEEN n':>6} {'win%':>5} {'PF':>5} {'net Rs':>8}  verdict"
    lines += [hdr, "-" * len(hdr)]

    def f(x, w, d=0):
        return f"{'-':>{w}}" if x is None else f"{x:>{w},.{d}f}"
    for r in res["rows"]:
        u, s = r["unseen"], r["seen"]
        lines.append(f"{r['label'][:44]:44} {f(u['n'], 8)} {f(u['per_day'], 5, 2)} {f(u['win_rate'], 5)} {f(u['pf'], 5, 2)} "
                     f"{f(u['rupees'], 8)} | {f(s['n'], 6)} {f(s['win_rate'], 5)} {f(s['pf'], 5, 2)} {f(s['rupees'], 8)}  {r['verdict']}")
    lines += ["", res["summary"]] + ["note: " + n for n in res["notes"]]
    return "\n".join(lines)


def main(args: list[str]) -> int:
    from . import instruments as I
    from .clock import IST
    from .config import STATE_DIR, Settings
    from .dhan_api import DhanApi

    def opt(name: str, default: int) -> int:
        return int(args[args.index(name) + 1]) if name in args and args.index(name) + 1 < len(args) else default
    s = Settings.load()
    today = dt.datetime.now(IST).date()
    inst = I.load(today)
    api = DhanApi(s.dhan_client_id, s.dhan_access_token)
    res = run(api, inst.stocks, inst.nifty_id, today, days=opt("--days", 250), seen=opt("--seen", 60),
              progress=lambda m: print("..", m, flush=True))
    res["ran_at"] = dt.datetime.now(IST).strftime("%d %b %H:%M")
    print()
    print(format_table(res))
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / "stock_research.json").write_text(json.dumps(res, default=str), encoding="utf-8")
    print(f"\nSaved to {STATE_DIR / 'stock_research.json'} (the page shows it too).")
    return 0
