from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any, Sequence

from . import candles as C
from .backtest import CHARGES, BtResult, _outcome
from .candles import Candle
from .clock import at
from .nifty import MAX_RISK_PTS, SLIPPAGE_PREM, MarketCtx, NiftyRules
from .signals import FAIL, PASS, WARN, Check, Signal, grade

CHUNK_DAYS = 85


@dataclasses.dataclass(frozen=True)
class Variant:
    key: str
    label: str
    stop: str = "candle"
    atr_mult: float = 1.0
    t1_r: float = 1.5
    t2_r: float = 2.5
    cutoff: str = "13:00"
    start: str = "09:30"
    first_only: bool = False
    trend: bool = True
    bnf: bool = True
    vix: bool = True
    max_per_day: int = 2
    pm_from: str | None = None
    pm_t1_r: float = 1.0
    pm_t2_r: float = 1.5
    tf: int = 15


VARIANTS: list[Variant] = [
    Variant("live", "Live rules (till 13:00 + trend)"),
    Variant("notrend", "Without the trend filter", trend=False),
    Variant("old1430", "Old rules: till 14:30, no trend", cutoff="14:30", trend=False),
    Variant("late", "Skip the 9:45 breakout (start 10:00)", start="10:00"),
    Variant("first", "First breakout of the day only", first_only=True),
    Variant("one", "Max 1 trade a day", max_per_day=1),
    Variant("atr", "ATR stop (1x 15m ATR, 20-80 pts)", stop="atr"),
    Variant("min30", "Min stop 30 pts (not 20)", stop="min30"),
    Variant("t1_1r", "T1 at 1R, T2 at 2R", t1_r=1.0, t2_r=2.0),
    Variant("nofilt", "No Bank Nifty / VIX filters", bnf=False, vix=False),
    Variant("tf25", "25-min candles", tf=25),
    Variant("tf60", "1-hour candles (paper alerts)", tf=60),
]


class VariantRules(NiftyRules):

    def __init__(self, v: Variant) -> None:
        super().__init__()
        self.v = v
        self.prev_tail: list[Candle] = []

    def _armed(self, side: str, cur: Candle, today: Sequence[Candle]) -> bool:
        if self.v.first_only and any(s.time.date() == cur.start.date() for s in self.signals):
            return False
        return super()._armed(side, cur, today)

    def _build(self, side: str, level: float, cur: Candle, ctx: MarketCtx) -> Signal:
        v = self.v
        sig = super()._build(side, level, cur, dataclasses.replace(ctx, cutoff=v.cutoff, trend_filter=v.trend))
        long = side == "LONG"
        entry = cur.close
        raw_r = (entry - cur.low if long else cur.high - entry) + 3
        checks = [c for c in sig.checks if not ((c.name == "Bank Nifty" and not v.bnf) or (c.name == "VIX" and not v.vix))]

        if v.stop == "atr":
            prior = [x for x in ctx.nifty if x.end < cur.end]
            a = C.atr(self.prev_tail + prior) if (self.prev_tail or prior) else raw_r
            r = min(max(raw_r, v.atr_mult * a, 20.0), 80.0)
            checks = [c for c in checks if c.name != "Stop size"]
            if raw_r > 80:
                checks.append(Check("Stop size", WARN, f"breakout candle is {raw_r - 3:.0f} pts tall; stop capped at 80 pts"))
        elif v.stop == "min30":
            r = min(max(raw_r, 30.0), MAX_RISK_PTS)
        else:
            r = min(max(raw_r, 20.0), MAX_RISK_PTS)
        sgn = 1 if long else -1
        t1_r, t2_r = v.t1_r, v.t2_r
        if v.pm_from and cur.end > at(cur.start.date(), v.pm_from):
            t1_r, t2_r = v.pm_t1_r, v.pm_t2_r
        sig.stop = round(entry - sgn * r, 1)
        sig.t1 = round(entry + sgn * t1_r * r, 1)
        sig.t2 = round(entry + sgn * t2_r * r, 1)
        sig.risk_per_lot = round((0.5 * r + SLIPPAGE_PREM) * ctx.lot)

        if cur.end < at(cur.start.date(), v.start):
            checks.append(Check("Start", FAIL, f"{cur.end:%H:%M} close - before {v.start}"))
        sig.checks = checks
        sig.grade, sig.action = grade(checks)
        return sig


def fetch_range(api: Any, sid: int, start: dt.date, end: dt.date, tf: int = 15) -> list[Candle]:
    out: dict[dt.datetime, Candle] = {}
    a = start
    while a <= end:
        b = min(a + dt.timedelta(days=CHUNK_DAYS), end)
        try:
            d = api.intraday(sid, "IDX_I", "INDEX", a, b + dt.timedelta(days=1), tf)
        except Exception as exc:
            if "auth" in type(exc).__name__.lower():
                raise
            d = None
        if d:
            for c in C.from_arrays(d, tf):
                out[c.start] = c
        a = b + dt.timedelta(days=1)
    return [out[k] for k in sorted(out)]


def replay(v: Variant, n_all: list[Candle], b_all: list[Candle], v_all: list[Candle], sessions: Sequence[dt.date],
           lot: int = 65, squareoff: str = "15:15") -> list[BtResult]:
    by_day: dict[dt.date, list[Candle]] = {}
    for c in n_all:
        by_day.setdefault(c.start.date(), []).append(c)
    b_day: dict[dt.date, list[Candle]] = {}
    for c in b_all:
        b_day.setdefault(c.start.date(), []).append(c)
    v_day: dict[dt.date, list[Candle]] = {}
    for c in v_all:
        v_day.setdefault(c.start.date(), []).append(c)
    all_days = sorted(by_day)
    results: list[BtResult] = []
    for d in sessions:
        day_n = by_day.get(d, [])
        tf = day_n[0].minutes if day_n else 15
        if len(day_n) < int(0.8 * 375 / tf):
            continue
        i_d = all_days.index(d)
        prev_d = all_days[i_d - 1] if i_d else None
        prev = C.daily_hlc(by_day[prev_d]) if prev_d else None
        day_b, day_v = b_day.get(d, []), v_day.get(d, [])
        prev_b = next((b_day[x][-1].close for x in reversed(all_days[:i_d]) if b_day.get(x)), None)
        prev_v = next((v_day[x][-1].close for x in reversed(all_days[:i_d]) if v_day.get(x)), None)
        rules = VariantRules(v)
        rules.prev_tail = by_day[prev_d][-14:] if prev_d else []
        taken = 0
        for i in range(2, len(day_n) + 1):
            cand = day_n[:i]
            now = cand[-1].end
            vix_now = next((c.close for c in reversed(day_v) if c.end <= now), None)
            ctx = MarketCtx(now=now, nifty=cand, prev_hlc=prev, bnf=[c for c in day_b if c.end <= now],
                            bnf_prev_close=prev_b, vix=vix_now, vix_prev_close=prev_v, oi_stats=None, trade_chain=None,
                            lot=lot, budget=1e9, blocked_reason=None if taken < v.max_per_day else "max trades",
                            cutoff=v.cutoff, backtest=True)
            for sig in rules.on_candle_close(ctx):
                outcome, pts, t2 = _outcome(sig, day_n[i:], at(d, squareoff))
                sig.status, sig.t2_hit = outcome, t2
                r = pts / sig.risk_pts if sig.risk_pts else 0.0
                results.append(BtResult(sig, outcome, round(r, 2), round(pts, 1), round(pts * 0.5 * lot - CHARGES)))
                if sig.action == "TAKE":
                    taken += 1
    return results


def stats(rs: Sequence[BtResult], sessions: int) -> dict:
    n = len(rs)
    if not n:
        return {"n": 0, "per_day": 0.0, "win_rate": None, "avg_r": None, "total_r": 0.0, "rupees": 0, "pf": None,
                "max_dd": 0, "pos_months": "-"}
    wins = sum(1 for x in rs if x.rupees > 0)
    gw = sum(x.rupees for x in rs if x.rupees > 0)
    gl = -sum(x.rupees for x in rs if x.rupees < 0)
    eq = peak = dd = 0.0
    months: dict[str, float] = {}
    for x in sorted(rs, key=lambda x: x.signal.time):
        eq += x.rupees
        peak = max(peak, eq)
        dd = min(dd, eq - peak)
        k = f"{x.signal.time:%Y-%m}"
        months[k] = months.get(k, 0.0) + x.rupees
    return {"n": n, "per_day": round(n / max(sessions, 1), 2), "win_rate": round(100 * wins / n, 1),
            "avg_r": round(sum(x.r for x in rs) / n, 3), "total_r": round(sum(x.r for x in rs), 1),
            "rupees": round(sum(x.rupees for x in rs)), "pf": round(gw / gl, 2) if gl else None, "max_dd": round(dd),
            "pos_months": f"{sum(1 for m in months.values() if m > 0)}/{len(months)}"}


def _verdict(row: dict, base: dict, min_n: int) -> str:
    u, s = row["unseen"]["take"], row["seen"]["take"]
    bu, bs = base["unseen"]["take"], base["seen"]["take"]
    if u["n"] < min_n:
        return "too few trades to judge"
    better_u = (u["avg_r"] or -9) > (bu["avg_r"] or -9) + 0.03
    better_s = (s["avg_r"] or -9) > (bs["avg_r"] or -9) + 0.03
    profitable = (u["avg_r"] or 0) > 0.05 and (u["pf"] or 0) >= 1.15 and (s["avg_r"] or 0) > 0
    if profitable and better_u and better_s:
        return "HOLDS UP - profitable and better than live in both periods"
    if better_u and better_s:
        return "better than live in both periods, but still not profitable"
    if better_s and not better_u:
        return "only looked better on the seen period - likely luck"
    if better_u and not better_s:
        return "better on unseen only - mixed"
    return "no improvement"


def run(api: Any, nifty_id: int, bnf_id: int, vix_id: int, today: dt.date, days: int = 250, seen: int = 60,
        lot: int = 65, squareoff: str = "15:15", variants: Sequence[Variant] = VARIANTS, progress=None) -> dict:
    start = today - dt.timedelta(days=int(days * 1.47) + 10)
    end = today - dt.timedelta(days=1)
    say = progress or (lambda _m: None)
    data: dict[int, tuple[list[Candle], list[Candle], list[Candle]]] = {}
    for tf in sorted({v.tf for v in variants} | {15}, key=lambda t: (t != 15, t)):
        say(f"fetching {tf}-min Nifty history")
        n_tf = fetch_range(api, nifty_id, start, end, tf)
        say(f"fetching {tf}-min Bank Nifty / VIX history")
        data[tf] = (n_tf, fetch_range(api, bnf_id, start, end, tf), fetch_range(api, vix_id, start, end, tf))
    n_all = data[15][0]
    sessions = sorted({c.start.date() for c in n_all if c.start.date() < today})
    full = [d for d in sessions if len(C.for_day(n_all, d)) >= 20][-days:]
    seen_days, unseen_days = full[-seen:], full[:-seen]
    min_n = max(15, len(unseen_days) // 6)
    rows = []
    for i, v in enumerate(variants):
        say(f"testing {i + 1}/{len(variants)}: {v.label}")
        rs = replay(v, *data[v.tf], full, lot=lot, squareoff=squareoff)
        sset, uset = set(seen_days), set(unseen_days)
        row = {"key": v.key, "label": v.label, "tf": v.tf}
        for name, days_set, n_days in (("seen", sset, len(seen_days)), ("unseen", uset, len(unseen_days))):
            part = [x for x in rs if x.signal.time.date() in days_set]
            row[name] = {"take": stats([x for x in part if x.signal.action == "TAKE"], n_days),
                         "a_only": stats([x for x in part if x.signal.action == "TAKE" and x.signal.grade == "A"], n_days),
                         "calls": stats([x for x in part if x.signal.action == "TAKE" and x.signal.side == "LONG"], n_days),
                         "puts": stats([x for x in part if x.signal.action == "TAKE" and x.signal.side == "SHORT"], n_days)}
        rows.append(row)
    base = rows[0]
    for row in rows:
        row["verdict"] = "baseline" if row is base else _verdict(row, base, min_n)
    winners = [r["label"] for r in rows if r["verdict"].startswith("HOLDS UP")]
    return {
        "status": "done",
        "sessions": len(full),
        "seen": {"n": len(seen_days), "from": str(seen_days[0]) if seen_days else None, "to": str(seen_days[-1]) if seen_days else None},
        "unseen": {"n": len(unseen_days), "from": str(unseen_days[0]) if unseen_days else None,
                   "to": str(unseen_days[-1]) if unseen_days else None},
        "rows": rows,
        "winners": winners,
        "summary": (("Holds up out of sample: " + ", ".join(winners)) if winners else
                    "No variant is profitable on BOTH periods - keep alerts as paper trades; do not trade real money on them yet."),
        "notes": ["'Seen' = the recent period the earlier backtest showed; 'unseen' = older sessions never looked at.",
                  f"With {len(variants)} variants, one can look good by luck - trust only what holds up on the unseen period.",
                  "OI-wall/OI-flow checks are skipped (no history). Rupees = points x 0.5 delta x lot - ₹60.",
                  "Same-candle stop/target ties count as stops (pessimistic)."],
    }


def format_table(res: dict) -> str:
    lines = [f"Sessions: {res['sessions']}  |  unseen {res['unseen']['from']} .. {res['unseen']['to']} ({res['unseen']['n']})"
             f"  |  seen {res['seen']['from']} .. {res['seen']['to']} ({res['seen']['n']})", ""]
    hdr = f"{'variant':38} {'UNSEEN n':>8} {'win%':>5} {'avgR':>6} {'PF':>5} {'Rs':>8} | {'SEEN n':>6} {'win%':>5} {'avgR':>6} {'PF':>5} {'Rs':>8}  verdict"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    def f(x, w, d=None):
        if x is None:
            return f"{'-':>{w}}"
        return f"{x:>{w},.{d}f}" if d is not None else f"{x:>{w}}"
    for r in res["rows"]:
        u, s = r["unseen"]["take"], r["seen"]["take"]
        lines.append(f"{r['label'][:38]:38} {f(u['n'], 8)} {f(u['win_rate'], 5, 0)} {f(u['avg_r'], 6, 2)} {f(u['pf'], 5, 2)} "
                     f"{f(u['rupees'], 8, 0)} | {f(s['n'], 6)} {f(s['win_rate'], 5, 0)} {f(s['avg_r'], 6, 2)} {f(s['pf'], 5, 2)} "
                     f"{f(s['rupees'], 8, 0)}  {r['verdict']}")
    lines += ["", res["summary"]] + ["note: " + n for n in res["notes"]]
    return "\n".join(lines)


def main(args: list[str]) -> int:
    import json

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
    res = run(api, inst.nifty_id, inst.banknifty_id, inst.vix_id, today, days=opt("--days", 250), seen=opt("--seen", 60),
              lot=inst.nifty_lot or 65, squareoff=s.squareoff, progress=lambda m: print("..", m, flush=True))
    res["ran_at"] = dt.datetime.now(IST).strftime("%d %b %H:%M")
    print()
    print(format_table(res))
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / "research.json").write_text(json.dumps(res, default=str), encoding="utf-8")
    print(f"\nSaved to {STATE_DIR / 'research.json'} (the page shows it too).")
    return 0
