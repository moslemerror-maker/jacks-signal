from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any

from . import candles as C
from .clock import at
from .nifty import MarketCtx, NiftyRules
from .signals import Signal

CHARGES = 60.0


@dataclasses.dataclass
class BtResult:
    signal: Signal
    outcome: str
    r: float
    points: float
    rupees: float


def _outcome(sig: Signal, after: list[C.Candle], squareoff: dt.datetime) -> tuple[str, float, bool]:
    long = sig.side == "LONG"
    sgn = 1 if long else -1
    hit_t1 = False
    last = sig.entry
    for c in after:
        if c.start >= squareoff:
            break
        last = c.close
        hi_fav = (c.high - sig.entry) * sgn if long else (sig.entry - c.low)
        lo_adv = (c.low - sig.entry) if long else (sig.entry - c.high)
        if not hit_t1:
            if lo_adv <= -(sig.risk_pts) + 1e-9:
                return "SL", -sig.risk_pts, False
            if hi_fav >= abs(sig.t1 - sig.entry):
                hit_t1 = True
                if hi_fav >= abs(sig.t2 - sig.entry):
                    return "T1", 0.5 * abs(sig.t1 - sig.entry) + 0.5 * abs(sig.t2 - sig.entry), True
                continue
        else:
            if hi_fav >= abs(sig.t2 - sig.entry):
                return "T1", 0.5 * abs(sig.t1 - sig.entry) + 0.5 * abs(sig.t2 - sig.entry), True
            if lo_adv <= 0:
                return "T1", 0.5 * abs(sig.t1 - sig.entry), False
    move = (last - sig.entry) * sgn
    if hit_t1:
        return "T1", 0.5 * abs(sig.t1 - sig.entry) + 0.5 * move, False
    return "timeout", move, False


def run(api: Any, nifty_id: int, bnf_id: int, vix_id: int, today: dt.date, days: int = 60, lot: int = 65,
        budget: float = 1500.0, cutoff: str = "14:30", squareoff: str = "15:15", max_per_day: int = 2,
        trend: bool = False) -> dict:
    start = today - dt.timedelta(days=min(int(days * 1.5) + 5, 88))

    def fetch(sid):
        d = api.intraday(sid, "IDX_I", "INDEX", start, today + dt.timedelta(days=1), 15)
        return C.from_arrays(d, 15)
    n_all, b_all, v_all = fetch(nifty_id), fetch(bnf_id), fetch(vix_id)
    sessions = sorted({c.start.date() for c in n_all if c.start.date() < today})[-days:]
    results: list[BtResult] = []
    for d in sessions:
        day_n = C.for_day(n_all, d)
        if len(day_n) < 20:
            continue
        prior = [x for x in sorted({c.start.date() for c in n_all}) if x < d]
        prev = C.daily_hlc(C.for_day(n_all, prior[-1])) if prior else None
        day_b, day_v = C.for_day(b_all, d), C.for_day(v_all, d)
        prev_b = [c for c in b_all if c.start.date() < d]
        prev_v = [c for c in v_all if c.start.date() < d]
        rules = NiftyRules()
        taken = 0
        for i in range(2, len(day_n) + 1):
            cand = day_n[:i]
            now = cand[-1].end
            vix_now = next((c.close for c in reversed(day_v) if c.end <= now), None)
            ctx = MarketCtx(now=now, nifty=cand, prev_hlc=prev, bnf=[c for c in day_b if c.end <= now],
                            bnf_prev_close=prev_b[-1].close if prev_b else None, vix=vix_now,
                            vix_prev_close=prev_v[-1].close if prev_v else None, oi_stats=None, trade_chain=None,
                            lot=lot, budget=budget, blocked_reason=None if taken < max_per_day else "max trades",
                            cutoff=cutoff, backtest=True, trend_filter=trend)
            for sig in rules.on_candle_close(ctx):
                outcome, pts, t2 = _outcome(sig, day_n[i:], at(d, squareoff))
                sig.status, sig.t2_hit = outcome, t2
                r = pts / sig.risk_pts if sig.risk_pts else 0.0
                rupees = pts * 0.5 * lot - CHARGES
                results.append(BtResult(sig, outcome, round(r, 2), round(pts, 1), round(rupees)))
                if sig.action == "TAKE":
                    taken += 1
    return summarize(results, sessions)


def _stats(rs: list[BtResult]) -> dict:
    if not rs:
        return {"n": 0, "win_rate": None, "avg_r": None, "total_r": 0.0, "rupees": 0, "t1": 0, "sl": 0, "timeout": 0,
                "profit_factor": None}
    wins = [x for x in rs if x.rupees > 0]
    gross_win = sum(x.rupees for x in rs if x.rupees > 0)
    gross_loss = -sum(x.rupees for x in rs if x.rupees < 0)
    return {
        "n": len(rs), "win_rate": round(100 * len(wins) / len(rs), 1), "avg_r": round(sum(x.r for x in rs) / len(rs), 2),
        "total_r": round(sum(x.r for x in rs), 1), "rupees": round(sum(x.rupees for x in rs)),
        "t1": sum(1 for x in rs if x.outcome == "T1"), "sl": sum(1 for x in rs if x.outcome == "SL"),
        "timeout": sum(1 for x in rs if x.outcome == "timeout"),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else None,
    }


def summarize(results: list[BtResult], sessions: list[dt.date]) -> dict:
    take = [x for x in results if x.signal.action == "TAKE"]
    by_grade = {g: _stats([x for x in results if x.signal.grade == g]) for g in ("A", "B", "C")}
    eq, peak, dd = 0.0, 0.0, 0.0
    for x in take:
        eq += x.rupees
        peak = max(peak, eq)
        dd = min(dd, eq - peak)
    return {
        "sessions": len(sessions), "from": str(sessions[0]) if sessions else None, "to": str(sessions[-1]) if sessions else None,
        "take": _stats(take), "all": _stats(results), "by_grade": by_grade, "max_drawdown": round(dd),
        "trades": [{"date": f"{x.signal.time:%Y-%m-%d}", "time": f"{x.signal.time:%H:%M}", "side": x.signal.side,
                    "grade": x.signal.grade, "action": x.signal.action, "entry": round(x.signal.entry, 1),
                    "stop": x.signal.stop, "t1": x.signal.t1, "outcome": x.outcome, "r": x.r, "points": x.points,
                    "rupees": x.rupees, "why_skip": "; ".join(c.detail for c in x.signal.checks if c.status == "fail")}
                   for x in results],
        "notes": ["Past option-chain OI is not available, so OI-wall checks are skipped in this test.",
                  "Rupees = index points x 0.5 delta x lot - ₹60; real premiums also lose time value.",
                  "Same-candle stop/target ties are counted as stops (pessimistic)."],
    }
