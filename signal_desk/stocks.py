from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Sequence

from .candles import Candle, atr, slot_avg_volume, vwap
from .clock import at
from .signals import FAIL, PASS, WARN, Check, Signal, grade


@dataclasses.dataclass
class DailyProfile:
    symbol: str
    security_id: int
    prev_high: float
    prev_low: float
    prev_close: float
    atr14: float
    avg_vol20: float
    prev_vol: float
    high20: float
    low20: float
    nr7: bool
    inside: bool
    setup: str = ""
    bias: str = ""
    score: float = 0.0

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


def daily_profile(symbol: str, sid: int, days: Sequence[Candle]) -> DailyProfile | None:
    if len(days) < 8:
        return None
    p = days[-1]
    last20 = days[-20:]
    ranges7 = [d.high - d.low for d in days[-7:]]
    prof = DailyProfile(
        symbol=symbol, security_id=sid, prev_high=p.high, prev_low=p.low, prev_close=p.close,
        atr14=atr(days[-15:], 14), avg_vol20=sum(d.volume for d in last20[:-1]) / max(len(last20) - 1, 1),
        prev_vol=p.volume, high20=max(d.high for d in last20), low20=min(d.low for d in last20),
        nr7=(p.high - p.low) <= min(ranges7) + 1e-9, inside=p.high <= days[-2].high and p.low >= days[-2].low,
    )
    rng = max(p.high - p.low, 1e-9)
    pos = (p.close - p.low) / rng
    vr = (p.volume / prof.avg_vol20) if prof.avg_vol20 else 0
    near_hi = (prof.high20 - p.close) <= 0.5 * prof.atr14
    near_lo = (p.close - prof.low20) <= 0.5 * prof.atr14
    if near_hi and pos >= 0.7:
        prof.setup, prof.bias, prof.score = "near 20-day high, strong close", "LONG", 2 + min(vr, 3)
    elif near_lo and pos <= 0.3:
        prof.setup, prof.bias, prof.score = "near 20-day low, weak close", "SHORT", 2 + min(vr, 3)
    elif prof.nr7 or prof.inside:
        prof.setup, prof.bias, prof.score = ("NR7" if prof.nr7 else "inside day") + " - range expansion likely", "BOTH", 1.5 + min(vr, 2) / 2
    else:
        prof.score = 0
    return prof


def watchlist(profiles: Sequence[DailyProfile], n: int = 10) -> list[DailyProfile]:
    return sorted([p for p in profiles if p.score > 0], key=lambda p: p.score, reverse=True)[:n]


def shortlist(quotes: dict[int, dict], profiles: dict[int, DailyProfile], n: int = 25,
              near_pct: float = 0.3, min_move_pct: float = 0.7) -> list[int]:
    scored = []
    for sid, q in quotes.items():
        ltp, hi, lo = q.get("ltp"), q.get("high"), q.get("low")
        prof = profiles.get(sid)
        if not ltp or not hi or not lo or not prof or not prof.prev_close:
            continue
        chg = (ltp - prof.prev_close) / prof.prev_close * 100
        near_hi = (hi - ltp) / ltp * 100 <= near_pct
        near_lo = (ltp - lo) / ltp * 100 <= near_pct
        crossed = ltp > prof.prev_high or ltp < prof.prev_low
        if (near_hi and chg >= min_move_pct) or (near_lo and chg <= -min_move_pct) or (crossed and prof.score > 0):
            scored.append((abs(chg) + (1.0 if prof.score > 0 else 0), sid))
    return [sid for _, sid in sorted(scored, reverse=True)[:n]]


class StockRules:
    def __init__(self) -> None:
        self.signals: list[Signal] = []

    def evaluate(self, symbol: str, today: list[Candle], history: list[Candle], prof: DailyProfile | None,
                 nifty_chg_pct: float | None, now: dt.datetime, budget: float, blocked: str | None,
                 cutoff: str = "14:30") -> Signal | None:
        if len(today) < 2:
            return None
        cur, prior = today[-1], today[:-1]
        hi, lo = max(c.high for c in prior), min(c.low for c in prior)
        side = "LONG" if cur.close > hi else "SHORT" if cur.close < lo else None
        if side is None:
            return None
        same = [s for s in self.signals if s.symbol == symbol and s.side == side and s.time.date() == cur.start.date()]
        if len(same) >= 2 or any(s.action == "TAKE" for s in same) or any(s.time == cur.end for s in same):
            return None
        long = side == "LONG"
        checks: list[Check] = []
        entry = cur.close
        level = hi if long else lo

        base = slot_avg_volume(history, cur.start.time())
        rvol = (cur.volume / base) if base else None
        if rvol is None:
            checks.append(Check("Volume", WARN, "no volume baseline"))
        else:
            checks.append(Check("Volume", PASS if rvol >= 1.5 else WARN if rvol >= 1.0 else FAIL,
                                f"{rvol:.1f}x normal for {cur.start:%H:%M}"))
        vw = vwap(today)
        if vw:
            ok = entry > vw if long else entry < vw
            checks.append(Check("VWAP", PASS if ok else FAIL, f"{'above' if entry > vw else 'below'} VWAP {vw:,.2f}"))
        rng = max(cur.high - cur.low, 1e-9)
        pos = (cur.close - cur.low) / rng if long else (cur.high - cur.close) / rng
        checks.append(Check("Candle", PASS if pos >= 0.6 else WARN, "strong close" if pos >= 0.6 else "weak close / wick"))
        a = atr(history[-40:] + today) if history or today else 0
        raw_r = (entry - cur.low if long else cur.high - entry) + 0.001 * entry
        r = min(max(raw_r, 0.004 * entry), 0.015 * entry)
        if prof:
            if long:
                if entry > prof.prev_high:
                    checks.append(Check("Prev day", PASS, f"above prev-day high {prof.prev_high:,.2f}"))
                elif prof.prev_high - entry < 1.5 * r:
                    checks.append(Check("Prev day", WARN, f"prev-day high {prof.prev_high:,.2f} just above"))
                else:
                    checks.append(Check("Prev day", PASS, f"room to prev-day high {prof.prev_high:,.2f}"))
            else:
                if entry < prof.prev_low:
                    checks.append(Check("Prev day", PASS, f"below prev-day low {prof.prev_low:,.2f}"))
                elif entry - prof.prev_low < 1.5 * r:
                    checks.append(Check("Prev day", WARN, f"prev-day low {prof.prev_low:,.2f} just below"))
                else:
                    checks.append(Check("Prev day", PASS, f"room to prev-day low {prof.prev_low:,.2f}"))
        if nifty_chg_pct is not None:
            with_mkt = nifty_chg_pct >= -0.15 if long else nifty_chg_pct <= 0.15
            checks.append(Check("Nifty", PASS if with_mkt else WARN, f"Nifty {nifty_chg_pct:+.2f}% today"))
        cut = at(cur.start.date(), cutoff)
        checks.append(Check("Time", PASS if cur.end <= cut else FAIL, f"{cur.end:%H:%M} close"))
        qty = max(1, int(budget // r)) if r > 0 else 1
        if budget < r:
            checks.append(Check("Worst case", "info", f"even 1 share risks ₹{r:,.2f} - more than ₹{budget:,.0f} left"))
        if blocked:
            checks.append(Check("Risk guard", FAIL, blocked))

        stop = entry - r if long else entry + r
        sig = Signal(id=f"{symbol}-{cur.end:%Y%m%d-%H%M}-{side}", kind="STOCK", symbol=symbol, side=side, time=cur.end,
                     level=round(level, 2), entry=round(entry, 2), stop=round(stop, 2),
                     t1=round(entry + 1.5 * r if long else entry - 1.5 * r, 2),
                     t2=round(entry + 2.5 * r if long else entry - 2.5 * r, 2), checks=checks,
                     qty=qty, risk_rupees=round(qty * r), security_id=prof.security_id if prof else None)
        sig.grade, sig.action = grade(checks)
        sig.note = f"ATR(15m) {a:,.2f}" + (f" | RVOL {rvol:.1f}x" if rvol else "")
        self.signals.append(sig)
        return sig


def rank(signals: Sequence[Signal]) -> list[Signal]:
    order = {"A": 0, "B": 1, "C": 2}

    def rv(s: Signal) -> float:
        for c in s.checks:
            if c.name == "Volume" and "x normal" in c.detail:
                try:
                    return float(c.detail.split("x")[0])
                except ValueError:
                    return 0.0
        return 0.0
    return sorted(signals, key=lambda s: (order.get(s.grade, 9), -rv(s)))
