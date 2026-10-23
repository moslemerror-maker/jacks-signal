from __future__ import annotations

import dataclasses
import datetime as dt
import math
from typing import Sequence

from .candles import Candle, atr
from .clock import at
from .optionchain import Chain, ChainStats, pick_strike
from .signals import FAIL, INFO, PASS, WARN, Check, Signal, grade

MIN_RISK_PTS = 20.0
MAX_RISK_PTS = 60.0
SLIPPAGE_PREM = 2.0
APPROACH_PTS = 10.0


@dataclasses.dataclass
class MarketCtx:
    now: dt.datetime
    nifty: list[Candle]
    prev_hlc: tuple[float, float, float] | None
    bnf: list[Candle]
    bnf_prev_close: float | None
    vix: float | None
    vix_prev_close: float | None
    oi_stats: ChainStats | None
    trade_chain: Chain | None
    lot: int
    budget: float
    blocked_reason: str | None
    cutoff: str = "14:30"
    backtest: bool = False
    trend_filter: bool = False
    prev_range_ratio: float | None = None


RANGE_LOOK = 10
RANGE_NARROW = 0.777
RANGE_WIDE = 1.157


def prev_range_ratio(hist: Sequence[Candle], day: dt.date, min_candles: int = 20) -> float | None:
    by: dict[dt.date, list[Candle]] = {}
    for c in hist:
        if c.start.date() < day:
            by.setdefault(c.start.date(), []).append(c)
    rngs = [max(c.high for c in by[d]) - min(c.low for c in by[d]) for d in sorted(by) if len(by[d]) >= min_candles]
    if len(rngs) < RANGE_LOOK + 1:
        return None
    avg = sum(rngs[-RANGE_LOOK - 1:-1]) / RANGE_LOOK
    return round(rngs[-1] / avg, 3) if avg > 0 else None


def _range_adjust(sig: Signal, ratio: float | None) -> None:
    if ratio is None:
        return
    sig.base_grade, sig.prev_range = sig.grade, ratio
    if ratio < RANGE_NARROW:
        up = sig.grade == "B"
        sig.checks.append(Check("Prev day", INFO, f"narrow day before ({ratio:.2f}x normal range)" + (" - grade B→A" if up else "")))
        if up:
            sig.grade = "A"
    elif ratio >= RANGE_WIDE:
        down = sig.grade == "A"
        sig.checks.append(Check("Prev day", INFO, f"wide day before ({ratio:.2f}x normal range)" + (" - grade A→B" if down else "")))
        if down:
            sig.grade = "B"
    else:
        sig.checks.append(Check("Prev day", INFO, f"normal day before ({ratio:.2f}x normal range)"))


def levels(ctx: MarketCtx) -> dict:
    c = ctx.nifty
    out: dict = {}
    if c:
        out.update(orh=c[0].high, orl=c[0].low, day_high=max(x.high for x in c), day_low=min(x.low for x in c),
                   last_close=c[-1].close, atr15=round(atr(c), 1))
    if ctx.prev_hlc:
        out.update(pdh=ctx.prev_hlc[0], pdl=ctx.prev_hlc[1], pdc=ctx.prev_hlc[2])
    if ctx.oi_stats:
        s = ctx.oi_stats
        out.update(call_wall=s.call_wall, put_wall=s.put_wall, call_wall_2=s.call_wall_2, put_wall_2=s.put_wall_2,
                   pcr=s.pcr, max_pain=s.max_pain, oi_bias=s.oi_bias_label)
    if c and len(c) >= 2 and ctx.now.time() <= dt.time(*map(int, ctx.cutoff.split(":"))):
        out["long_trigger"] = max(x.high for x in c)
        out["short_trigger"] = min(x.low for x in c)
    return out


class NiftyRules:
    def __init__(self) -> None:
        self.signals: list[Signal] = []
        self._approach_sent: set[str] = set()
        self._last_approach: dt.datetime | None = None

    def on_candle_close(self, ctx: MarketCtx) -> list[Signal]:
        c = ctx.nifty
        if len(c) < 2:
            return []
        cur, prior = c[-1], c[:-1]
        if any(s.time == cur.end for s in self.signals):
            return []
        hi, lo = max(x.high for x in prior), min(x.low for x in prior)
        out = []
        if cur.close > hi and self._armed("LONG", cur, c):
            out.append(self._build("LONG", hi, cur, ctx))
        elif cur.close < lo and self._armed("SHORT", cur, c):
            out.append(self._build("SHORT", lo, cur, ctx))
        self.signals.extend(out)
        return out

    def _armed(self, side: str, cur: Candle, today: Sequence[Candle]) -> bool:
        same = [s for s in self.signals if s.side == side and s.time.date() == cur.start.date()]
        if not same:
            return True
        if sum(1 for s in same if s.action == "TAKE") >= 2:
            return False
        last = same[-1]
        if last.status == "open" and last.action == "TAKE":
            return False
        base = 0
        for i, x in enumerate(today):
            if x.end <= last.time or x is cur:
                continue
            before = today[:i]
            if side == "LONG" and x.close <= max(b.high for b in before):
                base += 1
            if side == "SHORT" and x.close >= min(b.low for b in before):
                base += 1
        return base >= 2

    def _build(self, side: str, level: float, cur: Candle, ctx: MarketCtx) -> Signal:
        long = side == "LONG"
        checks: list[Check] = []
        entry = cur.close
        raw_r = (entry - cur.low if long else cur.high - entry) + 3
        r = min(max(raw_r, MIN_RISK_PTS), MAX_RISK_PTS)
        stop = entry - r if long else entry + r
        t1 = entry + 1.5 * r if long else entry - 1.5 * r
        t2 = entry + 2.5 * r if long else entry - 2.5 * r
        note = ""

        cutoff = at(cur.start.date(), ctx.cutoff)
        checks.append(Check("Time", PASS if cur.end <= cutoff else FAIL,
                            f"{cur.end:%H:%M} close" + ("" if cur.end <= cutoff else f" - after {ctx.cutoff} cutoff")))

        rng = max(cur.high - cur.low, 0.01)
        pos = (cur.close - cur.low) / rng if long else (cur.high - cur.close) / rng
        checks.append(Check("Candle", PASS if pos >= 0.6 else WARN,
                            f"strong close near the {'high' if long else 'low'} ({pos * 100:.0f}% of range)"
                            if pos >= 0.6 else f"weak close ({pos * 100:.0f}% of range) - long {'upper' if long else 'lower'} wick"))
        if raw_r > MAX_RISK_PTS:
            checks.append(Check("Stop size", WARN, f"breakout candle is {raw_r - 3:.0f} pts tall; stop capped at {MAX_RISK_PTS:.0f} pts"))

        s = ctx.oi_stats
        wall = (s.call_wall if long else s.put_wall) if s else None
        if wall is None:
            checks.append(Check("OI room", PASS, "no OI wall data"))
        else:
            room = (wall - entry) if long else (entry - wall)
            name = "call wall (resistance)" if long else "put wall (support)"
            if room <= 0:
                nxt = (s.call_wall_2 if long else s.put_wall_2) if s else None
                nroom = ((nxt - entry) if long else (entry - nxt)) if nxt else None
                if nroom is None or nroom >= 1.5 * r:
                    checks.append(Check("OI room", PASS, f"through the {wall:,.0f} {name}"))
                else:
                    checks.append(Check("OI room", WARN, f"next {name} at {nxt:,.0f}, only {nroom:.0f} pts away"))
            elif room >= 1.5 * r:
                checks.append(Check("OI room", PASS, f"{name} {wall:,.0f} is {room:.0f} pts away"))
            elif room >= 1.0 * r:
                checks.append(Check("OI room", WARN, f"{name} {wall:,.0f} only {room:.0f} pts away - T1 set below it"))
                t1 = wall - 5 if long else wall + 5
            else:
                checks.append(Check("OI room", FAIL, f"breaking straight into the {wall:,.0f} {name} ({room:.0f} pts) - low reward"))

        if ctx.trend_filter and ctx.nifty:
            day_open = ctx.nifty[0].open
            pdc = ctx.prev_hlc[2] if ctx.prev_hlc else None
            refs = f"open {day_open:,.0f}" + (f" / prev close {pdc:,.0f}" if pdc else "")
            if long:
                ok = entry > day_open and (pdc is None or entry > pdc)
                det = f"above {refs} - day is up" if ok else f"against the day: not above {refs} - no call"
            else:
                ok = entry < day_open and (pdc is None or entry < pdc)
                det = f"below {refs} - day is down" if ok else f"against the day: not below {refs} - no put"
            checks.append(Check("Trend", PASS if ok else FAIL, det))

        if ctx.bnf:
            b = ctx.bnf[-1]
            day_chg = ((b.close - ctx.bnf_prev_close) / ctx.bnf_prev_close * 100) if ctx.bnf_prev_close else 0.0
            with_us = (b.close >= b.open) if long else (b.close <= b.open)
            day_with = day_chg >= -0.1 if long else day_chg <= 0.1
            st = PASS if with_us and day_with else (FAIL if not with_us and not day_with else WARN)
            checks.append(Check("Bank Nifty", st, f"{'green' if b.close >= b.open else 'red'} candle, {day_chg:+.2f}% on day"))
        else:
            checks.append(Check("Bank Nifty", WARN, "no Bank Nifty data"))

        if ctx.vix and ctx.vix_prev_close:
            v = (ctx.vix - ctx.vix_prev_close) / ctx.vix_prev_close * 100
            if long:
                st = PASS if v <= 3 else (WARN if v <= 7 else FAIL)
                det = f"VIX {ctx.vix:.2f} ({v:+.1f}%)" + (" - fear rising, calls fight it" if v > 3 else "")
            else:
                st = PASS if v >= 0 else (WARN if v >= -5 else FAIL)
                det = f"VIX {ctx.vix:.2f} ({v:+.1f}%)" + (" - falling VIX shrinks put premiums" if v < 0 else "")
            checks.append(Check("VIX", st, det))

        if s and s.oi_bias is not None:
            contrary = (long and s.oi_bias_label.startswith("bearish")) or (not long and s.oi_bias_label.startswith("bullish"))
            checks.append(Check("OI flow", WARN if contrary else PASS, s.oi_bias_label + (f", PCR {s.pcr:.2f}" if s.pcr else "")))

        sig = Signal(id=f"NIFTY-{cur.end:%Y%m%d-%H%M}-{side}", kind="NIFTY", symbol="NIFTY", side=side, time=cur.end,
                     level=level, entry=entry, stop=round(stop, 1), t1=round(t1, 1), t2=round(t2, 1))

        ch = ctx.trade_chain
        if ch and ch.rows:
            opt = "CE" if long else "PE"
            k = pick_strike(ch, entry, opt)
            row = ch.row(k)
            leg = (row.ce if long else row.pe) if row else None
            ltp = leg.ltp if leg else None
            delta = abs(leg.delta) if leg and leg.delta else 0.5
            sig.option, sig.strike, sig.expiry, sig.opt_ltp, sig.delta, sig.lot = opt, k, ch.expiry, ltp, round(delta, 2), ctx.lot
            prem_risk = delta * r + SLIPPAGE_PREM
            sig.risk_per_lot = round(prem_risk * ctx.lot)
            if ltp:
                sig.prem_stop = round(max(ltp - delta * r, 0.05), 1)
                sig.prem_t1 = round(ltp + delta * abs(sig.t1 - entry), 1)
            if ltp is None:
                checks.append(Check("Option", WARN, f"no live price for {k:,.0f} {opt}"))
            checks.append(_worst_case(sig, ctx.budget))
        elif ctx.backtest:
            sig.option, sig.delta, sig.lot = ("CE" if long else "PE"), 0.5, ctx.lot
            sig.strike = (entry // 50) * 50 if long else -(-entry // 50) * 50
            sig.risk_per_lot = round((0.5 * r + SLIPPAGE_PREM) * ctx.lot)
            checks.append(_worst_case(sig, ctx.budget))
        else:
            checks.append(Check("Option", WARN, "option chain unavailable - pick ATM strike manually"))

        if ctx.blocked_reason:
            checks.append(Check("Risk guard", FAIL, ctx.blocked_reason))

        sig.checks = checks
        sig.grade, sig.action = grade(checks)
        _range_adjust(sig, ctx.prev_range_ratio)
        sig.note = note
        return sig

    def approach(self, ltp: float, ctx: MarketCtx) -> str | None:
        c = ctx.nifty
        if len(c) < 1 or ctx.blocked_reason:
            return None
        if ctx.now.time() > dt.time(*map(int, ctx.cutoff.split(":"))):
            return None
        if self._last_approach and (ctx.now - self._last_approach) < dt.timedelta(minutes=30):
            return None
        if any(s.status == "open" and s.time.date() == ctx.now.date() for s in self.signals):
            return None
        hi, lo = max(x.high for x in c), min(x.low for x in c)
        if hi - APPROACH_PTS <= ltp <= hi + 2:
            key = f"up-{hi:.0f}"
            if key not in self._approach_sent:
                self._approach_sent.add(key)
                self._last_approach = ctx.now
                return (f"👀 Nifty {ltp:,.1f} is near the breakout level {hi:,.1f} (day high). "
                        f"Get ready - a CE trade triggers only on a 15-min CLOSE above it (next close {self._next_close(ctx.now):%H:%M}).")
        if lo - 2 <= ltp <= lo + APPROACH_PTS:
            key = f"dn-{lo:.0f}"
            if key not in self._approach_sent:
                self._approach_sent.add(key)
                self._last_approach = ctx.now
                return (f"👀 Nifty {ltp:,.1f} is near the breakdown level {lo:,.1f} (day low). "
                        f"A PE trade triggers only on a 15-min CLOSE below it (next close {self._next_close(ctx.now):%H:%M}).")
        return None

    @staticmethod
    def _next_close(now: dt.datetime) -> dt.datetime:
        base = now.replace(second=0, microsecond=0)
        add = 15 - (base.minute % 15)
        return base + dt.timedelta(minutes=add)


def _worst_case(sig: Signal, room: float) -> Check:
    worst = (sig.risk_per_lot or 0) + 60
    sig.lots = max(1, int(room // worst)) if room >= worst else 1
    if worst <= room:
        return Check("Worst case", INFO, f"if SL hits: -₹{worst:,.0f} per lot (₹{room:,.0f} left of today's limit)")
    return Check("Worst case", INFO, f"if SL hits: -₹{worst:,.0f} for 1 lot - MORE than the ₹{room:,.0f} left of today's limit")


def lots_for(budget: float, risk_per_lot: float) -> int:
    return int(math.floor(budget / risk_per_lot)) if risk_per_lot > 0 else 0
