from __future__ import annotations

import datetime as dt
import random
import statistics as st
from typing import Sequence

from signal_desk.candles import Candle
from signal_desk.clock import IST, at

LOT, COST = 65, 60.0
Q1_MIN_N, Q1_BUY_PF, Q1_SELL_PF, MAX_P, SHUFFLES = 40, 1.2, 0.8, 0.10, 2000


STEP = 50.0
MAX_OFF = 10


class Book:

    def __init__(self, series: dict | None = None, fetch=None) -> None:
        self.px: dict = {}
        self.atm: dict = {}
        self.fetch = fetch
        self.tried: set = set()
        for key, ser in (series or {}).items():
            self.add(key, ser)

    def add(self, key, ser: dict) -> None:
        code, typ, off = key
        for t, (k, close, _spot) in ser.items():
            if k is None:
                continue
            self.px.setdefault((code, typ), {}).setdefault(t, {})[k] = close
            if off == "ATM":
                self.atm[(code, typ, t)] = k

    def _cached(self, code, typ, strike, t):
        return self.px.get((code, typ), {}).get(t, {}).get(strike)

    def price(self, code: int, typ: str, strike: float, when: dt.datetime) -> float | None:
        t = when - dt.timedelta(minutes=5)
        v = self._cached(code, typ, strike, t)
        if v is not None or self.fetch is None:
            return v
        atm = self.atm.get((code, typ, t))
        if atm is None:
            self.ensure_atm(code, typ, when.date())
            atm = self.atm.get((code, typ, t))
            v = self._cached(code, typ, strike, t)
            if v is not None or atm is None:
                return v
        n = int(round((strike - atm) / STEP))
        if abs(n) > MAX_OFF:
            return None
        for off in ((f"ATM+{n}", f"ATM-{n}") if n > 0 else (f"ATM-{-n}", f"ATM+{-n}") if n < 0 else ("ATM",)):
            k = (code, typ, off, when.date())
            if k in self.tried:
                continue
            self.tried.add(k)
            got = self.fetch(code, typ, off, when.date())
            self.add((code, typ, off), got)
            v = self._cached(code, typ, strike, t)
            if v is not None or not got:
                return v
        return None

    def ensure_atm(self, code: int, typ: str, day: dt.date) -> None:
        k = (code, typ, "ATM", day)
        if self.fetch is not None and k not in self.tried:
            self.tried.add(k)
            self.add((code, typ, "ATM"), self.fetch(code, typ, "ATM", day))

    def atm_strike(self, code: int, when: dt.datetime) -> float | None:
        t = when - dt.timedelta(minutes=5)
        k = self.atm.get((code, "CALL", t)) or self.atm.get((code, "PUT", t))
        if k is None and self.fetch is not None:
            self.ensure_atm(code, "CALL", when.date())
            k = self.atm.get((code, "CALL", t))
        return k


def detect_codes(book: Book, exp_days: Sequence[dt.date], codes: Sequence[int]) -> dict:
    med = {}
    for code in codes:
        vals = []
        for d in exp_days:
            t = at(d, "15:15")
            k = book.atm_strike(code, t)
            if k is None:
                continue
            c, p = book.price(code, "CALL", k, t), book.price(code, "PUT", k, t)
            if c is not None and p is not None:
                vals.append(c + p)
        if len(vals) >= 3:
            med[code] = st.median(vals)
    order = sorted(med, key=med.get)
    return {"zero": order[0] if order else None, "next": order[1] if len(order) > 1 else None,
            "median_atm_straddle_1515": {str(k): round(v, 1) for k, v in med.items()}}


def pf_stats(pnl: Sequence[float]) -> dict:
    n = len(pnl)
    if not n:
        return {"n": 0, "win": None, "avg": None, "pf": None, "total": 0, "max_dd": 0}
    gw, gl = sum(x for x in pnl if x > 0), -sum(x for x in pnl if x < 0)
    eq = peak = dd = 0.0
    for x in pnl:
        eq += x
        peak = max(peak, eq)
        dd = min(dd, eq - peak)
    return {"n": n, "win": round(100 * sum(1 for x in pnl if x > 0) / n, 1), "avg": round(st.mean(pnl)),
            "pf": round(gw / gl, 2) if gl else None, "total": round(sum(pnl)), "max_dd": round(dd)}


def signflip_p(vals: Sequence[float], seed: int = 3) -> float | None:
    if len(vals) < 5:
        return None
    real = abs(st.mean(vals))
    rnd = random.Random(seed)
    hits = sum(abs(st.mean(v * rnd.choice((1, -1)) for v in vals)) >= real - 1e-9 for _ in range(SHUFFLES))
    return round(hits / SHUFFLES, 3)


def straddle(book: Book, code: int, d: dt.date) -> float | None:
    t0, t1 = at(d, "09:20"), at(d, "15:15")
    k = book.atm_strike(code, t0)
    if k is None:
        return None
    p = [book.price(code, typ, k, t) for typ in ("CALL", "PUT") for t in (t0, t1)]
    if any(x is None for x in p):
        return None
    ce0, ce1, pe0, pe1 = p
    return ((ce1 - ce0) + (pe1 - pe0)) * LOT - 2 * COST


def exit_times(sig, after: Sequence[Candle], squareoff: dt.datetime):
    long = sig.side == "LONG"
    hit_t1, t1_time, last_end = False, None, sig.time
    for c in after:
        if c.start >= squareoff:
            break
        last_end = c.end
        fav = (c.high - sig.entry) if long else (sig.entry - c.low)
        adv = (c.low - sig.entry) if long else (sig.entry - c.high)
        if not hit_t1:
            if adv <= -sig.risk_pts + 1e-9:
                return None, c.end
            if fav >= abs(sig.t1 - sig.entry):
                hit_t1, t1_time = True, c.end
                if fav >= abs(sig.t2 - sig.entry):
                    return t1_time, c.end
        else:
            if fav >= abs(sig.t2 - sig.entry) or adv <= 0:
                return t1_time, c.end
    return t1_time, min(last_end, squareoff)


def option_trade(book: Book, code: int, sig, t1_time, t_exit) -> float | None:
    typ = "CALL" if sig.side == "LONG" else "PUT"
    k = book.atm_strike(code, sig.time)
    if k is None:
        return None
    e = book.price(code, typ, k, sig.time)
    x = book.price(code, typ, k, t_exit)
    if e is None or x is None:
        return None
    if t1_time is not None:
        h = book.price(code, typ, k, t1_time)
        if h is None:
            return None
        return (0.5 * (h - e) + 0.5 * (x - e)) * LOT - COST
    return (x - e) * LOT - COST


def q2_trades(book: Book, codes: dict, signals_with_after: list) -> list[dict]:
    out = []
    for sig, after in signals_with_after:
        t1t, tx = exit_times(sig, after, at(sig.time.date(), "15:15"))
        z = option_trade(book, codes["zero"], sig, t1t, tx) if codes.get("zero") is not None else None
        n = option_trade(book, codes["next"], sig, t1t, tx) if codes.get("next") is not None else None
        out.append({"date": sig.time.date(), "time": f"{sig.time:%H:%M}", "side": sig.side, "zero": z, "next": n,
                    "exit": f"{tx:%H:%M}"})
    return out


def verdict_q1(h: dict, pooled_p) -> tuple[str, list[str]]:
    why = [f"{k}: n={v['n']} win={v['win']}% avg=₹{v['avg']} PF={v['pf']} total=₹{v['total']:,}" for k, v in h.items()]
    why.append(f"two-sided sign-flip p (pooled) = {pooled_p}  (pass needs <= {MAX_P})")
    o, n = h["older"], h["newer"]
    enough = o["n"] >= Q1_MIN_N and n["n"] >= Q1_MIN_N
    sig = pooled_p is not None and pooled_p <= MAX_P
    if enough and sig and (o["pf"] or 0) >= Q1_BUY_PF and (n["pf"] or 0) >= Q1_BUY_PF:
        return "PASS (BUYERS' EDGE) - buying the expiry-day straddle made money in both years", why
    if enough and sig and o["pf"] is not None and n["pf"] is not None and o["pf"] <= Q1_SELL_PF and n["pf"] <= Q1_SELL_PF:
        return "PASS (SELLERS' EDGE) - the expiry-day straddle lost money for buyers in both years", why
    if not enough:
        return "INCONCLUSIVE - not enough expiry days with complete premium data", why
    return "FAIL - no consistent edge either way for an expiry-day straddle", why


def verdict_q2(h: dict, pooled_p) -> tuple[str, list[str]]:
    why = []
    signs = set()
    for half in ("older", "newer"):
        x = h[half]
        why.append(f"{half}: paired n={x['pairs']}  0DTE avg ₹{x['zero']['avg']} (PF {x['zero']['pf']})  vs  next-week "
                   f"avg ₹{x['next']['avg']} (PF {x['next']['pf']})  diff avg ₹{x['diff_avg']}")
        if x["diff_avg"] is not None and x["diff_avg"] != 0:
            signs.add(x["diff_avg"] > 0)
        else:
            signs.add(None)
    why.append(f"two-sided sign-flip p on paired differences (pooled) = {pooled_p}")
    if len(signs) == 1 and None not in signs and pooled_p is not None and pooled_p <= MAX_P:
        return ("0DTE BETTER in both years - worth reconsidering the next-week rule" if signs.pop()
                else "0DTE WORSE in both years - the next-week rule is right"), why
    return "NO CLEAR DIFFERENCE - keep the next-week rule (it was the tested choice)", why
