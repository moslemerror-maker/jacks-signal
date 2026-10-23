from __future__ import annotations

import datetime as dt
import math

from expiry_0dte_lab.analysis import Book, exit_times, pf_stats
from signal_desk.clock import at

LOT, COST = 65, 60.0
MIN_PF, MIN_COVER = 1.10, 0.80
CODE_NEAR, CODE_NEXT = 1, 2


def strike_for(side: str, entry: float) -> float:
    return math.floor(entry / 50) * 50 if side == "LONG" else math.ceil(entry / 50) * 50


def real_pnl(book: Book, code: int, sig, t1_time, t_exit) -> float | None:
    typ = "CALL" if sig.side == "LONG" else "PUT"
    k = strike_for(sig.side, sig.entry)
    e = book.price(code, typ, k, sig.time)
    if e is None:
        return None
    x = book.price(code, typ, k, t_exit)
    if x is None:
        return None
    if t1_time is not None:
        h = book.price(code, typ, k, t1_time)
        if h is None:
            return None
        return (0.5 * (h - e) + 0.5 * (x - e)) * LOT - COST
    return (x - e) * LOT - COST


def trades(book: Book, results: list, day_candles: dict, expiry: set) -> list[dict]:
    out = []
    for r in results:
        sig = r.signal
        d = sig.time.date()
        after = [c for c in day_candles.get(d, []) if c.start >= sig.time]
        t1, tx = exit_times(sig, after, at(d, "15:15"))
        code = CODE_NEXT if d in expiry else CODE_NEAR
        out.append({"date": d, "time": f"{sig.time:%H:%M}", "side": sig.side, "grade": sig.grade, "expiry_day": d in expiry,
                    "estimate": r.rupees, "real": real_pnl(book, code, sig, t1, tx), "exit": f"{tx:%H:%M}",
                    "outcome": r.outcome})
    return out


def summarise(tr: list[dict], older: set) -> dict:
    out = {"halves": {}}
    for half, cond in (("older", lambda d: d in older), ("newer", lambda d: d not in older)):
        part = [t for t in tr if cond(t["date"])]
        priced = [t for t in part if t["real"] is not None]
        row = {"signals": len(part), "priced": len(priced),
               "real": pf_stats([t["real"] for t in priced]), "estimate": pf_stats([t["estimate"] for t in priced])}
        for name, f in (("calls", lambda t: t["side"] == "LONG"), ("puts", lambda t: t["side"] == "SHORT"),
                        ("grade_A", lambda t: t["grade"] == "A"), ("grade_B", lambda t: t["grade"] == "B"),
                        ("expiry_day", lambda t: t["expiry_day"]), ("before_11", lambda t: t["time"] < "11:00"),
                        ("from_11", lambda t: t["time"] >= "11:00")):
            row[name] = pf_stats([t["real"] for t in priced if f(t)])
        out["halves"][half] = row
    out["verdict"], out["why"] = verdict(out)
    return out


def verdict(s: dict) -> tuple[str, list[str]]:
    why, good = [], 0
    for half in ("older", "newer"):
        h = s["halves"][half]
        r, e = h["real"], h["estimate"]
        why.append(f"{half}: priced {h['priced']}/{h['signals']}  REAL avg ₹{r['avg']} PF {r['pf']} total ₹{r['total']:,}"
                   f"  |  estimate avg ₹{e['avg']} PF {e['pf']} total ₹{e['total']:,}")
        if h["signals"] and h["priced"] / h["signals"] < MIN_COVER:
            return "INCOMPLETE - too many signals could not be priced (re-run to fetch the rest)", why
        if (r["avg"] or 0) > 0 and (r["pf"] or 0) >= MIN_PF:
            good += 1
    if good == 2:
        return "SURVIVES - the breakout rules stay profitable with real option premiums in both years", why
    if good == 1:
        return "MIXED - profitable with real premiums in one year only; treat the edge as unproven", why
    return "FAILS - with real option premiums the rules are not profitable; do not trade them with real money", why
