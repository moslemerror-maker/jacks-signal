from __future__ import annotations

import datetime as dt

from expiry_0dte_lab.analysis import Book, exit_times, pf_stats, signflip_p
from premium_lab.analysis import CODE_NEAR, CODE_NEXT, COST, LOT, real_pnl, strike_for
from signal_desk.clock import at

DELTA, MIN_PF, MAX_P = 0.5, 1.10, 0.10
COLLECT = False


def premium_exit(book: Book, code: int, sig) -> float | None:
    typ = "CALL" if sig.side == "LONG" else "PUT"
    k = strike_for(sig.side, sig.entry)
    e = book.price(code, typ, k, sig.time)
    if e is None:
        return None
    r = sig.risk_pts
    stop, t1, t2 = e - DELTA * r, e + DELTA * 1.5 * r, e + DELTA * 2.5 * r
    half = None
    t, end = sig.time + dt.timedelta(minutes=5), at(sig.time.date(), "15:15")
    p = e
    while t <= end:
        q = book.price(code, typ, k, t)
        if q is None:
            if COLLECT:
                t += dt.timedelta(minutes=5)
                continue
            return None
        p = q
        if half is None:
            if p <= stop:
                return (p - e) * LOT - COST
            if p >= t1:
                half = p
                if p >= t2:
                    return (p - e) * LOT - COST
        elif p >= t2 or p <= e:
            return (0.5 * (half - e) + 0.5 * (p - e)) * LOT - COST
        t += dt.timedelta(minutes=5)
    if half is not None:
        return (0.5 * (half - e) + 0.5 * (p - e)) * LOT - COST
    return (p - e) * LOT - COST


def trades(book: Book, results: list, day_candles: dict, expiry: set) -> list[dict]:
    out = []
    for x in results:
        sig = x.signal
        d = sig.time.date()
        code = CODE_NEXT if d in expiry else CODE_NEAR
        after = [c for c in day_candles.get(d, []) if c.start >= sig.time]
        t1, tx = exit_times(sig, after, at(d, "15:15"))
        idx = real_pnl(book, code, sig, t1, tx)
        prem = premium_exit(book, code, sig) if idx is not None else None
        out.append({"date": d, "time": f"{sig.time:%H:%M}", "side": sig.side, "index": idx, "premium": prem})
    return out


def summarise(tr: list[dict], older: set) -> dict:
    out: dict = {"halves": {}}
    all_d = []
    for half, cond in (("older", lambda d: d in older), ("newer", lambda d: d not in older)):
        both = [t for t in tr if cond(t["date"]) and t["index"] is not None and t["premium"] is not None]
        diffs = [t["premium"] - t["index"] for t in both]
        all_d += diffs
        out["halves"][half] = {"signals": sum(1 for t in tr if cond(t["date"])), "paired": len(both),
                               "index": pf_stats([t["index"] for t in both]),
                               "premium": pf_stats([t["premium"] for t in both]),
                               "diff_avg": round(sum(diffs) / len(diffs)) if diffs else None,
                               "premium_calls": pf_stats([t["premium"] for t in both if t["side"] == "LONG"]),
                               "premium_puts": pf_stats([t["premium"] for t in both if t["side"] == "SHORT"])}
    out["p"] = signflip_p(all_d)
    out["verdict"], out["why"] = verdict(out)
    return out


def verdict(s: dict) -> tuple[str, list[str]]:
    why, ok = [], True
    for half in ("older", "newer"):
        h = s["halves"][half]
        i, p = h["index"], h["premium"]
        why.append(f"{half}: paired {h['paired']}/{h['signals']}  PREMIUM exit avg ₹{p['avg']} PF {p['pf']} total "
                   f"₹{p['total']:,}  |  INDEX exit avg ₹{i['avg']} PF {i['pf']} total ₹{i['total']:,}  | diff ₹{h['diff_avg']}")
        if h["signals"] and h["paired"] / h["signals"] < 0.8:
            return "INCOMPLETE - too many trades without a full premium path (re-run to fetch the rest)", why
        if (h["diff_avg"] or 0) <= 0 or (p["avg"] or 0) <= 0 or (p["pf"] or 0) < MIN_PF:
            ok = False
    why.append(f"paired sign-flip p (both years) = {s['p']}  (pass needs <= {MAX_P})")
    if ok and s["p"] is not None and s["p"] <= MAX_P:
        return "PASS - exiting on the premium stop/target beats the index exits and is profitable in both years", why
    return "FAIL - the premium exit is not reliably better (or not profitable) in both years; keep the index exits", why
