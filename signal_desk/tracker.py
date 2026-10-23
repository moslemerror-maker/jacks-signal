from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from typing import Iterable

from .signals import Signal

JOURNAL_FIELDS = ["date", "time", "symbol", "side", "grade", "action", "level", "entry", "stop", "t1", "t2",
                  "status", "outcome_time", "t2_hit", "mfe_pts", "mae_pts", "r_pts", "result_r",
                  "option", "strike", "expiry", "opt_ltp", "lots", "risk_per_lot", "base_grade", "prev_range"]


def update(sig: Signal, price: float, now: dt.datetime) -> str | None:
    if price is None:
        return None
    long = sig.side == "LONG"
    fav = (price - sig.entry) if long else (sig.entry - price)
    sig.mfe = max(sig.mfe, fav)
    sig.mae = min(sig.mae, fav)
    if sig.status == "open":
        if (long and price <= sig.stop) or (not long and price >= sig.stop):
            sig.status, sig.outcome_time, sig.exit_ref = "SL", now, sig.stop
            return "SL"
        if (long and price >= sig.t1) or (not long and price <= sig.t1):
            sig.status, sig.outcome_time, sig.exit_ref = "T1", now, sig.t1
            return "T1"
    elif sig.status == "T1" and not sig.t2_hit:
        if (long and price >= sig.t2) or (not long and price <= sig.t2):
            sig.t2_hit = True
            return "T2"
    return None


def close_day(sig: Signal, last_price: float | None, now: dt.datetime) -> None:
    if sig.status == "open":
        sig.status, sig.outcome_time = "timeout", now
        sig.exit_ref = last_price


def result_r(sig: Signal) -> float | None:
    r = sig.risk_pts
    if not r:
        return None
    if sig.status == "SL":
        return -1.0
    if sig.status == "T1":
        return round(abs(sig.t2 - sig.entry) / r, 2) if sig.t2_hit else round(abs(sig.t1 - sig.entry) / r, 2)
    if sig.status == "timeout" and sig.exit_ref is not None:
        move = (sig.exit_ref - sig.entry) if sig.side == "LONG" else (sig.entry - sig.exit_ref)
        return round(move / r, 2)
    return None


def append_journal(path: Path, signals: Iterable[Signal]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            rows = list(rd)
            old_fields = rd.fieldnames or []
        for row in rows:
            existing.add(f"{row['date']}|{row['time']}|{row['symbol']}|{row['side']}")
        if old_fields != JOURNAL_FIELDS:
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS, extrasaction="ignore")
                w.writeheader()
                for row in rows:
                    w.writerow({k: row.get(k, "") or "" for k in JOURNAL_FIELDS})
    new = 0
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
        if write_header:
            w.writeheader()
        for s in signals:
            key = f"{s.time:%Y-%m-%d}|{s.time:%H:%M}|{s.symbol}|{s.side}"
            if key in existing:
                continue
            w.writerow({
                "date": f"{s.time:%Y-%m-%d}", "time": f"{s.time:%H:%M}", "symbol": s.symbol, "side": s.side,
                "grade": s.grade, "action": s.action, "level": s.level, "entry": s.entry, "stop": s.stop,
                "t1": s.t1, "t2": s.t2, "status": s.status,
                "outcome_time": f"{s.outcome_time:%H:%M}" if s.outcome_time else "", "t2_hit": s.t2_hit,
                "mfe_pts": round(s.mfe, 2), "mae_pts": round(s.mae, 2), "r_pts": round(s.risk_pts, 2),
                "result_r": result_r(s), "option": s.option or "", "strike": s.strike or "", "expiry": s.expiry or "",
                "opt_ltp": s.opt_ltp or "", "lots": s.lots if s.lots is not None else "", "risk_per_lot": s.risk_per_lot or "",
                "base_grade": s.base_grade or "", "prev_range": s.prev_range if s.prev_range is not None else "",
            })
            new += 1
    return new


def summary(signals: Iterable[Signal]) -> dict:
    sigs = list(signals)
    done = [s for s in sigs if s.status != "open"]
    taken = [s for s in done if s.action == "TAKE"]
    rs = [result_r(s) for s in taken if result_r(s) is not None]
    return {
        "signals": len(sigs), "take": sum(1 for s in sigs if s.action == "TAKE"),
        "t1": sum(1 for s in taken if s.status == "T1"), "sl": sum(1 for s in taken if s.status == "SL"),
        "timeout": sum(1 for s in taken if s.status == "timeout"),
        "net_r": round(sum(rs), 2) if rs else 0.0,
    }
