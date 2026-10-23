from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import io
import logging
from pathlib import Path

import requests

from .config import CACHE_DIR

log = logging.getLogger("signal_desk.instruments")
MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"


@dataclasses.dataclass
class Stock:
    symbol: str
    security_id: int
    lot: int


@dataclasses.dataclass
class Instruments:
    nifty_id: int = 13
    banknifty_id: int = 25
    vix_id: int = 21
    nifty_lot: int = 0
    stocks: list[Stock] = dataclasses.field(default_factory=list)
    source: str = "defaults"
    notes: list[str] = dataclasses.field(default_factory=list)

    def by_symbol(self) -> dict[str, Stock]:
        return {s.symbol: s for s in self.stocks}


def _download(today: dt.date, session: requests.Session | None = None) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"scrip-master-{today:%Y-%m-%d}.csv"
    if path.exists() and path.stat().st_size > 100_000:
        return path.read_text(encoding="utf-8", errors="replace")
    r = (session or requests).get(MASTER_URL, timeout=60)
    r.raise_for_status()
    text = r.content.decode("utf-8", errors="replace")
    path.write_text(text, encoding="utf-8")
    for old in CACHE_DIR.glob("scrip-master-*.csv"):
        if old != path:
            try:
                old.unlink()
            except OSError:
                pass
    return text


def _int(v) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def _date(v) -> dt.date | None:
    try:
        return dt.date.fromisoformat(str(v)[:10])
    except ValueError:
        return None


def parse_master(text: str, today: dt.date) -> Instruments:
    inst = Instruments(source="dhan scrip master")
    eq: dict[str, int] = {}
    fut_lot: dict[str, tuple[dt.date, int]] = {}
    nifty_fut: list[tuple[dt.date, int]] = []
    idx: dict[str, int] = {}
    for row in csv.DictReader(io.StringIO(text)):
        if row.get("SEM_EXM_EXCH_ID") != "NSE":
            continue
        name = row.get("SEM_INSTRUMENT_NAME", "")
        sym = (row.get("SEM_TRADING_SYMBOL") or "").strip()
        sid = _int(row.get("SEM_SMST_SECURITY_ID"))
        if name == "EQUITY" and row.get("SEM_SERIES") == "EQ":
            eq[sym] = sid
        elif name == "INDEX":
            idx[sym.upper()] = sid
        elif name in ("FUTSTK", "FUTIDX"):
            exp = _date(row.get("SEM_EXPIRY_DATE"))
            if not exp or exp < today:
                continue
            under = sym.split("-")[0]
            lot = _int(row.get("SEM_LOT_UNITS"))
            if name == "FUTSTK":
                if under not in fut_lot or exp < fut_lot[under][0]:
                    fut_lot[under] = (exp, lot)
            elif under == "NIFTY":
                nifty_fut.append((exp, lot))
    for key, attr in ((("NIFTY", "NIFTY 50"), "nifty_id"), (("BANKNIFTY", "NIFTY BANK"), "banknifty_id"),
                      (("INDIA VIX", "INDIAVIX"), "vix_id")):
        found = next((idx[k] for k in key if k in idx), None)
        if found:
            if found != getattr(inst, attr):
                inst.notes.append(f"{attr}: scrip master says {found} (default was {getattr(inst, attr)})")
            setattr(inst, attr, found)
        else:
            inst.notes.append(f"{attr}: not found in scrip master, using default {getattr(inst, attr)}")
    if nifty_fut:
        inst.nifty_lot = sorted(nifty_fut)[0][1]
    for under, (_, lot) in sorted(fut_lot.items()):
        if under in eq:
            inst.stocks.append(Stock(under, eq[under], lot))
    return inst


def load(today: dt.date, session: requests.Session | None = None) -> Instruments:
    try:
        return parse_master(_download(today, session), today)
    except Exception as exc:
        log.warning("scrip master unavailable: %s", exc)
        inst = Instruments()
        inst.notes.append(f"scrip master unavailable ({exc}); stock scanner disabled, Nifty uses default ids")
        return inst


def cache_path_for(today: dt.date) -> Path:
    return CACHE_DIR / f"scrip-master-{today:%Y-%m-%d}.csv"
