from __future__ import annotations

import csv
import datetime as dt
import gzip
from pathlib import Path

from signal_desk.candles import Candle
from signal_desk.clock import IST
from signal_desk.research import fetch_range

LAB_DIR = Path(__file__).resolve().parent
CACHE = LAB_DIR / "cache"
IDS = {"nifty": 13, "banknifty": 25, "vix": 21}


def _path(name: str) -> Path:
    return CACHE / f"{name}_15m.csv.gz"


def save(name: str, cs: list[Candle]) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    with gzip.open(_path(name), "wt", newline="") as f:
        w = csv.writer(f)
        for c in cs:
            w.writerow([c.start.isoformat(), c.open, c.high, c.low, c.close, c.volume])


def load(name: str) -> list[Candle]:
    p = _path(name)
    if not p.exists():
        return []
    out = []
    with gzip.open(p, "rt", newline="") as f:
        for row in csv.reader(f):
            out.append(Candle(dt.datetime.fromisoformat(row[0]).astimezone(IST), float(row[1]), float(row[2]),
                              float(row[3]), float(row[4]), float(row[5]), 15))
    return out


def update(api, start: dt.date, end: dt.date, say=print) -> dict[str, list[Candle]]:
    out = {}
    for name, sid in IDS.items():
        have = load(name)
        by = {c.start: c for c in have}
        first = min((c.start.date() for c in have), default=None)
        last = max((c.start.date() for c in have), default=None)
        mark = CACHE / f"{name}.from"
        asked = dt.date.fromisoformat(mark.read_text().strip()) if mark.exists() else None
        if first is None:
            spans = [(start, end)]
        else:
            lo = start if asked is None or start < asked else None
            spans = ([(lo, first - dt.timedelta(days=1))] if lo else []) + [(last + dt.timedelta(days=1), end)]
        for a, b in spans:
            if a <= b:
                say(f".. {name}: fetching {a} .. {b}")
                for c in fetch_range(api, sid, a, b, 15):
                    by[c.start] = c
        cs = [by[k] for k in sorted(by)]
        save(name, cs)
        if asked is None or start < asked:
            mark.write_text(start.isoformat())
        out[name] = cs
    return out
