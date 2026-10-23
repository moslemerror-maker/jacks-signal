from __future__ import annotations

import csv
import datetime as dt
import gzip
import io
from pathlib import Path
from typing import Any, Callable

import requests

LAB_DIR = Path(__file__).resolve().parent
CACHE = LAB_DIR / "cache"
M5 = CACHE / "m5"
CHUNK_DAYS = 85
MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"


def universe(today: dt.date) -> list[Any]:
    from signal_desk.instruments import parse_master
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"scrip-master-{today:%Y-%m-%d}.csv"
    if not p.exists():
        r = requests.get(MASTER_URL, timeout=90)
        r.raise_for_status()
        p.write_text(r.text, encoding="utf-8")
        for old in CACHE.glob("scrip-master-*.csv"):
            if old != p:
                old.unlink(missing_ok=True)
    return parse_master(p.read_text(encoding="utf-8"), today).stocks


def _read(path: Path) -> dict[int, tuple]:
    rows: dict[int, tuple] = {}
    if path.exists():
        try:
            with gzip.open(path, "rt") as f:
                for r in csv.reader(f):
                    rows[int(r[0])] = (int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]))
        except (OSError, ValueError, IndexError, EOFError):
            rows = {}
    return rows


def _write(path: Path, rows: dict[int, tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.writer(buf)
    for k in sorted(rows):
        r = rows[k]
        w.writerow([r[0], f"{r[1]:g}", f"{r[2]:g}", f"{r[3]:g}", f"{r[4]:g}", f"{int(r[5])}"])
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        f.write(buf.getvalue())
    tmp.replace(path)


def load_m5(api: Any, sid: int, start: dt.date, end: dt.date, segment: str = "NSE_EQ", instrument: str = "EQUITY",
            refresh: bool = True) -> list[tuple]:
    path = M5 / f"{sid}.csv.gz"
    meta = M5 / f"{sid}.from"
    rows = _read(path)
    have = sorted(rows)
    first = dt.datetime.fromtimestamp(have[0], dt.timezone.utc).date() if have else None
    last = dt.datetime.fromtimestamp(have[-1], dt.timezone.utc).date() if have else None
    try:
        asked_from = dt.date.fromisoformat(meta.read_text().strip())
    except (OSError, ValueError):
        asked_from = None
    ranges = []
    if not have or (first > start + dt.timedelta(days=5) and (asked_from is None or asked_from > start)):
        ranges.append((start, (first - dt.timedelta(days=1)) if have else end))
    if refresh and have and last < end:
        ranges.append((last, end))
    changed = False
    for a0, b0 in ranges:
        a = a0
        while a <= b0:
            b = min(a + dt.timedelta(days=CHUNK_DAYS), b0)
            try:
                d = api.intraday(sid, segment, instrument, a, b + dt.timedelta(days=1), 5)
            except Exception as exc:
                if "auth" in type(exc).__name__.lower():
                    raise
                d = None
            if d and d.get("close"):
                vols = d.get("volume") or [0] * len(d["close"])
                for i in range(len(d["close"])):
                    try:
                        ts = int(d["timestamp"][i])
                        rows[ts] = (ts, float(d["open"][i]), float(d["high"][i]), float(d["low"][i]),
                                    float(d["close"][i]), float(vols[i] or 0))
                        changed = True
                    except (TypeError, ValueError, IndexError):
                        continue
            a = b + dt.timedelta(days=1)
    if changed:
        _write(path, rows)
    if ranges and (asked_from is None or start < asked_from):
        M5.mkdir(parents=True, exist_ok=True)
        meta.write_text(start.isoformat())
    lo = dt.datetime.combine(start, dt.time(0)).timestamp() - 86400
    hi = dt.datetime.combine(end + dt.timedelta(days=1), dt.time(0)).timestamp() + 86400
    return [rows[k] for k in sorted(rows) if lo <= k <= hi]


def download_all(api: Any, stocks: list[Any], start: dt.date, end: dt.date,
                 progress: Callable[[str], None] = print) -> None:
    for i, st in enumerate(stocks):
        if i % 10 == 0:
            progress(f"downloading 5-min candles {i}/{len(stocks)} ({st.symbol})")
        load_m5(api, st.security_id, start, end)
