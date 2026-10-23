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
M1 = CACHE / "m1"
CHUNK_DAYS = 80
MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
WINDOWS = ((dt.time(9, 15), dt.time(11, 30)), (dt.time(15, 0), dt.time(15, 30)))


def universe(today: dt.date) -> list[Any]:
    from signal_desk.instruments import parse_master
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"scrip-master-{today:%Y-%m-%d}.csv"
    if not p.exists():
        r = requests.get(MASTER_URL, timeout=90)
        r.raise_for_status()
        p.write_text(r.text, encoding="utf-8")
    return parse_master(p.read_text(encoding="utf-8"), today).stocks


def _keep(ts: int) -> bool:
    t = dt.datetime.fromtimestamp(ts, IST).time()
    return any(a <= t < b for a, b in WINDOWS)


def read_bars(sid: int) -> dict[int, tuple]:
    rows: dict[int, tuple] = {}
    p = M1 / f"{sid}.csv.gz"
    if p.exists():
        try:
            with gzip.open(p, "rt", newline="") as f:
                for r in csv.reader(f):
                    if len(r) < 6:
                        continue
                    rows[int(r[0])] = (int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]))
        except (OSError, ValueError, IndexError, EOFError):
            rows = {}
    return rows


def read_turnover(sid: int) -> dict[dt.date, float]:
    out: dict[dt.date, float] = {}
    p = M1 / f"{sid}.turn.csv"
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                d, v = line.split(",")
                out[dt.date.fromisoformat(d)] = float(v)
            except ValueError:
                continue
    return out


def _write(sid: int, rows: dict[int, tuple], turn: dict[dt.date, float]) -> None:
    M1.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.writer(buf)
    for k in sorted(rows):
        r = rows[k]
        w.writerow([r[0], f"{r[1]:g}", f"{r[2]:g}", f"{r[3]:g}", f"{r[4]:g}", f"{int(r[5])}"])
    tmp = M1 / f"{sid}.tmp"
    with gzip.open(tmp, "wt", newline="") as f:
        f.write(buf.getvalue())
    tmp.replace(M1 / f"{sid}.csv.gz")
    (M1 / f"{sid}.turn.csv").write_text("".join(f"{d.isoformat()},{v:.0f}\n" for d, v in sorted(turn.items())))


def load_m1(api: Any, sid: int, start: dt.date, end: dt.date) -> None:
    rows, turn = read_bars(sid), read_turnover(sid)
    meta = M1 / f"{sid}.from"
    try:
        asked = dt.date.fromisoformat(meta.read_text().strip())
    except (OSError, ValueError):
        asked = None
    days = sorted(turn)
    ranges = []
    if not days:
        ranges.append((start, end))
    else:
        if days[0] > start + dt.timedelta(days=5) and (asked is None or asked > start):
            ranges.append((start, days[0] - dt.timedelta(days=1)))
        if days[-1] < end:
            ranges.append((days[-1] + dt.timedelta(days=1), end))
    changed = False
    for a0, b0 in ranges:
        a = a0
        while a <= b0:
            b = min(a + dt.timedelta(days=CHUNK_DAYS), b0)
            try:
                d = api.intraday(sid, "NSE_EQ", "EQUITY", a, b + dt.timedelta(days=1), 1)
            except Exception as exc:
                if "auth" in type(exc).__name__.lower():
                    raise
                d = None
            if d and d.get("close"):
                vols = d.get("volume") or [0] * len(d["close"])
                for i in range(len(d["close"])):
                    try:
                        ts = int(d["timestamp"][i])
                        c, v = float(d["close"][i]), float(vols[i] or 0)
                        day = dt.datetime.fromtimestamp(ts, IST).date()
                        turn[day] = turn.get(day, 0.0) + c * v
                        if _keep(ts):
                            rows[ts] = (ts, float(d["open"][i]), float(d["high"][i]), float(d["low"][i]), c, v)
                        changed = True
                    except (TypeError, ValueError, IndexError):
                        continue
            a = b + dt.timedelta(days=1)
    if changed:
        _write(sid, rows, turn)
    if ranges and (asked is None or start < asked):
        M1.mkdir(parents=True, exist_ok=True)
        meta.write_text(start.isoformat() + "\n")


def download_all(api: Any, stocks: list[Any], start: dt.date, end: dt.date, say: Callable[[str], None] = print) -> None:
    for i, st in enumerate(stocks):
        if i % 10 == 0:
            say(f".. 1-min candles {i}/{len(stocks)} ({st.symbol})")
        load_m1(api, st.security_id, start, end)
