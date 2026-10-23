from __future__ import annotations

import datetime as dt
import json
import time
from typing import Callable

import requests

from .data import CACHE, IST

NEWS = CACHE / "news"
HOME = "https://www.nseindia.com"
API = HOME + "/api/corporate-announcements?index=equities&from_date={a:%d-%m-%Y}&to_date={b:%d-%m-%Y}"
REFERER = HOME + "/companies-listing/corporate-filings-announcements"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/124.0 Safari/537.36",
           "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9"}
ROUTINE = ("trading window", "newspaper", "analysts/institutional", "analyst / investor meet", "analyst meet",
           "investor meet", "loss of share", "duplicate share", "certificate under", "reg. 74", "regulation 74(5)",
           "compliance certificate", "esop", "esos", "change in registrar", "postal ballot", "book closure",
           "record date", "annual general meeting", "voting results", "scrutinizer", "closure of trading",
           "shareholders meeting", "earnings call", "conference call", "transcript", "audio recording",
           "board meeting intimation", "general updates")


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    s.get(HOME, timeout=30)
    return s


def fetch(s: requests.Session, a: dt.date, b: dt.date) -> list[dict]:
    r = s.get(API.format(a=a, b=b), headers={"Referer": REFERER}, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else (data.get("data") or [])


def download(start: dt.date, end: dt.date, say: Callable[[str], None] = print) -> int:
    NEWS.mkdir(parents=True, exist_ok=True)
    s, n, a = None, 0, start
    today = dt.date.today()
    while a <= end:
        b = min(a + dt.timedelta(days=6), end)
        p = NEWS / f"{a:%Y%m%d}.json"
        if not p.exists():
            for attempt in range(4):
                try:
                    if s is None or attempt:
                        s = _session()
                        time.sleep(1.0)
                    rows = fetch(s, a, b)
                    if b < today:
                        p.write_text(json.dumps(rows), encoding="utf-8")
                    n += len(rows)
                    say(f".. NSE announcements {a} .. {b}: {len(rows)}")
                    break
                except Exception as exc:
                    say(f"   NSE busy/blocked ({str(exc)[:80]}) - retrying in {5 * (attempt + 1)} s")
                    time.sleep(5 * (attempt + 1))
            time.sleep(1.5)
        a = b + dt.timedelta(days=1)
    return n


def _when(row: dict) -> dt.datetime | None:
    for key in ("an_dt", "exchdisstime", "sort_date", "dt"):
        v = (row.get(key) or "").strip()
        for fmt in ("%d-%b-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d-%b-%Y %H:%M", "%d%m%Y%H%M%S"):
            try:
                return dt.datetime.strptime(v, fmt).replace(tzinfo=IST)
            except ValueError:
                continue
    return None


def routine(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ROUTINE)


def load_events() -> list[tuple[str, dt.datetime, bool]]:
    out = []
    for p in sorted(NEWS.glob("*.json")):
        try:
            rows = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        for r in rows:
            sym, t = (r.get("symbol") or "").strip().upper(), _when(r)
            if sym and t:
                text = f"{r.get('desc') or ''} {r.get('attchmntText') or ''}"
                out.append((sym, t, not routine(r.get("desc") or "") and not routine(text[:200])))
    return out


def check(say: Callable[[str], None] = print) -> bool:
    end = dt.date.today()
    try:
        rows = fetch(_session(), end - dt.timedelta(days=3), end)
    except Exception as exc:
        say(f"NSE check FAILED: {exc}")
        return False
    say(f"NSE check OK: {len(rows)} announcements in the last 3 days")
    for r in rows[:5]:
        say(f"   {r.get('symbol')}  {_when(r)}  {(r.get('desc') or '')[:70]}  material={not routine(r.get('desc') or '')}")
    if rows:
        say(f"   fields: {sorted(rows[0].keys())}")
    return bool(rows)
