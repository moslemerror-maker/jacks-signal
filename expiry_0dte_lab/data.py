from __future__ import annotations

import datetime as dt
import gzip
import json
import time
from pathlib import Path

from signal_desk.clock import from_epoch

LAB_DIR = Path(__file__).resolve().parent
CACHE = LAB_DIR / "cache"
NIFTY_ID = 13
WINDOW = 29
INTERVAL = 5


def offsets(n: int) -> list[str]:
    return ["ATM"] + [f"ATM+{i}" for i in range(1, n + 1)] + [f"ATM-{i}" for i in range(1, n + 1)]


def _key(code: int, typ: str, off: str, a: dt.date) -> Path:
    return CACHE / f"c{code}_{typ}_{off.replace('+', 'p').replace('-', 'm')}_{a:%Y%m%d}.json.gz"


def _day_path(code: int, typ: str, off: str, day: dt.date) -> Path:
    return CACHE / f"d{code}_{typ}_{off.replace('+', 'p').replace('-', 'm')}_{day:%Y%m%d}.json.gz"


TRANSIENT = ("network", "rate", "timeout", "timed out", "http 500", "http 502", "http 503", "http 504")


def _transient(msg: str) -> bool:
    return any(w in msg.lower() for w in TRANSIENT)


def _read(p: Path):
    if not p.exists():
        return None
    body = json.loads(gzip.decompress(p.read_bytes()))
    if isinstance(body, dict) and _transient(str(body.get("error", ""))):
        return None
    return body


def is_cached(code: int, typ: str, off: str, day: dt.date) -> bool:
    return _read(_day_path(code, typ, off, day)) is not None


def cached_day(code: int, typ: str, off: str, day: dt.date) -> dict:
    body = _read(_day_path(code, typ, off, day))
    return parse(body, typ) if body is not None else {}


def fetch_day(api, code: int, typ: str, off: str, day: dt.date, say=print) -> dict:
    p = _day_path(code, typ, off, day)
    cached = _read(p)
    if cached is not None:
        return parse(cached, typ)
    body = None
    for attempt in range(4):
        try:
            body = api.post("/charts/rollingoption", {
                "exchangeSegment": "NSE_FNO", "interval": str(INTERVAL), "securityId": NIFTY_ID, "instrument": "OPTIDX",
                "expiryFlag": "WEEK", "expiryCode": code, "strike": off, "drvOptionType": typ,
                "requiredData": ["close", "strike", "spot"], "fromDate": day.isoformat(),
                "toDate": (day + dt.timedelta(days=1)).isoformat()})
            break
        except Exception as exc:
            if "auth" in type(exc).__name__.lower():
                raise
            msg = str(exc)
            if _transient(msg) and attempt < 3:
                say(f"   .. Dhan busy ({msg[:40].strip()}) for {typ} {off} {day} - waiting {10 * (attempt + 1)} s, retrying")
                time.sleep(10 * (attempt + 1))
                continue
            say(f"   ! {typ} {off} code {code} {day}: {msg[:80]}")
            body = {"error": msg[:300]}
            if _transient(msg):
                return {}
            break
    if day < dt.date.today():
        CACHE.mkdir(parents=True, exist_ok=True)
        p.write_bytes(gzip.compress(json.dumps(body).encode()))
    return parse(body, typ)


def fetch_series(api, code: int, typ: str, off: str, start: dt.date, end: dt.date, say=print) -> dict:
    out: dict = {}
    a = start
    while a <= end:
        b = min(a + dt.timedelta(days=WINDOW), end + dt.timedelta(days=1))
        p = _key(code, typ, off, a)
        body = None
        if p.exists():
            body = json.loads(gzip.decompress(p.read_bytes()))
        else:
            try:
                body = api.post("/charts/rollingoption", {
                    "exchangeSegment": "NSE_FNO", "interval": str(INTERVAL), "securityId": NIFTY_ID, "instrument": "OPTIDX",
                    "expiryFlag": "WEEK", "expiryCode": code, "strike": off, "drvOptionType": typ,
                    "requiredData": ["close", "strike", "spot"], "fromDate": a.isoformat(), "toDate": b.isoformat()})
            except Exception as exc:
                if "auth" in type(exc).__name__.lower():
                    raise
                say(f"   ! {typ} {off} code {code} {a}: {exc}")
                body = {"error": str(exc)}
            transient = "error" in body and any(w in body["error"].lower() for w in ("network", "rate", "timeout"))
            if b <= dt.date.today() and not transient:
                CACHE.mkdir(parents=True, exist_ok=True)
                p.write_bytes(gzip.compress(json.dumps(body).encode()))
        out.update(parse(body, typ))
        a = b
    return out


def parse(body, typ: str) -> dict:
    d = (body or {}).get("data") if isinstance(body, dict) else None
    if isinstance(d, dict) and "data" in d and isinstance(d["data"], dict):
        d = d["data"]
    side = (d or {}).get("ce" if typ == "CALL" else "pe") if isinstance(d, dict) else None
    if not isinstance(side, dict):
        return {}
    ts, cl = side.get("timestamp") or [], side.get("close") or []
    sk, sp = side.get("strike") or [], side.get("spot") or []
    out = {}
    for i, t in enumerate(ts):
        try:
            out[from_epoch(t)] = (float(sk[i]) if i < len(sk) and sk[i] is not None else None, float(cl[i]),
                                  float(sp[i]) if i < len(sp) and sp[i] is not None else None)
        except (TypeError, ValueError, IndexError):
            continue
    return out
