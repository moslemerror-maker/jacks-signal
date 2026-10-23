from __future__ import annotations

import datetime as dt
import re
import threading
import time
from typing import Any, Mapping

import requests

BASE = "https://api.dhan.co/v2"
TIMEOUT = 20

_MIN_GAP = {"/optionchain": 3.2, "/optionchain/expirylist": 3.2, "/marketfeed/ohlc": 1.1, "/marketfeed/ltp": 1.1}
_DEFAULT_GAP = 0.25


class DhanError(RuntimeError):
    pass


class DhanAuthError(DhanError):
    pass


_JWT = re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")
_AUTH_CODES = ("DH-901", "DH-902", "DH-903", "DH-808", "DH-809", "DH-810")


def find_jwt(body: Any) -> str | None:
    if isinstance(body, str):
        m = _JWT.search(body)
        return m.group(0) if m else None
    if isinstance(body, dict):
        body = list(body.values())
    if isinstance(body, (list, tuple)):
        for v in body:
            t = find_jwt(v)
            if t:
                return t
    return None


class DhanApi:
    def __init__(self, client_id: str, access_token: str, session: requests.Session | None = None) -> None:
        self.client_id = client_id
        self.access_token = access_token
        self._session = session or requests.Session()
        self._lock = threading.Lock()
        self._last_call: dict[str, float] = {}
        self.calls = 0
        self.daily_mode: str | None = None

    def set_token(self, token: str) -> None:
        self.access_token = token

    def renew_token(self) -> str:
        if not self.client_id or not self.access_token:
            raise DhanAuthError("/RenewToken: DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN not set")
        headers = {"access-token": self.access_token, "dhanClientId": self.client_id, "client-id": self.client_id,
                   "Accept": "application/json"}
        with self._lock:
            self.calls += 1
            try:
                r = self._session.get(BASE + "/RenewToken", headers=headers, timeout=TIMEOUT)
                if r.status_code == 405:
                    r = self._session.post(BASE + "/RenewToken", headers=headers, timeout=TIMEOUT)
            except requests.RequestException as exc:
                raise DhanError(f"/RenewToken: network error: {exc}") from exc
        try:
            body = r.json()
        except ValueError:
            body = None
        code = body.get("errorCode") if isinstance(body, dict) else None
        if r.status_code == 401 or code in _AUTH_CODES:
            raise DhanAuthError(f"/RenewToken: refused ({code or r.status_code})")
        if not (200 <= r.status_code < 300):
            msg = body.get("errorMessage") if isinstance(body, dict) else ""
            raise DhanError(f"/RenewToken: HTTP {r.status_code} {code or ''} {msg or ''}".strip())
        tok = find_jwt(body if body is not None else (r.text or ""))
        if not tok:
            raise DhanError("/RenewToken: reply had no token in it")
        return tok

    def _throttle(self, endpoint: str) -> None:
        gap = _MIN_GAP.get(endpoint, _DEFAULT_GAP)
        key = endpoint if endpoint in _MIN_GAP else "_data"
        last = self._last_call.get(key, 0.0)
        wait = last + gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_call[key] = time.monotonic()

    def post(self, endpoint: str, payload: Mapping[str, Any]) -> Any:
        if not self.client_id or not self.access_token:
            raise DhanAuthError("DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN not set")
        headers = {
            "access-token": self.access_token, "client-id": self.client_id, "dhanClientId": self.client_id,
            "Content-Type": "application/json", "Accept": "application/json",
        }
        with self._lock:
            for attempt in range(3):
                self._throttle(endpoint)
                self.calls += 1
                try:
                    r = self._session.post(BASE + endpoint, json=dict(payload), headers=headers, timeout=TIMEOUT)
                except requests.RequestException as exc:
                    if attempt == 2:
                        raise DhanError(f"{endpoint}: network error: {exc}") from exc
                    time.sleep(1.5 * (attempt + 1))
                    continue
                try:
                    body = r.json()
                except ValueError:
                    body = None
                code = (body or {}).get("errorCode") if isinstance(body, dict) else None
                if r.status_code == 401 or code in _AUTH_CODES:
                    raise DhanAuthError(f"{endpoint}: authentication failed ({code or r.status_code}) - regenerate the Dhan access token")
                if r.status_code == 429 or code == "DH-904":
                    time.sleep(2.0 * (attempt + 1))
                    continue
                if not (200 <= r.status_code < 300):
                    msg = (body or {}).get("errorMessage") if isinstance(body, dict) else (r.text[:200] if r.text else "")
                    raise DhanError(f"{endpoint}: HTTP {r.status_code} {code or ''} {msg}")
                if isinstance(body, dict) and body.get("status") == "failure":
                    raise DhanError(f"{endpoint}: {body.get('remarks') or body}")
                return body
            raise DhanError(f"{endpoint}: rate-limited after retries")

    def intraday(self, security_id: int, segment: str, instrument: str, from_date: dt.date, to_date: dt.date,
                 interval: int = 15) -> dict[str, list]:
        body = self.post("/charts/intraday", {
            "securityId": str(security_id), "exchangeSegment": segment, "instrument": instrument,
            "interval": interval, "oi": False, "fromDate": from_date.isoformat(), "toDate": to_date.isoformat(),
        })
        return unwrap_candles(body)

    def daily(self, security_id: int, segment: str, instrument: str, from_date: dt.date, to_date: dt.date) -> dict[str, list]:
        base = {"securityId": str(security_id), "exchangeSegment": segment, "instrument": instrument, "oi": False,
                "fromDate": from_date.isoformat(), "toDate": to_date.isoformat()}
        if self.daily_mode in (None, "historical"):
            for body in ({**base, "expiryCode": 0}, base):
                try:
                    out = unwrap_candles(self.post("/charts/historical", body))
                    if out["close"]:
                        self.daily_mode = "historical"
                        return out
                except DhanAuthError:
                    raise
                except DhanError:
                    continue
            if self.daily_mode == "historical":
                raise DhanError(f"/charts/historical: no daily data for {security_id}")
        self.daily_mode = "intraday60"
        start = max(from_date, to_date - dt.timedelta(days=88))
        return daily_from_intraday(self.intraday(security_id, segment, instrument, start, to_date + dt.timedelta(days=1), 60))

    def ohlc(self, securities: Mapping[str, list[int]]) -> dict[str, dict[str, dict]]:
        body = self.post("/marketfeed/ohlc", {seg: [int(i) for i in ids] for seg, ids in securities.items()})
        return unwrap_feed(body)

    def expiry_list(self, security_id: int, segment: str) -> list[str]:
        body = self.post("/optionchain/expirylist", {"UnderlyingScrip": int(security_id), "UnderlyingSeg": segment})
        data = (body or {}).get("data") if isinstance(body, dict) else body
        if isinstance(data, dict):
            data = data.get("data")
        if not isinstance(data, list):
            raise DhanError(f"expirylist: unexpected shape {str(body)[:200]}")
        return sorted(str(d) for d in data)

    def option_chain(self, security_id: int, segment: str, expiry: str) -> dict[str, Any]:
        body = self.post("/optionchain", {"UnderlyingScrip": int(security_id), "UnderlyingSeg": segment, "Expiry": expiry})
        return unwrap_chain(body)


_CANDLE_KEYS = ("open", "high", "low", "close", "timestamp")


def unwrap_candles(body: Any) -> dict[str, list]:
    d = body
    for _ in range(3):
        if isinstance(d, dict) and all(k in d for k in _CANDLE_KEYS):
            break
        d = d.get("data") if isinstance(d, dict) else None
    if not isinstance(d, dict) or not all(k in d for k in _CANDLE_KEYS):
        raise DhanError(f"candles: unexpected shape {str(body)[:200]}")
    d = dict(d)
    d.setdefault("volume", [0] * len(d["close"]))
    return d


def unwrap_feed(body: Any) -> dict[str, dict[str, dict]]:
    d = body
    for _ in range(3):
        if isinstance(d, dict) and any(k in d for k in ("NSE_EQ", "IDX_I", "NSE_FNO")):
            return d
        d = d.get("data") if isinstance(d, dict) else None
    raise DhanError(f"ohlc: unexpected shape {str(body)[:200]}")


def unwrap_chain(body: Any) -> dict[str, Any]:
    d = body
    for _ in range(3):
        if isinstance(d, dict) and "oc" in d:
            return d
        d = d.get("data") if isinstance(d, dict) else None
    raise DhanError(f"optionchain: unexpected shape {str(body)[:200]}")


def daily_from_intraday(d: dict[str, list]) -> dict[str, list]:
    ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
    days: dict[dt.date, list[float]] = {}
    vols = d.get("volume") or [0] * len(d["close"])
    for i, ts in enumerate(d["timestamp"]):
        day = dt.datetime.fromtimestamp(float(ts), ist).date()
        o, h, l, c, v = (float(d["open"][i]), float(d["high"][i]), float(d["low"][i]), float(d["close"][i]), float(vols[i] or 0))
        if day not in days:
            days[day] = [o, h, l, c, v]
        else:
            b = days[day]
            b[1], b[2], b[3], b[4] = max(b[1], h), min(b[2], l), c, b[4] + v
    keys = sorted(days)
    return {"open": [days[k][0] for k in keys], "high": [days[k][1] for k in keys], "low": [days[k][2] for k in keys],
            "close": [days[k][3] for k in keys], "volume": [days[k][4] for k in keys],
            "timestamp": [dt.datetime.combine(k, dt.time(0, 0), ist).timestamp() for k in keys]}
