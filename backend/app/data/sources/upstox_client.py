from __future__ import annotations

import dataclasses
import urllib.parse
from typing import Any

import requests

from .provider_credentials import UpstoxCredentials

BASE = "https://api.upstox.com/v2"
_TIMEOUT_SECONDS = 30
_ACCEPT = "application/json"


@dataclasses.dataclass(frozen=True)
class ApiResponse:
    status_code: int
    content: bytes
    headers: dict[str, str | None]
    json: Any | None


def expiries_url(instrument_key: str) -> str:
    return f"{BASE}/expired-instruments/expiries?instrument_key={urllib.parse.quote(instrument_key, safe='')}"


def expired_option_contracts_url(instrument_key: str, expiry_date: str) -> str:
    return f"{BASE}/expired-instruments/option/contract?instrument_key={urllib.parse.quote(instrument_key, safe='')}&expiry_date={expiry_date}"


def expired_candle_url(expired_instrument_key: str, interval: str, to_date: str, from_date: str) -> str:
    if interval not in ("1minute", "3minute", "5minute", "15minute", "30minute", "day"):
        raise ValueError(f"unsupported interval {interval!r}")
    key = urllib.parse.quote(expired_instrument_key, safe="")
    return f"{BASE}/expired-instruments/historical-candle/{key}/{interval}/{to_date}/{from_date}"


class UpstoxClient:

    def __init__(self, credentials: UpstoxCredentials, session: requests.Session | None = None) -> None:
        self._creds = credentials
        self._session = session or requests.Session()

    def _get(self, url: str) -> ApiResponse:
        r = self._session.get(url, headers={**self._creds.auth_header(), "Accept": _ACCEPT}, timeout=_TIMEOUT_SECONDS)
        try:
            body = r.json()
        except ValueError:
            body = None
        return ApiResponse(r.status_code, r.content, dict(r.headers), body)

    def expiries(self, instrument_key: str) -> ApiResponse:
        return self._get(expiries_url(instrument_key))

    def expired_option_contracts(self, instrument_key: str, expiry_date: str) -> ApiResponse:
        return self._get(expired_option_contracts_url(instrument_key, expiry_date))

    def expired_candles(self, expired_instrument_key: str, interval: str, to_date: str, from_date: str) -> ApiResponse:
        return self._get(expired_candle_url(expired_instrument_key, interval, to_date, from_date))
