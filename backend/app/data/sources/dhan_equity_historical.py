from __future__ import annotations

import dataclasses
import json
import re
from datetime import date
from typing import Sequence

import requests

from ..raw.adapter import FetchedContent
from ..raw.schema_registry import ContentInfo
from .errors import InvalidContentError
from .provider_credentials import DhanCredentials

PROVIDER_ID = "dhan"
DATASET = "nse_eq_intraday_1m"
BASE = "https://api.dhan.co/v2"
_ENDPOINT = f"{BASE}/charts/intraday"
_TIMEOUT_SECONDS = 30
_REQUIRED_DATA_KEYS = ("open", "high", "low", "close", "volume", "timestamp")
_KEPT_HEADERS = ("Content-Type", "Content-Length", "Date")

_KEY = re.compile(r"^NSE_EQ-(\d+)-(\d+)m-(\d{8})-(\d{8})$")


@dataclasses.dataclass(frozen=True)
class DhanEquityHistoricalKey:
    security_id: int
    from_date: date
    to_date: date
    interval: int = 1


def source_key_for(security_id: int, from_date: date, to_date: date, interval: int = 1) -> str:
    return f"NSE_EQ-{security_id}-{interval}m-{from_date:%Y%m%d}-{to_date:%Y%m%d}"


def parse_source_key(source_key: str) -> DhanEquityHistoricalKey:
    m = _KEY.match(source_key)
    if not m:
        raise ValueError(f"not a {DATASET} key (expected NSE_EQ-<security_id>-<interval>m-<YYYYMMDD>-<YYYYMMDD>): {source_key!r}")
    security_id, interval, from_s, to_s = m.groups()
    return DhanEquityHistoricalKey(
        security_id=int(security_id), interval=int(interval),
        from_date=date(int(from_s[:4]), int(from_s[4:6]), int(from_s[6:8])),
        to_date=date(int(to_s[:4]), int(to_s[4:6]), int(to_s[6:8])),
    )


def validate_intraday_json_shape(content: bytes) -> None:
    try:
        parsed = json.loads(content)
    except (ValueError, UnicodeDecodeError) as exc:
        raise InvalidContentError(f"not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise InvalidContentError("top-level response is not a JSON object")
    missing = [k for k in _REQUIRED_DATA_KEYS if k not in parsed]
    if missing:
        raise InvalidContentError(f"response is missing required fields: {missing}")
    not_lists = [k for k in _REQUIRED_DATA_KEYS if not isinstance(parsed[k], list)]
    if not_lists:
        raise InvalidContentError(f"response fields are not lists: {not_lists}")
    lengths = {k: len(parsed[k]) for k in _REQUIRED_DATA_KEYS}
    if len(set(lengths.values())) > 1:
        raise InvalidContentError(f"response arrays have inconsistent lengths: {lengths}")


class DhanEquityHistoricalAdapter:

    provider_id = PROVIDER_ID
    fetcher_version = "dhan_equity_historical/1"
    license_status = "UNCLEAR"

    def __init__(self, credentials: DhanCredentials, session: requests.Session | None = None) -> None:
        self._creds = credentials
        self._session = session or requests.Session()

    def datasets(self) -> Sequence[str]:
        return (DATASET,)

    def source_url(self, dataset: str, source_key: str) -> str:
        if dataset != DATASET:
            raise ValueError(f"unknown dataset {dataset!r}")
        parse_source_key(source_key)
        return _ENDPOINT

    def fetch(self, dataset: str, source_key: str) -> FetchedContent | None:
        if dataset != DATASET:
            raise ValueError(f"unknown dataset {dataset!r}")
        key = parse_source_key(source_key)
        body = {
            "securityId": str(key.security_id), "exchangeSegment": "NSE_EQ", "instrument": "EQUITY",
            "interval": key.interval, "oi": False,
            "fromDate": key.from_date.isoformat(), "toDate": key.to_date.isoformat(),
        }
        r = self._session.post(_ENDPOINT, json=body, headers={**self._creds.auth_headers(), "Content-Type": "application/json"}, timeout=_TIMEOUT_SECONDS)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return FetchedContent(r.content, r.status_code, _ENDPOINT, {k: r.headers.get(k) for k in _KEPT_HEADERS}, "application/json")

    def validate(self, dataset: str, content: bytes) -> ContentInfo | None:
        validate_intraday_json_shape(content)
        return None
