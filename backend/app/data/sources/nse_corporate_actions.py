from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import requests

from backend.app.data.sources.errors import InvalidContentError
from backend.app.data.sources.fetch_result import FetchResult
from backend.app.data.sources.quarantine import quarantine

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
_TIMEOUT_SECONDS = 30
_CORPORATE_ACTIONS_URL = "https://www.nseindia.com/api/corporates-corporateActions"
_REFERER = "https://www.nseindia.com/companies-listing/corporate-filings-actions"
_PROVIDER = "NSE (nseindia.com, unofficial JSON API)"


def _new_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})
    session.get("https://www.nseindia.com/", timeout=_TIMEOUT_SECONDS)
    return session


def _validate_corporate_actions_content(content: bytes) -> list:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise InvalidContentError(f"not valid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise InvalidContentError(f"expected a JSON list, got {type(payload).__name__}")
    for i, row in enumerate(payload):
        if not isinstance(row, dict):
            raise InvalidContentError(
                f"expected every element to be a JSON object, element {i} is "
                f"{type(row).__name__}"
            )
    return payload


def open_session() -> requests.Session:
    return _new_session()


def _window_slug(from_date: date | None, to_date: date | None) -> str:
    return f"{from_date:%Y%m%d}-{to_date:%Y%m%d}"


def fetch_corporate_actions_snapshot(
    symbol: str,
    raw_dir: Path,
    *,
    from_date: date | None = None,
    to_date: date | None = None,
    purpose: str | None = None,
    session: requests.Session | None = None,
) -> FetchResult:
    if (from_date is None) != (to_date is None):
        raise ValueError("from_date and to_date must be given together or not at all")
    if from_date is not None and to_date < from_date:
        raise ValueError("to_date must be >= from_date")

    symbol_dir = raw_dir / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)

    params: dict[str, str] = {"index": "equities", "symbol": symbol}
    if from_date is not None:
        params["from_date"] = from_date.strftime("%d-%m-%Y")
        params["to_date"] = to_date.strftime("%d-%m-%Y")

    session = session or _new_session()
    response = session.get(
        _CORPORATE_ACTIONS_URL,
        params=params,
        headers={"Accept": "application/json", "Referer": _REFERER},
        timeout=_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    retrieved_at = datetime.now(timezone.utc)

    content = response.content
    sha256 = hashlib.sha256(content).hexdigest()
    prefix = f"{symbol}_{_window_slug(from_date, to_date)}" if from_date is not None else symbol
    filename = f"{prefix}_{sha256[:16]}.json"
    dest = symbol_dir / filename
    meta_path = dest.with_suffix(dest.suffix + ".meta.json")

    if dest.exists() and meta_path.exists():
        existing = json.loads(meta_path.read_text())
        return FetchResult(
            dest, False, sha256, existing["record_count"], existing["fetched_at"], from_date, to_date
        )

    try:
        payload = _validate_corporate_actions_content(content)
    except InvalidContentError as exc:
        quarantine(
            raw_dir,
            filename,
            content,
            source_url=response.url,
            trade_date=to_date or date.today(),
            http_status=response.status_code,
            reason=str(exc),
        )
        raise

    announcement_date_observed = any(
        isinstance(row, dict) and row.get("caBroadcastDate") for row in payload
    )

    dest.write_bytes(content)
    meta_path.write_text(
        json.dumps(
            {
                "symbol": symbol,
                "provider": _PROVIDER,
                "source_url": response.url,
                "query_params": params,
                "fetch_mode": "windowed" if from_date is not None else "default",
                "query_window": (
                    {"from_date": from_date.isoformat(), "to_date": to_date.isoformat()}
                    if from_date is not None
                    else None
                ),
                "purpose": purpose,
                "fetched_at": retrieved_at.isoformat(),
                "sha256": sha256,
                "http_status": response.status_code,
                "content_length": len(content),
                "record_count": len(payload),
                "announcement_date_observed": announcement_date_observed,
                "parsing_applied": False,
                "schema_confirmed": False,
                "note": (
                    "raw storage only - no field interpretation, no ratio "
                    "parsing, no adjustment factors. fetched_at is OUR clock, "
                    "not a claim about when the source published this data."
                ),
            },
            indent=2,
        )
    )
    return FetchResult(dest, True, sha256, len(payload), retrieved_at.isoformat(), from_date, to_date)


def fetch_corporate_actions(
    symbol: str,
    raw_dir: Path,
    *,
    from_date: date | None = None,
    to_date: date | None = None,
) -> Path | None:
    result = fetch_corporate_actions_snapshot(
        symbol, raw_dir, from_date=from_date, to_date=to_date
    )
    return result.path if result.created else None
