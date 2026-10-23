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
_BSE_ACTIONS_URL = "https://api.bseindia.com/BseIndiaAPI/api/DefaultData/w"
_REFERER = "https://www.bseindia.com/corporates/corporate_act.aspx"
_PROVIDER = "BSE (bseindia.com, unofficial JSON API) - investigation/cross-validation only"


def _new_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT, "Accept": "application/json"})
    return session


def _validate_content(content: bytes) -> list:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise InvalidContentError(f"not valid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise InvalidContentError(f"expected a JSON list, got {type(payload).__name__}")
    for i, row in enumerate(payload):
        if not isinstance(row, dict):
            raise InvalidContentError(f"element {i} is not an object")
    return payload


def fetch_bse_corporate_actions_snapshot(
    scrip_code: str,
    symbol: str,
    raw_dir: Path,
    from_date: date,
    to_date: date,
    *,
    purpose: str | None = None,
    session: requests.Session | None = None,
) -> FetchResult:
    if to_date < from_date:
        raise ValueError("to_date must be >= from_date")
    symbol_dir = raw_dir / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "ddlcategorys": "E", "ddlindustrys": "", "segment": "0", "strSearch": "D",
        "Fdate": from_date.strftime("%Y%m%d"), "TDate": to_date.strftime("%Y%m%d"),
        "scripcode": scrip_code,
    }
    response = (session or _new_session()).get(
        _BSE_ACTIONS_URL, params=params, headers={"Referer": _REFERER}, timeout=_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    retrieved_at = datetime.now(timezone.utc)

    content = response.content
    sha256 = hashlib.sha256(content).hexdigest()
    filename = f"{symbol}_{from_date:%Y%m%d}-{to_date:%Y%m%d}_{sha256[:16]}.json"
    dest = symbol_dir / filename
    meta_path = dest.with_suffix(dest.suffix + ".meta.json")

    if dest.exists() and meta_path.exists():
        existing = json.loads(meta_path.read_text())
        return FetchResult(
            dest, False, sha256, existing["record_count"], existing["fetched_at"], from_date, to_date
        )

    try:
        payload = _validate_content(content)
    except InvalidContentError as exc:
        quarantine(
            raw_dir, filename, content, source_url=response.url,
            trade_date=to_date, http_status=response.status_code, reason=str(exc),
        )
        raise

    dest.write_bytes(content)
    meta_path.write_text(
        json.dumps(
            {
                "symbol": symbol,
                "scrip_code": scrip_code,
                "provider": _PROVIDER,
                "source_url": response.url,
                "query_params": params,
                "query_window": {"from_date": from_date.isoformat(), "to_date": to_date.isoformat()},
                "purpose": purpose,
                "fetched_at": retrieved_at.isoformat(),
                "sha256": sha256,
                "http_status": response.status_code,
                "content_length": len(content),
                "record_count": len(payload),
                "note": (
                    "validation evidence only (in-memory reconciliation against NSE); never "
                    "inserted as first-class corporate-action events; carries no ISIN. "
                    "fetched_at is OUR clock."
                ),
            },
            indent=2,
        )
    )
    return FetchResult(dest, True, sha256, len(payload), retrieved_at.isoformat(), from_date, to_date)


def fetch_bse_corporate_actions(
    scrip_code: str, symbol: str, raw_dir: Path, from_date: date, to_date: date,
) -> Path | None:
    result = fetch_bse_corporate_actions_snapshot(scrip_code, symbol, raw_dir, from_date, to_date)
    return result.path if result.created else None
