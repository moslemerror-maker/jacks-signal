from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import requests

from backend.app.data.sources.errors import InvalidContentError
from backend.app.data.sources.quarantine import quarantine

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
_TIMEOUT_SECONDS = 30
_INDEX_HISTORY_URL = "https://www.nseindia.com/api/historicalOR/indicesHistory"
_REFERER = "https://www.nseindia.com/reports-indices-historical-index-data"


def _new_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})
    session.get("https://www.nseindia.com/", timeout=_TIMEOUT_SECONDS)
    return session


def _validate_index_history_content(content: bytes) -> None:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise InvalidContentError(f"not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise InvalidContentError(f"expected a JSON object, got {type(payload).__name__}")
    if "data" not in payload:
        raise InvalidContentError('response has no "data" key')
    if not isinstance(payload["data"], list):
        raise InvalidContentError(
            f'"data" is not a list (got {type(payload["data"]).__name__})'
        )


def fetch_index_history(
    index_type: str, from_date: date, to_date: date, raw_dir: Path
) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    slug = index_type.replace(" ", "_")
    filename = f"{slug}_{from_date.isoformat()}_{to_date.isoformat()}.json"
    dest = raw_dir / filename
    meta_path = dest.with_suffix(dest.suffix + ".meta.json")

    if dest.exists() and meta_path.exists():
        return dest

    session = _new_session()
    response = session.get(
        _INDEX_HISTORY_URL,
        params={
            "indexType": index_type,
            "from": from_date.strftime("%d-%m-%Y"),
            "to": to_date.strftime("%d-%m-%Y"),
        },
        headers={"Accept": "application/json", "Referer": _REFERER},
        timeout=_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    try:
        _validate_index_history_content(response.content)
    except InvalidContentError as exc:
        quarantine(
            raw_dir,
            filename,
            response.content,
            source_url=response.url,
            trade_date=from_date,
            http_status=response.status_code,
            reason=str(exc),
        )
        raise

    dest.write_bytes(response.content)
    meta_path.write_text(
        json.dumps(
            {
                "index_type": index_type,
                "from_date": from_date.isoformat(),
                "to_date": to_date.isoformat(),
                "source_url": response.url,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "sha256": hashlib.sha256(response.content).hexdigest(),
                "http_status": response.status_code,
                "content_length": len(response.content),
            },
            indent=2,
        )
    )
    return dest
