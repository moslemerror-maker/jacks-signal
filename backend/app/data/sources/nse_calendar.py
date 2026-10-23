from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
_TIMEOUT_SECONDS = 30
_HOLIDAY_URL = "https://www.nseindia.com/api/holiday-master?type=trading"

CM_SEGMENT_KEY = "CM"

FO_SEGMENT_KEY = "FO"


def fetch_holiday_master(raw_dir: Path, *, force: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / "holiday_master.json"
    meta_path = dest.with_suffix(dest.suffix + ".meta.json")

    if dest.exists() and meta_path.exists() and not force:
        return dest

    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})
    session.get("https://www.nseindia.com/", timeout=_TIMEOUT_SECONDS)

    response = session.get(
        _HOLIDAY_URL,
        headers={"Accept": "application/json", "Referer": "https://www.nseindia.com/"},
        timeout=_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    dest.write_bytes(response.content)
    meta_path.write_text(
        json.dumps(
            {
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


def equity_holiday_dates(holiday_master_path: Path) -> set[str]:
    payload = json.loads(holiday_master_path.read_text())
    return {row["tradingDate"] for row in payload.get(CM_SEGMENT_KEY, [])}


def fo_holiday_dates(holiday_master_path: Path) -> set[str]:
    payload = json.loads(holiday_master_path.read_text())
    return {row["tradingDate"] for row in payload.get(FO_SEGMENT_KEY, [])}
