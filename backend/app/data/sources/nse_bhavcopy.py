from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from backend.app.data.normalizers import UDIFF_CUTOVER_DATE
from backend.app.data.normalizers._io import read_csv_from_zip
from backend.app.data.sources.errors import InvalidContentError
from backend.app.data.sources.quarantine import quarantine

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
_REFERER = "https://www.nseindia.com/all-reports"
_TIMEOUT_SECONDS = 30

_RETRY = Retry(
    total=5,
    connect=5,
    read=5,
    backoff_factor=2,
    status_forcelist=(500, 502, 503, 504),
    allowed_methods=("GET",),
)

_session: requests.Session | None = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        adapter = HTTPAdapter(max_retries=_RETRY)
        _session.mount("https://", adapter)
        _session.mount("http://", adapter)
    return _session


def _legacy_url(trade_date: date) -> str:
    day = trade_date.strftime("%d")
    mon = trade_date.strftime("%b").upper()
    year = trade_date.strftime("%Y")
    filename = f"cm{day}{mon}{year}bhav.csv.zip"
    return (
        f"https://nsearchives.nseindia.com/content/historical/EQUITIES/"
        f"{year}/{mon}/{filename}"
    )


def _udiff_url(trade_date: date) -> str:
    ymd = trade_date.strftime("%Y%m%d")
    filename = f"BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip"
    return f"https://nsearchives.nseindia.com/content/cm/{filename}"


def bhavcopy_url(trade_date: date) -> str:
    if trade_date >= UDIFF_CUTOVER_DATE:
        return _udiff_url(trade_date)
    return _legacy_url(trade_date)


def _validate_bhavcopy_content(content: bytes) -> None:
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        read_csv_from_zip(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def fetch_bhavcopy(trade_date: date, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = bhavcopy_url(trade_date)
    filename = url.rsplit("/", 1)[-1]
    dest = raw_dir / filename
    meta_path = dest.with_suffix(dest.suffix + ".meta.json")

    if dest.exists() and meta_path.exists():
        return dest

    response = _get_session().get(
        url,
        headers={"User-Agent": _USER_AGENT, "Referer": _REFERER},
        timeout=_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    try:
        _validate_bhavcopy_content(response.content)
    except Exception as exc:
        quarantine(
            raw_dir,
            filename,
            response.content,
            source_url=url,
            trade_date=trade_date,
            http_status=response.status_code,
            reason=str(exc),
        )
        raise InvalidContentError(f"{filename}: {exc}") from exc

    dest.write_bytes(response.content)
    meta_path.write_text(
        json.dumps(
            {
                "trade_date": trade_date.isoformat(),
                "source_url": url,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "sha256": hashlib.sha256(response.content).hexdigest(),
                "http_status": response.status_code,
                "content_length": len(response.content),
            },
            indent=2,
        )
    )
    return dest
