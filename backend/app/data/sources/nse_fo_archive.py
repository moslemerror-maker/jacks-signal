from __future__ import annotations

import datetime as dt
import re
from typing import Sequence

from ..raw.adapter import FetchedContent
from ..raw.schema_registry import ContentInfo, SchemaRegistry, default_registry
from . import nse_bhavcopy

PROVIDER_ID = "nse_archives"
BASE = "https://nsearchives.nseindia.com/content"
_MON = {1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN", 7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC"}
_LEGACY_KEY = re.compile(r"^fo(\d{2})([A-Z]{3})(\d{4})bhav\.csv\.zip$")
_UDIFF_KEY = re.compile(r"^BhavCopy_NSE_FO_0_0_0_(\d{8})_F_0000\.csv\.zip$")
_MASTER_KEY = re.compile(r"^NSE_FO_contract_(\d{8})\.csv\.gz$")
_MEDIA = {"fo_bhavcopy_legacy": "application/zip", "fo_bhavcopy_udiff": "application/zip", "fo_contract_master": "application/gzip"}
_REFERER = "https://www.nseindia.com/all-reports-derivatives"
_TIMEOUT_SECONDS = 40
_KEPT_HEADERS = ("Last-Modified", "ETag", "Date", "Content-Type", "Content-Length", "Server", "Age", "Cache-Control")


def source_key_for(dataset: str, day: dt.date) -> str:
    if dataset == "fo_bhavcopy_legacy":
        return f"fo{day.day:02d}{_MON[day.month]}{day.year}bhav.csv.zip"
    if dataset == "fo_bhavcopy_udiff":
        return f"BhavCopy_NSE_FO_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"
    if dataset == "fo_contract_master":
        return f"NSE_FO_contract_{day:%d%m%Y}.csv.gz"
    raise ValueError(f"unknown dataset {dataset!r}")


class NseFoArchiveAdapter:
    provider_id = PROVIDER_ID
    fetcher_version = "nse_fo_archive/1"
    license_status = "UNCLEAR"

    def __init__(self, session=None, registry: SchemaRegistry | None = None) -> None:
        self._session = session
        self._registry = registry or default_registry()

    def datasets(self) -> Sequence[str]:
        return tuple(_MEDIA)

    def source_url(self, dataset: str, source_key: str) -> str:
        if dataset == "fo_bhavcopy_legacy":
            m = _LEGACY_KEY.match(source_key)
            if not m:
                raise ValueError(f"not a legacy F&O bhavcopy key: {source_key!r}")
            return f"{BASE}/historical/DERIVATIVES/{m.group(3)}/{m.group(2)}/{source_key}"
        pattern = {"fo_bhavcopy_udiff": _UDIFF_KEY, "fo_contract_master": _MASTER_KEY}.get(dataset)
        if pattern is None:
            raise ValueError(f"unknown dataset {dataset!r}")
        if not pattern.match(source_key):
            raise ValueError(f"not a {dataset} key: {source_key!r}")
        return f"{BASE}/fo/{source_key}"

    def fetch(self, dataset: str, source_key: str) -> FetchedContent | None:
        url = self.source_url(dataset, source_key)
        session = self._session or nse_bhavcopy._get_session()
        r = session.get(url, headers={"User-Agent": nse_bhavcopy._USER_AGENT, "Referer": _REFERER}, timeout=_TIMEOUT_SECONDS)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return FetchedContent(r.content, r.status_code, url, {k: r.headers.get(k) for k in _KEPT_HEADERS}, _MEDIA[dataset])

    def validate(self, dataset: str, content: bytes) -> ContentInfo | None:
        return self._registry.inspect(PROVIDER_ID, dataset, content)
