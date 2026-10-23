from __future__ import annotations

import dataclasses
import datetime
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class FetchResult:
    path: Path
    created: bool
    sha256: str
    record_count: int
    fetched_at: str
    query_from: datetime.date | None = None
    query_to: datetime.date | None = None
