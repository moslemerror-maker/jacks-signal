from __future__ import annotations

import dataclasses
import datetime as dt
import enum
from typing import Mapping, Protocol, Sequence, runtime_checkable

from ..sources.errors import InvalidContentError
from .schema_registry import ContentInfo, SchemaRegistry
from .store import PutStatus, RawStore


@dataclasses.dataclass(frozen=True)
class FetchedContent:
    content: bytes
    http_status: int
    source_url: str
    headers: Mapping[str, str | None]
    media_type: str | None = None


@runtime_checkable
class ProviderAdapter(Protocol):
    provider_id: str
    fetcher_version: str
    license_status: str

    def datasets(self) -> Sequence[str]: ...

    def source_url(self, dataset: str, source_key: str) -> str: ...

    def fetch(self, dataset: str, source_key: str) -> FetchedContent | None:
        pass

    def validate(self, dataset: str, content: bytes) -> ContentInfo | None:
        pass


class IngestStatus(enum.Enum):
    CREATED = "CREATED"
    UNCHANGED = "UNCHANGED"
    NEW_VERSION = "NEW_VERSION"
    SKIPPED_EXISTING = "SKIPPED_EXISTING"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    QUARANTINED = "QUARANTINED"


@dataclasses.dataclass(frozen=True)
class IngestResult:
    status: IngestStatus
    provider: str
    dataset: str
    source_key: str
    sha256: str | None = None
    version: int | None = None
    reason: str | None = None


def ingest(
    store: RawStore, adapter: ProviderAdapter, dataset: str, source_key: str, *, refresh: bool = False, now: dt.datetime | None = None,
) -> IngestResult:
    p = adapter.provider_id
    if dataset not in adapter.datasets():
        raise ValueError(f"{p} has no dataset {dataset!r}")
    if not refresh:
        have = store.latest(p, dataset, source_key)
        if have is not None:
            return IngestResult(IngestStatus.SKIPPED_EXISTING, p, dataset, source_key, have.sha256, have.version)
    got = adapter.fetch(dataset, source_key)
    if got is None:
        return IngestResult(IngestStatus.NOT_AVAILABLE, p, dataset, source_key, reason="provider reports the artifact does not exist")
    try:
        info = adapter.validate(dataset, got.content)
    except InvalidContentError as exc:
        store.quarantine(provider=p, dataset=dataset, source_key=source_key, content=got.content, source_url=got.source_url, http_status=got.http_status,
                         reason=str(exc), response_headers=got.headers, quarantined_at=now)
        return IngestResult(IngestStatus.QUARANTINED, p, dataset, source_key, reason=str(exc))
    put = store.put(
        provider=p, dataset=dataset, source_key=source_key, content=got.content, source_url=got.source_url, http_status=got.http_status,
        response_headers=got.headers, fetcher_version=adapter.fetcher_version, media_type=got.media_type,
        schema_id=info.schema_id if info else None, format_version=info.format_version if info else None, license_status=adapter.license_status, fetched_at=now,
    )
    return IngestResult(IngestStatus(put.status.value), p, dataset, source_key, put.meta.sha256, put.meta.version)
