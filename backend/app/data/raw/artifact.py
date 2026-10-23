from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, Mapping

SIDECAR_VERSION = 1
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._=-]{0,127}$")

REQUIRED_FIELDS = (
    "sidecar_version", "provider", "dataset", "source_key", "version", "source_url", "fetched_at", "http_status", "sha256", "size",
    "media_type", "response_headers", "fetcher_version", "schema_id", "format_version", "supersedes_sha256", "license_status",
)
LICENSE_STATUSES = ("CLEAR", "UNCLEAR", "REQUIRES_PROVIDER_CONFIRMATION")


class RawIntegrityError(Exception):
    pass


def check_component(name: str, value: str) -> str:
    if not isinstance(value, str) or not _SAFE.match(value) or ".." in value:
        raise ValueError(f"unsafe {name}: {value!r} (allowed: letters, digits, . _ = - ; no separators, must start alphanumeric, no '..')")
    return value


@dataclasses.dataclass(frozen=True)
class RawArtifactMeta:

    provider: str
    dataset: str
    source_key: str
    version: int
    source_url: str
    fetched_at: str
    http_status: int
    sha256: str
    size: int
    media_type: str | None
    response_headers: Mapping[str, str | None]
    fetcher_version: str
    schema_id: str | None = None
    format_version: str | None = None
    supersedes_sha256: str | None = None
    license_status: str = "UNCLEAR"
    sidecar_version: int = SIDECAR_VERSION

    def __post_init__(self) -> None:
        check_component("provider", self.provider)
        check_component("dataset", self.dataset)
        check_component("source_key", self.source_key)
        if self.version < 1:
            raise ValueError("version starts at 1")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        if self.size < 0:
            raise ValueError("size must be >= 0")
        if self.license_status not in LICENSE_STATUSES:
            raise ValueError(f"license_status must be one of {LICENSE_STATUSES}")
        if (self.version == 1) != (self.supersedes_sha256 is None):
            raise ValueError("version 1 supersedes nothing; every later version must name the sha256 it supersedes")

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "RawArtifactMeta":
        raw: dict[str, Any] = json.loads(text)
        missing = [f for f in REQUIRED_FIELDS if f not in raw]
        if missing:
            raise RawIntegrityError(f"sidecar is missing required fields: {missing}")
        unknown = set(raw) - set(REQUIRED_FIELDS)
        if unknown:
            raise RawIntegrityError(f"sidecar has unknown fields (written by a newer version?): {sorted(unknown)}")
        return cls(**raw)
