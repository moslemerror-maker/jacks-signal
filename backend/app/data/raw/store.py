from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import hashlib
import json
import os
import pathlib
from typing import Iterator, Mapping

from .artifact import RawArtifactMeta, RawIntegrityError, check_component

QUARANTINE_DIR = "_quarantine"
MANIFEST_NAME = "MANIFEST.jsonl"
_SIDECAR_SUFFIX = ".meta.json"


class PutStatus(enum.Enum):
    CREATED = "CREATED"
    UNCHANGED = "UNCHANGED"
    NEW_VERSION = "NEW_VERSION"


@dataclasses.dataclass(frozen=True)
class PutResult:
    status: PutStatus
    path: pathlib.Path
    meta: RawArtifactMeta


@dataclasses.dataclass(frozen=True)
class Problem:
    path: str
    kind: str
    detail: str


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class RawStore:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = pathlib.Path(root)

    def _dir(self, provider: str, dataset: str) -> pathlib.Path:
        return self.root / check_component("provider", provider) / check_component("dataset", dataset)

    def artifact_path(self, provider: str, dataset: str, source_key: str, version: int = 1) -> pathlib.Path:
        check_component("source_key", source_key)
        name = source_key if version == 1 else f"{source_key}.v{version}"
        return self._dir(provider, dataset) / name

    @staticmethod
    def sidecar_path(path: pathlib.Path) -> pathlib.Path:
        return path.with_name(path.name + _SIDECAR_SUFFIX)

    def versions(self, provider: str, dataset: str, source_key: str) -> list[RawArtifactMeta]:
        out: list[RawArtifactMeta] = []
        n = 1
        while True:
            path = self.artifact_path(provider, dataset, source_key, n)
            side = self.sidecar_path(path)
            if not path.exists() and not side.exists():
                break
            if path.exists() and not side.exists():
                raise RawIntegrityError(f"{path} exists without a provenance sidecar; refusing to trust or overwrite it")
            if side.exists() and not path.exists():
                raise RawIntegrityError(f"{side} exists without its data file")
            meta = RawArtifactMeta.from_json(side.read_text(encoding="utf-8"))
            if (meta.provider, meta.dataset, meta.source_key, meta.version) != (provider, dataset, source_key, n):
                raise RawIntegrityError(f"{side} describes {(meta.provider, meta.dataset, meta.source_key, meta.version)}, not the path it sits at")
            if n > 1 and meta.supersedes_sha256 != out[-1].sha256:
                raise RawIntegrityError(f"{side}: supersedes_sha256 does not match version {n - 1}")
            out.append(meta)
            n += 1
        return out

    def latest(self, provider: str, dataset: str, source_key: str) -> RawArtifactMeta | None:
        v = self.versions(provider, dataset, source_key)
        return v[-1] if v else None

    def exists(self, provider: str, dataset: str, source_key: str) -> bool:
        return self.latest(provider, dataset, source_key) is not None

    def read_bytes(self, provider: str, dataset: str, source_key: str, version: int | None = None) -> bytes:
        vs = self.versions(provider, dataset, source_key)
        if not vs:
            raise FileNotFoundError(f"{provider}/{dataset}/{source_key}")
        meta = vs[-1] if version is None else vs[version - 1]
        data = self.artifact_path(provider, dataset, source_key, meta.version).read_bytes()
        if sha256_bytes(data) != meta.sha256 or len(data) != meta.size:
            raise RawIntegrityError(f"{provider}/{dataset}/{source_key} v{meta.version} does not match its recorded sha256/size")
        return data

    def put(
        self, *, provider: str, dataset: str, source_key: str, content: bytes, source_url: str, http_status: int,
        response_headers: Mapping[str, str | None], fetcher_version: str, media_type: str | None = None,
        schema_id: str | None = None, format_version: str | None = None, license_status: str = "UNCLEAR", fetched_at: dt.datetime | None = None,
    ) -> PutResult:
        existing = self.versions(provider, dataset, source_key)
        digest = sha256_bytes(content)
        if existing:
            latest = existing[-1]
            path = self.artifact_path(provider, dataset, source_key, latest.version)
            data = path.read_bytes()
            if sha256_bytes(data) != latest.sha256:
                raise RawIntegrityError(f"{path.name} does not match its recorded sha256; raw files are immutable - investigate, do not overwrite")
            if latest.sha256 == digest:
                return PutResult(PutStatus.UNCHANGED, path, latest)
            version, supersedes, status = latest.version + 1, latest.sha256, PutStatus.NEW_VERSION
        else:
            version, supersedes, status = 1, None, PutStatus.CREATED
        when = (fetched_at or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
        meta = RawArtifactMeta(
            provider=provider, dataset=dataset, source_key=source_key, version=version, source_url=source_url, fetched_at=when.isoformat(),
            http_status=http_status, sha256=digest, size=len(content), media_type=media_type, response_headers=dict(response_headers),
            fetcher_version=fetcher_version, schema_id=schema_id, format_version=format_version, supersedes_sha256=supersedes, license_status=license_status,
        )
        path = self.artifact_path(provider, dataset, source_key, version)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "xb") as fh:
            fh.write(content)
        with open(self.sidecar_path(path), "xb") as fh:
            fh.write(meta.to_json().encode("utf-8"))
        return PutResult(status, path, meta)

    def quarantine(
        self, *, provider: str, dataset: str, source_key: str, content: bytes, source_url: str, http_status: int, reason: str,
        response_headers: Mapping[str, str | None] | None = None, quarantined_at: dt.datetime | None = None,
    ) -> pathlib.Path:
        digest = sha256_bytes(content)
        d = self.root / QUARANTINE_DIR / check_component("provider", provider) / check_component("dataset", dataset)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{digest[:16]}-{check_component('source_key', source_key)}"
        if not path.exists():
            path.write_bytes(content)
        side = self.sidecar_path(path)
        if not side.exists():
            when = (quarantined_at or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
            side.write_text(json.dumps({
                "provider": provider, "dataset": dataset, "source_key": source_key, "source_url": source_url, "http_status": http_status, "reason": reason,
                "quarantined_at": when.isoformat(), "sha256": digest, "size": len(content), "response_headers": dict(response_headers or {}),
            }, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _data_files(self) -> Iterator[pathlib.Path]:
        if not self.root.exists():
            return
        for p in sorted(self.root.rglob("*")):
            if not p.is_file() or QUARANTINE_DIR in p.relative_to(self.root).parts or p.name == MANIFEST_NAME:
                continue
            yield p

    def verify_all(self) -> list[Problem]:
        problems: list[Problem] = []
        for p in self._data_files():
            rel = p.relative_to(self.root).as_posix()
            if p.name.endswith(_SIDECAR_SUFFIX):
                if not p.with_name(p.name[: -len(_SIDECAR_SUFFIX)]).exists():
                    problems.append(Problem(rel, "ORPHAN_SIDECAR", "sidecar without a data file"))
                continue
            side = self.sidecar_path(p)
            if not side.exists():
                problems.append(Problem(rel, "ORPHAN_FILE", "data file without a sidecar"))
                continue
            try:
                meta = RawArtifactMeta.from_json(side.read_text(encoding="utf-8"))
            except (RawIntegrityError, ValueError, KeyError, TypeError) as exc:
                problems.append(Problem(rel, "BAD_SIDECAR", str(exc)))
                continue
            parts = p.relative_to(self.root).parts
            expected = self.artifact_path(meta.provider, meta.dataset, meta.source_key, meta.version)
            if len(parts) != 3 or expected != p:
                problems.append(Problem(rel, "PATH_MISMATCH", f"sidecar says {meta.provider}/{meta.dataset}/{meta.source_key} v{meta.version}"))
                continue
            data = p.read_bytes()
            if len(data) != meta.size:
                problems.append(Problem(rel, "SIZE_MISMATCH", f"{len(data)} bytes on disk, {meta.size} recorded"))
            if sha256_bytes(data) != meta.sha256:
                problems.append(Problem(rel, "HASH_MISMATCH", "content does not match the recorded sha256"))
        by_key: dict[tuple[str, str, str], list[int]] = {}
        for m in self.manifest_entries(_tolerant=True):
            by_key.setdefault((m.provider, m.dataset, m.source_key), []).append(m.version)
        for k, versions in by_key.items():
            if sorted(versions) != list(range(1, len(versions) + 1)):
                problems.append(Problem("/".join(k), "VERSION_GAP", f"versions on disk: {sorted(versions)}"))
                continue
            try:
                self.versions(*k)
            except RawIntegrityError as exc:
                problems.append(Problem("/".join(k), "CHAIN_BREAK", str(exc)))
        return problems

    def manifest_entries(self, *, _tolerant: bool = False) -> list[RawArtifactMeta]:
        out = []
        for p in self._data_files():
            if not p.name.endswith(_SIDECAR_SUFFIX):
                continue
            try:
                out.append(RawArtifactMeta.from_json(p.read_text(encoding="utf-8")))
            except (RawIntegrityError, ValueError, KeyError, TypeError):
                if not _tolerant:
                    raise
        return sorted(out, key=lambda m: (m.provider, m.dataset, m.source_key, m.version))

    def write_manifest(self) -> tuple[pathlib.Path, str]:
        lines = [json.dumps(dataclasses.asdict(m), sort_keys=True) for m in self.manifest_entries()]
        body = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / MANIFEST_NAME
        path.write_bytes(body)
        return path, sha256_bytes(body)
