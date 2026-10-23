from __future__ import annotations

import dataclasses
import enum
import json
import random
import time
from typing import Callable

import requests

from ..raw.adapter import IngestStatus, ProviderAdapter, ingest
from ..raw.store import RawStore
from ..sources import dhan_equity_historical as deh
from .chunk_planner import BackfillChunk


class ChunkOutcome(str, enum.Enum):
    SUCCESS = "SUCCESS"
    SKIPPED_EXISTING = "SKIPPED_EXISTING"
    HTTP_401_AUTHENTICATION = "HTTP_401_AUTHENTICATION"
    HTTP_429_RATE_LIMIT = "HTTP_429_RATE_LIMIT"
    HTTP_5XX_PROVIDER = "HTTP_5XX_PROVIDER"
    NETWORK_TIMEOUT = "NETWORK_TIMEOUT"
    NETWORK_ERROR = "NETWORK_ERROR"
    MALFORMED_RESPONSE = "MALFORMED_RESPONSE"
    EMPTY_RESPONSE = "EMPTY_RESPONSE"
    OTHER_FAILURE = "OTHER_FAILURE"


_RETRYABLE = frozenset({ChunkOutcome.HTTP_429_RATE_LIMIT, ChunkOutcome.HTTP_5XX_PROVIDER, ChunkOutcome.NETWORK_TIMEOUT, ChunkOutcome.NETWORK_ERROR})


@dataclasses.dataclass(frozen=True)
class ExecutorConfig:
    pace_seconds: float = 1.0
    max_retries: int = 5
    backoff_base_seconds: float = 2.0
    jitter_max_seconds: float = 1.0
    sleep_fn: Callable[[float], None] = time.sleep


@dataclasses.dataclass(frozen=True)
class ChunkResult:
    chunk: BackfillChunk
    source_key: str
    outcome: ChunkOutcome
    attempts: int
    elapsed_seconds: float
    bar_count: int | None = None
    detail: str | None = None


@dataclasses.dataclass(frozen=True)
class PlanResult:
    results: tuple[ChunkResult, ...]
    stopped_early: bool
    stop_reason: str | None

    @property
    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.outcome.value] = counts.get(r.outcome.value, 0) + 1
        counts["total_planned"] = len(self.results)
        counts["total_retry_attempts"] = sum(r.attempts - 1 for r in self.results if r.attempts > 1)
        return counts


def _classify_exception(exc: Exception) -> ChunkOutcome:
    if isinstance(exc, requests.HTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 401:
            return ChunkOutcome.HTTP_401_AUTHENTICATION
        if status == 429:
            return ChunkOutcome.HTTP_429_RATE_LIMIT
        if status is not None and 500 <= status < 600:
            return ChunkOutcome.HTTP_5XX_PROVIDER
        return ChunkOutcome.OTHER_FAILURE
    if isinstance(exc, requests.Timeout):
        return ChunkOutcome.NETWORK_TIMEOUT
    if isinstance(exc, requests.ConnectionError):
        return ChunkOutcome.NETWORK_ERROR
    return ChunkOutcome.OTHER_FAILURE


def _bar_count(store: RawStore, source_key: str) -> int:
    content = store.read_bytes(deh.PROVIDER_ID, deh.DATASET, source_key)
    return len(json.loads(content)["timestamp"])


@dataclasses.dataclass(frozen=True)
class _Partial:
    outcome: ChunkOutcome
    attempts: int
    elapsed_seconds: float
    bar_count: int | None
    detail: str | None


def _attempt_chunk(store: RawStore, adapter: ProviderAdapter, source_key: str, config: ExecutorConfig) -> _Partial:
    attempts = 0
    t0 = time.monotonic()
    while True:
        attempts += 1
        if config.pace_seconds > 0:
            config.sleep_fn(config.pace_seconds)
        try:
            result = ingest(store, adapter, deh.DATASET, source_key)
        except Exception as exc:  # noqa: BLE001 - classified immediately below, never swallowed silently
            outcome = _classify_exception(exc)
            if outcome in _RETRYABLE and attempts <= config.max_retries:
                backoff = config.backoff_base_seconds * (2 ** (attempts - 1)) + random.uniform(0, config.jitter_max_seconds)
                config.sleep_fn(backoff)
                continue
            return _Partial(outcome, attempts, time.monotonic() - t0, None, str(exc))

        if result.status is IngestStatus.QUARANTINED:
            return _Partial(ChunkOutcome.MALFORMED_RESPONSE, attempts, time.monotonic() - t0, None, result.reason)
        if result.status is IngestStatus.NOT_AVAILABLE:
            return _Partial(ChunkOutcome.EMPTY_RESPONSE, attempts, time.monotonic() - t0, 0, "provider reports no data for this range")

        n = _bar_count(store, source_key)
        outcome = ChunkOutcome.EMPTY_RESPONSE if n == 0 else ChunkOutcome.SUCCESS
        return _Partial(outcome, attempts, time.monotonic() - t0, n, None)


def execute_plan(store: RawStore, adapter: ProviderAdapter, chunks: list[BackfillChunk], *, config: ExecutorConfig | None = None) -> PlanResult:
    config = config or ExecutorConfig()
    results: list[ChunkResult] = []
    for chunk in chunks:
        source_key = deh.source_key_for(chunk.security_id, chunk.from_date, chunk.to_date, interval=1)
        if store.exists(deh.PROVIDER_ID, deh.DATASET, source_key):
            results.append(ChunkResult(chunk, source_key, ChunkOutcome.SKIPPED_EXISTING, attempts=0, elapsed_seconds=0.0, bar_count=_bar_count(store, source_key)))
            continue

        partial = _attempt_chunk(store, adapter, source_key, config)
        results.append(ChunkResult(chunk, source_key, partial.outcome, partial.attempts, partial.elapsed_seconds, partial.bar_count, partial.detail))

        if partial.outcome is ChunkOutcome.HTTP_401_AUTHENTICATION:
            return PlanResult(tuple(results), stopped_early=True, stop_reason="HTTP 401 - Dhan access token expired or invalid; manual token refresh required before re-invoking this plan")

    return PlanResult(tuple(results), stopped_early=False, stop_reason=None)
