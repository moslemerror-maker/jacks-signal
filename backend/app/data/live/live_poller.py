from __future__ import annotations

import datetime as dt
import random
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from sqlalchemy.orm import Session

from backend.app.data.backfill.executor import _RETRYABLE, ChunkOutcome, ExecutorConfig, _bar_count, _classify_exception
from backend.app.data.canonical.intraday_equity_bar import CanonicalIntradayEquityBar
from backend.app.data.canonical.intraday_loader import upsert_intraday_bars_batched
from backend.app.data.normalizers.dhan_intraday_equity import normalize_dhan_intraday_artifact
from backend.app.data.raw.adapter import IngestStatus, ProviderAdapter, ingest
from backend.app.data.raw.store import RawStore
from backend.app.data.sources import dhan_equity_historical as deh


def bars_up_to_now(
    bars: Sequence[CanonicalIntradayEquityBar], *, now_utc: dt.datetime,
) -> list[CanonicalIntradayEquityBar]:
    current_minute_start = now_utc.replace(second=0, microsecond=0)
    return [b for b in bars if b.bar_start_utc < current_minute_start]


@dataclass(frozen=True)
class _LivePartial:
    outcome: ChunkOutcome
    attempts: int
    elapsed_seconds: float
    bar_count: int | None
    detail: str | None


def _attempt_live_fetch(
    store: RawStore, adapter: ProviderAdapter, source_key: str, config: ExecutorConfig,
) -> _LivePartial:
    attempts = 0
    t0 = time.monotonic()
    while True:
        attempts += 1
        if config.pace_seconds > 0:
            config.sleep_fn(config.pace_seconds)
        try:
            result = ingest(store, adapter, deh.DATASET, source_key, refresh=True)
        except Exception as exc:  # noqa: BLE001 - classified immediately below, never swallowed silently
            outcome = _classify_exception(exc)
            if outcome in _RETRYABLE and attempts <= config.max_retries:
                backoff = config.backoff_base_seconds * (2 ** (attempts - 1)) + random.uniform(0, config.jitter_max_seconds)
                config.sleep_fn(backoff)
                continue
            return _LivePartial(outcome, attempts, time.monotonic() - t0, None, str(exc))

        if result.status is IngestStatus.QUARANTINED:
            return _LivePartial(ChunkOutcome.MALFORMED_RESPONSE, attempts, time.monotonic() - t0, None, result.reason)
        if result.status is IngestStatus.NOT_AVAILABLE:
            return _LivePartial(ChunkOutcome.EMPTY_RESPONSE, attempts, time.monotonic() - t0, 0, "provider reports no data for this range")

        n = _bar_count(store, source_key)
        outcome = ChunkOutcome.EMPTY_RESPONSE if n == 0 else ChunkOutcome.SUCCESS
        return _LivePartial(outcome, attempts, time.monotonic() - t0, n, None)


@dataclass(frozen=True)
class LivePollResult:
    security_id: int
    trading_symbol: str
    outcome: ChunkOutcome
    bars_fetched: int
    bars_persisted: int
    latest_completed_bar_utc: dt.datetime | None
    detail: str | None = None


def poll_symbol_once(
    session: Session, store: RawStore, adapter: ProviderAdapter, *,
    security_id: int, trading_symbol: str, today: dt.date, now_utc: dt.datetime, config: ExecutorConfig,
) -> LivePollResult:
    source_key = deh.source_key_for(security_id, today, today, interval=1)
    partial = _attempt_live_fetch(store, adapter, source_key, config)

    if partial.outcome not in (ChunkOutcome.SUCCESS,):
        return LivePollResult(
            security_id=security_id, trading_symbol=trading_symbol, outcome=partial.outcome,
            bars_fetched=0, bars_persisted=0, latest_completed_bar_utc=None, detail=partial.detail,
        )

    bars, rejected = normalize_dhan_intraday_artifact(store, source_key=source_key, trading_symbol=trading_symbol)
    safe_bars = bars_up_to_now(bars, now_utc=now_utc)
    n_persisted = upsert_intraday_bars_batched(session, safe_bars) if safe_bars else 0
    latest = max((b.bar_start_utc for b in safe_bars), default=None)
    return LivePollResult(
        security_id=security_id, trading_symbol=trading_symbol, outcome=partial.outcome,
        bars_fetched=len(bars), bars_persisted=n_persisted, latest_completed_bar_utc=latest,
        detail=(f"{len(rejected)} rejected" if rejected else None),
    )


def poll_universe_once(
    session: Session, store: RawStore, adapter: ProviderAdapter, *,
    securities: Sequence[tuple[int, str]], today: dt.date, now_utc_fn: Callable[[], dt.datetime],
    config: ExecutorConfig | None = None,
) -> list[LivePollResult]:
    config = config or ExecutorConfig()
    results: list[LivePollResult] = []
    for security_id, trading_symbol in securities:
        try:
            result = poll_symbol_once(
                session, store, adapter, security_id=security_id, trading_symbol=trading_symbol,
                today=today, now_utc=now_utc_fn(), config=config,
            )
        except Exception as exc:  # noqa: BLE001 - one symbol's unexpected failure must never abort the sweep
            result = LivePollResult(
                security_id=security_id, trading_symbol=trading_symbol, outcome=ChunkOutcome.OTHER_FAILURE,
                bars_fetched=0, bars_persisted=0, latest_completed_bar_utc=None, detail=str(exc),
            )
        results.append(result)
        if result.outcome is ChunkOutcome.HTTP_401_AUTHENTICATION:
            break
    return results
