from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Sequence

from sqlalchemy.orm import Session

from backend.app.data.backfill.executor import ExecutorConfig
from backend.app.data.calendar.equity_trading_calendar import EXPECTED_BARS_PER_COMPLETE_SESSION
from backend.app.data.live import events
from backend.app.data.live.live_poller import LivePollResult, poll_universe_once
from backend.app.data.raw.adapter import ProviderAdapter
from backend.app.data.raw.store import RawStore


@dataclass(frozen=True)
class FinalizationReport:
    trading_date: dt.date
    symbols_expected: int
    symbols_complete: int
    symbols_partial: int
    symbols_failed: int
    total_bars_persisted: int
    poll_results: tuple[LivePollResult, ...]
    is_session_complete: bool


def finalize_trading_day(
    session: Session, store: RawStore, adapter: ProviderAdapter, *,
    securities: Sequence[tuple[int, str]], trading_date: dt.date, now_utc_fn,
    expected_bars: int = EXPECTED_BARS_PER_COMPLETE_SESSION, config: ExecutorConfig | None = None,
) -> FinalizationReport:
    events.log_event(
        session, trading_date=trading_date, severity="INFO", category=events.CYCLE_START,
        detail=f"end-of-day finalization pass starting for {len(securities)} symbols",
        symbols_expected=len(securities),
    )

    poll_results = poll_universe_once(
        session, store, adapter, securities=securities, today=trading_date, now_utc_fn=now_utc_fn, config=config,
    )

    complete = sum(1 for r in poll_results if r.bars_fetched >= expected_bars)
    failed = sum(1 for r in poll_results if r.outcome.value not in ("SUCCESS", "SKIPPED_EXISTING"))
    partial = len(poll_results) - complete - failed
    total_bars = sum(r.bars_persisted for r in poll_results)
    session_complete = complete == len(securities)

    severity = "INFO" if session_complete else "WARNING"
    category = events.SESSION_COMPLETE if session_complete else events.RECONCILIATION_COMPLETE
    latest = max((r.latest_completed_bar_utc for r in poll_results if r.latest_completed_bar_utc), default=None)
    events.log_event(
        session, trading_date=trading_date, severity=severity, category=category,
        detail=(
            f"finalization complete: {complete} complete, {partial} partial, {failed} failed "
            f"of {len(securities)} symbols; {total_bars} bars persisted this pass"
        ),
        symbols_expected=len(securities), symbols_succeeded=complete, symbols_failed=failed,
        latest_completed_minute=latest, bars_persisted=total_bars,
    )

    return FinalizationReport(
        trading_date=trading_date, symbols_expected=len(securities), symbols_complete=complete,
        symbols_partial=partial, symbols_failed=failed, total_bars_persisted=total_bars,
        poll_results=tuple(poll_results), is_session_complete=session_complete,
    )
