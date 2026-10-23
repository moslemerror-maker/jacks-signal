from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import StockCollectionEvent

VALID_SEVERITIES = ("INFO", "WARNING", "ERROR")

CYCLE_START = "CYCLE_START"
CYCLE_COMPLETE = "CYCLE_COMPLETE"
BACKFILL_STARTED = "BACKFILL_STARTED"
BACKFILL_COMPLETE = "BACKFILL_COMPLETE"
BACKFILL_FAILED = "BACKFILL_FAILED"
SYMBOL_FAILED = "SYMBOL_FAILED"
TOKEN_EXPIRED = "TOKEN_EXPIRED"
API_ERROR = "API_ERROR"
SESSION_COMPLETE = "SESSION_COMPLETE"
RECONCILIATION_COMPLETE = "RECONCILIATION_COMPLETE"
NON_TRADING_DAY = "NON_TRADING_DAY"


def log_event(
    session: Session,
    *,
    trading_date: dt.date,
    severity: str,
    category: str,
    detail: str,
    event_timestamp: dt.datetime | None = None,
    symbols_expected: int | None = None,
    symbols_succeeded: int | None = None,
    symbols_failed: int | None = None,
    latest_completed_minute: dt.datetime | None = None,
    bars_persisted: int | None = None,
) -> StockCollectionEvent:
    if severity not in VALID_SEVERITIES:
        raise ValueError(f"severity must be one of {VALID_SEVERITIES}, got {severity!r}")
    event = StockCollectionEvent(
        trading_date=trading_date,
        event_timestamp=event_timestamp or dt.datetime.now(dt.timezone.utc),
        severity=severity,
        category=category,
        detail=detail,
        symbols_expected=symbols_expected,
        symbols_succeeded=symbols_succeeded,
        symbols_failed=symbols_failed,
        latest_completed_minute=latest_completed_minute,
        bars_persisted=bars_persisted,
    )
    session.add(event)
    return event


def events_for_date(session: Session, *, trading_date: dt.date) -> list[StockCollectionEvent]:
    stmt = (
        select(StockCollectionEvent)
        .where(StockCollectionEvent.trading_date == trading_date)
        .order_by(StockCollectionEvent.event_timestamp.asc())
    )
    return list(session.execute(stmt).scalars().all())


def is_session_complete(session: Session, *, trading_date: dt.date) -> bool:
    return any(e.category == SESSION_COMPLETE for e in events_for_date(session, trading_date=trading_date))
