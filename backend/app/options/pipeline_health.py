from __future__ import annotations

import dataclasses
import datetime as dt
import subprocess
from typing import Mapping, Sequence

from sqlalchemy import text
from sqlalchemy.orm import Session

from .pipeline_reliability import FETCH_FAILURE_CATEGORIES
from .session_schedule import session_start_utc
from .session_scheduler import SessionStatusApiState, session_status_snapshot
from .trading_calendar import is_trading_day

DEFAULT_SILENCE_THRESHOLD = dt.timedelta(minutes=15)


@dataclasses.dataclass(frozen=True, slots=True)
class PipelineHealthSnapshot:
    underlying: str
    as_of: dt.datetime
    trading_date: dt.date
    is_trading_day: bool
    session_state: SessionStatusApiState
    session_started: bool
    session_completed: bool
    session_failed: bool
    last_event_timestamp: dt.datetime | None
    silent_for: dt.timedelta | None
    is_silent_too_long: bool
    cycle_count_today: int
    last_successful_fetch_at: dt.datetime | None
    last_decision_at: dt.datetime | None
    fetch_failure_counts_today: Mapping[str, int]
    auth_failure_suspected_today: bool
    crash_events_today: int
    scheduled_task_state: str | None


def _fetch_failure_counts_today(db: Session, *, window_start: dt.datetime, window_end: dt.datetime) -> dict[str, int]:
    rows = db.execute(text(
        "SELECT category, count(*) FROM option_session_events "
        "WHERE event_timestamp >= :s AND event_timestamp <= :e AND category = ANY(:cats) "
        "GROUP BY category"
    ), {"s": window_start, "e": window_end, "cats": list(FETCH_FAILURE_CATEGORIES)}).all()
    return {category: int(count) for category, count in rows}


def _crash_events_today(db: Session, *, window_start: dt.datetime, window_end: dt.datetime) -> int:
    return db.execute(text(
        "SELECT count(*) FROM option_session_events WHERE event_timestamp >= :s AND event_timestamp <= :e AND category = 'CYCLE_CRASH'"
    ), {"s": window_start, "e": window_end}).scalar_one()


def _last_successful_fetch_at(db: Session, *, underlying: str) -> dt.datetime | None:
    return db.execute(text(
        "SELECT max(decision_timestamp) FROM option_recommendations WHERE underlying = :u"
    ), {"u": underlying}).scalar()


def pipeline_health_snapshot(
    db: Session, *, underlying: str, today: dt.date, now: dt.datetime, holiday_dates, task_name: str | None = None,
    silence_threshold: dt.timedelta = DEFAULT_SILENCE_THRESHOLD,
) -> PipelineHealthSnapshot:
    trading_day = is_trading_day(today, holiday_dates)
    status = session_status_snapshot(db, underlying=underlying, session_date=today, now=now)

    from .eod_report import session_window_utc
    window_start, window_end = session_window_utc(today)
    window_end = min(window_end, now)

    fetch_failures = _fetch_failure_counts_today(db, window_start=window_start, window_end=window_end)
    crash_count = _crash_events_today(db, window_start=window_start, window_end=window_end)
    last_fetch = _last_successful_fetch_at(db, underlying=underlying)

    silent_for = (now - status.last_event_timestamp) if status.last_event_timestamp is not None else None
    is_silent_too_long = bool(
        trading_day and status.session_state in ("STARTING", "LIVE", "DEGRADED", "PAUSED")
        and silent_for is not None and silent_for > silence_threshold
    )

    task_state = scheduled_task_state(task_name) if task_name else None

    return PipelineHealthSnapshot(
        underlying=underlying, as_of=now, trading_date=today, is_trading_day=trading_day, session_state=status.session_state,
        session_started=status.session_start_timestamp is not None, session_completed=status.session_state == "COMPLETED",
        session_failed=status.session_state == "FAILED", last_event_timestamp=status.last_event_timestamp,
        silent_for=silent_for, is_silent_too_long=is_silent_too_long, cycle_count_today=status.cycle_count,
        last_successful_fetch_at=last_fetch, last_decision_at=last_fetch,
        fetch_failure_counts_today=fetch_failures, auth_failure_suspected_today=fetch_failures.get("AUTH_FAILURE_SUSPECTED", 0) > 0,
        crash_events_today=crash_count, scheduled_task_state=task_state,
    )


def health_concerns(snapshot: PipelineHealthSnapshot) -> list[str]:
    concerns: list[str] = []
    if snapshot.session_failed:
        concerns.append(f"today's session reached FAILED (last event {snapshot.last_event_timestamp})")
    if snapshot.is_silent_too_long:
        concerns.append(f"session state is {snapshot.session_state} but silent for {snapshot.silent_for} (threshold exceeded)")
    grace_period = dt.timedelta(minutes=15)
    if (snapshot.is_trading_day and not snapshot.session_started
            and snapshot.as_of >= session_start_utc(snapshot.trading_date) + grace_period):
        concerns.append("today is a trading day, well past the scheduled start time, and no session has started yet")
    if snapshot.auth_failure_suspected_today:
        concerns.append(f"AUTH_FAILURE_SUSPECTED events logged today ({snapshot.fetch_failure_counts_today.get('AUTH_FAILURE_SUSPECTED')}) - the Dhan access token most likely needs to be refreshed")
    other_failures = {k: v for k, v in snapshot.fetch_failure_counts_today.items() if k != "AUTH_FAILURE_SUSPECTED" and v > 0}
    if other_failures:
        concerns.append(f"non-auth provider failures logged today: {other_failures}")
    if snapshot.crash_events_today > 0:
        concerns.append(f"{snapshot.crash_events_today} CYCLE_CRASH event(s) logged today")
    if snapshot.scheduled_task_state is not None and snapshot.scheduled_task_state.lower() != "ready" and snapshot.scheduled_task_state.lower() != "running":
        concerns.append(f"scheduled task state is {snapshot.scheduled_task_state!r} (expected Ready/Running)")
    return concerns


def scheduled_task_state(task_name: str, *, timeout_seconds: float = 5.0) -> str | None:
    try:
        proc = subprocess.run(
            ["schtasks", "/query", "/tn", task_name, "/fo", "csv"],
            capture_output=True, text=True, timeout=timeout_seconds,
        )
        if proc.returncode != 0:
            return None
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        if len(lines) < 2:
            return None
        header = [h.strip('"') for h in lines[0].split(",")]
        values = [v.strip('"') for v in lines[1].split(",")]
        if "Status" not in header:
            return None
        return values[header.index("Status")]
    except Exception:
        return None
