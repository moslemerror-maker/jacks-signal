from __future__ import annotations

import dataclasses
import datetime as dt
import time
from pathlib import Path
from typing import AbstractSet, Callable, Literal, Mapping, MutableMapping, Sequence, get_args

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from .eod_report import session_window_utc
from .live_session import CycleResult, OrchestratorConfig, RunnerConfig, log_session_event, run_session
from .pipeline_reliability import ended_on_failure_streak
from .trading_calendar import is_trading_day

SessionState = Literal["PRE_MARKET", "STARTING", "LIVE", "DEGRADED", "PAUSED", "COMPLETED", "FAILED"]
_SESSION_STATES: tuple[str, ...] = get_args(SessionState)
_TERMINAL_STATES = ("COMPLETED", "FAILED")

SessionStatus = Literal["NOT_STARTED", "IN_PROGRESS", "COMPLETED", "FAILED"]

_CATEGORY_PREFIX = "SESSION_"
NOT_TRADING_DAY_CATEGORY = "SESSION_NOT_TRADING_DAY"


def _category_for(state: SessionState) -> str:
    if state not in _SESSION_STATES:
        raise ValueError(f"state must be one of {_SESSION_STATES}, got {state!r}")
    return f"{_CATEGORY_PREFIX}{state}"


def _state_from_category(category: str) -> SessionState | None:
    if not category.startswith(_CATEGORY_PREFIX):
        return None
    candidate = category[len(_CATEGORY_PREFIX):]
    return candidate if candidate in _SESSION_STATES else None  # type: ignore[return-value]


def log_session_state(db: Session, *, state: SessionState, detail: str, event_timestamp: dt.datetime, severity: str = "INFO") -> None:
    log_session_event(db, event_timestamp=event_timestamp, severity=severity, category=_category_for(state), detail=detail)


def log_not_trading_day(db: Session, *, detail: str, event_timestamp: dt.datetime) -> None:
    log_session_event(db, event_timestamp=event_timestamp, severity="INFO", category=NOT_TRADING_DAY_CATEGORY, detail=detail)


def check_db_reachable(db: Session) -> bool:
    try:
        db.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False


def todays_session_state_events(db: Session, *, session_date: dt.date, now: dt.datetime) -> list[Mapping]:
    window_start, window_end = session_window_utc(session_date)
    window_end = min(window_end, now)
    rows = db.execute(text(
        "SELECT event_timestamp, severity, category, detail FROM option_session_events "
        "WHERE event_timestamp >= :s AND event_timestamp <= :e AND category LIKE :prefix "
        "ORDER BY event_timestamp"
    ), {"s": window_start, "e": window_end, "prefix": f"{_CATEGORY_PREFIX}%"}).mappings().all()
    return [r for r in rows if _state_from_category(r["category"]) is not None]


def determine_session_status(events: Sequence[Mapping]) -> SessionStatus:
    if not events:
        return "NOT_STARTED"
    last_state = _state_from_category(events[-1]["category"])
    assert last_state is not None
    if last_state in _TERMINAL_STATES:
        return last_state  # type: ignore[return-value]
    return "IN_PROGRESS"


SessionStatusApiState = Literal["NOT_STARTED", "STARTING", "LIVE", "DEGRADED", "PAUSED", "COMPLETED", "FAILED", "UNKNOWN"]


def latest_session_state(events: Sequence[Mapping]) -> SessionStatusApiState:
    if not events:
        return "NOT_STARTED"
    state = _state_from_category(events[-1]["category"])
    return state if state is not None else "UNKNOWN"


@dataclasses.dataclass(frozen=True, slots=True)
class SessionStatusSnapshot:

    underlying: str
    trading_date: dt.date
    session_state: SessionStatusApiState
    session_start_timestamp: dt.datetime | None
    last_event_timestamp: dt.datetime | None
    completion_timestamp: dt.datetime | None
    failure_timestamp: dt.datetime | None
    failure_detail: str | None
    cycle_count: int
    latest_recommendation_timestamp: dt.datetime | None
    latest_recommendation_status: str | None
    latest_recommendation_id: str | None
    data_quality: str | None
    source: str | None


def session_status_snapshot(db: Session, *, underlying: str, session_date: dt.date, now: dt.datetime) -> SessionStatusSnapshot:
    events = todays_session_state_events(db, session_date=session_date, now=now)
    state = latest_session_state(events)

    starting_events = [e for e in events if _state_from_category(e["category"]) == "STARTING"]
    session_start_timestamp = starting_events[0]["event_timestamp"] if starting_events else None
    last_event_timestamp = events[-1]["event_timestamp"] if events else None
    completion_timestamp = last_event_timestamp if state == "COMPLETED" else None
    failure_timestamp = last_event_timestamp if state == "FAILED" else None
    failure_detail = events[-1]["detail"] if state == "FAILED" else None

    window_start, window_end = session_window_utc(session_date)
    window_end = min(window_end, now)
    cycle_count = db.execute(text(
        "SELECT count(*) FROM option_recommendations WHERE underlying=:u AND decision_timestamp >= :s AND decision_timestamp < :e"
    ), {"u": underlying, "s": window_start, "e": window_end}).scalar_one()

    latest = db.execute(text(
        "SELECT id, decision_timestamp, status, data_quality, source FROM option_recommendations "
        "WHERE underlying=:u ORDER BY decision_timestamp DESC LIMIT 1"
    ), {"u": underlying}).mappings().first()

    return SessionStatusSnapshot(
        underlying=underlying, trading_date=session_date, session_state=state,
        session_start_timestamp=session_start_timestamp, last_event_timestamp=last_event_timestamp,
        completion_timestamp=completion_timestamp, failure_timestamp=failure_timestamp, failure_detail=failure_detail,
        cycle_count=cycle_count,
        latest_recommendation_timestamp=latest["decision_timestamp"] if latest else None,
        latest_recommendation_status=latest["status"] if latest else None,
        latest_recommendation_id=str(latest["id"]) if latest else None,
        data_quality=latest["data_quality"] if latest else None,
        source=latest["source"] if latest else None,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class SessionStartDecision:
    action: Literal["START", "RESUME", "REFUSE_ALREADY_COMPLETED", "REFUSE_ALREADY_FAILED"]
    detail: str
    status_before: SessionStatus


def decide_session_start(db: Session, *, session_date: dt.date, now: dt.datetime, force: bool = False) -> SessionStartDecision:
    events = todays_session_state_events(db, session_date=session_date, now=now)
    status = determine_session_status(events)
    if force:
        return SessionStartDecision("START", f"--force override (status before override was {status})", status)
    if status == "NOT_STARTED":
        return SessionStartDecision("START", "no session recorded yet today", status)
    if status == "IN_PROGRESS":
        return SessionStartDecision("RESUME", "a session already exists today and has not reached a terminal state - resuming, not restarting", status)
    if status == "COMPLETED":
        return SessionStartDecision("REFUSE_ALREADY_COMPLETED", "today's session already reached COMPLETED - refusing to start a second one", status)
    return SessionStartDecision("REFUSE_ALREADY_FAILED", "today's session already reached FAILED - refusing to silently retry; use --force after investigating", status)


DailySessionOutcome_ = Literal["NOT_TRADING_DAY", "REFUSE_ALREADY_COMPLETED", "REFUSE_ALREADY_FAILED", "COMPLETED", "FAILED"]


@dataclasses.dataclass(frozen=True, slots=True)
class DailySessionResult:
    outcome: DailySessionOutcome_
    detail: str
    decision_status_before: SessionStatus | None = None
    cycle_results: tuple[CycleResult, ...] = ()


def run_daily_session_workflow(
    *, db: Session, session_factory: Callable[[], Session], today: dt.date, holiday_dates: AbstractSet[str],
    config: OrchestratorConfig, runner_config: RunnerConfig, now_fn: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc),
    sleep_fn: Callable[[float], None] = time.sleep, stop_flag: Callable[[], bool] | None = None, force: bool = False,
) -> DailySessionResult:
    start_now = now_fn()
    if not is_trading_day(today, holiday_dates):
        log_not_trading_day(db, detail=f"{today.isoformat()} is not a NIFTY F&O trading day (weekend or FO holiday)", event_timestamp=start_now)
        db.commit()
        return DailySessionResult("NOT_TRADING_DAY", f"{today.isoformat()} is not a trading day")

    decision = decide_session_start(db, session_date=today, now=start_now, force=force)
    if decision.action == "REFUSE_ALREADY_COMPLETED":
        return DailySessionResult("REFUSE_ALREADY_COMPLETED", decision.detail, decision.status_before)
    if decision.action == "REFUSE_ALREADY_FAILED":
        return DailySessionResult("REFUSE_ALREADY_FAILED", decision.detail, decision.status_before)

    if decision.action == "START":
        log_session_state(db, state="STARTING", detail="automatic daily session starting", event_timestamp=start_now)
        db.commit()
    log_session_state(db, state="LIVE", detail="resumed after restart" if decision.action == "RESUME" else "polling started", event_timestamp=start_now)
    db.commit()

    try:
        results = run_session(session_factory=session_factory, config=config, runner_config=runner_config, now_fn=now_fn, sleep_fn=sleep_fn, stop_flag=stop_flag)
    except Exception as exc:
        log_session_state(db, state="FAILED", detail=f"uncaught exception during session: {exc}", event_timestamp=now_fn(), severity="ERROR")
        db.commit()
        return DailySessionResult("FAILED", str(exc), decision.status_before)

    status_counts: dict[str, int] = {}
    for r in results:
        if r.status:
            status_counts[r.status] = status_counts.get(r.status, 0) + 1
    summary = f"cycles={len(results)} status_counts={status_counts}"

    if ended_on_failure_streak([r.outcome for r in results], threshold=runner_config.max_consecutive_fetch_failures):
        detail = (f"stopped after {runner_config.max_consecutive_fetch_failures} consecutive FETCH_FAILURE cycles - "
                  f"{summary}. See option_session_events for the specific failure category(ies) logged this session.")
        log_session_state(db, state="FAILED", detail=detail, event_timestamp=now_fn(), severity="ERROR")
        db.commit()
        return DailySessionResult("FAILED", detail, decision.status_before, tuple(results))

    log_session_state(db, state="COMPLETED", detail=summary, event_timestamp=now_fn())
    db.commit()
    return DailySessionResult("COMPLETED", summary, decision.status_before, tuple(results))


def windows_safe_stop_flag(*, signal_state: MutableMapping[str, bool], stop_file: Path) -> Callable[[], bool]:
    def _stop_requested() -> bool:
        return bool(signal_state.get("requested", False)) or stop_file.exists()
    return _stop_requested
