from __future__ import annotations

import datetime as dt
import email.utils
from typing import Sequence

UTC = dt.timezone.utc
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
SESSION_OPEN_IST = dt.time(9, 15)
SESSION_CLOSE_IST = dt.time(15, 30)


def session_close_utc(session_date: dt.date) -> dt.datetime:
    return dt.datetime.combine(session_date, SESSION_CLOSE_IST, tzinfo=IST).astimezone(UTC)


def next_session_open_utc(session_date: dt.date, sessions: Sequence[dt.date] | None = None) -> tuple[dt.datetime, str]:
    later = [s for s in (sessions or ()) if s > session_date]
    if later:
        nxt, basis = min(later), "OBSERVED_SESSIONS"
    else:
        nxt, basis = session_date + dt.timedelta(1), "WEEKDAY_FALLBACK"
        while nxt.weekday() >= 5:
            nxt += dt.timedelta(1)
    return dt.datetime.combine(nxt, SESSION_OPEN_IST, tzinfo=IST).astimezone(UTC), basis


def availability_for_eod_file(session_date: dt.date, sessions: Sequence[dt.date] | None, last_modified: dt.datetime | None) -> dict:
    close = session_close_utc(session_date)
    nxt, cal = next_session_open_utc(session_date, sessions)
    if last_modified is not None and close <= last_modified < nxt:
        return {"available_at": last_modified, "availability_basis": "OBSERVED_HTTP_LAST_MODIFIED", "evidence": {"last_modified": last_modified.isoformat(), "calendar": cal}}
    if last_modified is None:
        why = "NO_LAST_MODIFIED"
    elif last_modified < close:
        why = "LAST_MODIFIED_BEFORE_SESSION_CLOSE_IMPLAUSIBLE"
    else:
        why = "LAST_MODIFIED_AFTER_NEXT_OPEN_REUPLOAD_OR_LATE"
    return {"available_at": nxt, "availability_basis": "RULE_NEXT_SESSION_OPEN",
            "evidence": {"last_modified": last_modified.isoformat() if last_modified else None, "reason": why, "calendar": cal}}


def parse_last_modified(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        t = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return t.astimezone(UTC) if t.tzinfo else t.replace(tzinfo=UTC)
