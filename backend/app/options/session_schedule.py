from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
UTC = dt.timezone.utc

SESSION_START_IST = dt.time(9, 15)
SESSION_STOP_IST = dt.time(15, 15)


def session_start_utc(session_date: dt.date) -> dt.datetime:
    return dt.datetime.combine(session_date, SESSION_START_IST, tzinfo=IST).astimezone(UTC)


def session_stop_utc(session_date: dt.date) -> dt.datetime:
    return dt.datetime.combine(session_date, SESSION_STOP_IST, tzinfo=IST).astimezone(UTC)


def ist_today(now: dt.datetime) -> dt.date:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(IST).date()
