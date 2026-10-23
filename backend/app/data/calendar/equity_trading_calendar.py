from __future__ import annotations

import datetime as dt
import enum
import json
from pathlib import Path
from typing import AbstractSet

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

SESSION_OPEN_IST = dt.time(9, 15)
SESSION_CLOSE_IST = dt.time(15, 30)

FIRST_BAR_IST = dt.time(9, 15)
LAST_BAR_IST = dt.time(15, 14)
EXPECTED_BARS_PER_COMPLETE_SESSION = 360

HISTORICAL_VARIANT_LAST_DATE = dt.date(2026, 7, 31)
HISTORICAL_VARIANT_FIRST_BAR_IST = dt.time(9, 15)
HISTORICAL_VARIANT_LAST_BAR_IST = dt.time(15, 29)
HISTORICAL_VARIANT_BARS_PER_SESSION = 375

_MONTH_ABBR = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


class HolidayCalendarUnavailable(RuntimeError):
    pass


def nse_date_string(day: dt.date) -> str:
    return f"{day.day:02d}-{_MONTH_ABBR[day.month]}-{day.year}"


def is_trading_day(day: dt.date, holiday_dates: AbstractSet[str]) -> bool:
    if day.weekday() >= 5:
        return False
    return nse_date_string(day) not in holiday_dates


def load_equity_holiday_dates(holiday_master_path: Path) -> frozenset[str]:
    if not holiday_master_path.exists():
        raise HolidayCalendarUnavailable(f"holiday-master file not found: {holiday_master_path}")
    try:
        payload = json.loads(holiday_master_path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HolidayCalendarUnavailable(f"holiday-master file at {holiday_master_path} is not valid JSON: {exc}") from exc
    from backend.app.data.sources.nse_calendar import CM_SEGMENT_KEY
    if CM_SEGMENT_KEY not in payload:
        raise HolidayCalendarUnavailable(f"holiday-master file at {holiday_master_path} has no {CM_SEGMENT_KEY!r} segment key")
    try:
        return frozenset(row["tradingDate"] for row in payload[CM_SEGMENT_KEY])
    except (KeyError, TypeError) as exc:
        raise HolidayCalendarUnavailable(f"holiday-master file at {holiday_master_path}'s {CM_SEGMENT_KEY!r} rows are not in the expected shape: {exc}") from exc


def expected_trading_days(start: dt.date, end: dt.date, holiday_dates: AbstractSet[str]) -> list[dt.date]:
    out, d = [], start
    while d <= end:
        if is_trading_day(d, holiday_dates):
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def _minute_grid(first: dt.time, last: dt.time) -> list[dt.time]:
    out, t = [], dt.datetime(2000, 1, 1, first.hour, first.minute)
    end = dt.datetime(2000, 1, 1, last.hour, last.minute)
    while t <= end:
        out.append(t.time())
        t += dt.timedelta(minutes=1)
    return out


def expected_minute_grid_ist() -> list[dt.time]:
    return _minute_grid(FIRST_BAR_IST, LAST_BAR_IST)


def historical_variant_minute_grid_ist() -> list[dt.time]:
    return _minute_grid(HISTORICAL_VARIANT_FIRST_BAR_IST, HISTORICAL_VARIANT_LAST_BAR_IST)


class SessionShape(str, enum.Enum):

    MATCHES_CURRENT_GRID = "MATCHES_CURRENT_GRID"
    MATCHES_KNOWN_HISTORICAL_VARIANT = "MATCHES_KNOWN_HISTORICAL_VARIANT"
    UNRECOGNIZED_SHAPE = "UNRECOGNIZED_SHAPE"


def classify_session_shape(trading_date: dt.date, bar_start_times_ist: AbstractSet[dt.time]) -> SessionShape:
    if bar_start_times_ist == set(expected_minute_grid_ist()):
        return SessionShape.MATCHES_CURRENT_GRID
    if trading_date <= HISTORICAL_VARIANT_LAST_DATE and bar_start_times_ist == set(historical_variant_minute_grid_ist()):
        return SessionShape.MATCHES_KNOWN_HISTORICAL_VARIANT
    return SessionShape.UNRECOGNIZED_SHAPE
