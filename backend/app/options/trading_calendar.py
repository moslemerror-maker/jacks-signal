from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import AbstractSet

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


def load_fo_holiday_dates(holiday_master_path: Path) -> frozenset[str]:
    if not holiday_master_path.exists():
        raise HolidayCalendarUnavailable(f"holiday-master file not found: {holiday_master_path}")
    try:
        payload = json.loads(holiday_master_path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HolidayCalendarUnavailable(f"holiday-master file at {holiday_master_path} is not valid JSON: {exc}") from exc
    from backend.app.data.sources.nse_calendar import FO_SEGMENT_KEY
    if FO_SEGMENT_KEY not in payload:
        raise HolidayCalendarUnavailable(f"holiday-master file at {holiday_master_path} has no {FO_SEGMENT_KEY!r} segment key")
    try:
        return frozenset(row["tradingDate"] for row in payload[FO_SEGMENT_KEY])
    except (KeyError, TypeError) as exc:
        raise HolidayCalendarUnavailable(f"holiday-master file at {holiday_master_path}'s {FO_SEGMENT_KEY!r} rows are not in the expected shape: {exc}") from exc
