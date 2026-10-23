from __future__ import annotations

import datetime as dt
import enum


class CollectionState(str, enum.Enum):
    LIVE = "LIVE"
    STALE = "STALE"
    PARTIAL = "PARTIAL"
    BACKFILLING = "BACKFILLING"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


DEFAULT_MAX_STALENESS = dt.timedelta(minutes=8)


def classify_date_state(
    *,
    trading_date: dt.date,
    today: dt.date,
    bar_count: int,
    expected_bars: int,
    last_bar_utc: dt.datetime | None,
    now_utc: dt.datetime | None = None,
    is_backfilling: bool = False,
    had_error: bool = False,
    max_staleness: dt.timedelta = DEFAULT_MAX_STALENESS,
) -> CollectionState:
    if had_error:
        return CollectionState.ERROR
    if is_backfilling:
        return CollectionState.BACKFILLING

    if trading_date == today:
        if last_bar_utc is None:
            return CollectionState.STALE
        if now_utc is None:
            raise ValueError("now_utc is required to classify today's freshness")
        age = now_utc - last_bar_utc
        return CollectionState.STALE if age > max_staleness else CollectionState.LIVE

    if bar_count >= expected_bars:
        return CollectionState.COMPLETE
    return CollectionState.PARTIAL
