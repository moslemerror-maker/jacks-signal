from __future__ import annotations

import datetime as dt
from typing import Sequence

from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.data.calendar.equity_trading_calendar import EXPECTED_BARS_PER_COMPLETE_SESSION, expected_trading_days


def per_symbol_watermark(
    session: Session, *, security_ids: Sequence[int], expected_bars: int = EXPECTED_BARS_PER_COMPLETE_SESSION,
) -> dict[int, dt.date | None]:
    watermark: dict[int, dt.date | None] = {sid: None for sid in security_ids}
    if not security_ids:
        return watermark

    rows = session.execute(
        text(
            """
            SELECT security_id, MAX(trading_date) AS last_complete_date
            FROM (
                SELECT security_id, trading_date, COUNT(*) AS n
                FROM intraday_equity_bars
                WHERE security_id = ANY(:security_ids)
                GROUP BY security_id, trading_date
                HAVING COUNT(*) >= :expected_bars
            ) complete_sessions
            GROUP BY security_id
            """
        ),
        {"security_ids": list(security_ids), "expected_bars": expected_bars},
    ).fetchall()
    for security_id, last_complete_date in rows:
        watermark[security_id] = last_complete_date
    return watermark


def missing_trading_dates(
    watermark: dt.date | None, *, today: dt.date, holiday_dates, bootstrap_from: dt.date | None = None,
) -> list[dt.date]:
    if watermark is None:
        if bootstrap_from is None:
            raise ValueError(
                "watermark is None (no complete session on record) and no bootstrap_from was supplied - "
                "this function will not silently choose a starting date for a symbol with zero history"
            )
        start = bootstrap_from
    else:
        start = watermark + dt.timedelta(days=1)

    end = today - dt.timedelta(days=1)
    if end < start:
        return []
    return expected_trading_days(start, end, holiday_dates)
