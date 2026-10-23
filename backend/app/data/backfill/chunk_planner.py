from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Sequence


@dataclasses.dataclass(frozen=True)
class BackfillChunk:

    security_id: int
    trading_symbol: str
    from_date: dt.date
    to_date: dt.date


def plan_chunks(
    securities: Sequence[tuple[int, str]], *, overall_from: dt.date, overall_to: dt.date, chunk_days: int = 90,
) -> list[BackfillChunk]:
    if chunk_days < 1:
        raise ValueError(f"chunk_days must be >= 1, got {chunk_days}")
    if overall_to < overall_from:
        raise ValueError(f"overall_to ({overall_to}) must be >= overall_from ({overall_from})")

    windows = _date_windows(overall_from, overall_to, chunk_days)
    return [
        BackfillChunk(security_id=security_id, trading_symbol=symbol, from_date=from_date, to_date=to_date)
        for security_id, symbol in securities
        for from_date, to_date in windows
    ]


def _date_windows(overall_from: dt.date, overall_to: dt.date, chunk_days: int) -> list[tuple[dt.date, dt.date]]:
    windows: list[tuple[dt.date, dt.date]] = []
    start = overall_from
    while start <= overall_to:
        end = min(start + dt.timedelta(days=chunk_days - 1), overall_to)
        windows.append((start, end))
        start = end + dt.timedelta(days=1)
    return windows


def describe_plan(chunks: Sequence[BackfillChunk], *, preview: int = 3) -> dict:
    securities = list(dict.fromkeys((c.security_id, c.trading_symbol) for c in chunks))
    return {
        "total_chunks": len(chunks),
        "distinct_securities": len(securities),
        "securities": securities,
        "first_chunks": [dataclasses.asdict(c) for c in chunks[:preview]],
        "last_chunks": [dataclasses.asdict(c) for c in chunks[-preview:]] if chunks else [],
    }
