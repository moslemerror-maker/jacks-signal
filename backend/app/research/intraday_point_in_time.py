from __future__ import annotations

import dataclasses
import datetime as dt

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import IntradayEquityBar


@dataclasses.dataclass(frozen=True)
class IntradayBarView:

    security_id: int
    trading_symbol: str
    bar_start_utc: dt.datetime
    trading_date: dt.date
    open: float
    high: float
    low: float
    close: float
    volume: float | None


def select_bars_as_of(
    session: Session, *, security_id: int, timeframe: str, as_of_utc: dt.datetime, provider: str = "DHAN",
    exchange_segment: str = "NSE_EQ", start_utc: dt.datetime | None = None,
) -> list[IntradayBarView]:
    if as_of_utc.tzinfo is None:
        raise ValueError("as_of_utc must be timezone-aware - a naive datetime cannot be safely compared to bar_start_utc")
    conditions = [
        IntradayEquityBar.provider == provider, IntradayEquityBar.exchange_segment == exchange_segment,
        IntradayEquityBar.security_id == security_id, IntradayEquityBar.timeframe == timeframe,
        IntradayEquityBar.bar_start_utc <= as_of_utc,
    ]
    if start_utc is not None:
        if start_utc.tzinfo is None:
            raise ValueError("start_utc must be timezone-aware - a naive datetime cannot be safely compared to bar_start_utc")
        conditions.append(IntradayEquityBar.bar_start_utc >= start_utc)
    rows = session.scalars(select(IntradayEquityBar).where(*conditions).order_by(IntradayEquityBar.bar_start_utc)).all()
    return [
        IntradayBarView(
            security_id=r.security_id, trading_symbol=r.trading_symbol, bar_start_utc=r.bar_start_utc, trading_date=r.trading_date,
            open=r.open, high=r.high, low=r.low, close=r.close, volume=r.volume,
        )
        for r in rows
    ]


def latest_bar_as_of(session: Session, *, security_id: int, timeframe: str, as_of_utc: dt.datetime, **kwargs) -> IntradayBarView | None:
    bars = select_bars_as_of(session, security_id=security_id, timeframe=timeframe, as_of_utc=as_of_utc, **kwargs)
    return bars[-1] if bars else None
