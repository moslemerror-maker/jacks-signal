from __future__ import annotations

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from backend.app.data.canonical.schema import CanonicalBar
from backend.app.data.instruments.matcher import get_or_create_instrument_for_bar
from backend.app.db.models import MarketBar

_UPDATE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "last_price",
    "prev_close",
    "volume",
    "turnover",
    "total_trades",
    "open_interest",
    "source_format",
    "source_file",
)


def upsert_bars(session: Session, bars: list[CanonicalBar]) -> int:
    upserted = 0
    for bar in bars:
        instrument = get_or_create_instrument_for_bar(session, bar)
        stmt = pg_insert(MarketBar).values(
            instrument_id=instrument.id,
            trade_date=bar.trade_date,
            series=bar.series,
            timeframe="1D",
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            last_price=bar.last_price,
            prev_close=bar.prev_close,
            volume=bar.total_traded_qty,
            turnover=bar.total_traded_val,
            total_trades=bar.total_trades,
            open_interest=bar.open_interest,
            source_format=bar.source_format.value,
            source_file=bar.source_file,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_market_bar_identity",
            set_={col: getattr(stmt.excluded, col) for col in _UPDATE_COLUMNS},
        )
        session.execute(stmt)
        upserted += 1
    return upserted
