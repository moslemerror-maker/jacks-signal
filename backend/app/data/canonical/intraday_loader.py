from __future__ import annotations

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from ..canonical.intraday_equity_bar import CanonicalIntradayEquityBar
from ...db.models import IntradayEquityBar

_UPDATE_COLUMNS = (
    "trading_symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trading_date",
    "raw_timestamp_epoch",
    "source_provider",
    "source_dataset",
    "source_key",
    "source_sha256",
    "normalizer_version",
)


def upsert_intraday_bars(session: Session, bars: list[CanonicalIntradayEquityBar], *, timeframe: str = "1m") -> int:
    upserted = 0
    for bar in bars:
        stmt = pg_insert(IntradayEquityBar).values(
            provider=bar.provider.value,
            exchange_segment=bar.exchange_segment,
            security_id=bar.security_id,
            trading_symbol=bar.trading_symbol,
            timeframe=timeframe,
            bar_start_utc=bar.bar_start_utc,
            trading_date=bar.trading_date,
            raw_timestamp_epoch=bar.raw_timestamp_epoch,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            source_provider=bar.source_provider,
            source_dataset=bar.source_dataset,
            source_key=bar.source_key,
            source_sha256=bar.source_sha256,
            normalizer_version=bar.normalizer_version,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_intraday_equity_bar_identity",
            set_={col: getattr(stmt.excluded, col) for col in _UPDATE_COLUMNS},
        )
        session.execute(stmt)
        upserted += 1
    return upserted


def _row_dict(bar: CanonicalIntradayEquityBar, timeframe: str) -> dict:
    return dict(
        provider=bar.provider.value, exchange_segment=bar.exchange_segment, security_id=bar.security_id,
        trading_symbol=bar.trading_symbol, timeframe=timeframe, bar_start_utc=bar.bar_start_utc,
        trading_date=bar.trading_date, raw_timestamp_epoch=bar.raw_timestamp_epoch, open=bar.open, high=bar.high,
        low=bar.low, close=bar.close, volume=bar.volume, source_provider=bar.source_provider,
        source_dataset=bar.source_dataset, source_key=bar.source_key, source_sha256=bar.source_sha256,
        normalizer_version=bar.normalizer_version,
    )


_PARAMS_PER_ROW = 19
_MAX_SAFE_BATCH_SIZE = 65_535 // _PARAMS_PER_ROW


def upsert_intraday_bars_batched(session: Session, bars: list[CanonicalIntradayEquityBar], *, timeframe: str = "1m", batch_size: int = 2000) -> int:
    if batch_size > _MAX_SAFE_BATCH_SIZE:
        raise ValueError(
            f"batch_size={batch_size} would exceed PostgreSQL's 65,535-bound-parameter limit at "
            f"{_PARAMS_PER_ROW} params/row ({batch_size * _PARAMS_PER_ROW} params) - use batch_size <= {_MAX_SAFE_BATCH_SIZE}"
        )
    upserted = 0
    for start in range(0, len(bars), batch_size):
        chunk = bars[start : start + batch_size]
        deduped: dict[tuple, CanonicalIntradayEquityBar] = {}
        for b in chunk:
            deduped[(b.provider.value, b.exchange_segment, b.security_id, timeframe, b.bar_start_utc)] = b
        stmt = pg_insert(IntradayEquityBar).values([_row_dict(b, timeframe) for b in deduped.values()])
        stmt = stmt.on_conflict_do_update(
            constraint="uq_intraday_equity_bar_identity",
            set_={col: getattr(stmt.excluded, col) for col in _UPDATE_COLUMNS},
        )
        session.execute(stmt)
        upserted += len(deduped)
    return upserted
