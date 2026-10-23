from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from backend.app.db.models import FeatureDefinition, FeatureValue, MarketBar
from backend.app.research.features.price import compute_atr, compute_return_n_day
from backend.app.research.features.trend import compute_close_vs_sma
from backend.app.research.features.types import Bar


def load_eq_bars(
    session: Session, instrument_id: uuid.UUID, timeframe: str = "1D"
) -> list[Bar]:
    rows = session.scalars(
        select(MarketBar)
        .where(
            MarketBar.instrument_id == instrument_id,
            MarketBar.series == "EQ",
            MarketBar.timeframe == timeframe,
        )
        .order_by(MarketBar.trade_date.asc())
    ).all()
    return [
        Bar(trade_date=r.trade_date, close=r.close, high=r.high, low=r.low) for r in rows
    ]


def _bulk_store_if_absent(session: Session, rows: list[dict]) -> int:
    if not rows:
        return 0
    stmt = (
        pg_insert(FeatureValue)
        .on_conflict_do_nothing(constraint="uq_feature_value_identity")
        .returning(FeatureValue.id)
    )
    return len(session.execute(stmt, rows).fetchall())


def compute_and_store_features(
    session: Session,
    instrument_id: uuid.UUID,
    definitions: dict[str, FeatureDefinition],
) -> dict[str, int]:
    bars = load_eq_bars(session, instrument_id)
    rows_by_feature: dict[str, list[dict]] = {
        "return_5d": [],
        "close_vs_sma20": [],
        "atr_14": [],
    }
    for index, bar in enumerate(bars):
        computations = {
            "return_5d": compute_return_n_day(bars, index, n=5),
            "close_vs_sma20": compute_close_vs_sma(bars, index, window=20),
            "atr_14": compute_atr(bars, index, window=14),
        }
        for name, value in computations.items():
            if value is None:
                continue
            fd = definitions[name]
            rows_by_feature[name].append(
                dict(
                    instrument_id=instrument_id,
                    feature_definition_id=fd.id,
                    computed_for_date=bar.trade_date,
                    timeframe=fd.timeframe,
                    value=value,
                    available_at=bar.trade_date,
                )
            )
    written = {
        name: _bulk_store_if_absent(session, rows) for name, rows in rows_by_feature.items()
    }
    session.flush()
    return written
