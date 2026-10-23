from __future__ import annotations

import datetime
import uuid
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from backend.app.db.models import LabelDefinition, LabelValue
from backend.app.research.features.compute import load_eq_bars
from backend.app.research.features.types import Bar

HORIZON_DAYS = 5

STAGE_3_LABEL_DEFINITION: dict = dict(
    name="forward_return_5d",
    version=1,
    description=(
        "5-trading-day forward close-to-close return (mainboard "
        "EQ-series bars only): realized_return = "
        "close[entry + 5 EQ bars] / close[entry] - 1. Pure two-point "
        "return, not a target/stop outcome - hit_target/hit_stop/mfe/mae "
        "are intentionally NULL. See forward_return.py's module "
        "docstring for the full exact definition."
    ),
    formula_ref="backend.app.research.labels.forward_return:compute_forward_return",
    horizon_days=HORIZON_DAYS,
    target_stop_definition=None,
    same_bar_resolution_rule=(
        "N/A - pure two-point return between two specific closes; no "
        "path-dependent target/stop check exists for this label."
    ),
    timeframe="1D",
)


def compute_forward_return(
    bars: Sequence[Bar], index: int, horizon_days: int = HORIZON_DAYS
) -> tuple[datetime.date, float] | None:
    target_index = index + horizon_days
    if target_index >= len(bars):
        return None
    entry_close = bars[index].close
    if entry_close == 0:
        return None
    horizon_bar = bars[target_index]
    return horizon_bar.trade_date, horizon_bar.close / entry_close - 1.0


def register_label_definition(
    session: Session,
    spec: dict = STAGE_3_LABEL_DEFINITION,
    created_by: str = "stage3",
) -> LabelDefinition:
    existing = session.scalar(
        select(LabelDefinition).where(
            LabelDefinition.name == spec["name"], LabelDefinition.version == spec["version"]
        )
    )
    if existing is not None:
        return existing
    ld = LabelDefinition(created_by=created_by, **spec)
    session.add(ld)
    session.flush()
    return ld


def _bulk_store_if_absent(session: Session, rows: list[dict]) -> int:
    if not rows:
        return 0
    stmt = (
        pg_insert(LabelValue)
        .on_conflict_do_nothing(constraint="uq_label_value_identity")
        .returning(LabelValue.id)
    )
    return len(session.execute(stmt, rows).fetchall())


def compute_and_store_labels(
    session: Session, instrument_id: uuid.UUID, label_definition: LabelDefinition
) -> int:
    bars = load_eq_bars(session, instrument_id)
    rows = []
    for index in range(len(bars)):
        result = compute_forward_return(bars, index, label_definition.horizon_days)
        if result is None:
            continue
        horizon_end_date, realized_return = result
        rows.append(
            dict(
                instrument_id=instrument_id,
                label_definition_id=label_definition.id,
                entry_date=bars[index].trade_date,
                horizon_end_date=horizon_end_date,
                timeframe=label_definition.timeframe,
                realized_return=realized_return,
                hit_target=None,
                hit_stop=None,
                mfe=None,
                mae=None,
                holding_time_days=label_definition.horizon_days,
            )
        )
    written = _bulk_store_if_absent(session, rows)
    session.flush()
    return written
