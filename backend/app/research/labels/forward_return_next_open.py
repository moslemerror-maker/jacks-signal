from __future__ import annotations

import dataclasses
import datetime
import uuid
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from backend.app.db.models import LabelDefinition, LabelValue, MarketBar

DEFAULT_HORIZON_DAYS = 5

STAGE_3_7_NEXT_OPEN_LABEL_DEFINITION: dict = dict(
    name="forward_return_5d_next_open",
    version=1,
    description=(
        "5-trading-session next-open-to-open return (mainboard EQ-series "
        "bars only): realized_return = open[signal + 1 + horizon_days EQ "
        "bars] / open[signal + 1 EQ bar] - 1. Captures the overnight gap "
        "between the signal-forming close and the next session's open - "
        "NOT fully execution-realistic (no market-impact/slippage "
        "modeled, prices unadjusted for corporate actions). See "
        "forward_return_next_open.py's module docstring for the full "
        "definition. forward_return_5d is unchanged by this label."
    ),
    formula_ref=(
        "backend.app.research.labels.forward_return_next_open:"
        "compute_forward_return_next_open"
    ),
    horizon_days=DEFAULT_HORIZON_DAYS,
    target_stop_definition=None,
    same_bar_resolution_rule=(
        "N/A - pure two-point open-to-open return; no path-dependent "
        "target/stop check exists for this label."
    ),
    timeframe="1D",
)


@dataclasses.dataclass(frozen=True)
class OpenBar:

    trade_date: datetime.date
    open: float


def load_eq_open_bars(
    session: Session, instrument_id: uuid.UUID, timeframe: str = "1D"
) -> list[OpenBar]:
    rows = session.scalars(
        select(MarketBar)
        .where(
            MarketBar.instrument_id == instrument_id,
            MarketBar.series == "EQ",
            MarketBar.timeframe == timeframe,
        )
        .order_by(MarketBar.trade_date.asc())
    ).all()
    return [OpenBar(trade_date=r.trade_date, open=r.open) for r in rows]


def compute_forward_return_next_open(
    bars: Sequence[OpenBar], index: int, horizon_days: int = DEFAULT_HORIZON_DAYS
) -> tuple[datetime.date, datetime.date, float, float, float] | None:
    entry_index = index + 1
    exit_index = entry_index + horizon_days
    if exit_index >= len(bars):
        return None
    entry_bar = bars[entry_index]
    exit_bar = bars[exit_index]
    if entry_bar.open == 0:
        return None
    return (
        entry_bar.trade_date,
        exit_bar.trade_date,
        entry_bar.open,
        exit_bar.open,
        exit_bar.open / entry_bar.open - 1.0,
    )


def register_next_open_label_definition(
    session: Session,
    spec: dict = STAGE_3_7_NEXT_OPEN_LABEL_DEFINITION,
    created_by: str = "stage3.7",
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


def compute_and_store_next_open_labels(
    session: Session, instrument_id: uuid.UUID, label_definition: LabelDefinition
) -> int:
    bars = load_eq_open_bars(session, instrument_id)
    rows = []
    for index in range(len(bars)):
        result = compute_forward_return_next_open(bars, index, label_definition.horizon_days)
        if result is None:
            continue
        _entry_date, exit_date, _entry_price, _exit_price, realized_return = result
        rows.append(
            dict(
                instrument_id=instrument_id,
                label_definition_id=label_definition.id,
                entry_date=bars[index].trade_date,
                horizon_end_date=exit_date,
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
