from __future__ import annotations

import datetime
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.data.instruments.corporate_actions import load_adjustment_gate_inputs
from backend.app.db.models import MarketBar
from backend.app.research.adjustment_eligibility import evaluate_adjustment_eligibility
from backend.app.research.adjustment_factors import AdjustedView, RawBar, build_adjusted_view


def read_adjusted_bars(
    session: Session,
    instrument_id: uuid.UUID,
    start: datetime.date,
    as_of: datetime.date,
    *,
    series: str = "EQ",
    timeframe: str = "1D",
    allow_missing_bars: bool = False,
) -> AdjustedView:
    if start > as_of:
        raise ValueError(f"start {start} is after as_of {as_of}")

    gate = load_adjustment_gate_inputs(session, instrument_id, start, as_of)
    eligibility = evaluate_adjustment_eligibility(
        start=start, end=as_of, coverage_status=gate.coverage_status, governing_assessment_id=gate.governing_assessment_id,
        events=gate.events, register_items=gate.register_items)

    M = MarketBar
    rows = session.execute(
        select(M.id, M.trade_date, M.series, M.timeframe, M.open, M.high, M.low, M.close)
        .where(M.instrument_id == instrument_id, M.series == series, M.timeframe == timeframe,
               M.trade_date >= start, M.trade_date <= as_of)
        .order_by(M.trade_date)
    ).all()
    raw = tuple(RawBar(str(r.id), r.trade_date, r.series, r.timeframe, r.open, r.high, r.low, r.close) for r in rows)

    sessions: tuple = ()
    if raw:
        sessions = tuple(session.scalars(
            select(M.trade_date)
            .where(M.series == series, M.timeframe == timeframe, M.trade_date >= raw[0].trade_date, M.trade_date <= as_of)
            .distinct().order_by(M.trade_date)
        ))

    return build_adjusted_view(
        instrument_id=str(instrument_id), series=series, timeframe=timeframe, requested_start=start, as_of=as_of,
        raw_bars=raw, session_dates=sessions, eligibility=eligibility, gate_events=gate.events,
        allow_missing_bars=allow_missing_bars)
