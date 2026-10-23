from __future__ import annotations

import logging
import uuid
from collections import Counter

from sqlalchemy import select

from backend.app.data.instruments.universe import EQUITY_BASKET
from backend.app.db.models import (
    AssetType,
    FeatureDefinition,
    FeatureValue,
    Instrument,
    InstrumentSymbol,
)
from backend.app.db.session import get_session
from backend.app.research.features.definitions import register_feature_definitions
from backend.app.research.labels.forward_return_next_open import (
    compute_and_store_next_open_labels,
    compute_forward_return_next_open,
    load_eq_open_bars,
    register_next_open_label_definition,
)
from backend.app.research.position_lifecycle import (
    NextOpenOutcome,
    SignalDay,
    simulate_position_lifecycle,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRANSACTION_COST_BPS_PER_SIDE = 10.0
SLIPPAGE_BPS_PER_SIDE = 5.0
MAX_SIMULTANEOUS_POSITIONS = 5


def _resolve_basket_instrument_ids(session) -> dict[str, uuid.UUID]:
    rows = session.execute(
        select(InstrumentSymbol.symbol, InstrumentSymbol.instrument_id)
        .join(Instrument, Instrument.id == InstrumentSymbol.instrument_id)
        .where(
            InstrumentSymbol.symbol.in_(EQUITY_BASKET),
            InstrumentSymbol.effective_to.is_(None),
            Instrument.asset_type == AssetType.EQUITY,
        )
    ).all()
    result = {symbol: instrument_id for symbol, instrument_id in rows}
    missing = set(EQUITY_BASKET) - result.keys()
    if missing:
        raise RuntimeError(
            f"could not resolve current instrument for basket symbols: {sorted(missing)}"
        )
    return result


def _build_signal_days(session, instrument_id, close_vs_sma20_id, horizon_days):
    feature_by_date = {
        r.computed_for_date: r.value
        for r in session.scalars(
            select(FeatureValue).where(
                FeatureValue.instrument_id == instrument_id,
                FeatureValue.feature_definition_id == close_vs_sma20_id,
            )
        ).all()
    }
    bars = load_eq_open_bars(session, instrument_id)

    signal_days = []
    for index, bar in enumerate(bars):
        feature_value = feature_by_date.get(bar.trade_date)
        if feature_value is None:
            continue
        is_long = feature_value > 0
        result = compute_forward_return_next_open(bars, index, horizon_days)
        next_open_outcome = None
        if result is not None:
            entry_date, exit_date, entry_price, exit_price, realized_return = result
            next_open_outcome = NextOpenOutcome(
                entry_date=entry_date,
                entry_price=entry_price,
                exit_date=exit_date,
                exit_price=exit_price,
                gross_return=realized_return,
                holding_sessions=horizon_days,
            )
        signal_days.append(SignalDay(instrument_id, bar.trade_date, is_long, next_open_outcome))
    return signal_days


def main() -> None:
    session = get_session()
    try:
        register_feature_definitions(session)
        label_definition = register_next_open_label_definition(session)
        session.commit()

        instrument_ids = _resolve_basket_instrument_ids(session)

        close_vs_sma20_id = session.scalar(
            select(FeatureDefinition.id).where(FeatureDefinition.name == "close_vs_sma20")
        )
        if close_vs_sma20_id is None:
            raise RuntimeError(
                "close_vs_sma20 feature definition not found - run "
                "scripts/compute_stage3_research_data.py first"
            )

        total_label_rows_written = 0
        all_signal_days: list[SignalDay] = []
        for symbol, instrument_id in sorted(instrument_ids.items()):
            written = compute_and_store_next_open_labels(session, instrument_id, label_definition)
            session.commit()
            total_label_rows_written += written
            logger.info("%s: forward_return_5d_next_open=+%d", symbol, written)
            all_signal_days.extend(
                _build_signal_days(
                    session, instrument_id, close_vs_sma20_id, label_definition.horizon_days
                )
            )

        logger.info(
            "total new forward_return_5d_next_open rows written: %d", total_label_rows_written
        )
        logger.info("total signal-day rows assembled across basket: %d", len(all_signal_days))

        result = simulate_position_lifecycle(
            all_signal_days,
            TRANSACTION_COST_BPS_PER_SIDE,
            SLIPPAGE_BPS_PER_SIDE,
            MAX_SIMULTANEOUS_POSITIONS,
        )

        n_completed = len(result.completed_events)
        n_open = len(result.open_at_cutoff)
        foregone_reasons = Counter(f.reason for f in result.foregone_signals)

        if n_completed:
            avg_gross = sum(e.gross_return for e in result.completed_events) / n_completed
            avg_net = sum(e.net_return for e in result.completed_events) / n_completed
            avg_holding = sum(e.holding_sessions for e in result.completed_events) / n_completed
        else:
            avg_gross = avg_net = avg_holding = float("nan")

        logger.info("completed positions: %d", n_completed)
        logger.info("open at cutoff (unresolved): %d", n_open)
        logger.info("foregone signals by reason: %s", dict(foregone_reasons))
        logger.info("avg gross_return (completed): %.6f", avg_gross)
        logger.info("avg net_return (completed):   %.6f", avg_net)
        logger.info("avg holding_sessions:         %.2f", avg_holding)
        logger.info(
            "note: cost/slippage params used here (%.1fbps/side each) are a demo "
            "assumption, not a calibrated real cost model - no trading conclusion "
            "is drawn from these numbers.",
            TRANSACTION_COST_BPS_PER_SIDE,
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
