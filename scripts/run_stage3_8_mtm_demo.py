from __future__ import annotations

import logging

from sqlalchemy import select

from backend.app.db.models import FeatureDefinition
from backend.app.db.session import get_session
from backend.app.research.features.compute import load_eq_bars
from backend.app.research.features.definitions import register_feature_definitions
from backend.app.research.labels.forward_return_next_open import (
    compute_and_store_next_open_labels,
    register_next_open_label_definition,
)
from backend.app.research.position_lifecycle import (
    MissingBarError,
    compute_daily_mark_to_market_portfolio_series,
    simulate_position_lifecycle,
)
from scripts.run_stage3_7_position_lifecycle_demo import (
    MAX_SIMULTANEOUS_POSITIONS,
    SLIPPAGE_BPS_PER_SIDE,
    TRANSACTION_COST_BPS_PER_SIDE,
    _build_signal_days,
    _resolve_basket_instrument_ids,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NUM_SIZING_SLOTS = MAX_SIMULTANEOUS_POSITIONS


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

        all_signal_days = []
        session_date_set = set()
        bars_by_instrument = {}
        for symbol, instrument_id in sorted(instrument_ids.items()):
            compute_and_store_next_open_labels(session, instrument_id, label_definition)
            session.commit()
            all_signal_days.extend(
                _build_signal_days(
                    session, instrument_id, close_vs_sma20_id, label_definition.horizon_days
                )
            )
            bars = load_eq_bars(session, instrument_id)
            bars_by_instrument[instrument_id] = bars
            session_date_set.update(b.trade_date for b in bars)

        session_dates = sorted(session_date_set)
        logger.info("session_dates span: %s .. %s (%d dates)",
                    session_dates[0], session_dates[-1], len(session_dates))

        lifecycle = simulate_position_lifecycle(
            all_signal_days,
            TRANSACTION_COST_BPS_PER_SIDE,
            SLIPPAGE_BPS_PER_SIDE,
            MAX_SIMULTANEOUS_POSITIONS,
        )
        logger.info(
            "lifecycle: completed=%d open_at_cutoff=%d foregone=%d",
            len(lifecycle.completed_events), len(lifecycle.open_at_cutoff),
            len(lifecycle.foregone_signals),
        )

        try:
            series = compute_daily_mark_to_market_portfolio_series(
                lifecycle,
                bars_by_instrument,
                session_dates,
                NUM_SIZING_SLOTS,
                MAX_SIMULTANEOUS_POSITIONS,
                TRANSACTION_COST_BPS_PER_SIDE,
                SLIPPAGE_BPS_PER_SIDE,
            )
        except MissingBarError as exc:
            logger.warning(
                "MTM evaluation aborted by a real data-quality incident (this IS the "
                "designed behavior, not a bug): instrument_id=%s trade_date=%s reason=%s",
                exc.incident.instrument_id, exc.incident.trade_date, exc.incident.reason,
            )
            return

        n = len(series.records)
        final = series.records[-1]
        max_exposure = max(r.exposure for r in series.records)
        avg_exposure = sum(r.exposure for r in series.records) / n
        total_turnover = sum(r.turnover for r in series.records)
        total_entries = sum(r.num_entries for r in series.records)
        total_exits = sum(r.num_exits for r in series.records)

        logger.info("price_basis=%s corporate_action_adjustment=%s",
                    series.price_basis, series.corporate_action_adjustment)
        logger.info("daily records: %d (%s .. %s)", n, series.records[0].trade_date, final.trade_date)
        logger.info("final cumulative_equity: %.6f", final.cumulative_equity)
        logger.info("max exposure: %.4f  avg exposure: %.4f", max_exposure, avg_exposure)
        logger.info("total turnover: %.4f  total entries: %d  total exits: %d",
                     total_turnover, total_entries, total_exits)
        logger.info(
            "note: unadjusted raw OHLC - any position spanning a real split/bonus/rights "
            "issue in the underlying data will show an economically wrong return; no "
            "conclusion should be drawn from these numbers until corporate-action "
            "adjustment is implemented in a future stage."
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
