from __future__ import annotations

import logging
from datetime import date

from sqlalchemy import select

from backend.app.data.instruments.universe import EQUITY_BASKET
from backend.app.db.models import AssetType, Instrument, InstrumentSymbol
from backend.app.db.session import get_session
from backend.app.research.datasets import build_dataset
from backend.app.research.experiments import (
    COST_MODEL_DESCRIPTION,
    HYPOTHESIS_TEXT,
    RULE_DESCRIPTION_TEXT,
    run_experiment,
)
from backend.app.research.features.definitions import register_feature_definitions
from backend.app.research.labels.forward_return import register_label_definition

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRAIN_START, TRAIN_END = date(2015, 1, 1), date(2021, 12, 31)
VALIDATION_START, VALIDATION_END = date(2022, 1, 1), date(2023, 12, 31)
TEST_START, TEST_END = date(2024, 1, 1), date(2026, 9, 11)

COST_BPS_PER_TRADE = 10.0


def _resolve_basket_instrument_ids(session) -> list:
    rows = session.execute(
        select(InstrumentSymbol.instrument_id)
        .join(Instrument, Instrument.id == InstrumentSymbol.instrument_id)
        .where(
            InstrumentSymbol.symbol.in_(EQUITY_BASKET),
            InstrumentSymbol.effective_to.is_(None),
            Instrument.asset_type == AssetType.EQUITY,
        )
    ).all()
    ids = [r[0] for r in rows]
    if len(ids) != len(EQUITY_BASKET):
        raise RuntimeError(
            f"resolved {len(ids)} instruments, expected {len(EQUITY_BASKET)} - "
            "run scripts/compute_stage3_research_data.py first"
        )
    return ids


def main() -> None:
    session = get_session()
    try:
        feature_definitions = register_feature_definitions(session)
        label_definition = register_label_definition(session)
        session.commit()

        instrument_ids = _resolve_basket_instrument_ids(session)

        logger.info(
            "building Stage 5 baseline dataset: %d instruments, %s to %s",
            len(instrument_ids),
            TRAIN_START,
            TEST_END,
        )
        dataset = build_dataset(
            session,
            name="stage5-baseline-25-stock-2015-2026",
            instrument_ids=instrument_ids,
            feature_definition_ids=[fd.id for fd in feature_definitions.values()],
            label_definition_id=label_definition.id,
            start_date=TRAIN_START,
            end_date=TEST_END,
            known_as_of=TEST_END,
            created_by="stage5-script",
        )
        session.commit()
        logger.info(
            "dataset built: id=%s row_count=%d checksum=%s "
            "rows_with_any_missing_feature=%d rows_with_unresolved_label=%d",
            dataset.id,
            dataset.row_count,
            dataset.checksum,
            dataset.rows_with_any_missing_feature,
            dataset.rows_with_unresolved_label,
        )

        run_kwargs = dict(
            dataset_id=dataset.id,
            rule_feature_definition_id=feature_definitions["close_vs_sma20"].id,
            name="stage5-baseline-trend-rule",
            hypothesis=HYPOTHESIS_TEXT,
            rule_description=RULE_DESCRIPTION_TEXT,
            cost_bps_per_trade=COST_BPS_PER_TRADE,
            cost_model_description=COST_MODEL_DESCRIPTION,
            train_start_date=TRAIN_START,
            train_end_date=TRAIN_END,
            validation_start_date=VALIDATION_START,
            validation_end_date=VALIDATION_END,
            test_start_date=TEST_START,
            test_end_date=TEST_END,
            created_by="stage5-script",
        )

        logger.info("running experiment (1st run)...")
        first = run_experiment(session, **run_kwargs)
        session.commit()

        logger.info("running experiment (2nd run, same inputs, to prove determinism)...")
        second = run_experiment(session, **run_kwargs)
        session.commit()

        logger.info("first.id=%s second.id=%s", first.id, second.id)
        logger.info("first.checksum=%s", first.checksum)
        logger.info("second.checksum=%s", second.checksum)
        logger.info("checksums match: %s", first.checksum == second.checksum)
        logger.info("results match:   %s", first.results == second.results)
        logger.info("conclusion: %s", first.conclusion.value)
        logger.info("conclusion_rationale: %s", first.conclusion_rationale)
        for split_name in ("train", "validation", "test"):
            logger.info("--- %s ---", split_name)
            for category, stats in first.results[split_name].items():
                logger.info("  %s: %s", category, stats)
    finally:
        session.close()


if __name__ == "__main__":
    main()
