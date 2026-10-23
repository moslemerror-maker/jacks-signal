from __future__ import annotations

import logging
from datetime import date

from sqlalchemy import select

from backend.app.db.models import Dataset, HypothesisFamily, RunPurpose
from backend.app.db.session import get_session
from backend.app.research.experiment_tracking import register_experiment_definition, run_experiment
from backend.app.research.features.definitions import register_feature_definitions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRAIN_START, TRAIN_END = date(2015, 1, 1), date(2021, 12, 31)
VALIDATION_START, VALIDATION_END = date(2022, 1, 1), date(2023, 12, 31)
TEST_START, TEST_END = date(2024, 1, 1), date(2026, 9, 11)
COST_BPS_PER_TRADE = 10.0


def main() -> None:
    session = get_session()
    try:
        feature_definitions = register_feature_definitions(session)
        session.commit()

        dataset = session.scalar(
            select(Dataset).where(Dataset.name == "stage5-baseline-25-stock-2015-2026")
        )
        if dataset is None:
            raise RuntimeError(
                "expected dataset 'stage5-baseline-25-stock-2015-2026' not found - "
                "run scripts/run_stage5_baseline_experiment.py first"
            )
        logger.info(
            "reusing existing real dataset: id=%s row_count=%d checksum=%s",
            dataset.id,
            dataset.row_count,
            dataset.checksum,
        )

        family = session.scalar(
            select(HypothesisFamily).where(
                HypothesisFamily.name == "stage5-6-trend-following-baseline"
            )
        )
        if family is None:
            raise RuntimeError(
                "expected hypothesis_family 'stage5-6-trend-following-baseline' "
                "not found - run the Stage 3.2 migration first"
            )

        definition = register_experiment_definition(
            session,
            name="stage6-baseline-trend-rule",
            hypothesis_family_id=family.id,
            hypothesis=(
                "A simple trend-following signal - long when close_vs_sma20 is "
                "positive, flat otherwise - has positive mean 5-day forward "
                "return, net of an assumed round-trip cost, in both validation "
                "and test periods."
            ),
            rule_feature_definition_id=feature_definitions["close_vs_sma20"].id,
            rule_description="LONG if close_vs_sma20 > 0 else FLAT. Fixed threshold, no search.",
            cost_model_description=(
                "Flat round-trip bps deducted per LONG trade. Modeled: brokerage "
                "+ STT + a crude flat slippage allowance. Not modeled: market "
                "impact, size-dependent slippage, financing cost, portfolio "
                "effects."
            ),
            created_by="stage6-script",
        )
        session.commit()
        logger.info("experiment_definition id=%s name=%s", definition.id, definition.name)

        base_kwargs = dict(
            experiment_definition_id=definition.id,
            dataset_id=dataset.id,
            rule_feature_definition_id=feature_definitions["close_vs_sma20"].id,
            configuration={"rule": "close_vs_sma20 > 0", "threshold": 0.0},
            cost_bps_per_trade=COST_BPS_PER_TRADE,
            cost_assumptions_description=(
                "10 bps flat round-trip cost per LONG trade; no market impact, "
                "size-dependent slippage, financing cost, or portfolio effects modeled."
            ),
            created_by="stage6-script",
        )

        logger.info(
            "attempting a second CONFIRMATION for this (definition, dataset) pair "
            "- expected to be REJECTED, since a real confirmation for it already "
            "exists from the original Stage 6 run (Stage 3.2's uniqueness rule)..."
        )
        try:
            run_experiment(
                session,
                run_purpose=RunPurpose.CONFIRMATION,
                train_start_date=TRAIN_START,
                train_end_date=TRAIN_END,
                validation_start_date=VALIDATION_START,
                validation_end_date=VALIDATION_END,
                test_start_date=TEST_START,
                test_end_date=TEST_END,
                **base_kwargs,
            )
            raise AssertionError("expected the second confirmation attempt to be rejected")
        except ValueError as exc:
            logger.info("correctly rejected: %s", exc)

        logger.info("running an EXPLORATION (train-only, repeatable) against real data...")
        exploration_run = run_experiment(
            session,
            run_purpose=RunPurpose.EXPLORATION,
            train_start_date=TRAIN_START,
            train_end_date=TRAIN_END,
            **base_kwargs,
        )
        session.commit()
        logger.info(
            "exploration_run.id=%s status=%s metrics=%d (splits=%s)",
            exploration_run.id,
            exploration_run.status.value,
            len(exploration_run.metrics),
            {m.split.value for m in exploration_run.metrics},
        )
        logger.info("git_commit=%s git_dirty=%s", exploration_run.git_commit, exploration_run.git_dirty)
        logger.info("dependency_versions=%s", exploration_run.dependency_versions)
    finally:
        session.close()


if __name__ == "__main__":
    main()
