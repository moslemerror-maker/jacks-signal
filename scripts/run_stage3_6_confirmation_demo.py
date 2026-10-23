from __future__ import annotations

import logging
from datetime import date

from backend.app.db.session import get_session
from backend.app.research.confirmation import MultipleTestingMethod, run_confirmation_evaluation
from backend.app.research.datasets import build_dataset  # noqa: F401 (documents provenance only)
from backend.app.research.experiment_tracking import (
    register_experiment_definition,
    register_hypothesis_family,
)
from backend.app.research.features.definitions import register_feature_definitions
from sqlalchemy import select

from backend.app.db.models import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRAIN_START, TRAIN_END = date(2015, 1, 1), date(2021, 12, 31)
VALIDATION_START, VALIDATION_END = date(2022, 1, 1), date(2023, 12, 31)
TEST_START, TEST_END = date(2024, 1, 1), date(2026, 9, 11)


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
            dataset.id, dataset.row_count, dataset.checksum,
        )

        family = register_hypothesis_family(
            session,
            name="stage3.6-pipeline-integration-demo",
            description=(
                "NOT a new or re-tuned trading hypothesis - a dedicated family "
                "for exercising the Stage 3.6 confirmation pipeline end-to-end "
                "against real data, reusing the exact same close_vs_sma20 rule "
                "as the original Stage 5/6 baseline (whose own confirmation "
                "slot for this dataset is already used)."
            ),
            created_by="stage3.6-script",
        )
        definition = register_experiment_definition(
            session,
            name="stage3.6-confirmation-pipeline-demo",
            hypothesis_family_id=family.id,
            hypothesis=(
                "Pipeline-integration demo only: identical rule to the Stage 5/6 "
                "baseline (LONG if close_vs_sma20 > 0 else FLAT), re-registered "
                "solely to produce a real confirmation_report via the new Stage "
                "3.6 pathway. Not a new research claim."
            ),
            rule_feature_definition_id=feature_definitions["close_vs_sma20"].id,
            rule_description="LONG if close_vs_sma20 > 0 else FLAT. Fixed threshold, no search.",
            cost_model_description=(
                "Simplified flat round-trip bps sweep - see confirmation_report."
                "cost_model_caveat for the explicit non-realism disclaimer."
            ),
            created_by="stage3.6-script",
        )
        session.commit()
        logger.info("hypothesis_family id=%s", family.id)
        logger.info("experiment_definition id=%s (first-ever run for this definition)", definition.id)

        run = run_confirmation_evaluation(
            session,
            experiment_definition_id=definition.id,
            dataset_id=dataset.id,
            rule_feature_definition_id=feature_definitions["close_vs_sma20"].id,
            configuration={"rule": "close_vs_sma20 > 0", "threshold": 0.0},
            train_start_date=TRAIN_START,
            train_end_date=TRAIN_END,
            validation_start_date=VALIDATION_START,
            validation_end_date=VALIDATION_END,
            test_start_date=TEST_START,
            test_end_date=TEST_END,
            alpha=0.05,
            multiple_testing_method=MultipleTestingMethod.BONFERRONI,
            cost_bps_sweep=[0.0, 5.0, 10.0, 20.0, 50.0],
            created_by="stage3.6-script",
        )
        session.commit()

        logger.info("run.id=%s status=%s run_purpose=%s", run.id, run.status.value, run.run_purpose.value)
        logger.info("conclusion=%s (must be None)", run.conclusion)
        logger.info("checksum=%s", run.checksum)

        report = run.confirmation_report
        logger.info("--- multiple_testing ---")
        logger.info("%s", report["multiple_testing"])
        logger.info("--- inference_method ---")
        logger.info("%s", report["inference_method"])
        for split in ("train", "validation", "test"):
            logger.info("--- descriptive_performance[%s] ---", split)
            logger.info("%s", report["descriptive_performance"][split])
            logger.info("--- portfolio_performance[%s] ---", split)
            logger.info("%s", report["portfolio_performance"][split])
        logger.info("--- statistical_inference[validation] ---")
        logger.info("%s", report["statistical_inference"]["validation"])
        logger.info("--- statistical_inference[test] ---")
        logger.info("%s", report["statistical_inference"]["test"])
        logger.info("--- cost_sensitivity ---")
        for level, stats in report["cost_sensitivity"].items():
            logger.info("  %s: mean=%s count=%s", level, stats["mean"], stats["count"])
        logger.info("--- sample_size ---")
        logger.info("%s", report["sample_size"])
        logger.info("--- unresolved_limitations ---")
        for item in report["unresolved_limitations"]:
            logger.info("  - %s", item)

        logger.info("attempting a second confirmation for this same (definition, dataset) - expecting rejection...")
        try:
            run_confirmation_evaluation(
                session,
                experiment_definition_id=definition.id,
                dataset_id=dataset.id,
                rule_feature_definition_id=feature_definitions["close_vs_sma20"].id,
                configuration={"rule": "close_vs_sma20 > 0", "threshold": 0.0},
                train_start_date=TRAIN_START,
                train_end_date=TRAIN_END,
                validation_start_date=VALIDATION_START,
                validation_end_date=VALIDATION_END,
                test_start_date=TEST_START,
                test_end_date=TEST_END,
                alpha=0.05,
                multiple_testing_method=MultipleTestingMethod.BONFERRONI,
                cost_bps_sweep=[0.0, 10.0],
                created_by="stage3.6-script",
            )
            raise AssertionError("expected the second confirmation attempt to be rejected")
        except ValueError as exc:
            logger.info("correctly rejected: %s", exc)
    finally:
        session.close()


if __name__ == "__main__":
    main()
