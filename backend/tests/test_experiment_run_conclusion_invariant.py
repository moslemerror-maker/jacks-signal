from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from backend.app.db.models import (
    AssetType,
    ExperimentConclusion,
    ExperimentRun,
    Instrument,
    MarketBar,
    RunPurpose,
    RunStatus,
    SourceFormat,
)
from backend.app.research.datasets import build_dataset
from backend.app.research.experiment_tracking import (
    register_experiment_definition,
    register_hypothesis_family,
    run_experiment,
)
from backend.app.research.features.compute import compute_and_store_features
from backend.app.research.features.definitions import register_feature_definitions
from backend.app.research.labels.forward_return import (
    compute_and_store_labels,
    register_label_definition,
)

_TRADING_DATES = [date(2024, 1, 1) + timedelta(days=i) for i in range(60)]
_CLOSES = [100.0 + i for i in range(60)]


@pytest.fixture
def instrument(db_session) -> Instrument:
    inst = Instrument(exchange="NSE", asset_type=AssetType.EQUITY)
    db_session.add(inst)
    db_session.flush()
    return inst


@pytest.fixture
def seeded(db_session, instrument):
    feature_definitions = register_feature_definitions(db_session)
    label_definition = register_label_definition(db_session)
    db_session.flush()

    for trade_date, close in zip(_TRADING_DATES, _CLOSES):
        db_session.add(
            MarketBar(
                instrument_id=instrument.id,
                trade_date=trade_date,
                series="EQ",
                timeframe="1D",
                open=close,
                high=close + 1.0,
                low=close - 1.0,
                close=close,
                source_format=SourceFormat.NSE_UDIFF,
                source_file="synthetic-test-fixture",
            )
        )
    db_session.flush()
    compute_and_store_features(db_session, instrument.id, feature_definitions)
    compute_and_store_labels(db_session, instrument.id, label_definition)

    dataset = build_dataset(
        db_session,
        name="stage3.2.1-test-dataset",
        instrument_ids=[instrument.id],
        feature_definition_ids=[fd.id for fd in feature_definitions.values()],
        label_definition_id=label_definition.id,
        start_date=_TRADING_DATES[0],
        end_date=_TRADING_DATES[-1],
        known_as_of=_TRADING_DATES[-1],
        created_by="test-suite",
    )
    family = register_hypothesis_family(
        db_session,
        name="stage3.2.1-test-family",
        description="test family for the conclusion-invariant suite",
        created_by="test-suite",
    )
    definition = register_experiment_definition(
        db_session,
        name="stage3.2.1-test-definition",
        hypothesis_family_id=family.id,
        hypothesis="test hypothesis",
        rule_feature_definition_id=feature_definitions["close_vs_sma20"].id,
        rule_description="LONG if close_vs_sma20 > 0 else FLAT",
        cost_model_description="flat bps deduction",
        created_by="test-suite",
    )
    return {
        "instrument": instrument,
        "feature_definitions": feature_definitions,
        "dataset": dataset,
        "definition": definition,
    }


def _bare_run_kwargs(seeded, **overrides) -> dict:
    kwargs = dict(
        id=uuid.uuid4(),
        experiment_definition_id=seeded["definition"].id,
        dataset_id=seeded["dataset"].id,
        rule_feature_definition_id=seeded["feature_definitions"]["close_vs_sma20"].id,
        run_purpose=RunPurpose.EXPLORATION,
        git_commit="0" * 40,
        git_dirty=False,
        dependency_versions={},
        configuration={},
        cost_assumptions={"cost_bps_per_trade": 10.0, "description": "x"},
        train_start_date=_TRADING_DATES[0],
        train_end_date=_TRADING_DATES[29],
        validation_start_date=None,
        validation_end_date=None,
        test_start_date=None,
        test_end_date=None,
        executed_at=datetime.now(timezone.utc),
        status=RunStatus.COMPLETED,
        failure_reason=None,
        conclusion=None,
        conclusion_rationale=None,
        checksum="0" * 64,
        created_by="test-suite",
    )
    kwargs.update(overrides)
    return kwargs


class TestExplorationCannotHaveAConclusionOnInsert:
    def test_completed_exploration_with_a_conclusion_is_rejected(self, db_session, seeded):
        db_session.add(
            ExperimentRun(
                **_bare_run_kwargs(
                    seeded,
                    conclusion=ExperimentConclusion.ACCEPT,
                    conclusion_rationale="smuggled in directly, bypassing run_experiment",
                )
            )
        )
        with pytest.raises(IntegrityError, match="ck_experiment_run_conclusion_matches_status"):
            db_session.flush()

    def test_completed_exploration_without_a_conclusion_is_accepted(self, db_session, seeded):
        run = ExperimentRun(**_bare_run_kwargs(seeded, conclusion=None))
        db_session.add(run)
        db_session.flush()
        assert run.conclusion is None


class TestConfirmationRequiresAConclusionOnInsert:
    def test_completed_confirmation_without_a_conclusion_is_rejected(self, db_session, seeded):
        db_session.add(
            ExperimentRun(
                **_bare_run_kwargs(
                    seeded,
                    run_purpose=RunPurpose.CONFIRMATION,
                    validation_start_date=_TRADING_DATES[30],
                    validation_end_date=_TRADING_DATES[39],
                    test_start_date=_TRADING_DATES[40],
                    test_end_date=_TRADING_DATES[54],
                    conclusion=None,
                )
            )
        )
        with pytest.raises(IntegrityError, match="ck_experiment_run_conclusion_matches_status"):
            db_session.flush()

    def test_completed_confirmation_with_a_conclusion_is_accepted(self, db_session, seeded):
        run = ExperimentRun(
            **_bare_run_kwargs(
                seeded,
                run_purpose=RunPurpose.CONFIRMATION,
                validation_start_date=_TRADING_DATES[30],
                validation_end_date=_TRADING_DATES[39],
                test_start_date=_TRADING_DATES[40],
                test_end_date=_TRADING_DATES[54],
                conclusion=ExperimentConclusion.NEEDS_MORE_TESTING,
                conclusion_rationale="genuine",
            )
        )
        db_session.add(run)
        db_session.flush()


class TestFailedRunCannotHaveAConclusionOnInsert:
    def test_failed_run_with_a_conclusion_is_rejected(self, db_session, seeded):
        db_session.add(
            ExperimentRun(
                **_bare_run_kwargs(
                    seeded,
                    status=RunStatus.FAILED,
                    failure_reason="synthetic",
                    conclusion=ExperimentConclusion.REJECT,
                )
            )
        )
        with pytest.raises(IntegrityError, match="ck_experiment_run_conclusion_matches_status"):
            db_session.flush()


class TestConstraintAlsoGuardsUpdatePath:

    def test_update_smuggling_a_conclusion_onto_an_exploration_run_is_rejected(
        self, db_session, seeded
    ):
        run = ExperimentRun(**_bare_run_kwargs(seeded, conclusion=None))
        db_session.add(run)
        db_session.flush()

        db_session.execute(
            text("ALTER TABLE experiment_runs DISABLE TRIGGER trg_experiment_runs_no_update")
        )
        with pytest.raises(IntegrityError, match="ck_experiment_run_conclusion_matches_status"):
            db_session.execute(
                text("UPDATE experiment_runs SET conclusion = 'ACCEPT' WHERE id = :id"),
                {"id": str(run.id)},
            )


class TestRealRunExperimentStillProducesValidRows:

    def test_exploration_run_via_run_experiment_has_no_conclusion(self, db_session, seeded):
        run = run_experiment(
            db_session,
            experiment_definition_id=seeded["definition"].id,
            dataset_id=seeded["dataset"].id,
            rule_feature_definition_id=seeded["feature_definitions"]["close_vs_sma20"].id,
            configuration={"rule": "close_vs_sma20 > 0"},
            cost_bps_per_trade=10.0,
            cost_assumptions_description="flat bps",
            run_purpose=RunPurpose.EXPLORATION,
            train_start_date=_TRADING_DATES[0],
            train_end_date=_TRADING_DATES[29],
            created_by="test-suite",
        )
        assert run.status == RunStatus.COMPLETED
        assert run.conclusion is None

    def test_confirmation_run_via_run_experiment_has_a_conclusion(self, db_session, seeded):
        run = run_experiment(
            db_session,
            experiment_definition_id=seeded["definition"].id,
            dataset_id=seeded["dataset"].id,
            rule_feature_definition_id=seeded["feature_definitions"]["close_vs_sma20"].id,
            configuration={"rule": "close_vs_sma20 > 0"},
            cost_bps_per_trade=10.0,
            cost_assumptions_description="flat bps",
            run_purpose=RunPurpose.CONFIRMATION,
            train_start_date=_TRADING_DATES[0],
            train_end_date=_TRADING_DATES[29],
            validation_start_date=_TRADING_DATES[30],
            validation_end_date=_TRADING_DATES[39],
            test_start_date=_TRADING_DATES[40],
            test_end_date=_TRADING_DATES[54],
            created_by="test-suite",
        )
        assert run.status == RunStatus.COMPLETED
        assert run.conclusion is not None
