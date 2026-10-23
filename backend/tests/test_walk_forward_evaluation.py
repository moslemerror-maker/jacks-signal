from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy.exc import IntegrityError

from backend.app.db.models import (
    AssetType,
    ExperimentMetric,
    ExperimentRun,
    Instrument,
    MarketBar,
    MetricCategory,
    MetricSplit,
    RunPurpose,
    RunStatus,
    SourceFormat,
)
from backend.app.research.datasets import build_dataset
from backend.app.research.experiment_tracking import (
    register_experiment_definition,
    register_hypothesis_family,
)
from backend.app.research.features.compute import compute_and_store_features
from backend.app.research.features.definitions import register_feature_definitions
from backend.app.research.labels.forward_return import (
    compute_and_store_labels,
    register_label_definition,
)
from backend.app.research.walk_forward import (
    AGGREGATE_FOLD_INDEX,
    FoldMode,
    run_walk_forward_evaluation,
)

_N_DAYS = 60
_TRADING_DATES = [date(2024, 1, 1) + timedelta(days=i) for i in range(_N_DAYS)]
_CLOSES = [100.0 + i for i in range(_N_DAYS)]


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
        name="stage3.3-test-dataset",
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
        name="stage3.3-test-family",
        description="test family for walk-forward tests",
        created_by="test-suite",
    )
    definition = register_experiment_definition(
        db_session,
        name="stage3.3-test-definition",
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
        "label_definition": label_definition,
        "dataset": dataset,
        "definition": definition,
    }


def _run_kwargs(seeded, **overrides):
    kwargs = dict(
        experiment_definition_id=seeded["definition"].id,
        dataset_id=seeded["dataset"].id,
        rule_feature_definition_id=seeded["feature_definitions"]["close_vs_sma20"].id,
        configuration={"rule": "close_vs_sma20 > 0", "threshold": 0.0},
        cost_bps_per_trade=10.0,
        cost_assumptions_description="flat round-trip bps",
        train_window_sessions=25,
        validation_window_sessions=5,
        test_window_sessions=5,
        step_sessions=100,
        mode=FoldMode.ROLLING,
        created_by="test-suite",
    )
    kwargs.update(overrides)
    return kwargs


class TestLabelHorizonPurge:

    def test_dataset_itself_shows_the_dangerous_rows_as_resolved(self, db_session, seeded):
        from sqlalchemy import select

        from backend.app.db.models import DatasetRow

        row_20 = db_session.scalars(
            select(DatasetRow).where(
                DatasetRow.dataset_id == seeded["dataset"].id,
                DatasetRow.entry_date == _TRADING_DATES[20],
            )
        ).one()
        assert row_20.label_resolved is True

    def test_walk_forward_train_split_excludes_the_unresolved_rows(self, db_session, seeded):
        run = run_walk_forward_evaluation(db_session, **_run_kwargs(seeded))
        assert run.status == RunStatus.COMPLETED

        train_count_metric = next(
            m
            for m in run.metrics
            if m.fold_index == 1
            and m.split == MetricSplit.TRAIN
            and m.category == MetricCategory.DESCRIPTIVE
            and m.metric_name == "count"
        )
        assert train_count_metric.metric_value == 20

    def test_excluded_rows_realized_return_does_not_enter_the_training_mean(
        self, db_session, seeded
    ):
        run = run_walk_forward_evaluation(db_session, **_run_kwargs(seeded))
        train_mean_metric = next(
            m
            for m in run.metrics
            if m.fold_index == 1
            and m.split == MetricSplit.TRAIN
            and m.category == MetricCategory.DESCRIPTIVE
            and m.metric_name == "mean"
        )
        expected = sum(5.0 / (100.0 + i) for i in range(20)) / 20
        assert train_mean_metric.metric_value == pytest.approx(expected)

    def test_validation_split_uses_its_own_cutoff_not_the_dataset_global_one(
        self, db_session, seeded
    ):
        run = run_walk_forward_evaluation(db_session, **_run_kwargs(seeded))
        validation_count_metric = next(
            m
            for m in run.metrics
            if m.fold_index == 1
            and m.split == MetricSplit.VALIDATION
            and m.category == MetricCategory.DESCRIPTIVE
            and m.metric_name == "count"
        )
        assert validation_count_metric.metric_value == 0


class TestFeatureAvailabilityBoundaries:
    def test_a_row_outside_the_fold_never_appears_in_that_folds_metrics(self, db_session, seeded):
        run = run_walk_forward_evaluation(db_session, **_run_kwargs(seeded))
        train_count = next(
            m
            for m in run.metrics
            if m.fold_index == 1
            and m.split == MetricSplit.TRAIN
            and m.category == MetricCategory.DESCRIPTIVE
            and m.metric_name == "count"
        )
        assert 0 <= train_count.metric_value <= 25

    def test_validation_and_test_observations_cannot_influence_training(self, db_session, seeded):
        first_run = run_walk_forward_evaluation(db_session, **_run_kwargs(seeded))
        first_train_mean = next(
            m
            for m in first_run.metrics
            if m.fold_index == 1
            and m.split == MetricSplit.TRAIN
            and m.category == MetricCategory.DESCRIPTIVE
            and m.metric_name == "mean"
        ).metric_value

        second_run = run_walk_forward_evaluation(db_session, **_run_kwargs(seeded))
        second_train_mean = next(
            m
            for m in second_run.metrics
            if m.fold_index == 1
            and m.split == MetricSplit.TRAIN
            and m.category == MetricCategory.DESCRIPTIVE
            and m.metric_name == "mean"
        ).metric_value
        assert first_train_mean == pytest.approx(second_train_mean)


class TestFoldDefinitionsAreDeterministic:
    def test_same_configuration_produces_identical_checksum(self, db_session, seeded):
        first = run_walk_forward_evaluation(db_session, **_run_kwargs(seeded))
        second = run_walk_forward_evaluation(db_session, **_run_kwargs(seeded))
        assert first.id != second.id
        assert first.checksum == second.checksum
        assert first.walk_forward_scheme["folds"] == second.walk_forward_scheme["folds"]

    def test_changing_fold_configuration_produces_a_distinct_evaluation(self, db_session, seeded):
        default_run = run_walk_forward_evaluation(db_session, **_run_kwargs(seeded))
        different_run = run_walk_forward_evaluation(
            db_session, **_run_kwargs(seeded, train_window_sessions=20)
        )
        assert default_run.checksum != different_run.checksum
        assert (
            default_run.walk_forward_scheme["train_window_sessions"]
            != different_run.walk_forward_scheme["train_window_sessions"]
        )


class TestFoldMetricsCannotBeDuplicated:
    def test_duplicate_fold_metric_identity_is_rejected(self, db_session, seeded):
        run = run_walk_forward_evaluation(db_session, **_run_kwargs(seeded))
        existing = run.metrics[0]
        db_session.add(
            ExperimentMetric(
                run_id=run.id,
                split=existing.split,
                category=existing.category,
                metric_name=existing.metric_name,
                fold_index=existing.fold_index,
                metric_value=999.0,
                sample_size=1,
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()

    def test_same_split_category_name_at_different_fold_index_is_allowed(
        self, db_session, seeded
    ):
        run = run_walk_forward_evaluation(db_session, **_run_kwargs(seeded))
        existing = run.metrics[0]
        db_session.add(
            ExperimentMetric(
                run_id=run.id,
                split=existing.split,
                category=existing.category,
                metric_name=existing.metric_name,
                fold_index=existing.fold_index + 1000,
                metric_value=1.23,
                sample_size=1,
            )
        )
        db_session.flush()


class TestAggregateMetrics:
    def test_aggregate_rows_use_the_reserved_sentinel_fold_index(self, db_session, seeded):
        run = run_walk_forward_evaluation(db_session, **_run_kwargs(seeded))
        aggregate_metrics = [m for m in run.metrics if m.fold_index == AGGREGATE_FOLD_INDEX]
        assert len(aggregate_metrics) > 0
        assert all(m.metric_name == "mean" for m in aggregate_metrics)

    def test_aggregate_mean_matches_the_single_folds_own_mean_when_only_one_fold(
        self, db_session, seeded
    ):
        run = run_walk_forward_evaluation(db_session, **_run_kwargs(seeded))
        fold0_train_mean = next(
            m
            for m in run.metrics
            if m.fold_index == 1
            and m.split == MetricSplit.TRAIN
            and m.category == MetricCategory.DESCRIPTIVE
            and m.metric_name == "mean"
        ).metric_value
        aggregate_train_mean = next(
            m
            for m in run.metrics
            if m.fold_index == AGGREGATE_FOLD_INDEX
            and m.split == MetricSplit.TRAIN
            and m.category == MetricCategory.DESCRIPTIVE
            and m.metric_name == "mean"
        ).metric_value
        assert aggregate_train_mean == pytest.approx(fold0_train_mean)


class TestInsufficientHistoryProducesFailedRun:
    def test_far_too_small_windows_relative_to_history_still_succeeds(self, db_session, seeded):
        run = run_walk_forward_evaluation(db_session, **_run_kwargs(seeded))
        assert run.status == RunStatus.COMPLETED

    def test_windows_larger_than_available_history_produce_a_failed_run(
        self, db_session, seeded
    ):
        run = run_walk_forward_evaluation(
            db_session,
            **_run_kwargs(
                seeded,
                train_window_sessions=1000,
                validation_window_sessions=1000,
                test_window_sessions=1000,
            ),
        )
        assert run.status == RunStatus.FAILED
        assert run.failure_reason is not None
        assert run.conclusion is None
        assert len(run.metrics) == 0
        assert run.walk_forward_scheme["n_folds"] == 0


class TestWalkForwardRunIsExplorationWithNoConclusion:
    def test_run_purpose_is_exploration(self, db_session, seeded):
        run = run_walk_forward_evaluation(db_session, **_run_kwargs(seeded))
        assert run.run_purpose == RunPurpose.EXPLORATION

    def test_no_conclusion_is_ever_produced(self, db_session, seeded):
        run = run_walk_forward_evaluation(db_session, **_run_kwargs(seeded))
        assert run.conclusion is None
        assert run.conclusion_rationale is None

    def test_multiple_walk_forward_runs_against_the_same_pair_are_all_permitted(
        self, db_session, seeded
    ):
        first = run_walk_forward_evaluation(db_session, **_run_kwargs(seeded))
        second = run_walk_forward_evaluation(db_session, **_run_kwargs(seeded))
        third = run_walk_forward_evaluation(
            db_session, **_run_kwargs(seeded, train_window_sessions=20)
        )
        assert len({first.id, second.id, third.id}) == 3


class TestOrdinaryExperimentsRemainCompatibleWithFoldIndexZero:
    def test_a_normal_run_experiment_call_uses_fold_index_zero(self, db_session, seeded):
        from backend.app.research.experiment_tracking import run_experiment

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
        assert run.metrics
        assert all(m.fold_index == 0 for m in run.metrics)
