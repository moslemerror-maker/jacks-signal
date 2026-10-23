from __future__ import annotations

import uuid
from datetime import date, timedelta

import pytest
from sqlalchemy.exc import ProgrammingError

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
        name="stage6-test-dataset",
        instrument_ids=[instrument.id],
        feature_definition_ids=[fd.id for fd in feature_definitions.values()],
        label_definition_id=label_definition.id,
        start_date=_TRADING_DATES[0],
        end_date=_TRADING_DATES[-1],
        known_as_of=_TRADING_DATES[-1],
        created_by="test-suite",
    )

    hypothesis_family = register_hypothesis_family(
        db_session,
        name="stage6-test-family",
        description="test family for Stage 6/3.2 experiment-tracking tests",
        created_by="test-suite",
    )
    experiment_definition = register_experiment_definition(
        db_session,
        name="stage6-test-definition",
        hypothesis_family_id=hypothesis_family.id,
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
        "experiment_definition": experiment_definition,
    }


def _run(db_session, ctx, cost_bps=10.0, **overrides):
    kwargs = dict(
        experiment_definition_id=ctx["experiment_definition"].id,
        dataset_id=ctx["dataset"].id,
        rule_feature_definition_id=ctx["feature_definitions"]["close_vs_sma20"].id,
        configuration={"rule": "close_vs_sma20 > 0", "threshold": 0.0},
        cost_bps_per_trade=cost_bps,
        cost_assumptions_description="flat round-trip bps, no market impact modeled",
        run_purpose=RunPurpose.CONFIRMATION,
        train_start_date=_TRADING_DATES[0],
        train_end_date=_TRADING_DATES[29],
        validation_start_date=_TRADING_DATES[30],
        validation_end_date=_TRADING_DATES[39],
        test_start_date=_TRADING_DATES[40],
        test_end_date=_TRADING_DATES[54],
        created_by="test-suite",
    )
    kwargs.update(overrides)
    return run_experiment(db_session, **kwargs)


class TestNonexistentReferences:
    def test_unknown_dataset_id_is_rejected(self, db_session, seeded):
        with pytest.raises(ValueError, match="unknown dataset_id"):
            _run(db_session, seeded, dataset_id=uuid.uuid4())

    def test_unknown_experiment_definition_id_is_rejected(self, db_session, seeded):
        with pytest.raises(ValueError, match="unknown experiment_definition_id"):
            _run(db_session, seeded, experiment_definition_id=uuid.uuid4())

    def test_rule_feature_not_bound_to_dataset_is_rejected(self, db_session, seeded):
        narrow_dataset = build_dataset(
            db_session,
            name="return-only-dataset",
            instrument_ids=[seeded["instrument"].id],
            feature_definition_ids=[seeded["feature_definitions"]["return_5d"].id],
            label_definition_id=seeded["label_definition"].id,
            start_date=_TRADING_DATES[0],
            end_date=_TRADING_DATES[-1],
            known_as_of=_TRADING_DATES[-1],
            created_by="test-suite",
        )
        with pytest.raises(ValueError, match="not bound"):
            _run(db_session, seeded, dataset_id=narrow_dataset.id)


class TestImmutability:
    def test_updating_a_run_is_rejected(self, db_session, seeded):
        run = _run(db_session, seeded)
        run.failure_reason = "tampered"
        with pytest.raises(ProgrammingError):
            db_session.flush()

    def test_deleting_a_run_is_rejected(self, db_session, seeded):
        run = _run(db_session, seeded)
        db_session.delete(run)
        with pytest.raises(ProgrammingError):
            db_session.flush()

    def test_updating_a_metric_is_rejected(self, db_session, seeded):
        run = _run(db_session, seeded)
        metric = run.metrics[0]
        metric.metric_value = 999.0
        with pytest.raises(ProgrammingError):
            db_session.flush()

    def test_deleting_a_metric_is_rejected(self, db_session, seeded):
        run = _run(db_session, seeded)
        metric = run.metrics[0]
        db_session.delete(metric)
        with pytest.raises(ProgrammingError):
            db_session.flush()


class TestReconstructability:
    def test_run_carries_everything_needed_to_reconstruct_what_was_tested(
        self, db_session, seeded
    ):
        run = _run(db_session, seeded, cost_bps=15.0)

        assert run.dataset_id == seeded["dataset"].id
        assert run.experiment_definition_id == seeded["experiment_definition"].id
        assert run.rule_feature_definition_id == seeded["feature_definitions"]["close_vs_sma20"].id
        assert run.configuration == {"rule": "close_vs_sma20 > 0", "threshold": 0.0}
        assert run.cost_assumptions["cost_bps_per_trade"] == 15.0
        assert run.train_start_date == _TRADING_DATES[0]
        assert run.test_end_date == _TRADING_DATES[54]
        assert len(run.git_commit) == 40
        assert isinstance(run.git_dirty, bool)
        assert "sqlalchemy" in run.dependency_versions
        assert "python" in run.dependency_versions
        assert run.executed_at is not None
        assert run.walk_forward_scheme is None


class TestDistinctRecordsForDistinctConfigurations:

    def test_different_cost_produces_a_distinct_run_with_different_checksum(
        self, db_session, seeded
    ):
        cheap = _run(
            db_session, seeded, cost_bps=1.0, run_purpose=RunPurpose.EXPLORATION,
            validation_start_date=None, validation_end_date=None,
            test_start_date=None, test_end_date=None,
        )
        expensive = _run(
            db_session, seeded, cost_bps=500.0, run_purpose=RunPurpose.EXPLORATION,
            validation_start_date=None, validation_end_date=None,
            test_start_date=None, test_end_date=None,
        )
        assert cheap.id != expensive.id
        assert cheap.checksum != expensive.checksum
        assert db_session.get(ExperimentRun, cheap.id) is not None
        assert db_session.get(ExperimentRun, expensive.id) is not None

    def test_rerunning_an_identical_configuration_creates_a_new_run_not_an_overwrite(
        self, db_session, seeded
    ):
        kwargs = dict(
            run_purpose=RunPurpose.EXPLORATION,
            validation_start_date=None, validation_end_date=None,
            test_start_date=None, test_end_date=None,
        )
        first = _run(db_session, seeded, **kwargs)
        second = _run(db_session, seeded, **kwargs)
        assert first.id != second.id
        assert first.checksum == second.checksum
        assert db_session.get(ExperimentRun, first.id) is not None
        assert db_session.get(ExperimentRun, second.id) is not None
        both_runs_for_definition = [
            r for r in seeded["experiment_definition"].runs if r.id in (first.id, second.id)
        ]
        assert len(both_runs_for_definition) == 2


class TestMetricIsolation:
    def test_metrics_are_only_reachable_through_their_own_run(self, db_session, seeded):
        kwargs = dict(
            cost_bps=5.0, run_purpose=RunPurpose.EXPLORATION,
            validation_start_date=None, validation_end_date=None,
            test_start_date=None, test_end_date=None,
        )
        run_a = _run(db_session, seeded, **kwargs)
        run_b = _run(db_session, seeded, **kwargs)

        metric_ids_a = {m.id for m in run_a.metrics}
        metric_ids_b = {m.id for m in run_b.metrics}
        assert metric_ids_a.isdisjoint(metric_ids_b)
        assert all(m.run_id == run_a.id for m in run_a.metrics)
        assert all(m.run_id == run_b.id for m in run_b.metrics)

    def test_uniqueness_prevents_duplicate_metric_identity_within_one_run(
        self, db_session, seeded
    ):
        run = _run(db_session, seeded)
        db_session.add(
            ExperimentMetric(
                run_id=run.id,
                split=MetricSplit.TEST,
                category=MetricCategory.DESCRIPTIVE,
                metric_name="mean",
                metric_value=999.0,
                sample_size=1,
            )
        )
        from sqlalchemy.exc import IntegrityError

        with pytest.raises(IntegrityError):
            db_session.flush()

    def test_every_split_and_category_combination_is_present(self, db_session, seeded):
        run = _run(db_session, seeded)
        seen = {(m.split, m.category, m.metric_name) for m in run.metrics}
        for split in MetricSplit:
            for category in MetricCategory:
                if category == MetricCategory.HYPOTHESIS_TEST:
                    continue
                for metric_name in ("count", "mean", "std", "win_rate"):
                    assert (split, category, metric_name) in seen


class TestChronologicalSplitValidation:
    def test_validation_overlapping_train_is_rejected(self, db_session, seeded):
        with pytest.raises(ValueError, match="validation_start_date"):
            _run(
                db_session,
                seeded,
                train_end_date=_TRADING_DATES[35],
                validation_start_date=_TRADING_DATES[30],
            )

    def test_test_overlapping_validation_is_rejected(self, db_session, seeded):
        with pytest.raises(ValueError, match="validation_end_date"):
            _run(
                db_session,
                seeded,
                validation_end_date=_TRADING_DATES[45],
                test_start_date=_TRADING_DATES[40],
            )


class TestFailedRunsRemainQueryable:
    def test_a_run_with_no_overlapping_rows_is_recorded_as_failed_and_queryable(
        self, db_session, seeded
    ):
        far_future = date(2099, 1, 1)
        run = _run(
            db_session,
            seeded,
            train_start_date=far_future,
            train_end_date=far_future + timedelta(days=10),
            validation_start_date=far_future + timedelta(days=11),
            validation_end_date=far_future + timedelta(days=20),
            test_start_date=far_future + timedelta(days=21),
            test_end_date=far_future + timedelta(days=30),
        )
        assert run.status == RunStatus.FAILED
        assert run.failure_reason is not None
        assert run.conclusion is None
        assert run.checksum is None
        assert len(run.metrics) == 0

        reloaded = db_session.get(ExperimentRun, run.id)
        assert reloaded is not None
        assert reloaded.status == RunStatus.FAILED


class TestReproducibility:
    def test_same_dataset_and_configuration_produce_deterministic_results(
        self, db_session, seeded
    ):
        kwargs = dict(
            cost_bps=8.0, run_purpose=RunPurpose.EXPLORATION,
            validation_start_date=None, validation_end_date=None,
            test_start_date=None, test_end_date=None,
        )
        first = _run(db_session, seeded, **kwargs)
        second = _run(db_session, seeded, **kwargs)

        assert first.checksum == second.checksum
        assert first.conclusion == second.conclusion

        first_values = {
            (m.split, m.category, m.metric_name): m.metric_value for m in first.metrics
        }
        second_values = {
            (m.split, m.category, m.metric_name): m.metric_value for m in second.metrics
        }
        assert first_values == second_values
