from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy.exc import ProgrammingError

from backend.app.db.models import (
    AssetType,
    Experiment,
    ExperimentConclusion,
    Instrument,
    MarketBar,
    SourceFormat,
)
from backend.app.research.datasets import build_dataset
from backend.app.research.experiments import (
    COST_MODEL_DESCRIPTION,
    HYPOTHESIS_TEXT,
    MIN_TRADES_FOR_DECISION,
    RULE_DESCRIPTION_TEXT,
    apply_rule,
    decide_conclusion,
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
def seeded_dataset(db_session, instrument):
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
        name="stage5-test-dataset",
        instrument_ids=[instrument.id],
        feature_definition_ids=[fd.id for fd in feature_definitions.values()],
        label_definition_id=label_definition.id,
        start_date=_TRADING_DATES[0],
        end_date=_TRADING_DATES[-1],
        known_as_of=_TRADING_DATES[-1],
        created_by="test-suite",
    )
    return dataset, feature_definitions, label_definition


def _run(db_session, dataset, feature_definitions, name="exp", cost_bps=10.0, **overrides):
    kwargs = dict(
        dataset_id=dataset.id,
        rule_feature_definition_id=feature_definitions["close_vs_sma20"].id,
        name=name,
        hypothesis=HYPOTHESIS_TEXT,
        rule_description=RULE_DESCRIPTION_TEXT,
        cost_bps_per_trade=cost_bps,
        cost_model_description=COST_MODEL_DESCRIPTION,
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


class TestApplyRule:
    def test_positive_value_is_long(self):
        assert apply_rule(0.01) == "LONG"

    def test_negative_value_is_flat(self):
        assert apply_rule(-0.01) == "FLAT"

    def test_zero_is_flat(self):
        assert apply_rule(0.0) == "FLAT"

    def test_none_is_flat(self):
        assert apply_rule(None) == "FLAT"


class TestDecideConclusion:
    def _split(self, mean, count):
        return {"realizable_strategy_performance": {"mean": mean, "count": count}}

    def test_accept_when_both_positive_with_enough_trades(self):
        conclusion, rationale = decide_conclusion(
            self._split(0.01, MIN_TRADES_FOR_DECISION),
            self._split(0.02, MIN_TRADES_FOR_DECISION),
        )
        assert conclusion == ExperimentConclusion.ACCEPT

    def test_reject_when_both_non_positive(self):
        conclusion, rationale = decide_conclusion(
            self._split(-0.01, MIN_TRADES_FOR_DECISION),
            self._split(-0.02, MIN_TRADES_FOR_DECISION),
        )
        assert conclusion == ExperimentConclusion.REJECT

    def test_needs_more_testing_when_signs_disagree(self):
        conclusion, rationale = decide_conclusion(
            self._split(0.01, MIN_TRADES_FOR_DECISION),
            self._split(-0.01, MIN_TRADES_FOR_DECISION),
        )
        assert conclusion == ExperimentConclusion.NEEDS_MORE_TESTING

    def test_needs_more_testing_when_sample_too_small(self):
        conclusion, rationale = decide_conclusion(
            self._split(0.05, MIN_TRADES_FOR_DECISION - 1),
            self._split(0.05, MIN_TRADES_FOR_DECISION),
        )
        assert conclusion == ExperimentConclusion.NEEDS_MORE_TESTING


class TestThreeWayMetricSeparation:
    def test_gross_minus_cost_equals_net(self, db_session, seeded_dataset):
        dataset, feature_definitions, label_definition = seeded_dataset
        cost_bps = 25.0
        experiment = _run(db_session, dataset, feature_definitions, cost_bps=cost_bps)

        test_split = experiment.results["test"]
        gross = test_split["hypothetical_signal_performance"]
        net = test_split["realizable_strategy_performance"]
        assert gross["count"] == net["count"]
        assert gross["count"] > 0
        assert net["mean"] == pytest.approx(gross["mean"] - cost_bps / 10000.0)

    def test_descriptive_stats_are_unconditional_on_the_signal(
        self, db_session, seeded_dataset
    ):
        dataset, feature_definitions, label_definition = seeded_dataset
        experiment = _run(db_session, dataset, feature_definitions)
        test_split = experiment.results["test"]
        descriptive = test_split["descriptive_label_stats"]
        hypothetical = test_split["hypothetical_signal_performance"]
        assert descriptive["count"] >= hypothetical["count"]


class TestDeterminism:
    def test_running_the_same_experiment_twice_is_byte_for_byte_identical(
        self, db_session, seeded_dataset
    ):
        dataset, feature_definitions, label_definition = seeded_dataset
        first = _run(db_session, dataset, feature_definitions, name="run-1")
        second = _run(db_session, dataset, feature_definitions, name="run-2")

        assert first.id != second.id
        assert first.checksum == second.checksum
        assert first.results == second.results
        assert first.conclusion == second.conclusion


class TestDistinguishableRecords:
    def test_changing_cost_assumption_changes_the_checksum(
        self, db_session, seeded_dataset
    ):
        dataset, feature_definitions, label_definition = seeded_dataset
        cheap = _run(db_session, dataset, feature_definitions, name="cheap", cost_bps=1.0)
        expensive = _run(
            db_session, dataset, feature_definitions, name="expensive", cost_bps=500.0
        )
        assert cheap.checksum != expensive.checksum
        assert cheap.id != expensive.id
        assert db_session.get(Experiment, cheap.id) is not None
        assert db_session.get(Experiment, expensive.id) is not None

    def test_changing_the_dataset_changes_the_checksum(
        self, db_session, instrument, seeded_dataset
    ):
        dataset, feature_definitions, label_definition = seeded_dataset

        other_dataset = build_dataset(
            db_session,
            name="narrower-dataset",
            instrument_ids=[instrument.id],
            feature_definition_ids=[fd.id for fd in feature_definitions.values()],
            label_definition_id=label_definition.id,
            start_date=_TRADING_DATES[0],
            end_date=_TRADING_DATES[45],
            known_as_of=_TRADING_DATES[-1],
            created_by="test-suite",
        )

        from_first = _run(db_session, dataset, feature_definitions, name="from-first")
        from_second = _run(
            db_session, other_dataset, feature_definitions, name="from-second"
        )
        assert from_first.checksum != from_second.checksum
        assert from_first.dataset_id != from_second.dataset_id


class TestChronologicalSplitValidation:
    def test_validation_overlapping_train_is_rejected(self, db_session, seeded_dataset):
        dataset, feature_definitions, label_definition = seeded_dataset
        with pytest.raises(ValueError, match="validation_start_date"):
            _run(
                db_session,
                dataset,
                feature_definitions,
                train_end_date=_TRADING_DATES[35],
                validation_start_date=_TRADING_DATES[30],
            )

    def test_test_overlapping_validation_is_rejected(self, db_session, seeded_dataset):
        dataset, feature_definitions, label_definition = seeded_dataset
        with pytest.raises(ValueError, match="validation_end_date"):
            _run(
                db_session,
                dataset,
                feature_definitions,
                validation_end_date=_TRADING_DATES[45],
                test_start_date=_TRADING_DATES[40],
            )


class TestRuleFeatureBinding:
    def test_rule_feature_not_bound_to_dataset_is_rejected(
        self, db_session, instrument, seeded_dataset
    ):
        dataset, feature_definitions, label_definition = seeded_dataset
        narrow_dataset = build_dataset(
            db_session,
            name="return-only-dataset",
            instrument_ids=[instrument.id],
            feature_definition_ids=[feature_definitions["return_5d"].id],
            label_definition_id=label_definition.id,
            start_date=_TRADING_DATES[0],
            end_date=_TRADING_DATES[-1],
            known_as_of=_TRADING_DATES[-1],
            created_by="test-suite",
        )
        with pytest.raises(ValueError, match="not bound"):
            _run(db_session, narrow_dataset, feature_definitions)


class TestUnknownDataset:
    def test_unknown_dataset_id_is_rejected(self, db_session, seeded_dataset):
        import uuid

        dataset, feature_definitions, label_definition = seeded_dataset
        with pytest.raises(ValueError, match="unknown dataset_id"):
            run_experiment(
                db_session,
                dataset_id=uuid.uuid4(),
                rule_feature_definition_id=feature_definitions["close_vs_sma20"].id,
                name="ghost-dataset",
                hypothesis=HYPOTHESIS_TEXT,
                rule_description=RULE_DESCRIPTION_TEXT,
                cost_bps_per_trade=10.0,
                cost_model_description=COST_MODEL_DESCRIPTION,
                train_start_date=_TRADING_DATES[0],
                train_end_date=_TRADING_DATES[29],
                validation_start_date=_TRADING_DATES[30],
                validation_end_date=_TRADING_DATES[39],
                test_start_date=_TRADING_DATES[40],
                test_end_date=_TRADING_DATES[54],
                created_by="test-suite",
            )


class TestImmutability:
    def test_updating_an_experiment_is_rejected(self, db_session, seeded_dataset):
        dataset, feature_definitions, label_definition = seeded_dataset
        experiment = _run(db_session, dataset, feature_definitions)
        experiment.conclusion_rationale = "tampered"
        with pytest.raises(ProgrammingError):
            db_session.flush()

    def test_deleting_an_experiment_is_rejected(self, db_session, seeded_dataset):
        dataset, feature_definitions, label_definition = seeded_dataset
        experiment = _run(db_session, dataset, feature_definitions)
        db_session.delete(experiment)
        with pytest.raises(ProgrammingError):
            db_session.flush()
