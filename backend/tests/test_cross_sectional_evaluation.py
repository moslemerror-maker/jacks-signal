from __future__ import annotations

from datetime import date, timedelta

import pytest

from backend.app.db.models import (
    AssetType,
    Instrument,
    MarketBar,
    MetricCategory,
    MetricSplit,
    RunPurpose,
    RunStatus,
    SourceFormat,
)
from backend.app.research.cross_sectional import run_cross_sectional_evaluation
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

_N_DAYS = 60
_TRADING_DATES = [date(2024, 1, 1) + timedelta(days=i) for i in range(_N_DAYS)]
_CLOSES = [100.0 + i for i in range(_N_DAYS)]


@pytest.fixture
def instruments(db_session):
    insts = [Instrument(exchange="NSE", asset_type=AssetType.EQUITY) for _ in range(3)]
    db_session.add_all(insts)
    db_session.flush()
    return insts


@pytest.fixture
def seeded(db_session, instruments):
    feature_definitions = register_feature_definitions(db_session)
    label_definition = register_label_definition(db_session)
    db_session.flush()

    for instrument in instruments:
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
        name="stage3.5-test-dataset",
        instrument_ids=[i.id for i in instruments],
        feature_definition_ids=[fd.id for fd in feature_definitions.values()],
        label_definition_id=label_definition.id,
        start_date=_TRADING_DATES[0],
        end_date=_TRADING_DATES[-1],
        known_as_of=_TRADING_DATES[-1],
        created_by="test-suite",
    )

    family = register_hypothesis_family(
        db_session,
        name="stage3.5-test-family",
        description="test family for cross-sectional evaluation tests",
        created_by="test-suite",
    )
    definition = register_experiment_definition(
        db_session,
        name="stage3.5-test-definition",
        hypothesis_family_id=family.id,
        hypothesis="test hypothesis",
        rule_feature_definition_id=feature_definitions["close_vs_sma20"].id,
        rule_description="LONG if close_vs_sma20 > 0 else FLAT",
        cost_model_description="flat bps deduction",
        created_by="test-suite",
    )

    return {
        "instruments": instruments,
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
        train_start_date=_TRADING_DATES[0],
        train_end_date=_TRADING_DATES[29],
        created_by="test-suite",
    )
    kwargs.update(overrides)
    return kwargs


class TestBasicWiring:
    def test_completes_and_produces_all_three_categories(self, db_session, seeded):
        run = run_cross_sectional_evaluation(db_session, **_run_kwargs(seeded))
        assert run.status == RunStatus.COMPLETED
        categories_present = {m.category for m in run.metrics}
        assert categories_present == {
            MetricCategory.DESCRIPTIVE,
            MetricCategory.HYPOTHETICAL_SIGNAL,
            MetricCategory.HYPOTHESIS_TEST,
        }
        assert all(m.split == MetricSplit.TRAIN for m in run.metrics)
        assert all(m.fold_index == 0 for m in run.metrics)

    def test_descriptive_category_is_the_signal_conditional_estimand(self, db_session, seeded):
        run = run_cross_sectional_evaluation(db_session, **_run_kwargs(seeded))
        descriptive_count = next(
            m for m in run.metrics if m.category == MetricCategory.DESCRIPTIVE and m.metric_name == "count"
        )
        hypothetical_count = next(
            m
            for m in run.metrics
            if m.category == MetricCategory.HYPOTHETICAL_SIGNAL and m.metric_name == "count"
        )
        assert hypothetical_count.metric_value >= descriptive_count.metric_value

    def test_hypothesis_test_metrics_are_present_with_explicit_method_names(
        self, db_session, seeded
    ):
        run = run_cross_sectional_evaluation(db_session, **_run_kwargs(seeded))
        hypothesis_metrics = {
            m.metric_name for m in run.metrics if m.category == MetricCategory.HYPOTHESIS_TEST
        }
        for prefix in ("newey_west", "non_overlapping"):
            for suffix in ("test_statistic", "p_value", "ci_low", "ci_high", "n"):
                assert f"{prefix}_{suffix}" in hypothesis_metrics


class TestPortfolioConventionRecorded:
    def test_configuration_records_the_full_portfolio_convention(self, db_session, seeded):
        run = run_cross_sectional_evaluation(db_session, **_run_kwargs(seeded))
        convention = run.configuration["portfolio_convention"]
        assert convention["weighting_scheme"] == "equal_weight_dynamic_across_active_signals"
        assert convention["zero_signal_day_treatment"] == "cash_zero_return"
        assert convention["equal_weighted_across_active_signals"] is True
        assert convention["unused_capital_in_cash"] is True
        assert convention["max_simultaneous_positions"] is None

    def test_configuration_records_a_nondefault_max_simultaneous_positions(self, db_session, seeded):
        run = run_cross_sectional_evaluation(
            db_session, **_run_kwargs(seeded, max_simultaneous_positions=2)
        )
        assert run.configuration["portfolio_convention"]["max_simultaneous_positions"] == 2

    def test_configuration_records_the_inference_method(self, db_session, seeded):
        run = run_cross_sectional_evaluation(db_session, **_run_kwargs(seeded))
        method = run.configuration["inference_method"]
        assert "signal_conditional" in method["estimand_1"]
        assert "portfolio" in method["estimand_2"]
        assert method["newey_west_max_lag"] == 4
        assert method["non_overlapping_stride"] == 5


class TestExplorationOnlyNoConclusion:
    def test_run_purpose_is_exploration(self, db_session, seeded):
        run = run_cross_sectional_evaluation(db_session, **_run_kwargs(seeded))
        assert run.run_purpose == RunPurpose.EXPLORATION

    def test_no_conclusion_is_ever_produced(self, db_session, seeded):
        run = run_cross_sectional_evaluation(db_session, **_run_kwargs(seeded))
        assert run.conclusion is None
        assert run.conclusion_rationale is None

    def test_repeated_evaluation_is_permitted_and_deterministic(self, db_session, seeded):
        first = run_cross_sectional_evaluation(db_session, **_run_kwargs(seeded))
        second = run_cross_sectional_evaluation(db_session, **_run_kwargs(seeded))
        assert first.id != second.id
        assert first.checksum == second.checksum


class TestFailedOnEmptyRange:
    def test_a_range_outside_the_dataset_produces_a_failed_run(self, db_session, seeded):
        far_future = date(2099, 1, 1)
        run = run_cross_sectional_evaluation(
            db_session,
            **_run_kwargs(
                seeded,
                train_start_date=far_future,
                train_end_date=far_future + timedelta(days=10),
            ),
        )
        assert run.status == RunStatus.FAILED
        assert run.failure_reason is not None
        assert run.conclusion is None
        assert len(run.metrics) == 0


class TestLabelHorizonPurgeReused:
    def test_a_cutoff_earlier_than_the_datasets_global_known_as_of_excludes_unresolved_rows(
        self, db_session, seeded
    ):
        run = run_cross_sectional_evaluation(
            db_session,
            **_run_kwargs(
                seeded, train_start_date=_TRADING_DATES[0], train_end_date=_TRADING_DATES[24]
            ),
        )
        descriptive_count = next(
            m
            for m in run.metrics
            if m.category == MetricCategory.DESCRIPTIVE and m.metric_name == "zero_signal_day_count"
        )
        assert descriptive_count.metric_value >= 5
