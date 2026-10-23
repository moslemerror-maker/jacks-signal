from __future__ import annotations

import uuid
from datetime import date, timedelta

import pytest

from backend.app.db.models import (
    AssetType,
    ExperimentRun,
    Instrument,
    LabelDefinition,
    MarketBar,
    MetricCategory,
    RunPurpose,
    RunStatus,
    SourceFormat,
)
from backend.app.research.confirmation import (
    MultipleTestingMethod,
    apply_simplified_flat_cost_to_portfolio_series,
    compute_adjusted_threshold,
    cost_sensitivity_sweep,
    run_confirmation_evaluation,
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
from backend.app.research.walk_forward import FoldMode, run_walk_forward_evaluation

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
        name="stage3.6-test-dataset",
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
        name="stage3.6-test-family",
        description="test family for confirmation pipeline tests",
        created_by="test-suite",
    )
    definition = register_experiment_definition(
        db_session,
        name="stage3.6-test-definition",
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
        "family": family,
        "definition": definition,
    }


def _confirm_kwargs(seeded, **overrides):
    kwargs = dict(
        experiment_definition_id=seeded["definition"].id,
        dataset_id=seeded["dataset"].id,
        rule_feature_definition_id=seeded["feature_definitions"]["close_vs_sma20"].id,
        configuration={"rule": "close_vs_sma20 > 0", "threshold": 0.0},
        train_start_date=_TRADING_DATES[0],
        train_end_date=_TRADING_DATES[29],
        validation_start_date=_TRADING_DATES[30],
        validation_end_date=_TRADING_DATES[39],
        test_start_date=_TRADING_DATES[40],
        test_end_date=_TRADING_DATES[54],
        alpha=0.05,
        multiple_testing_method=MultipleTestingMethod.BONFERRONI,
        cost_bps_sweep=[0.0, 10.0, 50.0],
        created_by="test-suite",
    )
    kwargs.update(overrides)
    return kwargs


class TestBasicConfirmation:
    def test_completes_with_a_confirmation_report_and_no_conclusion(self, db_session, seeded):
        run = run_confirmation_evaluation(db_session, **_confirm_kwargs(seeded))
        assert run.status == RunStatus.COMPLETED
        assert run.run_purpose == RunPurpose.CONFIRMATION
        assert run.conclusion is None
        assert run.conclusion_rationale is None
        assert run.confirmation_report is not None
        assert "descriptive_performance" in run.confirmation_report
        assert "statistical_inference" in run.confirmation_report
        assert "multiple_testing" in run.confirmation_report
        assert "cost_sensitivity" in run.confirmation_report


class TestMultipleTestingDenominatorCorrection:
    def test_current_hypothesis_counted_exactly_once_on_its_first_ever_run(
        self, db_session, seeded
    ):
        run = run_confirmation_evaluation(db_session, **_confirm_kwargs(seeded))
        mt = run.confirmation_report["multiple_testing"]
        assert mt["hypotheses_in_family"] == 1
        assert mt["adjusted_threshold"] == pytest.approx(0.05 / 1)

    def test_current_hypothesis_not_double_counted_if_already_run(self, db_session, seeded):
        run_cross_sectional_evaluation(
            db_session,
            experiment_definition_id=seeded["definition"].id,
            dataset_id=seeded["dataset"].id,
            rule_feature_definition_id=seeded["feature_definitions"]["close_vs_sma20"].id,
            configuration={"rule": "close_vs_sma20 > 0"},
            train_start_date=_TRADING_DATES[0],
            train_end_date=_TRADING_DATES[29],
            created_by="test-suite",
        )
        run = run_confirmation_evaluation(db_session, **_confirm_kwargs(seeded))
        mt = run.confirmation_report["multiple_testing"]
        assert mt["hypotheses_in_family"] == 1

    def test_a_second_family_member_increases_the_count_for_a_new_confirmation(
        self, db_session, seeded
    ):
        other_definition = register_experiment_definition(
            db_session,
            name="stage3.6-test-definition-2",
            hypothesis_family_id=seeded["family"].id,
            hypothesis="a second hypothesis in the same family",
            rule_feature_definition_id=seeded["feature_definitions"]["return_5d"].id,
            rule_description="a different rule",
            cost_model_description="flat bps",
            created_by="test-suite",
        )
        run_cross_sectional_evaluation(
            db_session,
            experiment_definition_id=other_definition.id,
            dataset_id=seeded["dataset"].id,
            rule_feature_definition_id=seeded["feature_definitions"]["return_5d"].id,
            configuration={"rule": "return_5d > 0"},
            train_start_date=_TRADING_DATES[0],
            train_end_date=_TRADING_DATES[29],
            created_by="test-suite",
        )
        run = run_confirmation_evaluation(db_session, **_confirm_kwargs(seeded))
        mt = run.confirmation_report["multiple_testing"]
        assert mt["hypotheses_in_family"] == 2
        assert mt["adjusted_threshold"] == pytest.approx(0.05 / 2)


class TestFamilyCountFrozenInReport:
    def test_a_later_hypothesis_does_not_alter_an_already_frozen_confirmation(
        self, db_session, seeded
    ):
        first_confirmation = run_confirmation_evaluation(db_session, **_confirm_kwargs(seeded))
        assert first_confirmation.confirmation_report["multiple_testing"]["hypotheses_in_family"] == 1

        later_definition = register_experiment_definition(
            db_session,
            name="stage3.6-later-definition",
            hypothesis_family_id=seeded["family"].id,
            hypothesis="registered after the confirmation above",
            rule_feature_definition_id=seeded["feature_definitions"]["atr_14"].id,
            rule_description="another rule",
            cost_model_description="flat bps",
            created_by="test-suite",
        )
        run_cross_sectional_evaluation(
            db_session,
            experiment_definition_id=later_definition.id,
            dataset_id=seeded["dataset"].id,
            rule_feature_definition_id=seeded["feature_definitions"]["atr_14"].id,
            configuration={"rule": "atr_14 > 0"},
            train_start_date=_TRADING_DATES[0],
            train_end_date=_TRADING_DATES[29],
            created_by="test-suite",
        )

        reloaded = db_session.get(ExperimentRun, first_confirmation.id)
        assert reloaded.confirmation_report["multiple_testing"]["hypotheses_in_family"] == 1
        assert reloaded.confirmation_report == first_confirmation.confirmation_report


class TestLabelsCannotBeMixed:
    def test_two_datasets_with_different_label_horizons_never_share_inference_parameters(
        self, db_session, instruments, seeded
    ):
        other_label_definition = LabelDefinition(
            name="stage3.6-alternate-label",
            version=1,
            description="a different label with a different horizon, for isolation testing",
            formula_ref="test:not_a_real_formula",
            horizon_days=3,
            timeframe="1D",
            created_by="test-suite",
        )
        db_session.add(other_label_definition)
        db_session.flush()

        other_dataset = build_dataset(
            db_session,
            name="stage3.6-other-label-dataset",
            instrument_ids=[i.id for i in instruments],
            feature_definition_ids=[fd.id for fd in seeded["feature_definitions"].values()],
            label_definition_id=other_label_definition.id,
            start_date=_TRADING_DATES[0],
            end_date=_TRADING_DATES[-1],
            known_as_of=_TRADING_DATES[-1],
            created_by="test-suite",
        )
        other_family = register_hypothesis_family(
            db_session,
            name="stage3.6-other-label-family",
            description="family for the alternate-label dataset",
            created_by="test-suite",
        )
        other_definition = register_experiment_definition(
            db_session,
            name="stage3.6-other-label-definition",
            hypothesis_family_id=other_family.id,
            hypothesis="hypothesis against the alternate-label dataset",
            rule_feature_definition_id=seeded["feature_definitions"]["close_vs_sma20"].id,
            rule_description="rule",
            cost_model_description="flat bps",
            created_by="test-suite",
        )

        real_confirmation = run_confirmation_evaluation(db_session, **_confirm_kwargs(seeded))
        other_confirmation = run_confirmation_evaluation(
            db_session,
            **_confirm_kwargs(
                seeded,
                experiment_definition_id=other_definition.id,
                dataset_id=other_dataset.id,
            ),
        )

        real_method = real_confirmation.confirmation_report["inference_method"]
        other_method = other_confirmation.confirmation_report["inference_method"]
        assert real_method["newey_west_max_lag"] == 4
        assert real_method["non_overlapping_stride"] == 5
        assert other_method["newey_west_max_lag"] == 2
        assert other_method["non_overlapping_stride"] == 3


class TestSimplifiedCostModel:
    def test_cost_is_applied_exactly_as_configured(self):
        series = {date(2024, 1, 1): 0.05, date(2024, 1, 2): 0.0, date(2024, 1, 3): -0.02}
        adjusted = apply_simplified_flat_cost_to_portfolio_series(series, cost_bps=10.0)
        assert adjusted[date(2024, 1, 1)] == pytest.approx(0.05 - 0.001)
        assert adjusted[date(2024, 1, 2)] == 0.0
        assert adjusted[date(2024, 1, 3)] == pytest.approx(-0.02 - 0.001)

    def test_sweep_requires_an_explicit_non_empty_list(self):
        with pytest.raises(ValueError, match="cost_bps_values"):
            cost_sensitivity_sweep({date(2024, 1, 1): 0.01}, [])

    def test_confirmation_report_carries_the_non_realism_caveat(self, db_session, seeded):
        run = run_confirmation_evaluation(db_session, **_confirm_kwargs(seeded))
        caveat = run.confirmation_report["cost_model_caveat"]
        assert "SIMPLIFIED" in caveat
        assert "not" in caveat.lower() and "realistic" in caveat.lower()
        assert "overlapping" in caveat.lower()
        assert caveat in run.confirmation_report["unresolved_limitations"]

    def test_no_claim_of_execution_realism_appears_anywhere_in_the_report(self, db_session, seeded):
        run = run_confirmation_evaluation(db_session, **_confirm_kwargs(seeded))
        categories_used = {m.category for m in run.metrics}
        assert MetricCategory.REALIZABLE_STRATEGY not in categories_used
        report_text = str(run.confirmation_report).lower()
        start = 0
        while True:
            idx = report_text.find("realizable", start)
            if idx == -1:
                break
            preceding = report_text[max(0, idx - 4) : idx]
            assert "not " in preceding, f"unqualified 'realizable' claim near: {report_text[idx-20:idx+30]!r}"
            start = idx + 1


class TestHACOnlyOnPortfolioSeries:
    def test_hypothesis_test_n_matches_daily_series_length_not_pooled_row_count(
        self, db_session, seeded
    ):
        run = run_confirmation_evaluation(db_session, **_confirm_kwargs(seeded))
        test_inference = run.confirmation_report["statistical_inference"]["test"]
        newey_west_n = test_inference["newey_west"]["n"]
        n_trading_dates = run.confirmation_report["sample_size"]["test"]["total_trading_dates"]
        assert newey_west_n == n_trading_dates
        assert newey_west_n < n_trading_dates * len(seeded["instruments"])


class TestSignalConditionalNeverBecomesInferential:
    def test_no_hypothesis_test_metric_is_tagged_with_signal_conditional_naming(
        self, db_session, seeded
    ):
        run = run_confirmation_evaluation(db_session, **_confirm_kwargs(seeded))
        hypothesis_test_names = {
            m.metric_name for m in run.metrics if m.category == MetricCategory.HYPOTHESIS_TEST
        }
        assert not any("signal_conditional" in name for name in hypothesis_test_names)

    def test_descriptive_category_never_contains_inferential_metric_names(self, db_session, seeded):
        run = run_confirmation_evaluation(db_session, **_confirm_kwargs(seeded))
        descriptive_names = {
            m.metric_name for m in run.metrics if m.category == MetricCategory.DESCRIPTIVE
        }
        for forbidden in ("p_value", "test_statistic", "ci_low", "ci_high"):
            assert not any(forbidden in name for name in descriptive_names)


class TestWalkForwardSummaryStaysDescriptive:
    def test_walk_forward_consistency_contains_no_inferential_fields(self, db_session, seeded):
        wf_run = run_walk_forward_evaluation(
            db_session,
            experiment_definition_id=seeded["definition"].id,
            dataset_id=seeded["dataset"].id,
            rule_feature_definition_id=seeded["feature_definitions"]["close_vs_sma20"].id,
            configuration={"rule": "close_vs_sma20 > 0"},
            cost_bps_per_trade=10.0,
            cost_assumptions_description="flat bps",
            train_window_sessions=20,
            validation_window_sessions=5,
            test_window_sessions=5,
            step_sessions=100,
            mode=FoldMode.ROLLING,
            created_by="test-suite",
        )
        assert wf_run.status == RunStatus.COMPLETED

        run = run_confirmation_evaluation(
            db_session, **_confirm_kwargs(seeded, walk_forward_run_id=wf_run.id)
        )
        summary = run.confirmation_report["walk_forward_consistency"]
        assert summary is not None
        forbidden_keys = {"p_value", "test_statistic", "conclusion", "accept", "reject"}
        assert not (forbidden_keys & set(summary.keys()))

    def test_walk_forward_run_from_a_different_definition_is_rejected(self, db_session, seeded):
        other_definition = register_experiment_definition(
            db_session,
            name="stage3.6-mismatched-definition",
            hypothesis_family_id=seeded["family"].id,
            hypothesis="a different hypothesis",
            rule_feature_definition_id=seeded["feature_definitions"]["return_5d"].id,
            rule_description="rule",
            cost_model_description="flat bps",
            created_by="test-suite",
        )
        wf_run = run_walk_forward_evaluation(
            db_session,
            experiment_definition_id=other_definition.id,
            dataset_id=seeded["dataset"].id,
            rule_feature_definition_id=seeded["feature_definitions"]["return_5d"].id,
            configuration={"rule": "return_5d > 0"},
            cost_bps_per_trade=10.0,
            cost_assumptions_description="flat bps",
            train_window_sessions=20,
            validation_window_sessions=5,
            test_window_sessions=5,
            step_sessions=100,
            mode=FoldMode.ROLLING,
            created_by="test-suite",
        )
        with pytest.raises(ValueError, match="walk_forward_run_id"):
            run_confirmation_evaluation(
                db_session, **_confirm_kwargs(seeded, walk_forward_run_id=wf_run.id)
            )


class TestConfirmationUniquenessAppliesToThisPathwayToo:
    def test_second_confirmation_via_this_pathway_is_rejected(self, db_session, seeded):
        run_confirmation_evaluation(db_session, **_confirm_kwargs(seeded))
        with pytest.raises(ValueError, match="a CONFIRMATION run already exists"):
            run_confirmation_evaluation(db_session, **_confirm_kwargs(seeded))

    def test_a_confirmation_via_this_pathway_blocks_the_old_run_experiment_pathway_too(
        self, db_session, seeded
    ):
        from backend.app.research.experiment_tracking import run_experiment

        run_confirmation_evaluation(db_session, **_confirm_kwargs(seeded))
        with pytest.raises(ValueError, match="a CONFIRMATION run already exists"):
            run_experiment(
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


class TestComputeAdjustedThreshold:
    def test_bonferroni_divides_alpha_by_n(self):
        assert compute_adjusted_threshold(MultipleTestingMethod.BONFERRONI, 0.05, 5) == pytest.approx(0.01)

    def test_fdr_bh_is_a_reserved_not_yet_implemented_extension_point(self):
        with pytest.raises(NotImplementedError):
            compute_adjusted_threshold(MultipleTestingMethod.FDR_BH, 0.05, 5)

    def test_rejects_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            compute_adjusted_threshold(MultipleTestingMethod.BONFERRONI, 1.5, 5)

    def test_rejects_zero_hypotheses(self):
        with pytest.raises(ValueError, match="n_hypotheses"):
            compute_adjusted_threshold(MultipleTestingMethod.BONFERRONI, 0.05, 0)
