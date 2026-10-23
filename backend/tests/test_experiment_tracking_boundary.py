from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy.exc import IntegrityError, ProgrammingError

from backend.app.db.models import (
    AssetType,
    ExperimentMetric,
    ExperimentRun,
    Instrument,
    MarketBar,
    MetricSplit,
    RunPurpose,
    RunStatus,
    SourceFormat,
)
from backend.app.research.datasets import build_dataset
from backend.app.research.experiment_tracking import (
    count_tested_hypotheses_in_family,
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
        name="stage3.2-test-dataset",
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
        name="stage3.2-test-family",
        description="test family for the Exploration/Confirmation boundary suite",
        created_by="test-suite",
    )

    definition_v1 = register_experiment_definition(
        db_session,
        name="stage3.2-test-definition",
        hypothesis_family_id=family.id,
        hypothesis="test hypothesis v1",
        rule_feature_definition_id=feature_definitions["close_vs_sma20"].id,
        rule_description="LONG if close_vs_sma20 > 0 else FLAT",
        cost_model_description="flat bps deduction",
        created_by="test-suite",
        version=1,
    )

    return {
        "instrument": instrument,
        "feature_definitions": feature_definitions,
        "label_definition": label_definition,
        "dataset": dataset,
        "family": family,
        "definition_v1": definition_v1,
    }


def _confirmation_kwargs(ctx, **overrides):
    kwargs = dict(
        experiment_definition_id=ctx["definition_v1"].id,
        dataset_id=ctx["dataset"].id,
        rule_feature_definition_id=ctx["feature_definitions"]["close_vs_sma20"].id,
        configuration={"rule": "close_vs_sma20 > 0", "threshold": 0.0},
        cost_bps_per_trade=10.0,
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
    return kwargs


def _exploration_kwargs(ctx, **overrides):
    kwargs = dict(
        experiment_definition_id=ctx["definition_v1"].id,
        dataset_id=ctx["dataset"].id,
        rule_feature_definition_id=ctx["feature_definitions"]["close_vs_sma20"].id,
        configuration={"rule": "close_vs_sma20 > 0", "threshold": 0.0},
        cost_bps_per_trade=10.0,
        cost_assumptions_description="flat round-trip bps, no market impact modeled",
        run_purpose=RunPurpose.EXPLORATION,
        train_start_date=_TRADING_DATES[0],
        train_end_date=_TRADING_DATES[29],
        created_by="test-suite",
    )
    kwargs.update(overrides)
    return kwargs


class TestExplorationCannotEmitConfirmationMetrics:
    def test_exploration_with_validation_dates_is_rejected_before_anything_runs(
        self, db_session, seeded
    ):
        with pytest.raises(ValueError, match="EXPLORATION runs must not specify"):
            run_experiment(
                db_session,
                **_exploration_kwargs(
                    seeded,
                    validation_start_date=_TRADING_DATES[30],
                    validation_end_date=_TRADING_DATES[39],
                ),
            )

    def test_exploration_run_only_ever_produces_train_split_metrics(self, db_session, seeded):
        run = run_experiment(db_session, **_exploration_kwargs(seeded))
        assert run.status == RunStatus.COMPLETED
        splits_present = {m.split for m in run.metrics}
        assert splits_present == {MetricSplit.TRAIN}

    def test_directly_inserting_a_validation_metric_on_an_exploration_run_is_rejected(
        self, db_session, seeded
    ):
        run = run_experiment(db_session, **_exploration_kwargs(seeded))
        db_session.add(
            ExperimentMetric(
                run_id=run.id,
                split=MetricSplit.VALIDATION,
                category=run.metrics[0].category,
                metric_name="a_smuggled_metric",
                metric_value=1.0,
                sample_size=1,
            )
        )
        with pytest.raises(ProgrammingError, match="must not emit"):
            db_session.flush()

    def test_exploration_run_never_acquires_a_conclusion(self, db_session, seeded):
        run = run_experiment(db_session, **_exploration_kwargs(seeded))
        assert run.conclusion is None
        assert run.conclusion_rationale is None


class TestConfirmationRequiresAppropriatePeriods:
    def test_confirmation_missing_validation_dates_is_rejected(self, db_session, seeded):
        with pytest.raises(ValueError, match="CONFIRMATION runs require"):
            run_experiment(
                db_session,
                **_confirmation_kwargs(
                    seeded, validation_start_date=None, validation_end_date=None
                ),
            )

    def test_confirmation_missing_test_dates_is_rejected(self, db_session, seeded):
        with pytest.raises(ValueError, match="CONFIRMATION runs require"):
            run_experiment(
                db_session,
                **_confirmation_kwargs(seeded, test_start_date=None, test_end_date=None),
            )

    def test_confirmation_with_all_periods_succeeds_and_has_a_conclusion(
        self, db_session, seeded
    ):
        run = run_experiment(db_session, **_confirmation_kwargs(seeded))
        assert run.status == RunStatus.COMPLETED
        assert run.conclusion is not None
        splits_present = {m.split for m in run.metrics}
        assert splits_present == {MetricSplit.TRAIN, MetricSplit.VALIDATION, MetricSplit.TEST}


class TestConfirmationUniqueness:
    def test_second_confirmation_attempt_is_rejected_at_the_application_level(
        self, db_session, seeded
    ):
        run_experiment(db_session, **_confirmation_kwargs(seeded))
        with pytest.raises(ValueError, match="a CONFIRMATION run already exists"):
            run_experiment(db_session, **_confirmation_kwargs(seeded))

    def test_second_confirmation_attempt_is_rejected_at_the_database_level(
        self, db_session, seeded
    ):
        common = dict(
            experiment_definition_id=seeded["definition_v1"].id,
            dataset_id=seeded["dataset"].id,
            rule_feature_definition_id=seeded["feature_definitions"]["close_vs_sma20"].id,
            run_purpose=RunPurpose.CONFIRMATION,
            git_commit="0" * 40,
            git_dirty=False,
            dependency_versions={},
            configuration={},
            cost_assumptions={"cost_bps_per_trade": 10.0, "description": "x"},
            train_start_date=_TRADING_DATES[0],
            train_end_date=_TRADING_DATES[29],
            validation_start_date=_TRADING_DATES[30],
            validation_end_date=_TRADING_DATES[39],
            test_start_date=_TRADING_DATES[40],
            test_end_date=_TRADING_DATES[54],
            executed_at=datetime.now(timezone.utc),
            status=RunStatus.FAILED,
            failure_reason="synthetic - forcing the DB constraint directly",
            conclusion=None,
            conclusion_rationale=None,
            checksum=None,
            created_by="test-suite",
        )
        db_session.add(ExperimentRun(**common))
        db_session.add(ExperimentRun(**common))
        with pytest.raises(IntegrityError):
            db_session.flush()

    def test_a_failed_confirmation_still_consumes_the_one_shot_slot(self, db_session, seeded):
        far_future = date(2099, 1, 1)
        failed = run_experiment(
            db_session,
            **_confirmation_kwargs(
                seeded,
                train_start_date=far_future,
                train_end_date=far_future + timedelta(days=5),
                validation_start_date=far_future + timedelta(days=6),
                validation_end_date=far_future + timedelta(days=10),
                test_start_date=far_future + timedelta(days=11),
                test_end_date=far_future + timedelta(days=15),
            ),
        )
        assert failed.status == RunStatus.FAILED
        with pytest.raises(ValueError, match="a CONFIRMATION run already exists"):
            run_experiment(db_session, **_confirmation_kwargs(seeded))

    def test_exploration_runs_are_unlimited_and_never_hit_the_uniqueness_rule(
        self, db_session, seeded
    ):
        first = run_experiment(db_session, **_exploration_kwargs(seeded))
        second = run_experiment(db_session, **_exploration_kwargs(seeded))
        third = run_experiment(db_session, **_exploration_kwargs(seeded))
        assert len({first.id, second.id, third.id}) == 3


class TestNewDefinitionVersionCanConfirmAgain:
    def test_a_new_version_creates_an_independent_confirmable_definition(
        self, db_session, seeded
    ):
        run_experiment(db_session, **_confirmation_kwargs(seeded))

        definition_v2 = register_experiment_definition(
            db_session,
            name="stage3.2-test-definition",
            hypothesis_family_id=seeded["family"].id,
            hypothesis="test hypothesis v2 - genuinely different rule threshold reasoning",
            rule_feature_definition_id=seeded["feature_definitions"]["close_vs_sma20"].id,
            rule_description="LONG if close_vs_sma20 > 0 else FLAT (v2)",
            cost_model_description="flat bps deduction",
            created_by="test-suite",
            version=2,
        )
        assert definition_v2.id != seeded["definition_v1"].id

        run_v2 = run_experiment(
            db_session,
            **_confirmation_kwargs(seeded, experiment_definition_id=definition_v2.id),
        )
        assert run_v2.status == RunStatus.COMPLETED
        assert run_v2.experiment_definition_id == definition_v2.id


class TestFailedRunsRemainQueryable:
    def test_a_failed_confirmation_run_remains_queryable(self, db_session, seeded):
        far_future = date(2099, 1, 1)
        run = run_experiment(
            db_session,
            **_confirmation_kwargs(
                seeded,
                train_start_date=far_future,
                train_end_date=far_future + timedelta(days=5),
                validation_start_date=far_future + timedelta(days=6),
                validation_end_date=far_future + timedelta(days=10),
                test_start_date=far_future + timedelta(days=11),
                test_end_date=far_future + timedelta(days=15),
            ),
        )
        assert run.status == RunStatus.FAILED
        reloaded = db_session.get(ExperimentRun, run.id)
        assert reloaded is not None
        assert reloaded.status == RunStatus.FAILED

    def test_a_failed_exploration_run_remains_queryable(self, db_session, seeded):
        far_future = date(2099, 1, 1)
        run = run_experiment(
            db_session,
            **_exploration_kwargs(
                seeded,
                train_start_date=far_future,
                train_end_date=far_future + timedelta(days=5),
            ),
        )
        assert run.status == RunStatus.FAILED
        reloaded = db_session.get(ExperimentRun, run.id)
        assert reloaded is not None


class TestImmutabilityBothPurposes:
    def test_exploration_run_is_immutable(self, db_session, seeded):
        run = run_experiment(db_session, **_exploration_kwargs(seeded))
        run.failure_reason = "tampered"
        with pytest.raises(ProgrammingError):
            db_session.flush()

    def test_confirmation_run_is_immutable(self, db_session, seeded):
        run = run_experiment(db_session, **_confirmation_kwargs(seeded))
        run.conclusion_rationale = "tampered"
        with pytest.raises(ProgrammingError):
            db_session.flush()

    def test_exploration_run_cannot_be_deleted(self, db_session, seeded):
        run = run_experiment(db_session, **_exploration_kwargs(seeded))
        db_session.delete(run)
        with pytest.raises(ProgrammingError):
            db_session.flush()

    def test_confirmation_run_cannot_be_deleted(self, db_session, seeded):
        run = run_experiment(db_session, **_confirmation_kwargs(seeded))
        db_session.delete(run)
        with pytest.raises(ProgrammingError):
            db_session.flush()


class TestHypothesisFamilyExplicitMembership:
    def test_definition_family_membership_is_explicit_and_queryable(self, db_session, seeded):
        assert seeded["definition_v1"].hypothesis_family_id == seeded["family"].id

    def test_family_identity_is_immutable_for_a_definition(self, db_session, seeded):
        other_family = register_hypothesis_family(
            db_session,
            name="stage3.2-other-family",
            description="a different family",
            created_by="test-suite",
        )
        seeded["definition_v1"].hypothesis_family_id = other_family.id
        with pytest.raises(ProgrammingError):
            db_session.flush()

    def test_hypothesis_family_itself_is_immutable(self, db_session, seeded):
        seeded["family"].description = "tampered"
        with pytest.raises(ProgrammingError):
            db_session.flush()

    def test_hypothesis_family_cannot_be_deleted(self, db_session, seeded):
        db_session.delete(seeded["family"])
        with pytest.raises(ProgrammingError):
            db_session.flush()

    def test_family_count_is_derived_from_persisted_records_not_a_counter(
        self, db_session, seeded
    ):
        assert count_tested_hypotheses_in_family(db_session, seeded["family"].id) == 0

        run_experiment(db_session, **_exploration_kwargs(seeded))
        assert count_tested_hypotheses_in_family(db_session, seeded["family"].id) == 1

        definition_v2 = register_experiment_definition(
            db_session,
            name="stage3.2-test-definition",
            hypothesis_family_id=seeded["family"].id,
            hypothesis="v2",
            rule_feature_definition_id=seeded["feature_definitions"]["close_vs_sma20"].id,
            rule_description="v2 rule",
            cost_model_description="flat bps",
            created_by="test-suite",
            version=2,
        )
        assert count_tested_hypotheses_in_family(db_session, seeded["family"].id) == 1

        run_experiment(
            db_session,
            **_exploration_kwargs(seeded, experiment_definition_id=definition_v2.id),
        )
        assert count_tested_hypotheses_in_family(db_session, seeded["family"].id) == 2

    def test_family_count_ignores_definitions_in_other_families(self, db_session, seeded):
        other_family = register_hypothesis_family(
            db_session,
            name="stage3.2-unrelated-family",
            description="unrelated",
            created_by="test-suite",
        )
        other_definition = register_experiment_definition(
            db_session,
            name="stage3.2-unrelated-definition",
            hypothesis_family_id=other_family.id,
            hypothesis="unrelated hypothesis",
            rule_feature_definition_id=seeded["feature_definitions"]["close_vs_sma20"].id,
            rule_description="unrelated rule",
            cost_model_description="flat bps",
            created_by="test-suite",
        )
        run_experiment(
            db_session,
            **_exploration_kwargs(seeded, experiment_definition_id=other_definition.id),
        )
        assert count_tested_hypotheses_in_family(db_session, seeded["family"].id) == 0
        assert count_tested_hypotheses_in_family(db_session, other_family.id) == 1


class TestConfirmationTimeFamilyCountIncludesCurrentHypothesis:

    def test_a_freshly_registered_never_run_definition_is_included_via_include_definition_id(
        self, db_session, seeded
    ):
        fresh_family = register_hypothesis_family(
            db_session,
            name="stage3.6-fresh-family",
            description="a family with one brand-new, never-run definition",
            created_by="test-suite",
        )
        fresh_definition = register_experiment_definition(
            db_session,
            name="stage3.6-fresh-definition",
            hypothesis_family_id=fresh_family.id,
            hypothesis="never run before",
            rule_feature_definition_id=seeded["feature_definitions"]["close_vs_sma20"].id,
            rule_description="rule",
            cost_model_description="flat bps",
            created_by="test-suite",
        )
        assert count_tested_hypotheses_in_family(db_session, fresh_family.id) == 0
        assert (
            count_tested_hypotheses_in_family(
                db_session, fresh_family.id, include_definition_id=fresh_definition.id
            )
            == 1
        )

    def test_a_definition_with_prior_exploration_runs_is_not_double_counted(
        self, db_session, seeded
    ):
        run_experiment(db_session, **_exploration_kwargs(seeded))
        assert count_tested_hypotheses_in_family(db_session, seeded["family"].id) == 1

        assert (
            count_tested_hypotheses_in_family(
                db_session,
                seeded["family"].id,
                include_definition_id=seeded["definition_v1"].id,
            )
            == 1
        )

    def test_include_definition_id_for_an_unrelated_definition_in_the_same_family(
        self, db_session, seeded
    ):
        run_experiment(db_session, **_exploration_kwargs(seeded))

        never_run_definition = register_experiment_definition(
            db_session,
            name="stage3.2-test-definition",
            hypothesis_family_id=seeded["family"].id,
            hypothesis="v2, never run",
            rule_feature_definition_id=seeded["feature_definitions"]["close_vs_sma20"].id,
            rule_description="v2 rule",
            cost_model_description="flat bps",
            created_by="test-suite",
            version=2,
        )
        count = count_tested_hypotheses_in_family(
            db_session, seeded["family"].id, include_definition_id=never_run_definition.id
        )
        assert count == 2


class TestHypothesisCannotChangeAfterConfirmation:
    def test_mutating_the_confirmed_definitions_rule_is_rejected(self, db_session, seeded):
        run_experiment(db_session, **_confirmation_kwargs(seeded))
        seeded["definition_v1"].rule_description = "a silently different rule"
        with pytest.raises(ProgrammingError):
            db_session.flush()

    def test_mutating_the_confirmed_definitions_hypothesis_text_is_rejected(
        self, db_session, seeded
    ):
        run_experiment(db_session, **_confirmation_kwargs(seeded))
        seeded["definition_v1"].hypothesis = "a silently different hypothesis"
        with pytest.raises(ProgrammingError):
            db_session.flush()

    def test_the_only_legitimate_path_is_a_new_version(self, db_session, seeded):
        run_experiment(db_session, **_confirmation_kwargs(seeded))
        definition_v2 = register_experiment_definition(
            db_session,
            name="stage3.2-test-definition",
            hypothesis_family_id=seeded["family"].id,
            hypothesis="a genuinely revised hypothesis",
            rule_feature_definition_id=seeded["feature_definitions"]["close_vs_sma20"].id,
            rule_description="a genuinely revised rule",
            cost_model_description="flat bps deduction",
            created_by="test-suite",
            version=2,
        )
        assert definition_v2.id != seeded["definition_v1"].id
        assert definition_v2.hypothesis == "a genuinely revised hypothesis"
        assert seeded["definition_v1"].hypothesis == "test hypothesis v1"


class TestConfirmationFreezing:
    def test_dataset_referenced_by_a_confirmation_cannot_be_mutated(self, db_session, seeded):
        run_experiment(db_session, **_confirmation_kwargs(seeded))
        seeded["dataset"].name = "renamed-after-confirmation"
        with pytest.raises(ProgrammingError):
            db_session.flush()

    def test_confirmation_run_itself_records_the_exact_definition_and_dataset(
        self, db_session, seeded
    ):
        run = run_experiment(db_session, **_confirmation_kwargs(seeded))
        assert run.experiment_definition_id == seeded["definition_v1"].id
        assert run.dataset_id == seeded["dataset"].id
