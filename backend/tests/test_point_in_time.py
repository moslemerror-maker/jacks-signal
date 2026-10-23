from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import select
from sqlalchemy.exc import ProgrammingError

from backend.app.db.models import (
    AssetType,
    DefinitionStatus,
    FeatureDefinition,
    FeatureGroup,
    FeatureValue,
    Instrument,
    InstrumentSymbol,
    LabelDefinition,
    LabelValue,
)
from backend.app.research.point_in_time import (
    assemble_point_in_time_features,
    assemble_training_row,
    resolve_label_outcome,
)


@pytest.fixture
def instrument(db_session) -> Instrument:
    inst = Instrument(exchange="NSE", asset_type=AssetType.EQUITY)
    db_session.add(inst)
    db_session.flush()
    return inst


def _feature_definition(**overrides) -> FeatureDefinition:
    fields = dict(
        name="test_feature",
        version=1,
        feature_group=FeatureGroup.PRICE,
        description="test feature",
        formula_ref="backend.app.research.features.price:test",
        input_dependencies="market_bars.close",
        availability_lag=0,
        timeframe="1D",
        status=DefinitionStatus.ACTIVE,
        created_by="test-suite",
    )
    fields.update(overrides)
    return FeatureDefinition(**fields)


def _label_definition(**overrides) -> LabelDefinition:
    fields = dict(
        name="test_label",
        version=1,
        description="test label",
        formula_ref="backend.app.research.labels.forward_return:label",
        horizon_days=5,
        timeframe="1D",
        status=DefinitionStatus.ACTIVE,
        created_by="test-suite",
    )
    fields.update(overrides)
    return LabelDefinition(**fields)


def _feature_value(instrument_id, feature_definition_id, **overrides) -> FeatureValue:
    fields = dict(
        instrument_id=instrument_id,
        feature_definition_id=feature_definition_id,
        computed_for_date=date(2024, 3, 1),
        timeframe="1D",
        value=1.0,
        available_at=date(2024, 3, 1),
    )
    fields.update(overrides)
    return FeatureValue(**fields)


def _label_value(instrument_id, label_definition_id, **overrides) -> LabelValue:
    fields = dict(
        instrument_id=instrument_id,
        label_definition_id=label_definition_id,
        entry_date=date(2024, 3, 1),
        horizon_end_date=date(2024, 3, 8),
        timeframe="1D",
        realized_return=0.02,
    )
    fields.update(overrides)
    return LabelValue(**fields)


class TestBasicFeatureAssembly:
    def test_returns_value_when_available(self, db_session, instrument):
        fd = _feature_definition()
        db_session.add(fd)
        db_session.flush()
        db_session.add(
            _feature_value(instrument.id, fd.id, value=42.0, computed_for_date=date(2024, 3, 1))
        )
        db_session.flush()

        [row] = assemble_point_in_time_features(
            db_session, [instrument.id], [fd.id], date(2024, 3, 1)
        )
        assert row.values[fd.id] == 42.0

    def test_returns_none_when_no_row_exists(self, db_session, instrument):
        fd = _feature_definition()
        db_session.add(fd)
        db_session.flush()

        [row] = assemble_point_in_time_features(
            db_session, [instrument.id], [fd.id], date(2024, 3, 1)
        )
        assert row.values[fd.id] is None


class TestBasicLabelResolution:
    def test_returns_outcome_once_horizon_elapsed(self, db_session, instrument):
        ld = _label_definition()
        db_session.add(ld)
        db_session.flush()
        db_session.add(
            _label_value(
                instrument.id,
                ld.id,
                entry_date=date(2024, 3, 1),
                horizon_end_date=date(2024, 3, 8),
                realized_return=0.05,
            )
        )
        db_session.flush()

        outcome = resolve_label_outcome(
            db_session, instrument.id, ld.id, date(2024, 3, 1), known_as_of=date(2024, 3, 8)
        )
        assert outcome is not None
        assert outcome.realized_return == 0.05

    def test_returns_none_when_no_row_exists(self, db_session, instrument):
        ld = _label_definition()
        db_session.add(ld)
        db_session.flush()

        outcome = resolve_label_outcome(
            db_session, instrument.id, ld.id, date(2024, 3, 1), known_as_of=date(2024, 3, 8)
        )
        assert outcome is None


class TestAvailableAtBoundary:
    def test_available_at_equal_to_entry_date_is_included(self, db_session, instrument):
        fd = _feature_definition()
        db_session.add(fd)
        db_session.flush()
        db_session.add(
            _feature_value(
                instrument.id,
                fd.id,
                computed_for_date=date(2024, 3, 1),
                available_at=date(2024, 3, 1),
                value=7.0,
            )
        )
        db_session.flush()

        [row] = assemble_point_in_time_features(
            db_session, [instrument.id], [fd.id], date(2024, 3, 1)
        )
        assert row.values[fd.id] == 7.0

    def test_available_at_after_entry_date_is_excluded(self, db_session, instrument):
        fd = _feature_definition(availability_lag=1)
        db_session.add(fd)
        db_session.flush()
        db_session.add(
            _feature_value(
                instrument.id,
                fd.id,
                computed_for_date=date(2024, 3, 1),
                available_at=date(2024, 3, 2),
                value=7.0,
            )
        )
        db_session.flush()

        [row] = assemble_point_in_time_features(
            db_session, [instrument.id], [fd.id], date(2024, 3, 1)
        )
        assert row.values[fd.id] is None


class TestHorizonEndDateBoundary:
    def test_horizon_end_date_equal_to_known_as_of_is_resolved(self, db_session, instrument):
        ld = _label_definition()
        db_session.add(ld)
        db_session.flush()
        db_session.add(
            _label_value(
                instrument.id,
                ld.id,
                entry_date=date(2024, 3, 1),
                horizon_end_date=date(2024, 3, 8),
            )
        )
        db_session.flush()

        outcome = resolve_label_outcome(
            db_session, instrument.id, ld.id, date(2024, 3, 1), known_as_of=date(2024, 3, 8)
        )
        assert outcome is not None

    def test_horizon_end_date_after_known_as_of_is_not_resolved(self, db_session, instrument):
        ld = _label_definition()
        db_session.add(ld)
        db_session.flush()
        db_session.add(
            _label_value(
                instrument.id,
                ld.id,
                entry_date=date(2024, 3, 1),
                horizon_end_date=date(2024, 3, 8),
            )
        )
        db_session.flush()

        outcome = resolve_label_outcome(
            db_session,
            instrument.id,
            ld.id,
            date(2024, 3, 1),
            known_as_of=date(2024, 3, 7),
        )
        assert outcome is None


class TestAdversarialFutureFeature:

    def test_future_dated_feature_never_appears_in_an_earlier_row(self, db_session, instrument):
        fd = _feature_definition()
        db_session.add(fd)
        db_session.flush()
        db_session.add(
            _feature_value(
                instrument.id,
                fd.id,
                computed_for_date=date(2024, 3, 10),
                available_at=date(2024, 3, 10),
                value=999.0,
            )
        )
        db_session.flush()

        [row] = assemble_point_in_time_features(
            db_session, [instrument.id], [fd.id], date(2024, 3, 1)
        )
        assert row.values[fd.id] is None

    def test_available_at_in_the_future_relative_to_entry_date_is_excluded(
        self, db_session, instrument
    ):
        fd = _feature_definition(availability_lag=9)
        db_session.add(fd)
        db_session.flush()
        db_session.add(
            _feature_value(
                instrument.id,
                fd.id,
                computed_for_date=date(2024, 3, 1),
                available_at=date(2024, 3, 10),
                value=999.0,
            )
        )
        db_session.flush()

        [row] = assemble_point_in_time_features(
            db_session, [instrument.id], [fd.id], date(2024, 3, 1)
        )
        assert row.values[fd.id] is None


class TestAdversarialLaggedFeature:

    def test_lagged_feature_excluded_from_its_own_entry_date_row(self, db_session, instrument):
        fd = _feature_definition(availability_lag=5)
        db_session.add(fd)
        db_session.flush()
        entry_date = date(2024, 3, 1)
        db_session.add(
            _feature_value(
                instrument.id,
                fd.id,
                computed_for_date=entry_date,
                available_at=date(2024, 3, 6),
                value=42.0,
            )
        )
        db_session.flush()

        [row] = assemble_point_in_time_features(
            db_session, [instrument.id], [fd.id], entry_date
        )
        assert row.values[fd.id] is None

    def test_lagged_feature_also_absent_from_the_later_date_it_becomes_known_on(
        self, db_session, instrument
    ):
        fd = _feature_definition(availability_lag=5)
        db_session.add(fd)
        db_session.flush()
        db_session.add(
            _feature_value(
                instrument.id,
                fd.id,
                computed_for_date=date(2024, 3, 1),
                available_at=date(2024, 3, 6),
                value=42.0,
            )
        )
        db_session.flush()

        [row] = assemble_point_in_time_features(
            db_session, [instrument.id], [fd.id], date(2024, 3, 6)
        )
        assert row.values[fd.id] is None


class TestAdversarialLabelHorizonInFuture:

    def test_unresolved_label_outcome_excluded_from_training_row(self, db_session, instrument):
        fd = _feature_definition()
        ld = _label_definition()
        db_session.add_all([fd, ld])
        db_session.flush()

        entry_date = date(2024, 3, 1)
        db_session.add(
            _feature_value(instrument.id, fd.id, computed_for_date=entry_date, value=1.0)
        )
        db_session.add(
            _label_value(
                instrument.id,
                ld.id,
                entry_date=entry_date,
                horizon_end_date=date(2024, 3, 20),
                realized_return=0.10,
            )
        )
        db_session.flush()

        row = assemble_training_row(
            db_session,
            instrument.id,
            [fd.id],
            ld.id,
            entry_date,
            known_as_of=date(2024, 3, 5),
        )
        assert row.label is None
        assert row.label_resolved is False
        assert row.features[fd.id] == 1.0


class TestAdversarialFeatureChangesAfterTargetDate:

    def test_value_rows_cannot_be_mutated_in_place(self, db_session, instrument):
        fd = _feature_definition()
        db_session.add(fd)
        db_session.flush()
        fv = _feature_value(instrument.id, fd.id, computed_for_date=date(2024, 3, 1), value=1.0)
        db_session.add(fv)
        db_session.flush()

        fv.value = 999.0
        with pytest.raises(ProgrammingError, match="immutable"):
            db_session.flush()

    def test_recomputation_under_a_new_version_does_not_alter_the_old_versions_row(
        self, db_session, instrument
    ):
        fd_v1 = _feature_definition(name="recomputed_feature", version=1)
        db_session.add(fd_v1)
        db_session.flush()
        entry_date = date(2024, 3, 1)
        db_session.add(
            _feature_value(instrument.id, fd_v1.id, computed_for_date=entry_date, value=1.0)
        )
        db_session.flush()

        [row_before] = assemble_point_in_time_features(
            db_session, [instrument.id], [fd_v1.id], entry_date
        )
        assert row_before.values[fd_v1.id] == 1.0

        fd_v2 = _feature_definition(name="recomputed_feature", version=2)
        db_session.add(fd_v2)
        db_session.flush()
        db_session.add(
            _feature_value(instrument.id, fd_v2.id, computed_for_date=entry_date, value=2.0)
        )
        db_session.flush()

        [row_after] = assemble_point_in_time_features(
            db_session, [instrument.id], [fd_v1.id], entry_date
        )
        assert row_after.values[fd_v1.id] == 1.0

        [row_v2] = assemble_point_in_time_features(
            db_session, [instrument.id], [fd_v2.id], entry_date
        )
        assert row_v2.values[fd_v2.id] == 2.0


class TestAdversarialMissingValuesAroundBoundary:

    def test_missing_date_is_not_filled_from_the_day_before(self, db_session, instrument):
        fd = _feature_definition()
        db_session.add(fd)
        db_session.flush()
        db_session.add(
            _feature_value(instrument.id, fd.id, computed_for_date=date(2024, 3, 1), value=5.0)
        )
        db_session.flush()

        [row] = assemble_point_in_time_features(
            db_session, [instrument.id], [fd.id], date(2024, 3, 4)
        )
        assert row.values[fd.id] is None

    def test_missing_date_is_not_filled_from_the_day_after(self, db_session, instrument):
        fd = _feature_definition()
        db_session.add(fd)
        db_session.flush()
        db_session.add(
            _feature_value(
                instrument.id,
                fd.id,
                computed_for_date=date(2024, 3, 4),
                available_at=date(2024, 3, 4),
                value=5.0,
            )
        )
        db_session.flush()

        [row] = assemble_point_in_time_features(
            db_session, [instrument.id], [fd.id], date(2024, 3, 1)
        )
        assert row.values[fd.id] is None


class TestAdversarialSimilarSymbolsDifferentInstruments:

    def test_no_cross_contamination_between_instruments_sharing_a_symbol(self, db_session):
        equity = Instrument(exchange="NSE", asset_type=AssetType.EQUITY)
        bond = Instrument(exchange="NSE", asset_type=AssetType.EQUITY)
        db_session.add_all([equity, bond])
        db_session.flush()
        db_session.add_all(
            [
                InstrumentSymbol(
                    instrument_id=equity.id, symbol="TESTCO", effective_from=date(2020, 1, 1)
                ),
                InstrumentSymbol(
                    instrument_id=bond.id, symbol="TESTCO", effective_from=date(2020, 1, 1)
                ),
            ]
        )
        db_session.flush()

        fd = _feature_definition()
        db_session.add(fd)
        db_session.flush()
        entry_date = date(2024, 3, 1)
        db_session.add(
            _feature_value(equity.id, fd.id, computed_for_date=entry_date, value=100.0)
        )
        db_session.add(_feature_value(bond.id, fd.id, computed_for_date=entry_date, value=-50.0))
        db_session.flush()

        [equity_row] = assemble_point_in_time_features(
            db_session, [equity.id], [fd.id], entry_date
        )
        [bond_row] = assemble_point_in_time_features(db_session, [bond.id], [fd.id], entry_date)

        assert equity_row.values[fd.id] == 100.0
        assert bond_row.values[fd.id] == -50.0


class TestAdversarialSameDayEodConvention:

    def test_same_day_feature_is_usable_for_a_same_day_decision(self, db_session, instrument):
        fd = _feature_definition()
        db_session.add(fd)
        db_session.flush()
        entry_date = date(2024, 3, 1)
        db_session.add(
            _feature_value(
                instrument.id,
                fd.id,
                computed_for_date=entry_date,
                available_at=entry_date,
                value=3.14,
            )
        )
        db_session.flush()

        [row] = assemble_point_in_time_features(
            db_session, [instrument.id], [fd.id], entry_date
        )
        assert row.values[fd.id] == 3.14


def _naive_get_feature_value(session, instrument_id, feature_definition_id, entry_date):
    fv = session.scalar(
        select(FeatureValue)
        .where(
            FeatureValue.instrument_id == instrument_id,
            FeatureValue.feature_definition_id == feature_definition_id,
            FeatureValue.computed_for_date <= entry_date,
        )
        .order_by(FeatureValue.computed_for_date.desc())
        .limit(1)
    )
    return fv.value if fv is not None else None


class TestNaiveImplementationWouldLeak:
    def test_naive_lookup_leaks_a_lagged_value_the_correct_one_excludes(
        self, db_session, instrument
    ):
        fd = _feature_definition(availability_lag=5)
        db_session.add(fd)
        db_session.flush()
        entry_date = date(2024, 3, 1)
        db_session.add(
            _feature_value(
                instrument.id,
                fd.id,
                computed_for_date=entry_date,
                available_at=date(2024, 3, 6),
                value=999.0,
            )
        )
        db_session.flush()

        leaked = _naive_get_feature_value(db_session, instrument.id, fd.id, entry_date)
        assert leaked == 999.0

        [row] = assemble_point_in_time_features(
            db_session, [instrument.id], [fd.id], entry_date
        )
        assert row.values[fd.id] is None
