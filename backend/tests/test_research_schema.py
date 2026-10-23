from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

import pytest
from sqlalchemy.exc import IntegrityError, ProgrammingError

from backend.app.db.models import (
    AssetType,
    DefinitionStatus,
    FeatureDefinition,
    FeatureGroup,
    FeatureValue,
    Instrument,
    LabelDefinition,
    LabelValue,
)


@pytest.fixture
def instrument(db_session) -> Instrument:
    inst = Instrument(exchange="NSE", asset_type=AssetType.EQUITY)
    db_session.add(inst)
    db_session.flush()
    return inst


def _feature_definition(**overrides) -> FeatureDefinition:
    fields = dict(
        name="return_5d",
        version=1,
        feature_group=FeatureGroup.PRICE,
        description="5-trading-day forward-looking return placeholder for tests",
        formula_ref="backend.app.research.features.price:return_n_day",
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
        name="fwd_return_5d",
        version=1,
        description="Forward 5-trading-day realized return",
        formula_ref="backend.app.research.labels.forward_return:label",
        horizon_days=5,
        timeframe="1D",
        status=DefinitionStatus.ACTIVE,
        created_by="test-suite",
    )
    fields.update(overrides)
    return LabelDefinition(**fields)


class TestFeatureDefinition:
    def test_creates_successfully(self, db_session):
        fd = _feature_definition()
        db_session.add(fd)
        db_session.flush()
        assert fd.id is not None
        assert fd.status is DefinitionStatus.ACTIVE

    def test_duplicate_name_version_rejected(self, db_session):
        db_session.add(_feature_definition())
        db_session.flush()
        db_session.add(_feature_definition())
        with pytest.raises(IntegrityError):
            db_session.flush()

    def test_same_name_different_version_allowed(self, db_session):
        db_session.add(_feature_definition(version=1))
        db_session.flush()
        db_session.add(_feature_definition(version=2))
        db_session.flush()


class TestFeatureDefinitionImmutability:

    def test_status_update_is_allowed(self, db_session):
        fd = _feature_definition()
        db_session.add(fd)
        db_session.flush()

        fd.status = DefinitionStatus.DEPRECATED
        db_session.flush()

        db_session.refresh(fd)
        assert fd.status is DefinitionStatus.DEPRECATED

    def test_content_update_is_rejected(self, db_session):
        fd = _feature_definition()
        db_session.add(fd)
        db_session.flush()

        fd.formula_ref = "sneaky:overwrite"
        with pytest.raises(ProgrammingError, match="immutable"):
            db_session.flush()

    def test_delete_is_rejected(self, db_session):
        fd = _feature_definition()
        db_session.add(fd)
        db_session.flush()

        db_session.delete(fd)
        with pytest.raises(ProgrammingError, match="cannot be deleted"):
            db_session.flush()


class TestFeatureValue:
    def test_creates_successfully(self, db_session, instrument):
        fd = _feature_definition()
        db_session.add(fd)
        db_session.flush()

        fv = FeatureValue(
            instrument_id=instrument.id,
            feature_definition_id=fd.id,
            computed_for_date=date(2024, 1, 2),
            timeframe="1D",
            value=1.23,
            available_at=date(2024, 1, 2),
        )
        db_session.add(fv)
        db_session.flush()
        assert fv.id is not None

    def test_available_at_before_computed_for_date_is_rejected(self, db_session, instrument):
        fd = _feature_definition()
        db_session.add(fd)
        db_session.flush()

        fv = FeatureValue(
            instrument_id=instrument.id,
            feature_definition_id=fd.id,
            computed_for_date=date(2024, 1, 10),
            timeframe="1D",
            value=1.23,
            available_at=date(2024, 1, 5),
        )
        db_session.add(fv)
        with pytest.raises(IntegrityError, match="ck_feature_value_available_at_order"):
            db_session.flush()

    def test_available_at_equal_to_computed_for_date_is_allowed(self, db_session, instrument):
        fd = _feature_definition()
        db_session.add(fd)
        db_session.flush()

        fv = FeatureValue(
            instrument_id=instrument.id,
            feature_definition_id=fd.id,
            computed_for_date=date(2024, 1, 10),
            timeframe="1D",
            value=1.23,
            available_at=date(2024, 1, 10),
        )
        db_session.add(fv)
        db_session.flush()

    def test_available_at_after_computed_for_date_is_allowed(self, db_session, instrument):
        fd = _feature_definition(availability_lag=2)
        db_session.add(fd)
        db_session.flush()

        fv = FeatureValue(
            instrument_id=instrument.id,
            feature_definition_id=fd.id,
            computed_for_date=date(2024, 1, 10),
            timeframe="1D",
            value=1.23,
            available_at=date(2024, 1, 12),
        )
        db_session.add(fv)
        db_session.flush()

    def test_duplicate_identity_rejected(self, db_session, instrument):
        fd = _feature_definition()
        db_session.add(fd)
        db_session.flush()

        common = dict(
            instrument_id=instrument.id,
            feature_definition_id=fd.id,
            computed_for_date=date(2024, 1, 2),
            timeframe="1D",
            available_at=date(2024, 1, 2),
        )
        db_session.add(FeatureValue(value=1.0, **common))
        db_session.flush()
        db_session.add(FeatureValue(value=2.0, **common))
        with pytest.raises(IntegrityError):
            db_session.flush()

    def test_unknown_instrument_rejected(self, db_session):
        fd = _feature_definition()
        db_session.add(fd)
        db_session.flush()

        fv = FeatureValue(
            instrument_id=uuid.uuid4(),
            feature_definition_id=fd.id,
            computed_for_date=date(2024, 1, 2),
            timeframe="1D",
            value=1.0,
            available_at=date(2024, 1, 2),
        )
        db_session.add(fv)
        with pytest.raises(IntegrityError):
            db_session.flush()

    def test_unknown_feature_definition_rejected(self, db_session, instrument):
        fv = FeatureValue(
            instrument_id=instrument.id,
            feature_definition_id=uuid.uuid4(),
            computed_for_date=date(2024, 1, 2),
            timeframe="1D",
            value=1.0,
            available_at=date(2024, 1, 2),
        )
        db_session.add(fv)
        with pytest.raises(IntegrityError):
            db_session.flush()


class TestLabelDefinitionImmutability:
    def test_status_update_is_allowed(self, db_session):
        ld = _label_definition()
        db_session.add(ld)
        db_session.flush()

        ld.status = DefinitionStatus.RETIRED
        db_session.flush()

    def test_content_update_is_rejected(self, db_session):
        ld = _label_definition()
        db_session.add(ld)
        db_session.flush()

        ld.horizon_days = 999
        with pytest.raises(ProgrammingError, match="immutable"):
            db_session.flush()

    def test_delete_is_rejected(self, db_session):
        ld = _label_definition()
        db_session.add(ld)
        db_session.flush()

        db_session.delete(ld)
        with pytest.raises(ProgrammingError, match="cannot be deleted"):
            db_session.flush()

    def test_duplicate_name_version_rejected(self, db_session):
        db_session.add(_label_definition())
        db_session.flush()
        db_session.add(_label_definition())
        with pytest.raises(IntegrityError):
            db_session.flush()


class TestLabelValue:
    def test_creates_successfully(self, db_session, instrument):
        ld = _label_definition()
        db_session.add(ld)
        db_session.flush()

        lv = LabelValue(
            instrument_id=instrument.id,
            label_definition_id=ld.id,
            entry_date=date(2024, 1, 2),
            horizon_end_date=date(2024, 1, 9),
            timeframe="1D",
            realized_return=0.023,
            holding_time_days=5,
        )
        db_session.add(lv)
        db_session.flush()
        assert lv.id is not None

    def test_horizon_before_entry_date_is_rejected(self, db_session, instrument):
        ld = _label_definition()
        db_session.add(ld)
        db_session.flush()

        lv = LabelValue(
            instrument_id=instrument.id,
            label_definition_id=ld.id,
            entry_date=date(2024, 1, 10),
            horizon_end_date=date(2024, 1, 5),
            timeframe="1D",
        )
        db_session.add(lv)
        with pytest.raises(IntegrityError, match="ck_label_value_horizon_order"):
            db_session.flush()

    def test_horizon_equal_to_entry_date_is_allowed(self, db_session, instrument):
        ld = _label_definition(horizon_days=0)
        db_session.add(ld)
        db_session.flush()

        lv = LabelValue(
            instrument_id=instrument.id,
            label_definition_id=ld.id,
            entry_date=date(2024, 1, 10),
            horizon_end_date=date(2024, 1, 10),
            timeframe="1D",
        )
        db_session.add(lv)
        db_session.flush()

    def test_duplicate_identity_rejected(self, db_session, instrument):
        ld = _label_definition()
        db_session.add(ld)
        db_session.flush()

        common = dict(
            instrument_id=instrument.id,
            label_definition_id=ld.id,
            entry_date=date(2024, 1, 2),
            horizon_end_date=date(2024, 1, 9),
            timeframe="1D",
        )
        db_session.add(LabelValue(realized_return=0.01, **common))
        db_session.flush()
        db_session.add(LabelValue(realized_return=0.02, **common))
        with pytest.raises(IntegrityError):
            db_session.flush()


class TestExtensibilityToFutureAndOption:

    def test_feature_value_accepts_a_future_instrument(self, db_session):
        future_instrument = Instrument(exchange="NSE", asset_type=AssetType.FUTURE)
        db_session.add(future_instrument)
        db_session.flush()

        fd = _feature_definition(name="future_test_feature")
        db_session.add(fd)
        db_session.flush()

        fv = FeatureValue(
            instrument_id=future_instrument.id,
            feature_definition_id=fd.id,
            computed_for_date=date(2024, 1, 2),
            timeframe="1D",
            value=1.0,
            available_at=date(2024, 1, 2),
        )
        db_session.add(fv)
        db_session.flush()

    def test_label_value_accepts_an_option_instrument(self, db_session):
        option_instrument = Instrument(exchange="NSE", asset_type=AssetType.OPTION)
        db_session.add(option_instrument)
        db_session.flush()

        ld = _label_definition(name="option_test_label")
        db_session.add(ld)
        db_session.flush()

        lv = LabelValue(
            instrument_id=option_instrument.id,
            label_definition_id=ld.id,
            entry_date=date(2024, 1, 2),
            horizon_end_date=date(2024, 1, 9),
            timeframe="1D",
        )
        db_session.add(lv)
        db_session.flush()
