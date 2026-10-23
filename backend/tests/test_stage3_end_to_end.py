from __future__ import annotations

from datetime import date, timedelta

import pytest

from backend.app.db.models import AssetType, Instrument, MarketBar, SourceFormat
from backend.app.research.features.compute import compute_and_store_features
from backend.app.research.features.definitions import register_feature_definitions
from backend.app.research.labels.forward_return import (
    compute_and_store_labels,
    register_label_definition,
)
from backend.app.research.point_in_time import (
    assemble_point_in_time_features,
    assemble_training_row,
    resolve_label_outcome,
)

_TRADING_DATES = [date(2024, 1, 1) + timedelta(days=i) for i in range(30)]
_CLOSES = [100.0 + i for i in range(30)]


@pytest.fixture
def instrument(db_session) -> Instrument:
    inst = Instrument(exchange="NSE", asset_type=AssetType.EQUITY)
    db_session.add(inst)
    db_session.flush()
    return inst


@pytest.fixture
def seeded_bars(db_session, instrument) -> Instrument:
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
    return instrument


class TestEndToEndComputeAndAssemble:
    def test_features_and_label_are_computed_and_assembled_correctly(
        self, db_session, seeded_bars
    ):
        instrument = seeded_bars
        feature_definitions = register_feature_definitions(db_session)
        label_definition = register_label_definition(db_session)
        db_session.flush()

        written = compute_and_store_features(db_session, instrument.id, feature_definitions)
        label_written = compute_and_store_labels(db_session, instrument.id, label_definition)

        assert written["return_5d"] == 25
        assert written["close_vs_sma20"] == 11
        assert written["atr_14"] == 16
        assert label_written == 25

        entry_date = _TRADING_DATES[20]
        feature_ids = [
            feature_definitions["return_5d"].id,
            feature_definitions["close_vs_sma20"].id,
            feature_definitions["atr_14"].id,
        ]
        [row] = assemble_point_in_time_features(
            db_session, [instrument.id], feature_ids, entry_date
        )

        expected_return_5d = _CLOSES[20] / _CLOSES[15] - 1.0
        expected_close_vs_sma20 = _CLOSES[20] / (sum(_CLOSES[1:21]) / 20) - 1.0
        expected_atr_14 = 2.0

        assert row.values[feature_definitions["return_5d"].id] == pytest.approx(
            expected_return_5d
        )
        assert row.values[feature_definitions["close_vs_sma20"].id] == pytest.approx(
            expected_close_vs_sma20
        )
        assert row.values[feature_definitions["atr_14"].id] == pytest.approx(
            expected_atr_14
        )

        horizon_end_date = _TRADING_DATES[25]
        outcome = resolve_label_outcome(
            db_session,
            instrument.id,
            label_definition.id,
            entry_date,
            known_as_of=horizon_end_date,
        )
        assert outcome is not None
        expected_realized_return = _CLOSES[25] / _CLOSES[20] - 1.0
        assert outcome.realized_return == pytest.approx(expected_realized_return)
        assert outcome.hit_target is None
        assert outcome.hit_stop is None

    def test_label_outcome_not_yet_resolved_is_withheld_even_though_the_row_already_exists(
        self, db_session, seeded_bars
    ):
        instrument = seeded_bars
        label_definition = register_label_definition(db_session)
        db_session.flush()
        compute_and_store_labels(db_session, instrument.id, label_definition)

        entry_date = _TRADING_DATES[20]
        horizon_end_date = _TRADING_DATES[25]
        day_before_horizon = _TRADING_DATES[24]

        assert (
            resolve_label_outcome(
                db_session,
                instrument.id,
                label_definition.id,
                entry_date,
                known_as_of=day_before_horizon,
            )
            is None
        )
        assert (
            resolve_label_outcome(
                db_session,
                instrument.id,
                label_definition.id,
                entry_date,
                known_as_of=horizon_end_date,
            )
            is not None
        )

    def test_features_requiring_more_lookback_than_available_are_absent_not_none_valued(
        self, db_session, seeded_bars
    ):
        instrument = seeded_bars
        feature_definitions = register_feature_definitions(db_session)
        db_session.flush()
        compute_and_store_features(db_session, instrument.id, feature_definitions)

        entry_date = _TRADING_DATES[0]
        feature_ids = [fd.id for fd in feature_definitions.values()]
        [row] = assemble_point_in_time_features(
            db_session, [instrument.id], feature_ids, entry_date
        )
        assert all(value is None for value in row.values.values())

    def test_assemble_training_row_combines_real_features_and_label(
        self, db_session, seeded_bars
    ):
        instrument = seeded_bars
        feature_definitions = register_feature_definitions(db_session)
        label_definition = register_label_definition(db_session)
        db_session.flush()
        compute_and_store_features(db_session, instrument.id, feature_definitions)
        compute_and_store_labels(db_session, instrument.id, label_definition)

        entry_date = _TRADING_DATES[20]
        horizon_end_date = _TRADING_DATES[25]
        row = assemble_training_row(
            db_session,
            instrument.id,
            [fd.id for fd in feature_definitions.values()],
            label_definition.id,
            entry_date,
            known_as_of=horizon_end_date,
        )
        assert row.label_resolved is True
        assert row.label is not None
        assert all(v is not None for v in row.features.values())

    def test_idempotent_rerun_does_not_duplicate_or_error(self, db_session, seeded_bars):
        instrument = seeded_bars
        feature_definitions = register_feature_definitions(db_session)
        label_definition = register_label_definition(db_session)
        db_session.flush()

        first_written = compute_and_store_features(db_session, instrument.id, feature_definitions)
        first_label_written = compute_and_store_labels(db_session, instrument.id, label_definition)

        second_written = compute_and_store_features(db_session, instrument.id, feature_definitions)
        second_label_written = compute_and_store_labels(db_session, instrument.id, label_definition)

        assert sum(first_written.values()) > 0
        assert first_label_written > 0
        assert sum(second_written.values()) == 0
        assert second_label_written == 0
