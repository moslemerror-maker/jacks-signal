from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy.exc import ProgrammingError

from backend.app.db.models import AssetType, Instrument, MarketBar, SourceFormat
from backend.app.research.datasets import build_dataset, compute_dataset_checksum
from backend.app.research.features.compute import compute_and_store_features
from backend.app.research.features.definitions import register_feature_definitions
from backend.app.research.labels.forward_return import (
    compute_and_store_labels,
    register_label_definition,
)

_TRADING_DATES = [date(2024, 1, 1) + timedelta(days=i) for i in range(40)]
_CLOSES = [100.0 + i for i in range(40)]


@pytest.fixture
def two_instruments(db_session) -> list[Instrument]:
    instruments = [
        Instrument(exchange="NSE", asset_type=AssetType.EQUITY) for _ in range(2)
    ]
    db_session.add_all(instruments)
    db_session.flush()
    return instruments


@pytest.fixture
def seeded_universe(db_session, two_instruments):
    feature_definitions = register_feature_definitions(db_session)
    label_definition = register_label_definition(db_session)
    db_session.flush()

    for instrument in two_instruments:
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

    return two_instruments, feature_definitions, label_definition


class TestChecksumReproducibility:
    def test_identical_canonical_rows_produce_identical_checksum(self):
        rows = [
            {"instrument_id": "a", "entry_date": "2024-01-01", "features": {"f": 1.0}, "label": None}
        ]
        assert compute_dataset_checksum(rows) == compute_dataset_checksum(list(rows))

    def test_building_twice_from_unchanged_inputs_matches_checksum_and_row_count(
        self, db_session, seeded_universe
    ):
        instruments, feature_definitions, label_definition = seeded_universe
        feature_ids = [fd.id for fd in feature_definitions.values()]
        kwargs = dict(
            instrument_ids=[i.id for i in instruments],
            feature_definition_ids=feature_ids,
            label_definition_id=label_definition.id,
            start_date=_TRADING_DATES[20],
            end_date=_TRADING_DATES[30],
            known_as_of=_TRADING_DATES[39],
            created_by="test-suite",
        )
        first = build_dataset(db_session, name="ds-1", **kwargs)
        second = build_dataset(db_session, name="ds-2", **kwargs)

        assert first.checksum == second.checksum
        assert first.row_count == second.row_count

    def test_different_row_content_produces_a_different_checksum(
        self, db_session, seeded_universe
    ):
        instruments, feature_definitions, label_definition = seeded_universe
        feature_ids = [fd.id for fd in feature_definitions.values()]
        narrow = build_dataset(
            db_session,
            name="narrow",
            instrument_ids=[instruments[0].id],
            feature_definition_ids=feature_ids,
            label_definition_id=label_definition.id,
            start_date=_TRADING_DATES[20],
            end_date=_TRADING_DATES[25],
            known_as_of=_TRADING_DATES[39],
            created_by="test-suite",
        )
        wide = build_dataset(
            db_session,
            name="wide",
            instrument_ids=[instruments[0].id],
            feature_definition_ids=feature_ids,
            label_definition_id=label_definition.id,
            start_date=_TRADING_DATES[20],
            end_date=_TRADING_DATES[30],
            known_as_of=_TRADING_DATES[39],
            created_by="test-suite",
        )
        assert narrow.checksum != wide.checksum


class TestDatasetIdentity:
    def test_two_builds_of_the_same_binding_are_distinct_dataset_rows(
        self, db_session, seeded_universe
    ):
        instruments, feature_definitions, label_definition = seeded_universe
        feature_ids = [fd.id for fd in feature_definitions.values()]
        kwargs = dict(
            instrument_ids=[instruments[0].id],
            feature_definition_ids=feature_ids,
            label_definition_id=label_definition.id,
            start_date=_TRADING_DATES[20],
            end_date=_TRADING_DATES[25],
            known_as_of=_TRADING_DATES[39],
            created_by="test-suite",
        )
        first = build_dataset(db_session, name="same-binding", **kwargs)
        second = build_dataset(db_session, name="same-binding", **kwargs)
        assert first.id != second.id
        assert first.created_at != second.created_at or first.id != second.id


class TestFeatureLabelVersionBinding:
    def test_dataset_records_exactly_the_requested_feature_definition_ids(
        self, db_session, seeded_universe
    ):
        instruments, feature_definitions, label_definition = seeded_universe
        chosen = [feature_definitions["return_5d"].id, feature_definitions["atr_14"].id]
        dataset = build_dataset(
            db_session,
            name="subset-features",
            instrument_ids=[instruments[0].id],
            feature_definition_ids=chosen,
            label_definition_id=label_definition.id,
            start_date=_TRADING_DATES[20],
            end_date=_TRADING_DATES[25],
            known_as_of=_TRADING_DATES[39],
            created_by="test-suite",
        )
        bound_ids = {dfd.feature_definition_id for dfd in dataset.feature_definitions}
        assert bound_ids == set(chosen)
        for row in dataset.rows:
            assert set(row.features.keys()) == {str(fid) for fid in chosen}

    def test_dataset_records_the_exact_label_definition_id(
        self, db_session, seeded_universe
    ):
        instruments, feature_definitions, label_definition = seeded_universe
        dataset = build_dataset(
            db_session,
            name="label-binding",
            instrument_ids=[instruments[0].id],
            feature_definition_ids=[feature_definitions["return_5d"].id],
            label_definition_id=label_definition.id,
            start_date=_TRADING_DATES[20],
            end_date=_TRADING_DATES[25],
            known_as_of=_TRADING_DATES[39],
            created_by="test-suite",
        )
        assert dataset.label_definition_id == label_definition.id

    def test_a_different_feature_definition_version_produces_a_different_dataset(
        self, db_session, seeded_universe
    ):
        instruments, feature_definitions, label_definition = seeded_universe
        from backend.app.db.models import FeatureDefinition, FeatureGroup

        v2 = FeatureDefinition(
            name="return_5d",
            version=2,
            feature_group=FeatureGroup.PRICE,
            description="v2, deliberately different for this test",
            formula_ref="backend.app.research.features.price:compute_return_n_day",
            input_dependencies="market_bars.close (series=EQ)",
            lookback_window=5,
            availability_lag=0,
            timeframe="1D",
            created_by="test-suite",
        )
        db_session.add(v2)
        db_session.flush()

        v1_dataset = build_dataset(
            db_session,
            name="v1",
            instrument_ids=[instruments[0].id],
            feature_definition_ids=[feature_definitions["return_5d"].id],
            label_definition_id=label_definition.id,
            start_date=_TRADING_DATES[20],
            end_date=_TRADING_DATES[25],
            known_as_of=_TRADING_DATES[39],
            created_by="test-suite",
        )
        v2_dataset = build_dataset(
            db_session,
            name="v2",
            instrument_ids=[instruments[0].id],
            feature_definition_ids=[v2.id],
            label_definition_id=label_definition.id,
            start_date=_TRADING_DATES[20],
            end_date=_TRADING_DATES[25],
            known_as_of=_TRADING_DATES[39],
            created_by="test-suite",
        )
        v1_ids = {dfd.feature_definition_id for dfd in v1_dataset.feature_definitions}
        v2_ids = {dfd.feature_definition_id for dfd in v2_dataset.feature_definitions}
        assert v1_ids != v2_ids
        assert all(
            row.features[str(v2.id)] is None for row in v2_dataset.rows
        )


class TestDateRangeBinding:
    def test_row_dates_are_exactly_within_the_requested_range(
        self, db_session, seeded_universe
    ):
        instruments, feature_definitions, label_definition = seeded_universe
        start_date, end_date = _TRADING_DATES[20], _TRADING_DATES[25]
        dataset = build_dataset(
            db_session,
            name="date-range",
            instrument_ids=[instruments[0].id],
            feature_definition_ids=[feature_definitions["return_5d"].id],
            label_definition_id=label_definition.id,
            start_date=start_date,
            end_date=end_date,
            known_as_of=_TRADING_DATES[39],
            created_by="test-suite",
        )
        assert dataset.start_date == start_date
        assert dataset.end_date == end_date
        assert all(start_date <= row.entry_date <= end_date for row in dataset.rows)
        assert dataset.row_count == 6

    def test_a_narrower_range_yields_fewer_rows(self, db_session, seeded_universe):
        instruments, feature_definitions, label_definition = seeded_universe
        wide = build_dataset(
            db_session,
            name="wide-range",
            instrument_ids=[instruments[0].id],
            feature_definition_ids=[feature_definitions["return_5d"].id],
            label_definition_id=label_definition.id,
            start_date=_TRADING_DATES[20],
            end_date=_TRADING_DATES[30],
            known_as_of=_TRADING_DATES[39],
            created_by="test-suite",
        )
        narrow = build_dataset(
            db_session,
            name="narrow-range",
            instrument_ids=[instruments[0].id],
            feature_definition_ids=[feature_definitions["return_5d"].id],
            label_definition_id=label_definition.id,
            start_date=_TRADING_DATES[20],
            end_date=_TRADING_DATES[22],
            known_as_of=_TRADING_DATES[39],
            created_by="test-suite",
        )
        assert narrow.row_count < wide.row_count


class TestUniverseBinding:
    def test_dataset_instruments_match_exactly_what_was_requested(
        self, db_session, seeded_universe
    ):
        instruments, feature_definitions, label_definition = seeded_universe
        dataset = build_dataset(
            db_session,
            name="single-instrument-universe",
            instrument_ids=[instruments[0].id],
            feature_definition_ids=[feature_definitions["return_5d"].id],
            label_definition_id=label_definition.id,
            start_date=_TRADING_DATES[20],
            end_date=_TRADING_DATES[25],
            known_as_of=_TRADING_DATES[39],
            created_by="test-suite",
        )
        bound = {di.instrument_id for di in dataset.instruments}
        assert bound == {instruments[0].id}
        assert all(row.instrument_id == instruments[0].id for row in dataset.rows)

    def test_two_instrument_universe_produces_rows_for_both(
        self, db_session, seeded_universe
    ):
        instruments, feature_definitions, label_definition = seeded_universe
        dataset = build_dataset(
            db_session,
            name="two-instrument-universe",
            instrument_ids=[i.id for i in instruments],
            feature_definition_ids=[feature_definitions["return_5d"].id],
            label_definition_id=label_definition.id,
            start_date=_TRADING_DATES[20],
            end_date=_TRADING_DATES[25],
            known_as_of=_TRADING_DATES[39],
            created_by="test-suite",
        )
        seen_instruments = {row.instrument_id for row in dataset.rows}
        assert seen_instruments == {i.id for i in instruments}
        assert dataset.row_count == 12

    def test_universe_caveat_is_recorded_on_every_dataset(
        self, db_session, seeded_universe
    ):
        instruments, feature_definitions, label_definition = seeded_universe
        dataset = build_dataset(
            db_session,
            name="caveat-check",
            instrument_ids=[instruments[0].id],
            feature_definition_ids=[feature_definitions["return_5d"].id],
            label_definition_id=label_definition.id,
            start_date=_TRADING_DATES[20],
            end_date=_TRADING_DATES[25],
            known_as_of=_TRADING_DATES[39],
            created_by="test-suite",
        )
        assert "NOT historical point-in-time NIFTY-50" in dataset.universe_caveat


class TestDeterministicRowOrdering:
    def test_row_index_is_sorted_by_instrument_then_entry_date(
        self, db_session, seeded_universe
    ):
        instruments, feature_definitions, label_definition = seeded_universe
        sorted_instrument_ids = sorted([i.id for i in instruments], key=str)
        dataset = build_dataset(
            db_session,
            name="ordering",
            instrument_ids=[i.id for i in instruments],
            feature_definition_ids=[feature_definitions["return_5d"].id],
            label_definition_id=label_definition.id,
            start_date=_TRADING_DATES[20],
            end_date=_TRADING_DATES[22],
            known_as_of=_TRADING_DATES[39],
            created_by="test-suite",
        )
        rows_by_index = sorted(dataset.rows, key=lambda r: r.row_index)
        expected_keys = [
            (instrument_id, entry_date)
            for instrument_id in sorted_instrument_ids
            for entry_date in (_TRADING_DATES[20], _TRADING_DATES[21], _TRADING_DATES[22])
        ]
        actual_keys = [(r.instrument_id, r.entry_date) for r in rows_by_index]
        assert actual_keys == expected_keys
        assert [r.row_index for r in rows_by_index] == list(range(len(rows_by_index)))

    def test_row_order_is_independent_of_caller_supplied_instrument_order(
        self, db_session, seeded_universe
    ):
        instruments, feature_definitions, label_definition = seeded_universe
        forward = build_dataset(
            db_session,
            name="forward-order",
            instrument_ids=[i.id for i in instruments],
            feature_definition_ids=[feature_definitions["return_5d"].id],
            label_definition_id=label_definition.id,
            start_date=_TRADING_DATES[20],
            end_date=_TRADING_DATES[22],
            known_as_of=_TRADING_DATES[39],
            created_by="test-suite",
        )
        reversed_order = build_dataset(
            db_session,
            name="reversed-order",
            instrument_ids=list(reversed([i.id for i in instruments])),
            feature_definition_ids=[feature_definitions["return_5d"].id],
            label_definition_id=label_definition.id,
            start_date=_TRADING_DATES[20],
            end_date=_TRADING_DATES[22],
            known_as_of=_TRADING_DATES[39],
            created_by="test-suite",
        )
        assert forward.checksum == reversed_order.checksum


class TestMissingValueBehavior:
    def test_features_with_insufficient_lookback_are_explicit_none_not_omitted(
        self, db_session, seeded_universe
    ):
        instruments, feature_definitions, label_definition = seeded_universe
        dataset = build_dataset(
            db_session,
            name="missing-features",
            instrument_ids=[instruments[0].id],
            feature_definition_ids=[fd.id for fd in feature_definitions.values()],
            label_definition_id=label_definition.id,
            start_date=_TRADING_DATES[0],
            end_date=_TRADING_DATES[0],
            known_as_of=_TRADING_DATES[39],
            created_by="test-suite",
        )
        assert dataset.row_count == 1
        [row] = dataset.rows
        assert set(row.features.keys()) == {str(fd.id) for fd in feature_definitions.values()}
        assert all(v is None for v in row.features.values())
        assert dataset.rows_with_any_missing_feature == 1

    def test_unresolved_labels_leave_all_outcome_columns_null_and_are_counted(
        self, db_session, seeded_universe
    ):
        instruments, feature_definitions, label_definition = seeded_universe
        dataset = build_dataset(
            db_session,
            name="unresolved-label",
            instrument_ids=[instruments[0].id],
            feature_definition_ids=[feature_definitions["return_5d"].id],
            label_definition_id=label_definition.id,
            start_date=_TRADING_DATES[39],
            end_date=_TRADING_DATES[39],
            known_as_of=_TRADING_DATES[39],
            created_by="test-suite",
        )
        [row] = dataset.rows
        assert row.label_resolved is False
        assert row.realized_return is None
        assert row.hit_target is None
        assert row.hit_stop is None
        assert row.mfe is None
        assert row.mae is None
        assert row.holding_time_days is None
        assert dataset.rows_with_unresolved_label == 1

    def test_known_as_of_before_horizon_leaves_an_otherwise_computed_label_unresolved(
        self, db_session, seeded_universe
    ):
        instruments, feature_definitions, label_definition = seeded_universe
        entry_date = _TRADING_DATES[20]
        horizon_end_date = _TRADING_DATES[25]
        dataset = build_dataset(
            db_session,
            name="known-as-of-too-early",
            instrument_ids=[instruments[0].id],
            feature_definition_ids=[feature_definitions["return_5d"].id],
            label_definition_id=label_definition.id,
            start_date=entry_date,
            end_date=entry_date,
            known_as_of=_TRADING_DATES[24],
            created_by="test-suite",
        )
        [row] = dataset.rows
        assert row.label_resolved is False
        assert row.realized_return is None


class TestProtectionAgainstAccidentalMutation:
    def test_updating_a_dataset_row_is_rejected(self, db_session, seeded_universe):
        instruments, feature_definitions, label_definition = seeded_universe
        dataset = build_dataset(
            db_session,
            name="immutability-check",
            instrument_ids=[instruments[0].id],
            feature_definition_ids=[feature_definitions["return_5d"].id],
            label_definition_id=label_definition.id,
            start_date=_TRADING_DATES[20],
            end_date=_TRADING_DATES[20],
            known_as_of=_TRADING_DATES[39],
            created_by="test-suite",
        )
        dataset.name = "renamed"
        with pytest.raises(ProgrammingError):
            db_session.flush()

    def test_deleting_a_dataset_row_is_rejected(self, db_session, seeded_universe):
        instruments, feature_definitions, label_definition = seeded_universe
        dataset = build_dataset(
            db_session,
            name="delete-check",
            instrument_ids=[instruments[0].id],
            feature_definition_ids=[feature_definitions["return_5d"].id],
            label_definition_id=label_definition.id,
            start_date=_TRADING_DATES[20],
            end_date=_TRADING_DATES[20],
            known_as_of=_TRADING_DATES[39],
            created_by="test-suite",
        )
        db_session.delete(dataset)
        with pytest.raises(ProgrammingError):
            db_session.flush()

    def test_updating_a_dataset_row_content_is_rejected(self, db_session, seeded_universe):
        instruments, feature_definitions, label_definition = seeded_universe
        dataset = build_dataset(
            db_session,
            name="row-immutability-check",
            instrument_ids=[instruments[0].id],
            feature_definition_ids=[feature_definitions["return_5d"].id],
            label_definition_id=label_definition.id,
            start_date=_TRADING_DATES[20],
            end_date=_TRADING_DATES[20],
            known_as_of=_TRADING_DATES[39],
            created_by="test-suite",
        )
        row = dataset.rows[0]
        row.label_resolved = not row.label_resolved
        with pytest.raises(ProgrammingError):
            db_session.flush()


class TestInputValidation:
    def test_empty_instrument_ids_rejected(self, db_session, seeded_universe):
        instruments, feature_definitions, label_definition = seeded_universe
        with pytest.raises(ValueError, match="instrument_ids"):
            build_dataset(
                db_session,
                name="empty-universe",
                instrument_ids=[],
                feature_definition_ids=[feature_definitions["return_5d"].id],
                label_definition_id=label_definition.id,
                start_date=_TRADING_DATES[20],
                end_date=_TRADING_DATES[20],
                known_as_of=_TRADING_DATES[39],
                created_by="test-suite",
            )

    def test_end_date_before_start_date_rejected(self, db_session, seeded_universe):
        instruments, feature_definitions, label_definition = seeded_universe
        with pytest.raises(ValueError, match="end_date"):
            build_dataset(
                db_session,
                name="inverted-range",
                instrument_ids=[instruments[0].id],
                feature_definition_ids=[feature_definitions["return_5d"].id],
                label_definition_id=label_definition.id,
                start_date=_TRADING_DATES[25],
                end_date=_TRADING_DATES[20],
                known_as_of=_TRADING_DATES[39],
                created_by="test-suite",
            )
