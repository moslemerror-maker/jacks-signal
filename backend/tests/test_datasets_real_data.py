from __future__ import annotations

import datetime

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from backend.app.config import database_url
from backend.app.db.models import (
    AssetType,
    FeatureValue,
    Instrument,
    LabelValue,
    MarketBar,
    SourceFormat,
)
from backend.app.research.datasets import build_dataset, compute_dataset_checksum
from backend.app.research.features.definitions import register_feature_definitions
from backend.app.research.labels.forward_return import register_label_definition


def _fetch_real_reliance_data():
    try:
        engine = create_engine(database_url())
        with engine.connect() as conn:
            instrument_id = conn.execute(
                text(
                    """
                    select i.id from instruments i
                    join instrument_symbols s
                        on s.instrument_id = i.id and s.effective_to is null
                    where s.symbol = 'RELIANCE' and i.asset_type = 'EQUITY'
                    """
                )
            ).scalar()
            if instrument_id is None:
                return None

            bars = conn.execute(
                text(
                    """
                    select trade_date, close, high, low from market_bars
                    where instrument_id = :iid and series = 'EQ' and timeframe = '1D'
                      and trade_date >= '2026-01-01'
                    order by trade_date asc
                    """
                ),
                {"iid": instrument_id},
            ).fetchall()
            if len(bars) < 60:
                return None

            feature_rows = conn.execute(
                text(
                    """
                    select fd.name, fd.version, fv.computed_for_date, fv.value,
                           fv.available_at
                    from feature_values fv
                    join feature_definitions fd on fd.id = fv.feature_definition_id
                    where fv.instrument_id = :iid and fv.computed_for_date >= '2026-01-01'
                    """
                ),
                {"iid": instrument_id},
            ).fetchall()
            label_rows = conn.execute(
                text(
                    """
                    select ld.name, ld.version, lv.entry_date, lv.horizon_end_date,
                           lv.realized_return, lv.hit_target, lv.hit_stop, lv.mfe,
                           lv.mae, lv.holding_time_days
                    from label_values lv
                    join label_definitions ld on ld.id = lv.label_definition_id
                    where lv.instrument_id = :iid and lv.entry_date >= '2026-01-01'
                      and ld.name = 'forward_return_5d'
                    """
                ),
                {"iid": instrument_id},
            ).fetchall()
            if not feature_rows or not label_rows:
                return None
            return {"bars": bars, "features": feature_rows, "labels": label_rows}
    except OperationalError:
        return None


@pytest.fixture(scope="module")
def real_data():
    data = _fetch_real_reliance_data()
    if data is None:
        pytest.skip(
            "dev DB unreachable or missing real Phase 1 / Stage 3 data for "
            "RELIANCE - this integration test requires a completed "
            "scripts/backfill_phase1.py + scripts/compute_stage3_research_data.py run"
        )
    return data


@pytest.fixture
def seeded_real_instrument(db_session, real_data):
    instrument = Instrument(exchange="NSE", asset_type=AssetType.EQUITY)
    db_session.add(instrument)
    db_session.flush()

    for bar in real_data["bars"]:
        db_session.add(
            MarketBar(
                instrument_id=instrument.id,
                trade_date=bar.trade_date,
                series="EQ",
                timeframe="1D",
                open=bar.close,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                source_format=SourceFormat.NSE_UDIFF,
                source_file="real-dev-db-copy-for-stage4-integration-test",
            )
        )
    db_session.flush()

    feature_definitions = register_feature_definitions(db_session)
    label_definition = register_label_definition(db_session)
    db_session.flush()

    definition_by_name_version = {
        (fd.name, fd.version): fd for fd in feature_definitions.values()
    }
    for row in real_data["features"]:
        fd = definition_by_name_version[(row.name, row.version)]
        db_session.add(
            FeatureValue(
                instrument_id=instrument.id,
                feature_definition_id=fd.id,
                computed_for_date=row.computed_for_date,
                timeframe="1D",
                value=row.value,
                available_at=row.available_at,
            )
        )

    for row in real_data["labels"]:
        assert (row.name, row.version) == (label_definition.name, label_definition.version)
        db_session.add(
            LabelValue(
                instrument_id=instrument.id,
                label_definition_id=label_definition.id,
                entry_date=row.entry_date,
                horizon_end_date=row.horizon_end_date,
                timeframe="1D",
                realized_return=row.realized_return,
                hit_target=row.hit_target,
                hit_stop=row.hit_stop,
                mfe=row.mfe,
                mae=row.mae,
                holding_time_days=row.holding_time_days,
            )
        )
    db_session.flush()

    return instrument, feature_definitions, label_definition


class TestRealDataIntegration:
    def test_dataset_built_from_real_data_has_expected_shape(
        self, db_session, real_data, seeded_real_instrument
    ):
        instrument, feature_definitions, label_definition = seeded_real_instrument
        bars = real_data["bars"]
        start_date = bars[-30].trade_date
        end_date = bars[-1].trade_date
        known_as_of = end_date

        dataset = build_dataset(
            db_session,
            name="reliance-real-data-integration",
            instrument_ids=[instrument.id],
            feature_definition_ids=[fd.id for fd in feature_definitions.values()],
            label_definition_id=label_definition.id,
            start_date=start_date,
            end_date=end_date,
            known_as_of=known_as_of,
            created_by="test-suite",
        )

        expected_row_count = sum(1 for b in bars if start_date <= b.trade_date <= end_date)
        assert dataset.row_count == expected_row_count
        assert len(dataset.rows) == expected_row_count

        real_feature_dates_per_name = {}
        for row in real_data["features"]:
            real_feature_dates_per_name.setdefault(row.name, set()).add(row.computed_for_date)
        for dataset_row in dataset.rows:
            for name, fd in feature_definitions.items():
                fd_id_str = str(fd.id)
                has_real_value = dataset_row.entry_date in real_feature_dates_per_name.get(
                    name, set()
                )
                stored_value = dataset_row.features[fd_id_str]
                if has_real_value:
                    assert stored_value is not None
                else:
                    assert stored_value is None

        real_labels_by_entry_date = {row.entry_date: row for row in real_data["labels"]}
        for dataset_row in dataset.rows:
            real_label = real_labels_by_entry_date.get(dataset_row.entry_date)
            should_be_resolved = (
                real_label is not None and real_label.horizon_end_date <= known_as_of
            )
            assert dataset_row.label_resolved == should_be_resolved
            if should_be_resolved:
                assert dataset_row.realized_return == pytest.approx(real_label.realized_return)

        assert dataset.rows_with_unresolved_label > 0
        assert dataset.rows_with_any_missing_feature == 0

    def test_rebuilding_from_the_same_real_data_reproduces_the_checksum(
        self, db_session, real_data, seeded_real_instrument
    ):
        instrument, feature_definitions, label_definition = seeded_real_instrument
        bars = real_data["bars"]
        start_date = bars[-30].trade_date
        end_date = bars[-1].trade_date
        feature_ids = [fd.id for fd in feature_definitions.values()]

        first = build_dataset(
            db_session,
            name="reliance-real-data-repro-1",
            instrument_ids=[instrument.id],
            feature_definition_ids=feature_ids,
            label_definition_id=label_definition.id,
            start_date=start_date,
            end_date=end_date,
            known_as_of=end_date,
            created_by="test-suite",
        )
        second = build_dataset(
            db_session,
            name="reliance-real-data-repro-2",
            instrument_ids=[instrument.id],
            feature_definition_ids=feature_ids,
            label_definition_id=label_definition.id,
            start_date=start_date,
            end_date=end_date,
            known_as_of=end_date,
            created_by="test-suite",
        )

        assert first.id != second.id
        assert first.checksum == second.checksum
        assert first.row_count == second.row_count
