from __future__ import annotations

import datetime
import hashlib
import json
import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.db.models import (
    Dataset,
    DatasetFeatureDefinition,
    DatasetInstrument,
    DatasetRow,
    FeatureValue,
    LabelDefinition,
    LabelValue,
)
from backend.app.research.features.compute import load_eq_bars
from backend.app.research.point_in_time import LabelOutcome, assemble_training_row

UNIVERSE_CAVEAT_TEXT = (
    "This dataset's instrument universe is the Phase 1 25-stock equity "
    "basket (backend/app/data/instruments/universe.py:EQUITY_BASKET) - a "
    "fixed, current-day-selected liquid basket used consistently across "
    "the whole backfill. It is NOT historical point-in-time NIFTY-50 "
    "constituent membership; a stock's presence here does not mean it "
    "was actually a NIFTY-50 member throughout [start_date, end_date]. "
    "See docs/16 (open questions) for the unresolved point-in-time "
    "constituent history gap this caveat exists because of."
)


def _canonical_row_dict(
    instrument_id: uuid.UUID,
    entry_date: datetime.date,
    features: dict[uuid.UUID, float | None],
    label: LabelOutcome | None,
) -> dict:
    return {
        "instrument_id": str(instrument_id),
        "entry_date": entry_date.isoformat(),
        "features": {
            str(k): v for k, v in sorted(features.items(), key=lambda kv: str(kv[0]))
        },
        "label": (
            None
            if label is None
            else {
                "realized_return": label.realized_return,
                "hit_target": label.hit_target,
                "hit_stop": label.hit_stop,
                "mfe": label.mfe,
                "mae": label.mae,
                "holding_time_days": label.holding_time_days,
            }
        ),
    }


def compute_dataset_checksum(canonical_rows: list[dict]) -> str:
    serialized = json.dumps(canonical_rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _source_data_watermark(
    session: Session,
    instrument_ids: list[uuid.UUID],
    feature_definition_ids: list[uuid.UUID],
    label_definition_id: uuid.UUID,
    start_date: datetime.date,
    end_date: datetime.date,
) -> datetime.datetime:
    feature_max = session.scalar(
        select(func.max(FeatureValue.computed_at)).where(
            FeatureValue.instrument_id.in_(instrument_ids),
            FeatureValue.feature_definition_id.in_(feature_definition_ids),
            FeatureValue.computed_for_date.between(start_date, end_date),
        )
    )
    label_max = session.scalar(
        select(func.max(LabelValue.computed_at)).where(
            LabelValue.instrument_id.in_(instrument_ids),
            LabelValue.label_definition_id == label_definition_id,
            LabelValue.entry_date.between(start_date, end_date),
        )
    )
    candidates = [ts for ts in (feature_max, label_max) if ts is not None]
    if not candidates:
        return datetime.datetime.now(datetime.timezone.utc)
    return max(candidates)


def build_dataset(
    session: Session,
    name: str,
    instrument_ids: list[uuid.UUID],
    feature_definition_ids: list[uuid.UUID],
    label_definition_id: uuid.UUID,
    start_date: datetime.date,
    end_date: datetime.date,
    known_as_of: datetime.date,
    created_by: str,
    timeframe: str = "1D",
) -> Dataset:
    if not instrument_ids:
        raise ValueError("instrument_ids must not be empty")
    if not feature_definition_ids:
        raise ValueError("feature_definition_ids must not be empty")
    if end_date < start_date:
        raise ValueError(f"end_date {end_date} is before start_date {start_date}")

    label_definition = session.get(LabelDefinition, label_definition_id)
    if label_definition is None:
        raise ValueError(f"unknown label_definition_id {label_definition_id}")

    sorted_instrument_ids = sorted(set(instrument_ids), key=str)

    pairs: list[tuple[uuid.UUID, datetime.date]] = []
    for instrument_id in sorted_instrument_ids:
        bars = load_eq_bars(session, instrument_id, timeframe=timeframe)
        for bar in bars:
            if start_date <= bar.trade_date <= end_date:
                pairs.append((instrument_id, bar.trade_date))
    pairs.sort(key=lambda p: (str(p[0]), p[1]))

    canonical_rows: list[dict] = []
    row_records: list[dict] = []
    rows_with_any_missing_feature = 0
    rows_with_unresolved_label = 0

    for row_index, (instrument_id, entry_date) in enumerate(pairs):
        training_row = assemble_training_row(
            session,
            instrument_id,
            feature_definition_ids,
            label_definition_id,
            entry_date,
            known_as_of,
            timeframe,
        )
        if any(v is None for v in training_row.features.values()):
            rows_with_any_missing_feature += 1
        if not training_row.label_resolved:
            rows_with_unresolved_label += 1

        canonical_rows.append(
            _canonical_row_dict(
                instrument_id, entry_date, training_row.features, training_row.label
            )
        )
        label = training_row.label
        row_records.append(
            dict(
                row_index=row_index,
                instrument_id=instrument_id,
                entry_date=entry_date,
                features={str(k): v for k, v in training_row.features.items()},
                label_resolved=training_row.label_resolved,
                realized_return=None if label is None else label.realized_return,
                hit_target=None if label is None else label.hit_target,
                hit_stop=None if label is None else label.hit_stop,
                mfe=None if label is None else label.mfe,
                mae=None if label is None else label.mae,
                holding_time_days=None if label is None else label.holding_time_days,
            )
        )

    checksum = compute_dataset_checksum(canonical_rows)
    watermark = _source_data_watermark(
        session,
        sorted_instrument_ids,
        feature_definition_ids,
        label_definition_id,
        start_date,
        end_date,
    )

    dataset = Dataset(
        name=name,
        label_definition_id=label_definition_id,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        known_as_of=known_as_of,
        universe_caveat=UNIVERSE_CAVEAT_TEXT,
        source_data_watermark=watermark,
        row_count=len(row_records),
        rows_with_any_missing_feature=rows_with_any_missing_feature,
        rows_with_unresolved_label=rows_with_unresolved_label,
        checksum=checksum,
        created_by=created_by,
    )
    session.add(dataset)
    session.flush()

    for instrument_id in sorted_instrument_ids:
        session.add(DatasetInstrument(dataset_id=dataset.id, instrument_id=instrument_id))
    for feature_definition_id in feature_definition_ids:
        session.add(
            DatasetFeatureDefinition(
                dataset_id=dataset.id, feature_definition_id=feature_definition_id
            )
        )
    for record in row_records:
        session.add(DatasetRow(dataset_id=dataset.id, **record))

    session.flush()
    return dataset
