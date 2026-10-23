from __future__ import annotations

import dataclasses
import datetime
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import FeatureValue, LabelValue


@dataclasses.dataclass(frozen=True)
class PointInTimeFeatureRow:

    instrument_id: uuid.UUID
    entry_date: datetime.date
    values: dict[uuid.UUID, float | None]


@dataclasses.dataclass(frozen=True)
class LabelOutcome:

    realized_return: float | None
    hit_target: bool | None
    hit_stop: bool | None
    mfe: float | None
    mae: float | None
    holding_time_days: int | None


@dataclasses.dataclass(frozen=True)
class TrainingRow:

    instrument_id: uuid.UUID
    entry_date: datetime.date
    features: dict[uuid.UUID, float | None]
    label: LabelOutcome | None
    label_resolved: bool


def assemble_point_in_time_features(
    session: Session,
    instrument_ids: list[uuid.UUID],
    feature_definition_ids: list[uuid.UUID],
    entry_date: datetime.date,
    timeframe: str = "1D",
) -> list[PointInTimeFeatureRow]:
    rows: list[PointInTimeFeatureRow] = []
    for instrument_id in instrument_ids:
        values: dict[uuid.UUID, float | None] = {}
        for feature_definition_id in feature_definition_ids:
            fv = session.scalar(
                select(FeatureValue).where(
                    FeatureValue.instrument_id == instrument_id,
                    FeatureValue.feature_definition_id == feature_definition_id,
                    FeatureValue.computed_for_date == entry_date,
                    FeatureValue.timeframe == timeframe,
                    FeatureValue.available_at <= entry_date,
                )
            )
            values[feature_definition_id] = fv.value if fv is not None else None
        rows.append(
            PointInTimeFeatureRow(
                instrument_id=instrument_id, entry_date=entry_date, values=values
            )
        )
    return rows


def resolve_label_outcome(
    session: Session,
    instrument_id: uuid.UUID,
    label_definition_id: uuid.UUID,
    entry_date: datetime.date,
    known_as_of: datetime.date,
    timeframe: str = "1D",
) -> LabelOutcome | None:
    lv = session.scalar(
        select(LabelValue).where(
            LabelValue.instrument_id == instrument_id,
            LabelValue.label_definition_id == label_definition_id,
            LabelValue.entry_date == entry_date,
            LabelValue.timeframe == timeframe,
        )
    )
    if lv is None:
        return None
    if lv.horizon_end_date > known_as_of:
        return None
    return LabelOutcome(
        realized_return=lv.realized_return,
        hit_target=lv.hit_target,
        hit_stop=lv.hit_stop,
        mfe=lv.mfe,
        mae=lv.mae,
        holding_time_days=lv.holding_time_days,
    )


def assemble_training_row(
    session: Session,
    instrument_id: uuid.UUID,
    feature_definition_ids: list[uuid.UUID],
    label_definition_id: uuid.UUID,
    entry_date: datetime.date,
    known_as_of: datetime.date,
    timeframe: str = "1D",
) -> TrainingRow:
    (feature_row,) = assemble_point_in_time_features(
        session, [instrument_id], feature_definition_ids, entry_date, timeframe
    )
    label = resolve_label_outcome(
        session, instrument_id, label_definition_id, entry_date, known_as_of, timeframe
    )
    return TrainingRow(
        instrument_id=instrument_id,
        entry_date=entry_date,
        features=feature_row.values,
        label=label,
        label_resolved=label is not None,
    )
