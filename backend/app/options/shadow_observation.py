from __future__ import annotations

import dataclasses
import datetime as dt
import uuid

from sqlalchemy import text
from sqlalchemy.orm import Session

SHADOW_OBSERVATION_STATUSES = ("AWAITING_OUTCOME", "OUTCOME_RECORDED")
QUOTE_STATUSES = ("QUOTE_AVAILABLE", "QUOTE_MISSING")

_NO_USABLE_DATA = "UNAVAILABLE_INSUFFICIENT_DATA"


class UnknownRecommendationError(ValueError):
    pass


class NotAForwardObservationError(ValueError):
    pass


def create_shadow_observation(
    db: Session, *, recommendation_id: uuid.UUID, research_phase_id: str, data_version: str, feature_version: str,
    cost_model_version: str, hypothesis_id: str | None = None,
) -> uuid.UUID:
    exists = db.execute(text("SELECT 1 FROM option_recommendations WHERE id = :id"), {"id": recommendation_id}).scalar()
    if exists is None:
        raise UnknownRecommendationError(f"no option_recommendations row with id={recommendation_id}")
    obs_id = uuid.uuid4()
    db.execute(text(
        "INSERT INTO option_shadow_observations (id, recommendation_id, research_phase_id, hypothesis_id, "
        "data_version, feature_version, cost_model_version) VALUES (:id, :rid, :rpid, :hid, :dv, :fv, :cv)"
    ), {"id": obs_id, "rid": recommendation_id, "rpid": research_phase_id, "hid": hypothesis_id,
        "dv": data_version, "fv": feature_version, "cv": cost_model_version})
    return obs_id


def shadow_observation_status(db: Session, shadow_observation_id: uuid.UUID) -> str:
    n = db.execute(text(
        "SELECT count(*) FROM option_shadow_forward_observations WHERE shadow_observation_id = :id"
    ), {"id": shadow_observation_id}).scalar_one()
    return "OUTCOME_RECORDED" if n > 0 else "AWAITING_OUTCOME"


def attach_forward_observation(db: Session, *, shadow_observation_id: uuid.UUID, forward_recommendation_id: uuid.UUID) -> uuid.UUID:
    row = db.execute(text(
        "SELECT r1.decision_timestamp AS origin_ts "
        "FROM option_shadow_observations so JOIN option_recommendations r1 ON r1.id = so.recommendation_id "
        "WHERE so.id = :soid"
    ), {"soid": shadow_observation_id}).mappings().first()
    if row is None:
        raise UnknownRecommendationError(f"no shadow observation with id={shadow_observation_id}")
    origin_ts = row["origin_ts"]

    fwd = db.execute(text(
        "SELECT decision_timestamp, data_quality FROM option_recommendations WHERE id = :frid"
    ), {"frid": forward_recommendation_id}).mappings().first()
    if fwd is None:
        raise UnknownRecommendationError(f"no option_recommendations row with id={forward_recommendation_id}")
    forward_ts = fwd["decision_timestamp"]

    if forward_ts <= origin_ts:
        raise NotAForwardObservationError(
            f"forward_recommendation_id's decision_timestamp ({forward_ts}) is not after the shadow "
            f"observation's own cycle ({origin_ts}) - no future data at entry, no backward attachment either"
        )

    elapsed_seconds = (forward_ts - origin_ts).total_seconds()
    quote_status = "QUOTE_MISSING" if fwd["data_quality"] == _NO_USABLE_DATA else "QUOTE_AVAILABLE"

    fwd_id = uuid.uuid4()
    db.execute(text(
        "INSERT INTO option_shadow_forward_observations (id, shadow_observation_id, forward_recommendation_id, "
        "observed_at, elapsed_seconds, quote_status) VALUES (:id, :soid, :frid, :obs, :el, :qs)"
    ), {"id": fwd_id, "soid": shadow_observation_id, "frid": forward_recommendation_id,
        "obs": forward_ts, "el": elapsed_seconds, "qs": quote_status})
    return fwd_id


@dataclasses.dataclass(frozen=True, slots=True)
class ForwardObservationRecord:
    id: uuid.UUID
    forward_recommendation_id: uuid.UUID
    observed_at: dt.datetime
    elapsed_seconds: float
    quote_status: str


def list_forward_observations(db: Session, shadow_observation_id: uuid.UUID) -> tuple[ForwardObservationRecord, ...]:
    rows = db.execute(text(
        "SELECT id, forward_recommendation_id, observed_at, elapsed_seconds, quote_status "
        "FROM option_shadow_forward_observations WHERE shadow_observation_id = :id ORDER BY observed_at"
    ), {"id": shadow_observation_id}).mappings().all()
    return tuple(ForwardObservationRecord(**r) for r in rows)
