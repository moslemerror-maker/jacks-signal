from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

SUPPORTED_UNDERLYINGS = ("NIFTY",)

_RAW_HASH_DESCRIPTION = (
    "Project-level digest of this project's own re-serialization of the parsed provider response - NOT a "
    "cryptographic digest of the provider's original wire-level HTTP response bytes (see docs/52's already-"
    "documented raw_response_sha256 limitation)."
)


def check_db_reachable(db: Session) -> bool:
    try:
        db.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False


def _leg_snapshot_by_contract(decision_snapshot_legs: list[dict] | None) -> dict[tuple, dict]:
    if not decision_snapshot_legs:
        return {}
    out = {}
    for leg in decision_snapshot_legs:
        c = leg["contract"]
        key = (c["underlying"], c["derivative_class"], c["expiry"], c["strike_paise"], c["option_type"])
        out[key] = leg
    return out


def _merge_leg(structure_leg: dict, snapshot_by_contract: dict[tuple, dict]) -> dict[str, Any]:
    c = structure_leg["contract"]
    key = (c["underlying"], c["derivative_class"], c["expiry"], c["strike_paise"], c["option_type"])
    snap = snapshot_by_contract.get(key)
    merged: dict[str, Any] = {
        "contract": c,
        "side": structure_leg["side"],
        "entry_price": structure_leg["entry_price"],
        "entry_price_source": structure_leg["entry_price_source"],
        "quote_timestamp": None, "quote_status": None, "last_price": None, "bid": None, "ask": None,
        "volume": None, "open_interest": None, "provider_reported_iv": None, "provider_reported_greeks": None,
        "provenance_class": None, "liquidity_assessment": None,
    }
    if snap is not None:
        merged.update({
            "quote_timestamp": snap["quote_observed_at"], "quote_status": snap["quote_status"],
            "last_price": snap["last_price"], "bid": snap["bid"], "ask": snap["ask"], "volume": snap["volume"],
            "open_interest": snap["open_interest"], "provider_reported_iv": snap["provider_reported_iv"],
            "provider_reported_greeks": snap["provider_reported_greeks"], "provenance_class": snap["provenance_class"],
            "liquidity_assessment": snap["liquidity_assessment"],
        })
    return merged


def _candidate_snapshot(entry: dict) -> dict[str, Any]:
    candidate = entry.get("candidate")
    if candidate is None:
        return {"shape": entry["shape"], "formed": False, "rejected_reason": entry["rejected_reason"], "data_quality": None, "legs": []}
    snapshot_by_contract = _leg_snapshot_by_contract(entry.get("decision_snapshot_legs"))
    legs = [_merge_leg(leg, snapshot_by_contract) for leg in candidate["structure"]["legs"]]
    return {"shape": entry["shape"], "formed": True, "rejected_reason": None, "data_quality": candidate.get("data_quality"), "legs": legs}


def _market_snapshot(payload: dict) -> dict[str, Any]:
    cycle = payload.get("cycle") or {}
    notes = payload.get("validation_notes") or {}
    forward_value = underlying_last_price = None
    for entry in payload.get("per_shape_results") or ():
        candidate = entry.get("candidate")
        fr = (candidate or {}).get("forward_resolution")
        if fr is not None:
            forward_value, underlying_last_price = fr["forward"], fr["underlying_last_price"]
            break
    return {
        "cycle_observed_at": cycle.get("observed_at"),
        "selected_expiry": notes.get("expiry_used"),
        "forward_mode": notes.get("forward_resolution_source"),
        "forward_value": forward_value,
        "underlying_last_price": underlying_last_price,
    }


def _probability(payload: dict, selected: dict) -> dict[str, Any]:
    return {
        "risk_neutral_probability_metrics": selected.get("probability_metrics"),
        "predictive_probability_status": payload.get("predictive_probability_status"),
        "note": payload.get("predictive_probability_status_note"),
    }


def _cost_risk_status(payload: dict) -> dict[str, Any]:
    risk_ev = payload.get("risk_and_expected_value_status") or {}
    return {
        "costs_slippage_assumptions": payload.get("costs_slippage_assumptions"),
        "risk": risk_ev.get("risk"),
        "expected_value": risk_ev.get("expected_value"),
    }


def shape_recommendation_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    payload = row["payload"] or {}
    selected = payload.get("selected_recommendation") or {}
    cycle = payload.get("cycle") or {}
    return {
        "metadata": {
            "recommendation_id": str(row["id"]), "decision_timestamp": row["decision_timestamp"].isoformat(),
            "underlying": row["underlying"], "status": row["status"], "shape": row["shape"],
            "data_quality": row["data_quality"], "model_version": row["model_version"],
            "strategy_version": row["strategy_version"], "source": row["source"],
        },
        "market_snapshot": _market_snapshot(payload),
        "selected_recommendation": {
            "status": selected.get("status", row["status"]), "selected_shape": payload.get("selected_shape"),
            "reasons": list(selected.get("reasons") or ()), "decision_thresholds": payload.get("decision_thresholds"),
        },
        "candidates": [_candidate_snapshot(e) for e in (payload.get("per_shape_results") or ())],
        "probability": _probability(payload, selected),
        "cost_risk_status": _cost_risk_status(payload),
        "data_quality": row["data_quality"],
        "provenance": {"raw_response_sha256": cycle.get("raw_response_sha256"), "description": _RAW_HASH_DESCRIPTION},
    }


def latest_recommendation(db: Session, *, underlying: str) -> dict[str, Any] | None:
    row = db.execute(text(
        "SELECT id, decision_timestamp, underlying, status, shape, data_quality, model_version, strategy_version, "
        "source, payload FROM option_recommendations WHERE underlying = :u ORDER BY decision_timestamp DESC LIMIT 1"
    ), {"u": underlying}).mappings().first()
    if row is None:
        return None
    return shape_recommendation_snapshot(dict(row))


def _iso(value: dt.datetime | dt.date | None) -> str | None:
    return value.isoformat() if value is not None else None


def session_status(db: Session, *, underlying: str, now: dt.datetime | None = None) -> dict[str, Any]:
    from backend.app.options.session_schedule import ist_today
    from backend.app.options.session_scheduler import session_status_snapshot

    now = now if now is not None else dt.datetime.now(dt.timezone.utc)
    today = ist_today(now)
    snap = session_status_snapshot(db, underlying=underlying, session_date=today, now=now)
    return {
        "underlying": snap.underlying,
        "trading_date": snap.trading_date.isoformat(),
        "session_state": snap.session_state,
        "session_start_timestamp": _iso(snap.session_start_timestamp),
        "last_event_timestamp": _iso(snap.last_event_timestamp),
        "completion_timestamp": _iso(snap.completion_timestamp),
        "failure_timestamp": _iso(snap.failure_timestamp),
        "failure_detail": snap.failure_detail,
        "cycle_count": snap.cycle_count,
        "latest_recommendation_timestamp": _iso(snap.latest_recommendation_timestamp),
        "latest_recommendation_status": snap.latest_recommendation_status,
        "latest_recommendation_id": snap.latest_recommendation_id,
        "data_quality": snap.data_quality,
        "source": snap.source,
    }
