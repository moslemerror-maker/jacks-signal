from __future__ import annotations

import dataclasses
import datetime as dt
import json
import uuid
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

from sqlalchemy import text
from sqlalchemy.orm import Session

from .contracts import ContractIdentity
from .live_quote import LiveQuote
from .mtm import LegValuation, mark_to_market_leg, valuation_from_live_quote
from .structures import OptionLeg, OptionStructure, structure_expiry_pnl


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


@dataclasses.dataclass(frozen=True, slots=True)
class OpenPaperPosition:
    recommendation_id: uuid.UUID
    decision_timestamp: dt.datetime
    entry_payload: Mapping[str, Any]


def open_paper_positions(db: Session, *, underlying: str) -> tuple[OpenPaperPosition, ...]:
    rows = db.execute(text(
        "SELECT r.id AS id, r.decision_timestamp AS decision_timestamp, e.payload AS payload "
        "FROM option_recommendations r "
        "JOIN option_paper_trade_events e ON e.recommendation_id = r.id AND e.event_type = 'PAPER_ENTRY' "
        "WHERE r.underlying = :u AND r.status = 'TRADE' "
        "AND NOT EXISTS (SELECT 1 FROM option_paper_trade_events x WHERE x.recommendation_id = r.id AND x.event_type = 'EXIT') "
        "ORDER BY r.decision_timestamp"
    ), {"u": underlying}).mappings().all()
    return tuple(
        OpenPaperPosition(recommendation_id=r["id"], decision_timestamp=r["decision_timestamp"], entry_payload=r["payload"])
        for r in rows
    )


def _reconstruct_legs(entry_payload: Mapping[str, Any]) -> list[OptionLeg]:
    legs = []
    for leg_snap in entry_payload.get("legs") or ():
        c = leg_snap["contract"]
        contract = ContractIdentity(
            underlying=c["underlying"], derivative_class=c["derivative_class"], expiry=dt.date.fromisoformat(c["expiry"]),
            strike_paise=c["strike_paise"], option_type=c["option_type"],
        )
        entry_observed_at = dt.datetime.fromisoformat(leg_snap["entry_observed_at"]) if leg_snap.get("entry_observed_at") else None
        legs.append(OptionLeg(
            contract=contract, side=leg_snap["side"], lots=leg_snap["lots"], lot_size=leg_snap["lot_size"],
            lot_size_source=leg_snap["lot_size_source"], entry_price=Decimal(leg_snap["entry_price"]),
            entry_price_source=leg_snap["entry_price_source"], provenance_class=leg_snap["provenance_class"],
            entry_observed_at=entry_observed_at,
        ))
    return legs


def _quote_for_contract(quotes: Sequence[LiveQuote], contract: ContractIdentity) -> LiveQuote | None:
    for q in quotes:
        if q.contract == contract:
            return q
    return None


def _observation_leg_snapshot(leg: OptionLeg, quote: LiveQuote | None) -> dict[str, Any]:
    if quote is None:
        return {
            "contract": _json_safe(dataclasses.asdict(leg.contract)), "quote_timestamp": None, "quote_status": None,
            "ltp": None, "bid": None, "ask": None, "volume": None, "open_interest": None,
            "provider_reported_iv": None, "provider_reported_greeks": None, "valuation_source": None,
            "mtm_pnl": None, "data_quality": "UNAVAILABLE_INSUFFICIENT_DATA",
            "unavailable_reason": "no live quote for this contract in the current snapshot",
        }
    valuation = valuation_from_live_quote(quote, price_kind="LAST_PRICE")
    mtm_pnl: Decimal | None = None
    unavailable_reason: str | None = None
    if valuation is None:
        unavailable_reason = f"quote status is {quote.status!r} or last_price unavailable - no valid mark"
    else:
        mtm_pnl = mark_to_market_leg(leg, valuation).mtm_pnl
    return {
        "contract": _json_safe(dataclasses.asdict(leg.contract)),
        "quote_timestamp": quote.observed_at.isoformat(),
        "quote_status": quote.status,
        "ltp": _json_safe(quote.last_price),
        "bid": _json_safe(quote.bid),
        "ask": _json_safe(quote.ask),
        "volume": quote.volume,
        "open_interest": quote.open_interest,
        "provider_reported_iv": _json_safe(quote.provider_reported_iv),
        "provider_reported_greeks": {k: _json_safe(v) for k, v in quote.provider_reported_greeks.items()} if quote.provider_reported_greeks else None,
        "valuation_source": valuation.source if valuation is not None else None,
        "mtm_pnl": _json_safe(mtm_pnl),
        "data_quality": "OK" if valuation is not None else "UNAVAILABLE_INSUFFICIENT_DATA",
        "unavailable_reason": unavailable_reason,
    }


def _existing_observation_id(db: Session, recommendation_id: uuid.UUID, event_timestamp: dt.datetime) -> uuid.UUID | None:
    return db.execute(text(
        "SELECT id FROM option_paper_trade_events WHERE recommendation_id=:rid AND event_type='OBSERVATION' AND event_timestamp=:ts LIMIT 1"
    ), {"rid": recommendation_id, "ts": event_timestamp}).scalar()


def _create_observation(db: Session, position: OpenPaperPosition, snapshot, now: dt.datetime) -> uuid.UUID | None:
    legs = _reconstruct_legs(position.entry_payload)
    if not legs:
        return None

    existing = _existing_observation_id(db, position.recommendation_id, snapshot.observed_at)
    if existing is not None:
        return existing

    leg_snapshots = []
    mtm_values: list[Decimal] = []
    all_valued = True
    for leg in legs:
        quote = _quote_for_contract(snapshot.quotes, leg.contract)
        snap = _observation_leg_snapshot(leg, quote)
        leg_snapshots.append(snap)
        if snap["mtm_pnl"] is not None:
            mtm_values.append(Decimal(snap["mtm_pnl"]))
        else:
            all_valued = False

    structure_mtm_pnl = sum(mtm_values, Decimal(0)) if all_valued and mtm_values else None
    payload = {
        "legs": leg_snapshots,
        "structure_mtm_pnl": _json_safe(structure_mtm_pnl),
        "structure_gross_unrealized_pnl": _json_safe(structure_mtm_pnl),
        "data_quality": "OK" if all_valued else "UNAVAILABLE_INSUFFICIENT_DATA",
        "snapshot_source": snapshot.source,
        "snapshot_raw_response_sha256": snapshot.raw_response_sha256,
    }
    event_id = uuid.uuid4()
    db.execute(text(
        "INSERT INTO option_paper_trade_events (id, recommendation_id, event_type, event_timestamp, payload) "
        "VALUES (:id, :rid, 'OBSERVATION', :ts, CAST(:payload AS JSONB))"
    ), {"id": event_id, "rid": position.recommendation_id, "ts": snapshot.observed_at, "payload": json.dumps(payload)})
    return event_id


def run_observation_cycle(db: Session, *, underlying: str, snapshot, now: dt.datetime) -> tuple[uuid.UUID, ...]:
    created: list[uuid.UUID] = []
    for position in open_paper_positions(db, underlying=underlying):
        event_id = _create_observation(db, position, snapshot, now)
        if event_id is not None:
            created.append(event_id)
    return tuple(created)


def _existing_exit_id(db: Session, recommendation_id: uuid.UUID) -> uuid.UUID | None:
    return db.execute(text(
        "SELECT id FROM option_paper_trade_events WHERE recommendation_id=:rid AND event_type='EXIT' LIMIT 1"
    ), {"rid": recommendation_id}).scalar()


def _insert_exit(db: Session, recommendation_id: uuid.UUID, event_timestamp: dt.datetime, payload: dict[str, Any]) -> uuid.UUID:
    event_id = uuid.uuid4()
    db.execute(text(
        "INSERT INTO option_paper_trade_events (id, recommendation_id, event_type, event_timestamp, payload) "
        "VALUES (:id, :rid, 'EXIT', :ts, CAST(:payload AS JSONB))"
    ), {"id": event_id, "rid": recommendation_id, "ts": event_timestamp, "payload": json.dumps(payload)})
    return event_id


def create_manual_paper_exit(
    db: Session, *, recommendation_id: uuid.UUID, entry_payload: Mapping[str, Any], now: dt.datetime,
    exit_valuations: Mapping[ContractIdentity, LegValuation], reason: str = "MANUAL_PAPER_EXIT",
) -> uuid.UUID:
    existing = _existing_exit_id(db, recommendation_id)
    if existing is not None:
        return existing

    legs = _reconstruct_legs(entry_payload)
    OptionStructure(legs=tuple(legs))
    missing = [leg.contract for leg in legs if leg.contract not in exit_valuations]
    if missing:
        raise ValueError(f"exit_valuations is missing a valuation for {missing} - every leg must be valued to record a manual exit")

    leg_results = [mark_to_market_leg(leg, exit_valuations[leg.contract]) for leg in legs]
    gross_pnl = sum((r.mtm_pnl for r in leg_results), Decimal(0))
    payload = {
        "exit_mode": "MANUAL_PAPER_EXIT",
        "exit_reason": reason,
        "gross_pnl": _json_safe(gross_pnl),
        "net_pnl": "COSTS_NOT_MODELED",
        "exit_prices": [
            {"contract": _json_safe(dataclasses.asdict(v.contract)), "price": _json_safe(v.price), "source": v.source, "valuation_at": v.valuation_at.isoformat()}
            for v in exit_valuations.values()
        ],
    }
    return _insert_exit(db, recommendation_id, now, payload)


def create_ambiguous_unavailable_exit(db: Session, *, recommendation_id: uuid.UUID, now: dt.datetime, reason: str) -> uuid.UUID:
    existing = _existing_exit_id(db, recommendation_id)
    if existing is not None:
        return existing
    payload = {"exit_mode": "AMBIGUOUS_UNAVAILABLE", "exit_reason": reason, "gross_pnl": None, "net_pnl": None}
    return _insert_exit(db, recommendation_id, now, payload)


def create_expiry_settlement_exit(
    db: Session, *, recommendation_id: uuid.UUID, entry_payload: Mapping[str, Any], now: dt.datetime,
    settlement_price: Decimal, settlement_source_detail: str,
) -> uuid.UUID:
    if settlement_price <= 0:
        raise ValueError(f"settlement_price must be positive, got {settlement_price!r}")
    if not settlement_source_detail or not settlement_source_detail.strip():
        raise ValueError("settlement_source_detail must describe the verified source - never left blank")
    existing = _existing_exit_id(db, recommendation_id)
    if existing is not None:
        return existing

    legs = _reconstruct_legs(entry_payload)
    structure = OptionStructure(legs=tuple(legs))
    gross_pnl = structure_expiry_pnl(structure, settlement_price)
    payload = {
        "exit_mode": "EXPIRY_EXIT", "exit_reason": "EXPIRY_SETTLEMENT",
        "settlement_price": _json_safe(settlement_price), "settlement_source_detail": settlement_source_detail,
        "gross_pnl": _json_safe(gross_pnl), "net_pnl": "COSTS_NOT_MODELED",
    }
    return _insert_exit(db, recommendation_id, now, payload)


def _decimal_or_none(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def compute_mfe_mae(observation_payloads: Sequence[Mapping[str, Any]]) -> tuple[Decimal | None, Decimal | None]:
    values = [v for v in (_decimal_or_none(p.get("structure_mtm_pnl")) for p in observation_payloads) if v is not None]
    if not values:
        return None, None
    mfe = max(max(values), Decimal(0))
    mae = min(min(values), Decimal(0))
    return mfe, mae
