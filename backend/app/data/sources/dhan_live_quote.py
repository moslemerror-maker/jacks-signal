from __future__ import annotations

import datetime as dt
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from backend.app.options.contracts import ContractIdentity
from backend.app.options.live_quote import LiveQuote
from backend.app.options.provenance import ProvenanceClass

_GREEK_FIELDS = ("delta", "theta", "gamma", "vega")


def _decimal_or_none(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        d = Decimal(str(value))
    except InvalidOperation:
        return None
    return d if d != 0 else None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _greeks_or_none(leg: Mapping[str, Any]) -> Mapping[str, Decimal] | None:
    raw = leg.get("greeks")
    if not isinstance(raw, Mapping):
        return None
    out = {f: _decimal_or_none(raw.get(f)) for f in _GREEK_FIELDS}
    out = {k: v for k, v in out.items() if v is not None}
    return out or None


def parse_dhan_optionchain(
    raw: Mapping[str, Any], *, underlying: str, expiry: dt.date, observed_at: dt.datetime, source_row_hash: str | None,
    provenance_class: ProvenanceClass,
) -> list[LiveQuote]:
    if observed_at.tzinfo is None:
        raise ValueError("observed_at must be timezone-aware")
    outer = raw.get("data") or {}
    data = outer.get("data") if isinstance(outer.get("data"), Mapping) else outer
    underlying_last_price = _decimal_or_none(data.get("last_price"))
    oc = data.get("oc") or {}
    quotes: list[LiveQuote] = []
    for strike_str, legs in oc.items():
        try:
            strike_paise = round(float(strike_str) * 100)
        except (TypeError, ValueError):
            continue
        if strike_paise <= 0:
            continue
        for side, option_type in (("ce", "CE"), ("pe", "PE")):
            leg = legs.get(side) if isinstance(legs, Mapping) else None
            if leg is None:
                continue
            contract = ContractIdentity(underlying, "IDO", expiry, strike_paise, option_type)
            quotes.append(LiveQuote.from_raw(
                contract=contract, observed_at=observed_at,
                last_price=_decimal_or_none(leg.get("last_price")),
                bid=_decimal_or_none(leg.get("top_bid_price")),
                ask=_decimal_or_none(leg.get("top_ask_price")),
                volume=_int_or_none(leg.get("volume")),
                open_interest=_int_or_none(leg.get("oi")),
                underlying_last_price=underlying_last_price,
                provider_reported_iv=_decimal_or_none(leg.get("implied_volatility")),
                provider_reported_greeks=_greeks_or_none(leg),
                source="dhan", source_row_hash=source_row_hash, provenance_class=provenance_class,
            ))
    return quotes
