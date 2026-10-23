from __future__ import annotations

import dataclasses
import datetime as dt
from decimal import Decimal
from typing import Literal, Mapping, get_args

from .contracts import ContractIdentity
from .provenance import ProvenanceClass, check_provenance_class

LiveQuoteStatus = Literal["QUOTED", "NO_QUOTE", "ANOMALY_NONPOSITIVE_PRICE", "ANOMALY_BID_ABOVE_ASK"]
_LIVE_QUOTE_STATUSES: tuple[str, ...] = get_args(LiveQuoteStatus)


def classify_live_quote(last_price: Decimal | None, bid: Decimal | None, ask: Decimal | None) -> LiveQuoteStatus:
    for v in (last_price, bid, ask):
        if v is not None and v <= 0:
            return "ANOMALY_NONPOSITIVE_PRICE"
    if bid is not None and ask is not None and bid > ask:
        return "ANOMALY_BID_ABOVE_ASK"
    if last_price is None and bid is None and ask is None:
        return "NO_QUOTE"
    return "QUOTED"


@dataclasses.dataclass(frozen=True, slots=True)
class LiveQuote:

    contract: ContractIdentity
    observed_at: dt.datetime
    last_price: Decimal | None
    bid: Decimal | None
    ask: Decimal | None
    volume: int | None
    open_interest: int | None
    underlying_last_price: Decimal | None
    provider_reported_iv: Decimal | None
    provider_reported_greeks: Mapping[str, Decimal] | None
    status: LiveQuoteStatus
    source: str
    source_row_hash: str | None
    provenance_class: ProvenanceClass

    def __post_init__(self) -> None:
        check_provenance_class(self.provenance_class)
        if self.status not in _LIVE_QUOTE_STATUSES:
            raise ValueError(f"status must be one of {_LIVE_QUOTE_STATUSES}, got {self.status!r}")
        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        expected = classify_live_quote(self.last_price, self.bid, self.ask)
        if self.status != expected:
            raise ValueError(f"status {self.status!r} is inconsistent with the raw values (classify_live_quote says {expected!r})")
        if self.underlying_last_price is not None and self.underlying_last_price <= 0:
            raise ValueError(f"underlying_last_price must be positive if given, got {self.underlying_last_price!r}")
        if self.provider_reported_iv is not None and self.provider_reported_iv <= 0:
            raise ValueError(f"provider_reported_iv must be positive if given, got {self.provider_reported_iv!r}")
        if not self.source or not self.source.strip():
            raise ValueError("source must be a non-empty provider name")

    @classmethod
    def from_raw(
        cls, *, contract: ContractIdentity, observed_at: dt.datetime, last_price: Decimal | None, bid: Decimal | None,
        ask: Decimal | None, volume: int | None, open_interest: int | None, underlying_last_price: Decimal | None,
        provider_reported_iv: Decimal | None, provider_reported_greeks: Mapping[str, Decimal] | None, source: str,
        source_row_hash: str | None, provenance_class: ProvenanceClass,
    ) -> "LiveQuote":
        status = classify_live_quote(last_price, bid, ask)
        return cls(
            contract=contract, observed_at=observed_at, last_price=last_price, bid=bid, ask=ask, volume=volume,
            open_interest=open_interest, underlying_last_price=underlying_last_price,
            provider_reported_iv=provider_reported_iv, provider_reported_greeks=provider_reported_greeks,
            status=status, source=source, source_row_hash=source_row_hash, provenance_class=provenance_class,
        )

    @property
    def is_usable(self) -> bool:
        return self.status == "QUOTED"


def quote_age(quote: LiveQuote, as_of: dt.datetime) -> dt.timedelta:
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    if as_of < quote.observed_at:
        raise ValueError(f"as_of ({as_of}) precedes quote.observed_at ({quote.observed_at}) - no look-ahead")
    return as_of - quote.observed_at


def is_stale(quote: LiveQuote, as_of: dt.datetime, max_age: dt.timedelta) -> bool:
    if max_age < dt.timedelta(0):
        raise ValueError("max_age must not be negative")
    return quote_age(quote, as_of) > max_age
