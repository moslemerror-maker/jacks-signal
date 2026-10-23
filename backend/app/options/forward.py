from __future__ import annotations

import dataclasses
from decimal import Decimal
from typing import Literal, get_args

from .live_quote import LiveQuote

ForwardSource = Literal["VERIFIED_LIVE_FUTURES", "SPOT_PROXY_UNVERIFIED"]
_FORWARD_SOURCES: tuple[str, ...] = get_args(ForwardSource)


@dataclasses.dataclass(frozen=True, slots=True)
class ForwardResolution:

    forward: Decimal
    source: ForwardSource
    underlying_last_price: Decimal
    detail: str

    def __post_init__(self) -> None:
        if self.source not in _FORWARD_SOURCES:
            raise ValueError(f"source must be one of {_FORWARD_SOURCES}, got {self.source!r}")
        if self.forward <= 0:
            raise ValueError(f"forward must be positive, got {self.forward!r}")
        if self.underlying_last_price <= 0:
            raise ValueError(f"underlying_last_price must be positive, got {self.underlying_last_price!r}")

    @property
    def is_verified(self) -> bool:
        return self.source == "VERIFIED_LIVE_FUTURES"


def resolve_forward_spot_proxy(underlying_last_price: Decimal) -> ForwardResolution:
    if underlying_last_price <= 0:
        raise ValueError(f"underlying_last_price must be positive, got {underlying_last_price!r}")
    return ForwardResolution(
        forward=underlying_last_price, source="SPOT_PROXY_UNVERIFIED", underlying_last_price=underlying_last_price,
        detail="Spot used directly as forward - cost-of-carry (financing/dividends) ignored. Not a verified market "
               "observation; use only when a verified live futures quote is unavailable.",
    )


def resolve_forward_from_live_future(future_quote: LiveQuote) -> ForwardResolution:
    if future_quote.contract.derivative_class not in ("IDF", "STF"):
        raise ValueError(f"future_quote must be a futures contract (IDF/STF), got derivative_class={future_quote.contract.derivative_class!r}")
    if not future_quote.is_usable:
        raise ValueError(f"future_quote is not usable (status={future_quote.status!r}) - cannot resolve a verified forward from it")
    assert future_quote.last_price is not None
    spot = future_quote.underlying_last_price if future_quote.underlying_last_price is not None else future_quote.last_price
    return ForwardResolution(
        forward=future_quote.last_price, source="VERIFIED_LIVE_FUTURES", underlying_last_price=spot,
        detail=f"live traded futures price for {future_quote.contract} observed at {future_quote.observed_at.isoformat()}",
    )
