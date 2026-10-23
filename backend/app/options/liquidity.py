from __future__ import annotations

import dataclasses
import datetime as dt
from decimal import Decimal

from .live_quote import LiveQuote, is_stale, quote_age


@dataclasses.dataclass(frozen=True, slots=True)
class LiquidityAssessment:
    is_acceptable: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.is_acceptable != (len(self.reasons) == 0):
            raise ValueError("is_acceptable must be True iff reasons is empty")


def assess_leg_liquidity(
    quote: LiveQuote, *, as_of: dt.datetime, max_quote_age: dt.timedelta, max_spread_fraction: Decimal, min_volume: int | None,
) -> LiquidityAssessment:
    reasons: list[str] = []
    if not quote.is_usable:
        reasons.append(f"quote status is {quote.status!r}, not QUOTED")
        return LiquidityAssessment(is_acceptable=False, reasons=tuple(reasons))

    if is_stale(quote, as_of, max_quote_age):
        reasons.append(f"quote is stale: {quote_age(quote, as_of)} old (max allowed {max_quote_age})")

    if quote.bid is not None and quote.ask is not None:
        mid = (quote.bid + quote.ask) / 2
        if mid > 0:
            spread_fraction = (quote.ask - quote.bid) / mid
            if spread_fraction > max_spread_fraction:
                reasons.append(f"spread too wide: {spread_fraction:.4f} of mid > max {max_spread_fraction}")
    else:
        reasons.append("bid/ask not both present - cannot assess spread")

    if min_volume is not None:
        if quote.volume is None:
            reasons.append(f"volume not reported - cannot verify minimum volume {min_volume}")
        elif quote.volume < min_volume:
            reasons.append(f"volume {quote.volume} below minimum {min_volume}")

    return LiquidityAssessment(is_acceptable=not reasons, reasons=tuple(reasons))
