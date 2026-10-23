from __future__ import annotations

import dataclasses
import datetime as dt
from decimal import Decimal
from typing import Literal, get_args

from .contracts import ContractIdentity
from .provenance import ProvenanceClass, check_provenance_class

TradeState = Literal["TRADED", "NOT_TRADED_NO_PRICE", "NOT_TRADED_PRICE_CARRIED", "ANOMALY_TRADED_NONPOSITIVE_PRICE",
                    "ANOMALY_TRADED_OHLC_ORDER", "ANOMALY_NOT_TRADED_WITH_OHLC", "ANOMALY_NEGATIVE_VOLUME"]
_TRADE_STATES: tuple[str, ...] = get_args(TradeState)

SettleSemantic = Literal["LIVE_DAILY_SETTLEMENT", "ZERO", "UNDERLYING_SETTLEMENT_PRICE", "INTRINSIC_VALUE", "EQUALS_LAST_CLOSE", "OTHER", "UNKNOWN"]
_SETTLE_SEMANTICS: tuple[str, ...] = get_args(SettleSemantic)


def classify_trade_state(volume: int, o: Decimal, h: Decimal, l: Decimal, c: Decimal) -> tuple[bool, TradeState]:
    if volume > 0:
        if min(o, h, l, c) <= 0:
            return False, "ANOMALY_TRADED_NONPOSITIVE_PRICE"
        if not (l <= o <= h and l <= c <= h):
            return False, "ANOMALY_TRADED_OHLC_ORDER"
        return True, "TRADED"
    if volume == 0:
        if o == 0 and h == 0 and l == 0 and c == 0:
            return False, "NOT_TRADED_NO_PRICE"
        if o == 0 and h == 0 and l == 0 and c > 0:
            return False, "NOT_TRADED_PRICE_CARRIED"
        return False, "ANOMALY_NOT_TRADED_WITH_OHLC"
    return False, "ANOMALY_NEGATIVE_VOLUME"


def classify_expiry_settle(settle: Decimal, close: Decimal, spot: Decimal, strike: Decimal, cp: str | None) -> str:
    intrinsic = max(spot - strike, Decimal(0)) if cp in ("CE", "CA") else max(strike - spot, Decimal(0))
    if settle == 0:
        return "ZERO"
    if settle == spot:
        return "UNDERLYING_SETTLEMENT_PRICE"
    if settle == intrinsic:
        return "INTRINSIC_VALUE"
    if settle == close:
        return "EQUALS_LAST_CLOSE"
    return "OTHER"


def settle_semantic_for(settle_raw: Decimal | None, close: Decimal, spot: Decimal | None, strike: Decimal | None, cp: str | None, is_expiry_day: bool) -> SettleSemantic | None:
    if settle_raw is None:
        return None
    if not is_expiry_day:
        return "LIVE_DAILY_SETTLEMENT"
    if spot is None or strike is None:
        return "UNKNOWN"
    return classify_expiry_settle(settle_raw, close, spot, strike, cp)  # type: ignore[return-value]


@dataclasses.dataclass(frozen=True, slots=True)
class DerivativeBar:

    contract: ContractIdentity
    session_date: dt.date
    bar_end_at: dt.datetime
    interval: str

    raw_open: Decimal
    raw_high: Decimal
    raw_low: Decimal
    raw_close: Decimal
    volume: int
    open_interest: int | None
    oi_change: int | None
    turnover: Decimal | None
    settle_raw: Decimal | None
    underlying_price: Decimal | None
    observed_lot_size: int | None

    is_traded: bool
    trade_state: TradeState
    open: Decimal | None
    high: Decimal | None
    low: Decimal | None
    close: Decimal | None
    carried_close: Decimal | None
    settle_semantic: SettleSemantic | None

    available_at: dt.datetime
    availability_basis: str
    source: str
    source_row_hash: str | None
    provenance_class: ProvenanceClass

    def __post_init__(self) -> None:
        check_provenance_class(self.provenance_class)
        if self.trade_state not in _TRADE_STATES:
            raise ValueError(f"trade_state must be one of {_TRADE_STATES}, got {self.trade_state!r}")
        if self.settle_semantic is not None and self.settle_semantic not in _SETTLE_SEMANTICS:
            raise ValueError(f"settle_semantic must be one of {_SETTLE_SEMANTICS} or None, got {self.settle_semantic!r}")
        if self.is_traded != (self.trade_state == "TRADED"):
            raise ValueError(f"is_traded ({self.is_traded}) is inconsistent with trade_state ({self.trade_state!r})")
        if self.bar_end_at.tzinfo is None:
            raise ValueError("bar_end_at must be timezone-aware")
        if self.available_at.tzinfo is None:
            raise ValueError("available_at must be timezone-aware")
        if self.available_at < self.bar_end_at:
            raise ValueError(f"available_at ({self.available_at}) is before bar_end_at ({self.bar_end_at}) - a bar cannot be known before it happened (no look-ahead)")
        if self.is_traded:
            if None in (self.open, self.high, self.low, self.close):
                raise ValueError("a TRADED bar must have open/high/low/close populated")
            if self.carried_close is not None:
                raise ValueError("a TRADED bar must not carry a carried_close")
            if not (self.low <= self.open <= self.high and self.low <= self.close <= self.high):  # type: ignore[operator]
                raise ValueError(f"TRADED bar's canonical OHLC is out of order: O={self.open} H={self.high} L={self.low} C={self.close}")
        else:
            if (self.open, self.high, self.low, self.close) != (None, None, None, None):
                raise ValueError("a non-traded bar must not have canonical open/high/low/close populated")
            if self.trade_state == "NOT_TRADED_PRICE_CARRIED":
                if self.carried_close is None:
                    raise ValueError("NOT_TRADED_PRICE_CARRIED requires carried_close")
            elif self.carried_close is not None:
                raise ValueError(f"trade_state {self.trade_state!r} must not have a carried_close")

    @classmethod
    def from_raw(
        cls, *, contract: ContractIdentity, session_date: dt.date, bar_end_at: dt.datetime, interval: str,
        open_: Decimal, high: Decimal, low: Decimal, close: Decimal, volume: int, open_interest: int | None, oi_change: int | None,
        turnover: Decimal | None, settle_raw: Decimal | None, underlying_price: Decimal | None, observed_lot_size: int | None,
        available_at: dt.datetime, availability_basis: str, source: str, source_row_hash: str | None, provenance_class: ProvenanceClass,
        is_expiry_day: bool = False,
    ) -> "DerivativeBar":
        is_traded, state = classify_trade_state(volume, open_, high, low, close)
        strike = Decimal(contract.strike_paise) / 100 if contract.strike_paise is not None else None
        semantic = settle_semantic_for(settle_raw, close, underlying_price, strike, contract.option_type, is_expiry_day)
        return cls(
            contract=contract, session_date=session_date, bar_end_at=bar_end_at, interval=interval,
            raw_open=open_, raw_high=high, raw_low=low, raw_close=close, volume=volume, open_interest=open_interest, oi_change=oi_change,
            turnover=turnover, settle_raw=settle_raw, underlying_price=underlying_price, observed_lot_size=observed_lot_size,
            is_traded=is_traded, trade_state=state,
            open=open_ if is_traded else None, high=high if is_traded else None, low=low if is_traded else None, close=close if is_traded else None,
            carried_close=close if state == "NOT_TRADED_PRICE_CARRIED" else None, settle_semantic=semantic,
            available_at=available_at, availability_basis=availability_basis, source=source, source_row_hash=source_row_hash,
            provenance_class=provenance_class,
        )
