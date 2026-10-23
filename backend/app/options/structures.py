from __future__ import annotations

import dataclasses
import datetime as dt
from decimal import Decimal
from typing import Literal, Sequence, get_args

from .contracts import ContractIdentity, ContractSpec, spec_as_of
from .provenance import ProvenanceClass, check_provenance_class

Side = Literal["BUY", "SELL"]
_SIDES: tuple[str, ...] = get_args(Side)

LotSizeSource = Literal["CONTRACT_SPEC", "CALLER_SUPPLIED"]
_LOT_SIZE_SOURCES: tuple[str, ...] = get_args(LotSizeSource)

EntryPriceSource = Literal["TRADED_BAR", "LIVE_QUOTE", "CALLER_SUPPLIED"]
_ENTRY_PRICE_SOURCES: tuple[str, ...] = get_args(EntryPriceSource)


@dataclasses.dataclass(frozen=True, slots=True)
class OptionLeg:

    contract: ContractIdentity
    side: Side
    lots: int
    lot_size: int
    lot_size_source: LotSizeSource
    entry_price: Decimal
    entry_price_source: EntryPriceSource
    provenance_class: ProvenanceClass
    entry_observed_at: dt.datetime | None = None

    def __post_init__(self) -> None:
        check_provenance_class(self.provenance_class)
        if not self.contract.is_option:
            raise ValueError(f"OptionLeg requires an option contract (IDO/STO), got derivative_class={self.contract.derivative_class!r} - futures legs are not supported in O2-D")
        if self.side not in _SIDES:
            raise ValueError(f"side must be one of {_SIDES}, got {self.side!r}")
        if not isinstance(self.lots, int) or isinstance(self.lots, bool) or self.lots <= 0:
            raise ValueError(f"lots must be a positive int, got {self.lots!r}")
        if not isinstance(self.lot_size, int) or isinstance(self.lot_size, bool) or self.lot_size <= 0:
            raise ValueError(f"lot_size must be a positive int, got {self.lot_size!r}")
        if self.lot_size_source not in _LOT_SIZE_SOURCES:
            raise ValueError(f"lot_size_source must be one of {_LOT_SIZE_SOURCES}, got {self.lot_size_source!r}")
        if self.entry_price_source not in _ENTRY_PRICE_SOURCES:
            raise ValueError(f"entry_price_source must be one of {_ENTRY_PRICE_SOURCES}, got {self.entry_price_source!r}")
        if self.entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {self.entry_price!r} - a zero/negative premium is never a valid entry observation")
        if self.entry_price_source == "TRADED_BAR" and self.entry_observed_at is None:
            raise ValueError("entry_price_source is TRADED_BAR but entry_observed_at is missing - a market-observed entry price must carry its PIT timestamp")
        if self.entry_observed_at is not None and self.entry_observed_at.tzinfo is None:
            raise ValueError("entry_observed_at must be timezone-aware")

    @property
    def quantity_sign(self) -> int:
        return 1 if self.side == "BUY" else -1

    @property
    def total_units(self) -> int:
        return self.lots * self.lot_size

    @property
    def entry_cash_flow(self) -> Decimal:
        return -Decimal(self.quantity_sign) * self.entry_price * self.total_units

    @property
    def is_non_authoritative(self) -> bool:
        return self.lot_size_source != "CONTRACT_SPEC"


def lot_size_from_spec(specs: Sequence[ContractSpec], contract: ContractIdentity, as_of: dt.date) -> tuple[int, LotSizeSource]:
    spec = spec_as_of([s for s in specs if s.contract == contract], as_of)
    if spec is None:
        raise ValueError(f"no ContractSpec covers {contract} as of {as_of} - lot size is not authoritatively known; do not guess it")
    return spec.lot_size, "CONTRACT_SPEC"


def leg_from_traded_bar(bar, *, side: Side, lots: int, lot_size: int, lot_size_source: LotSizeSource) -> OptionLeg:
    if not bar.is_traded:
        raise ValueError(f"bar is not TRADED (trade_state={bar.trade_state!r}) - refusing to use its close as an entry price")
    assert bar.close is not None
    return OptionLeg(
        contract=bar.contract, side=side, lots=lots, lot_size=lot_size, lot_size_source=lot_size_source,
        entry_price=bar.close, entry_price_source="TRADED_BAR", provenance_class=bar.provenance_class,
        entry_observed_at=bar.available_at,
    )


def leg_from_live_quote(quote, *, side: Side, lots: int, lot_size: int, lot_size_source: LotSizeSource) -> OptionLeg:
    if not quote.is_usable:
        raise ValueError(f"quote is not usable (status={quote.status!r}) - refusing to use it as an entry price")
    assert quote.last_price is not None
    return OptionLeg(
        contract=quote.contract, side=side, lots=lots, lot_size=lot_size, lot_size_source=lot_size_source,
        entry_price=quote.last_price, entry_price_source="LIVE_QUOTE", provenance_class=quote.provenance_class,
        entry_observed_at=quote.observed_at,
    )


def option_expiry_intrinsic_value_per_unit(contract: ContractIdentity, settlement_price: Decimal) -> Decimal:
    if not contract.is_option:
        raise ValueError(f"intrinsic value requires an option contract, got derivative_class={contract.derivative_class!r} (futures legs are not supported in O2-D)")
    if settlement_price < 0:
        raise ValueError(f"settlement_price must not be negative, got {settlement_price!r} (an index/underlying price cannot be negative)")
    strike = Decimal(contract.strike_paise) / 100  # type: ignore[arg-type]
    if contract.option_type in ("CE", "CA"):
        return max(settlement_price - strike, Decimal(0))
    return max(strike - settlement_price, Decimal(0))


def leg_expiry_payoff_per_lot(leg: OptionLeg, settlement_price: Decimal) -> Decimal:
    per_unit = option_expiry_intrinsic_value_per_unit(leg.contract, settlement_price)
    return Decimal(leg.quantity_sign) * per_unit * leg.lot_size


def leg_expiry_payoff(leg: OptionLeg, settlement_price: Decimal) -> Decimal:
    return leg_expiry_payoff_per_lot(leg, settlement_price) * leg.lots


def leg_expiry_pnl(leg: OptionLeg, settlement_price: Decimal) -> Decimal:
    return leg_expiry_payoff(leg, settlement_price) + leg.entry_cash_flow


@dataclasses.dataclass(frozen=True, slots=True)
class OptionStructure:

    legs: tuple[OptionLeg, ...]

    def __post_init__(self) -> None:
        if not self.legs:
            raise ValueError("an OptionStructure requires at least one leg")
        underlyings = {leg.contract.underlying for leg in self.legs}
        if len(underlyings) > 1:
            raise ValueError(f"all legs of a structure must share one underlying, got {underlyings}")
        expiries = {leg.contract.expiry for leg in self.legs}
        if len(expiries) > 1:
            raise ValueError(f"all legs of a structure must share one expiry, got {expiries} - calendar/diagonal spreads are not supported in O2-D")
        provenances = {leg.provenance_class for leg in self.legs}
        if len(provenances) > 1:
            raise ValueError(f"all legs of a structure must share one provenance_class, got {provenances}")

    @property
    def underlying(self) -> str:
        return self.legs[0].contract.underlying

    @property
    def expiry(self) -> dt.date:
        return self.legs[0].contract.expiry

    @property
    def provenance_class(self) -> ProvenanceClass:
        return self.legs[0].provenance_class

    @property
    def uses_non_authoritative_lot_size(self) -> bool:
        return any(leg.is_non_authoritative for leg in self.legs)


def structure_entry_cash_flow(structure: OptionStructure) -> Decimal:
    return sum((leg.entry_cash_flow for leg in structure.legs), Decimal(0))


def structure_expiry_payoff(structure: OptionStructure, settlement_price: Decimal) -> Decimal:
    return sum((leg_expiry_payoff(leg, settlement_price) for leg in structure.legs), Decimal(0))


def structure_expiry_pnl(structure: OptionStructure, settlement_price: Decimal) -> Decimal:
    return structure_expiry_payoff(structure, settlement_price) + structure_entry_cash_flow(structure)


@dataclasses.dataclass(frozen=True, slots=True)
class PayoffCurvePoint:
    settlement_price: Decimal
    payoff: Decimal
    pnl: Decimal


def expiry_pnl_curve(structure: OptionStructure, settlement_prices: Sequence[Decimal]) -> tuple[PayoffCurvePoint, ...]:
    if not settlement_prices:
        raise ValueError("settlement_prices must not be empty")
    entry_cf = structure_entry_cash_flow(structure)
    return tuple(
        PayoffCurvePoint(settlement_price=s, payoff=structure_expiry_payoff(structure, s), pnl=structure_expiry_payoff(structure, s) + entry_cf)
        for s in settlement_prices
    )


Boundedness = Literal["BOUNDED", "UNBOUNDED"]


@dataclasses.dataclass(frozen=True, slots=True)
class PnLExtremum:

    boundedness: Boundedness
    value: Decimal | None
    at_settlement_price: Decimal | None

    def __post_init__(self) -> None:
        if (self.boundedness == "BOUNDED") != (self.value is not None and self.at_settlement_price is not None):
            raise ValueError("value and at_settlement_price must be set iff boundedness is BOUNDED")


@dataclasses.dataclass(frozen=True, slots=True)
class BreakEvenResult:

    points: tuple[Decimal, ...]
    ranges: tuple[tuple[Decimal, Decimal | None], ...]


def _vertices_and_tail_slope(structure: OptionStructure) -> tuple[list[Decimal], list[Decimal], Decimal]:
    strikes = sorted({Decimal(leg.contract.strike_paise) / 100 for leg in structure.legs})  # type: ignore[arg-type]
    xs = sorted({Decimal(0)} | {k for k in strikes if k > 0})
    ys = [structure_expiry_pnl(structure, x) for x in xs]
    tail_slope = sum(
        (Decimal(leg.quantity_sign) * leg.total_units for leg in structure.legs if leg.contract.option_type in ("CE", "CA")),
        Decimal(0),
    )
    return xs, ys, tail_slope


def structure_max_profit(structure: OptionStructure) -> PnLExtremum:
    xs, ys, tail_slope = _vertices_and_tail_slope(structure)
    if tail_slope > 0:
        return PnLExtremum(boundedness="UNBOUNDED", value=None, at_settlement_price=None)
    i = max(range(len(ys)), key=lambda i: ys[i])
    return PnLExtremum(boundedness="BOUNDED", value=ys[i], at_settlement_price=xs[i])


def structure_max_loss(structure: OptionStructure) -> PnLExtremum:
    xs, ys, tail_slope = _vertices_and_tail_slope(structure)
    if tail_slope < 0:
        return PnLExtremum(boundedness="UNBOUNDED", value=None, at_settlement_price=None)
    i = min(range(len(ys)), key=lambda i: ys[i])
    return PnLExtremum(boundedness="BOUNDED", value=ys[i], at_settlement_price=xs[i])


def structure_break_even(structure: OptionStructure) -> BreakEvenResult:
    xs, ys, tail_slope = _vertices_and_tail_slope(structure)
    points: list[Decimal] = []
    ranges: list[tuple[Decimal, Decimal | None]] = []
    n = len(xs)
    i = 0
    while i < n:
        if ys[i] == 0:
            if i + 1 < n and ys[i + 1] == 0:
                ranges.append((xs[i], xs[i + 1]))
                i += 1
                continue
            points.append(xs[i])
        elif i + 1 < n and (ys[i] < 0) != (ys[i + 1] < 0):
            x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
            points.append(x0 + (x1 - x0) * (-y0) / (y1 - y0))
        i += 1
    if tail_slope == 0:
        if ys[-1] == 0 and xs[-1] not in points:
            pass
        if ys[-1] == 0:
            if xs[-1] in points:
                points.remove(xs[-1])
            ranges.append((xs[-1], None))
    else:
        if ys[-1] != 0 and ((ys[-1] < 0 and tail_slope > 0) or (ys[-1] > 0 and tail_slope < 0)):
            points.append(xs[-1] + (-ys[-1]) / tail_slope)
    return BreakEvenResult(points=tuple(sorted(set(points))), ranges=tuple(ranges))
