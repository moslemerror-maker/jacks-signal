from __future__ import annotations

import dataclasses
import datetime as dt
from decimal import Decimal
from typing import Literal, Mapping, get_args

from .bars import DerivativeBar
from .contracts import ContractIdentity
from .structures import OptionLeg, OptionStructure
from .valuation import ValuationInputs, black76_price

ValuationSource = Literal["OBSERVED_MARKET_PRICE", "MODEL_VALUATION"]
_VALUATION_SOURCES: tuple[str, ...] = get_args(ValuationSource)


@dataclasses.dataclass(frozen=True, slots=True)
class LegValuation:

    contract: ContractIdentity
    price: Decimal
    source: ValuationSource
    valuation_at: dt.datetime
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.source not in _VALUATION_SOURCES:
            raise ValueError(f"source must be one of {_VALUATION_SOURCES}, got {self.source!r}")
        if self.price <= 0:
            raise ValueError(f"price must be positive, got {self.price!r}")
        if self.valuation_at.tzinfo is None:
            raise ValueError("valuation_at must be timezone-aware")


def valuation_from_traded_bar(bar: DerivativeBar, *, valuation_at: dt.datetime | None = None) -> LegValuation:
    if not bar.is_traded:
        raise ValueError(f"bar is not TRADED (trade_state={bar.trade_state!r}) - refusing to use its close as a market valuation")
    assert bar.close is not None
    v_at = valuation_at if valuation_at is not None else bar.available_at
    if v_at.tzinfo is None:
        raise ValueError("valuation_at must be timezone-aware")
    if v_at < bar.available_at:
        raise ValueError(f"valuation_at ({v_at}) precedes this bar's available_at ({bar.available_at}) - no look-ahead")
    return LegValuation(
        contract=bar.contract, price=bar.close, source="OBSERVED_MARKET_PRICE", valuation_at=v_at,
        detail=f"bar session_date={bar.session_date.isoformat()} availability_basis={bar.availability_basis!r}",
    )


def valuation_from_model(contract: ContractIdentity, inputs: ValuationInputs, volatility: Decimal) -> LegValuation:
    if contract.option_type != inputs.option_type:
        raise ValueError(f"contract.option_type ({contract.option_type!r}) does not match inputs.option_type ({inputs.option_type!r})")
    contract_strike = Decimal(contract.strike_paise) / 100 if contract.strike_paise is not None else None
    if contract_strike != inputs.strike:
        raise ValueError(f"contract strike ({contract_strike}) does not match inputs.strike ({inputs.strike})")
    price = black76_price(inputs, volatility)
    return LegValuation(
        contract=contract, price=price, source="MODEL_VALUATION", valuation_at=inputs.valuation_at,
        detail=f"BLACK_76 volatility={volatility} forward={inputs.forward} risk_free_rate={inputs.risk_free_rate}",
    )


LiveQuotePriceKind = Literal["LAST_PRICE", "BID", "ASK"]
_LIVE_QUOTE_PRICE_KINDS: tuple[str, ...] = get_args(LiveQuotePriceKind)


def valuation_from_live_quote(
    quote, *, price_kind: LiveQuotePriceKind = "LAST_PRICE", valuation_at: "dt.datetime | None" = None,
) -> "LegValuation | None":
    if price_kind not in _LIVE_QUOTE_PRICE_KINDS:
        raise ValueError(f"price_kind must be one of {_LIVE_QUOTE_PRICE_KINDS}, got {price_kind!r}")
    if quote.status != "QUOTED":
        return None
    price = {"LAST_PRICE": quote.last_price, "BID": quote.bid, "ASK": quote.ask}[price_kind]
    if price is None or price <= 0:
        return None
    v_at = valuation_at if valuation_at is not None else quote.observed_at
    if v_at.tzinfo is None:
        raise ValueError("valuation_at must be timezone-aware")
    if v_at < quote.observed_at:
        raise ValueError(f"valuation_at ({v_at}) precedes this quote's observed_at ({quote.observed_at}) - no look-ahead")
    return LegValuation(
        contract=quote.contract, price=price, source="OBSERVED_MARKET_PRICE", valuation_at=v_at,
        detail=f"live quote price_kind={price_kind!r} source={quote.source!r} status={quote.status!r}",
    )


@dataclasses.dataclass(frozen=True, slots=True)
class LegMTMResult:

    leg: OptionLeg
    valuation: LegValuation
    mtm_pnl: Decimal
    entry_and_valuation_share_same_availability_instant: bool


def mark_to_market_leg(leg: OptionLeg, valuation: LegValuation) -> LegMTMResult:
    if valuation.contract != leg.contract:
        raise ValueError(f"valuation is for {valuation.contract} but leg is for {leg.contract}")
    if leg.entry_observed_at is not None and valuation.valuation_at < leg.entry_observed_at:
        raise ValueError(f"valuation_at ({valuation.valuation_at}) precedes entry_observed_at ({leg.entry_observed_at}) - impossible chronology")
    same_instant = leg.entry_observed_at is not None and valuation.valuation_at == leg.entry_observed_at
    pnl = Decimal(leg.quantity_sign) * (valuation.price - leg.entry_price) * leg.total_units
    return LegMTMResult(leg=leg, valuation=valuation, mtm_pnl=pnl, entry_and_valuation_share_same_availability_instant=same_instant)


@dataclasses.dataclass(frozen=True, slots=True)
class StructureMTMResult:
    structure: OptionStructure
    leg_results: tuple[LegMTMResult, ...]
    mtm_pnl: Decimal


def mark_to_market_structure(structure: OptionStructure, valuations: Mapping[ContractIdentity, LegValuation]) -> StructureMTMResult:
    results = []
    for leg in structure.legs:
        v = valuations.get(leg.contract)
        if v is None:
            raise ValueError(f"no valuation supplied for {leg.contract} - every leg must be explicitly valued")
        results.append(mark_to_market_leg(leg, v))
    total = sum((r.mtm_pnl for r in results), Decimal(0))
    return StructureMTMResult(structure=structure, leg_results=tuple(results), mtm_pnl=total)


CostKind = Literal["BROKERAGE", "EXCHANGE_TRANSACTION_CHARGE", "STT", "GST", "STAMP_DUTY", "SEBI_CHARGE", "OTHER"]
_COST_KINDS: tuple[str, ...] = get_args(CostKind)

SlippageModel = Literal["FIXED_PRICE", "BASIS_POINTS"]
_SLIPPAGE_MODELS: tuple[str, ...] = get_args(SlippageModel)


@dataclasses.dataclass(frozen=True, slots=True)
class CostItem:

    kind: CostKind
    amount: Decimal
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _COST_KINDS:
            raise ValueError(f"kind must be one of {_COST_KINDS}, got {self.kind!r}")
        if self.amount < 0:
            raise ValueError(f"amount must not be negative, got {self.amount!r} - a rebate/credit is not representable as a negative cost in O2-E")


@dataclasses.dataclass(frozen=True, slots=True)
class SlippageAssumption:

    model: SlippageModel
    value: Decimal
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.model not in _SLIPPAGE_MODELS:
            raise ValueError(f"model must be one of {_SLIPPAGE_MODELS}, got {self.model!r}")
        if self.value < 0:
            raise ValueError(f"value must not be negative, got {self.value!r}")


def slippage_cost_per_unit(assumption: SlippageAssumption, reference_price: Decimal) -> Decimal:
    if reference_price <= 0:
        raise ValueError(f"reference_price must be positive, got {reference_price!r}")
    if assumption.model == "FIXED_PRICE":
        return assumption.value
    return reference_price * assumption.value / Decimal(10000)


def slippage_amount(leg: OptionLeg, assumption: SlippageAssumption, reference_price: Decimal) -> Decimal:
    return slippage_cost_per_unit(assumption, reference_price) * leg.total_units


@dataclasses.dataclass(frozen=True, slots=True)
class LegCosts:

    contract: ContractIdentity
    entry_costs: tuple[CostItem, ...] = ()
    entry_slippage: Decimal = Decimal(0)
    exit_costs: tuple[CostItem, ...] = ()
    exit_slippage: Decimal = Decimal(0)

    def __post_init__(self) -> None:
        if self.entry_slippage < 0:
            raise ValueError(f"entry_slippage must not be negative, got {self.entry_slippage!r}")
        if self.exit_slippage < 0:
            raise ValueError(f"exit_slippage must not be negative, got {self.exit_slippage!r}")

    @property
    def entry_total(self) -> Decimal:
        return sum((c.amount for c in self.entry_costs), Decimal(0)) + self.entry_slippage

    @property
    def exit_total(self) -> Decimal:
        return sum((c.amount for c in self.exit_costs), Decimal(0)) + self.exit_slippage

    @property
    def total(self) -> Decimal:
        return self.entry_total + self.exit_total


@dataclasses.dataclass(frozen=True, slots=True)
class StructureCosts:
    leg_costs: tuple[LegCosts, ...]

    @property
    def entry_total(self) -> Decimal:
        return sum((lc.entry_total for lc in self.leg_costs), Decimal(0))

    @property
    def exit_total(self) -> Decimal:
        return sum((lc.exit_total for lc in self.leg_costs), Decimal(0))

    @property
    def total(self) -> Decimal:
        return sum((lc.total for lc in self.leg_costs), Decimal(0))


def structure_costs(structure: OptionStructure, costs_by_contract: Mapping[ContractIdentity, LegCosts]) -> StructureCosts:
    per_leg = []
    for leg in structure.legs:
        lc = costs_by_contract.get(leg.contract)
        if lc is None:
            raise ValueError(f"no costs supplied for {leg.contract} - every leg must have explicit costs (zero costs still require an explicit LegCosts)")
        per_leg.append(lc)
    return StructureCosts(leg_costs=tuple(per_leg))


def net_pnl(gross_pnl: Decimal, costs_total: Decimal) -> Decimal:
    return gross_pnl - costs_total
