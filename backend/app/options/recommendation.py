from __future__ import annotations

import dataclasses
import datetime as dt
from decimal import Decimal
from typing import Literal, Mapping, get_args

from .structures import BreakEvenResult, OptionStructure, PnLExtremum, structure_entry_cash_flow
from .forward import ForwardResolution
from .mtm import LegValuation, StructureCosts
from .valuation import Greeks, ImpliedVolatilityResult

DataQuality = Literal["EOD_RESOLVED", "AMBIGUOUS_EOD_CANNOT_DETERMINE_ORDER", "UNAVAILABLE_INSUFFICIENT_DATA"]
_DATA_QUALITIES: tuple[str, ...] = get_args(DataQuality)


ExitMode = Literal["EXPIRY_EXIT", "TARGET_EXIT", "STOP_EXIT", "TIME_EXIT", "AMBIGUOUS_UNAVAILABLE", "OPEN_NOT_YET_EXITED"]
_EXIT_MODES: tuple[str, ...] = get_args(ExitMode)

_RESOLVED_EXIT_MODES = ("EXPIRY_EXIT", "TARGET_EXIT", "STOP_EXIT", "TIME_EXIT")


@dataclasses.dataclass(frozen=True, slots=True)
class StrategyOutcome:

    structure: OptionStructure
    entry_at: dt.datetime
    exit_mode: ExitMode
    exit_at: dt.datetime | None
    exit_prices: tuple[LegValuation, ...]
    gross_pnl: Decimal | None
    costs: StructureCosts | None
    net_pnl: Decimal | None
    capital_basis: CapitalBasisKind | None = None
    return_on_capital_basis: Decimal | None = None
    mfe: Decimal | None = None
    mae: Decimal | None = None
    data_quality: DataQuality = "EOD_RESOLVED"

    def __post_init__(self) -> None:
        if self.exit_mode not in _EXIT_MODES:
            raise ValueError(f"exit_mode must be one of {_EXIT_MODES}, got {self.exit_mode!r}")
        if self.data_quality not in _DATA_QUALITIES:
            raise ValueError(f"data_quality must be one of {_DATA_QUALITIES}, got {self.data_quality!r}")
        if self.entry_at.tzinfo is None:
            raise ValueError("entry_at must be timezone-aware")
        if self.exit_mode in ("AMBIGUOUS_UNAVAILABLE", "OPEN_NOT_YET_EXITED"):
            if self.exit_at is not None:
                raise ValueError(f"exit_mode {self.exit_mode!r} must not carry an exit_at (no resolved exit instant exists)")
        else:
            if self.exit_at is None:
                raise ValueError(f"exit_mode {self.exit_mode!r} requires an exit_at")
            if self.exit_at.tzinfo is None:
                raise ValueError("exit_at must be timezone-aware")
            if self.exit_at < self.entry_at:
                raise ValueError(f"exit_at ({self.exit_at}) precedes entry_at ({self.entry_at}) - impossible chronology")
        if self.exit_mode == "AMBIGUOUS_UNAVAILABLE" and self.gross_pnl is not None:
            raise ValueError("exit_mode is AMBIGUOUS_UNAVAILABLE but gross_pnl is set - an undetermined outcome must never carry an invented P&L")
        if (self.gross_pnl is None) != (self.net_pnl is None):
            raise ValueError("net_pnl must be set iff gross_pnl is set")
        structure_contracts = {leg.contract for leg in self.structure.legs}
        for v in self.exit_prices:
            if v.contract not in structure_contracts:
                raise ValueError(f"exit_prices contains a valuation for {v.contract}, which is not a leg of this structure")
        if self.mae is not None and self.mae > 0:
            raise ValueError(f"mae (maximum adverse excursion) must be <= 0, got {self.mae!r}")
        if self.mfe is not None and self.mfe < 0:
            raise ValueError(f"mfe (maximum favorable excursion) must be >= 0, got {self.mfe!r}")


ProbabilityPropositionKind = Literal[
    "POSITIVE_NET_PNL_BY_EXPIRY", "TARGET_BEFORE_STOP", "RETURN_ABOVE_THRESHOLD", "MAX_LOSS_OR_INVALIDATION",
]
_PROBABILITY_PROPOSITION_KINDS: tuple[str, ...] = get_args(ProbabilityPropositionKind)

EODSupport = Literal["EOD_SUFFICIENT", "PARTIALLY_SUPPORTED_EOD", "REQUIRES_INTRADAY"]


@dataclasses.dataclass(frozen=True, slots=True)
class ProbabilityProposition:

    kind: ProbabilityPropositionKind
    description: str
    eod_support: EODSupport
    required_observations: tuple[str, ...]
    limitation: str


PROBABILITY_PROPOSITIONS: tuple[ProbabilityProposition, ...] = (
    ProbabilityProposition(
        kind="POSITIVE_NET_PNL_BY_EXPIRY",
        description="P(net P&L > 0) when the structure is held to expiry, evaluated via O2-D expiry payoff net of O2-E costs.",
        eod_support="EOD_SUFFICIENT",
        required_observations=("entry prices for every leg (O2-B TRADED)", "expiry-day settlement/underlying close", "O2-E cost/slippage assumptions"),
        limitation="None specific to this proposition - EXPIRY_EXIT is always EOD-resolvable (docs/36 cash-settled semantics).",
    ),
    ProbabilityProposition(
        kind="TARGET_BEFORE_STOP",
        description="P(a pre-defined target level is reached before a pre-defined stop level, prior to expiry).",
        eod_support="REQUIRES_INTRADAY",
        required_observations=("an intraday price path or at minimum a reliable intrabar high/low touch sequence for every leg",),
        limitation="This project's current EOD fixture cannot order same-day target/stop touches - see ExitMode.AMBIGUOUS_UNAVAILABLE and docs/49 Part F.",
    ),
    ProbabilityProposition(
        kind="RETURN_ABOVE_THRESHOLD",
        description="P(return, on an explicit capital basis - docs/49 Part G - is >= a specified threshold) at a specific evaluation point.",
        eod_support="PARTIALLY_SUPPORTED_EOD",
        required_observations=("an explicit evaluation point (e.g. a specific EOD close or expiry)", "an explicit capital_basis (docs/49 Part G)"),
        limitation="EOD-resolvable only when evaluated AT a specific EOD close/expiry; a continuous 'at any point before expiry' version requires intraday data, same as TARGET_BEFORE_STOP.",
    ),
    ProbabilityProposition(
        kind="MAX_LOSS_OR_INVALIDATION",
        description="P(the structure reaches its defined maximum loss, or an explicit invalidation condition, before expiry).",
        eod_support="PARTIALLY_SUPPORTED_EOD",
        required_observations=("O2-D structure_max_loss (when BOUNDED)", "an explicit invalidation rule", "the same path-ordering data TARGET_BEFORE_STOP needs if checked intraday"),
        limitation="Only meaningful for structures where O2-D's structure_max_loss is BOUNDED; for UNBOUNDED structures this proposition is undefined without an explicit invalidation rule substituted for 'maximum loss'.",
    ),
)


def probability_proposition(kind: ProbabilityPropositionKind) -> ProbabilityProposition:
    for p in PROBABILITY_PROPOSITIONS:
        if p.kind == kind:
            return p
    raise ValueError(f"kind must be one of {_PROBABILITY_PROPOSITION_KINDS}, got {kind!r}")


CapitalBasisKind = Literal["NET_ENTRY_DEBIT", "MAX_DEFINED_LOSS", "STRATEGY_NOTIONAL", "MARGIN_REQUIREMENT"]
_CAPITAL_BASIS_KINDS: tuple[str, ...] = get_args(CapitalBasisKind)


def available_capital_bases(structure: OptionStructure, max_loss: PnLExtremum) -> tuple[CapitalBasisKind, ...]:
    bases: list[CapitalBasisKind] = ["STRATEGY_NOTIONAL"]
    if structure_entry_cash_flow(structure) < 0:
        bases.append("NET_ENTRY_DEBIT")
    if max_loss.boundedness == "BOUNDED":
        bases.append("MAX_DEFINED_LOSS")
    return tuple(bases)


@dataclasses.dataclass(frozen=True, slots=True)
class CandidateStrategy:

    timestamp: dt.datetime
    structure: OptionStructure
    rationale: str
    market_context: str | None
    iv_context: tuple[ImpliedVolatilityResult, ...]
    greeks_context: tuple[Greeks, ...]
    max_profit: PnLExtremum
    max_loss: PnLExtremum
    break_even: BreakEvenResult
    target: Decimal | None
    stop: Decimal | None
    time_exit_at: dt.datetime | None
    expected_holding_period: dt.timedelta | None
    required_probability_propositions: tuple[ProbabilityPropositionKind, ...]
    expected_value_inputs: str | None
    data_quality: DataQuality = "EOD_RESOLVED"
    forward_resolution: ForwardResolution | None = None

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        if self.time_exit_at is not None and self.time_exit_at.tzinfo is None:
            raise ValueError("time_exit_at must be timezone-aware")
        if self.expected_holding_period is not None and self.expected_holding_period < dt.timedelta(0):
            raise ValueError(f"expected_holding_period must not be negative, got {self.expected_holding_period!r}")
        if self.iv_context and len(self.iv_context) != len(self.structure.legs):
            raise ValueError(f"iv_context has {len(self.iv_context)} entries but structure has {len(self.structure.legs)} legs (must be empty or one per leg)")
        if self.greeks_context and len(self.greeks_context) != len(self.structure.legs):
            raise ValueError(f"greeks_context has {len(self.greeks_context)} entries but structure has {len(self.structure.legs)} legs (must be empty or one per leg)")
        for kind in self.required_probability_propositions:
            probability_proposition(kind)
        if self.data_quality not in _DATA_QUALITIES:
            raise ValueError(f"data_quality must be one of {_DATA_QUALITIES}, got {self.data_quality!r}")


RecommendationStatus = Literal["TRADE", "WAIT", "NO_TRADE"]
_RECOMMENDATION_STATUSES: tuple[str, ...] = get_args(RecommendationStatus)


@dataclasses.dataclass(frozen=True, slots=True)
class RecommendationResult:

    status: RecommendationStatus
    timestamp: dt.datetime
    underlying: str
    candidate: CandidateStrategy | None
    probability_metrics: Mapping[ProbabilityPropositionKind, Decimal] | None = None
    expected_value: Decimal | None = None
    risk: str | None = None
    liquidity_context: str | None = None
    data_quality: DataQuality = "EOD_RESOLVED"
    reasons: tuple[str, ...] = ()
    model_version: str | None = None
    strategy_version: str | None = None

    def __post_init__(self) -> None:
        if self.status not in _RECOMMENDATION_STATUSES:
            raise ValueError(f"status must be one of {_RECOMMENDATION_STATUSES}, got {self.status!r}")
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        if self.status == "TRADE" and self.candidate is None:
            raise ValueError("status TRADE requires a candidate")
        if self.status != "TRADE" and self.candidate is not None:
            raise ValueError(f"status {self.status!r} must not carry a concrete candidate structure")
        if self.probability_metrics is not None:
            for kind in self.probability_metrics:
                probability_proposition(kind)
        if self.data_quality not in _DATA_QUALITIES:
            raise ValueError(f"data_quality must be one of {_DATA_QUALITIES}, got {self.data_quality!r}")
