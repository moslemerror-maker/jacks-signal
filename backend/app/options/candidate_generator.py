from __future__ import annotations

import dataclasses
import datetime as dt
from decimal import Decimal
from typing import Literal, Sequence, get_args

from .availability import session_close_utc
from .forward import ForwardResolution
from .liquidity import LiquidityAssessment, assess_leg_liquidity
from .live_quote import LiveQuote
from .recommendation import CandidateStrategy
from .structures import OptionStructure, leg_from_live_quote, structure_break_even, structure_max_loss, structure_max_profit
from .valuation import Greeks, ImpliedVolatilityResult, ValuationInputs, black76_greeks, black76_implied_volatility

CandidateShape = Literal["LONG_CALL_ATM", "LONG_PUT_ATM", "LONG_STRADDLE_ATM"]
_SHAPES: tuple[str, ...] = get_args(CandidateShape)


@dataclasses.dataclass(frozen=True, slots=True)
class CandidateGenerationOutcome:
    shape: CandidateShape
    candidate: CandidateStrategy | None
    rejected_reason: str | None
    leg_quotes: tuple[LiveQuote, ...] = ()
    leg_liquidity: tuple[LiquidityAssessment, ...] = ()

    def __post_init__(self) -> None:
        if self.shape not in _SHAPES:
            raise ValueError(f"shape must be one of {_SHAPES}, got {self.shape!r}")
        if (self.candidate is None) == (self.rejected_reason is None):
            raise ValueError("exactly one of candidate/rejected_reason must be set")
        if self.candidate is not None:
            n = len(self.candidate.structure.legs)
            if self.leg_quotes and len(self.leg_quotes) != n:
                raise ValueError(f"leg_quotes has {len(self.leg_quotes)} entries but the candidate has {n} legs (must be empty or one per leg)")
            if self.leg_liquidity and len(self.leg_liquidity) != n:
                raise ValueError(f"leg_liquidity has {len(self.leg_liquidity)} entries but the candidate has {n} legs (must be empty or one per leg)")


@dataclasses.dataclass(frozen=True, slots=True)
class CandidateGenerationConfig:

    lots: int
    lot_size: int
    lot_size_source: str
    risk_free_rate: Decimal
    max_quote_age: dt.timedelta
    max_spread_fraction: Decimal
    min_volume: int | None


def _nearest_usable_atm_strikes(quotes: Sequence[LiveQuote], forward: Decimal, n: int = 3) -> list[int]:
    usable_strikes = sorted({q.contract.strike_paise for q in quotes if q.is_usable})
    return sorted(usable_strikes, key=lambda sp: abs(Decimal(sp) / 100 - forward))[:n]


def _quote_for(quotes: Sequence[LiveQuote], strike_paise: int, option_type: str) -> LiveQuote | None:
    for q in quotes:
        if q.contract.strike_paise == strike_paise and q.contract.option_type == option_type:
            return q
    return None


def _leg_iv_and_greeks(quote: LiveQuote, forward: Decimal, expiry_at: dt.datetime, risk_free_rate: Decimal) -> tuple[ImpliedVolatilityResult, Greeks | None]:
    inputs = ValuationInputs(
        option_type=quote.contract.option_type, forward=forward, strike=Decimal(quote.contract.strike_paise) / 100,
        valuation_at=quote.observed_at, expiry_at=expiry_at, risk_free_rate=risk_free_rate,
    )
    iv_result = black76_implied_volatility(inputs, quote.last_price)
    if iv_result.status != "OK":
        return iv_result, None
    return iv_result, black76_greeks(inputs, iv_result.iv)


def _build_structure_candidate(
    *, shape: CandidateShape, legs_and_quotes: list[tuple], forward_resolution: ForwardResolution, config: CandidateGenerationConfig,
    expiry: dt.date, timestamp: dt.datetime, rationale: str,
) -> CandidateGenerationOutcome:
    bad = [(q.contract, la) for _, q, la in legs_and_quotes if not la.is_acceptable]
    if bad:
        reasons = "; ".join(f"{c}: {', '.join(la.reasons)}" for c, la in bad)
        return CandidateGenerationOutcome(shape=shape, candidate=None, rejected_reason=f"liquidity/data-quality rejection - {reasons}")

    legs = tuple(leg_from_live_quote(q, side=side, lots=config.lots, lot_size=config.lot_size, lot_size_source=config.lot_size_source) for side, q, _ in legs_and_quotes)
    structure = OptionStructure(legs)
    expiry_at = session_close_utc(expiry)

    iv_context: list[ImpliedVolatilityResult] = []
    greeks_context: list[Greeks] = []
    all_iv_ok = True
    for _, q, _ in legs_and_quotes:
        iv_result, greeks = _leg_iv_and_greeks(q, forward_resolution.forward, expiry_at, config.risk_free_rate)
        iv_context.append(iv_result)
        if greeks is None:
            all_iv_ok = False
        else:
            greeks_context.append(greeks)

    data_quality = "EOD_RESOLVED" if all_iv_ok else "UNAVAILABLE_INSUFFICIENT_DATA"
    candidate = CandidateStrategy(
        timestamp=timestamp, structure=structure, rationale=rationale, market_context=None,
        iv_context=tuple(iv_context) if all_iv_ok else (), greeks_context=tuple(greeks_context) if all_iv_ok else (),
        max_profit=structure_max_profit(structure), max_loss=structure_max_loss(structure), break_even=structure_break_even(structure),
        target=None, stop=None, time_exit_at=None, expected_holding_period=None,
        required_probability_propositions=("POSITIVE_NET_PNL_BY_EXPIRY",), expected_value_inputs=None,
        data_quality=data_quality, forward_resolution=forward_resolution,
    )
    leg_quotes = tuple(q for _, q, _ in legs_and_quotes)
    leg_liquidity = tuple(la for _, _, la in legs_and_quotes)
    return CandidateGenerationOutcome(shape=shape, candidate=candidate, rejected_reason=None, leg_quotes=leg_quotes, leg_liquidity=leg_liquidity)


def generate_candidates(
    quotes: Sequence[LiveQuote], *, expiry: dt.date, forward_resolution: ForwardResolution, config: CandidateGenerationConfig,
    timestamp: dt.datetime, as_of: dt.datetime,
) -> tuple[CandidateGenerationOutcome, ...]:
    strikes = _nearest_usable_atm_strikes(quotes, forward_resolution.forward)
    if not strikes:
        reason = "no strike in the supplied chain has any usable (QUOTED) leg"
        return tuple(CandidateGenerationOutcome(shape=s, candidate=None, rejected_reason=reason) for s in _SHAPES)  # type: ignore[arg-type]

    outcomes: list[CandidateGenerationOutcome] = []
    for shape in _SHAPES:
        built: CandidateGenerationOutcome | None = None
        last_reason = "no usable strike/leg combination found for this shape"
        for strike_paise in strikes:
            ce = _quote_for(quotes, strike_paise, "CE")
            pe = _quote_for(quotes, strike_paise, "PE")
            if shape == "LONG_CALL_ATM":
                if ce is None:
                    last_reason = f"no CE quote at strike {strike_paise}"
                    continue
                la = assess_leg_liquidity(ce, as_of=as_of, max_quote_age=config.max_quote_age, max_spread_fraction=config.max_spread_fraction, min_volume=config.min_volume)
                outcome = _build_structure_candidate(shape=shape, legs_and_quotes=[("BUY", ce, la)], forward_resolution=forward_resolution, config=config, expiry=expiry, timestamp=timestamp, rationale=f"single-leg long ATM call, strike {strike_paise}")
            elif shape == "LONG_PUT_ATM":
                if pe is None:
                    last_reason = f"no PE quote at strike {strike_paise}"
                    continue
                la = assess_leg_liquidity(pe, as_of=as_of, max_quote_age=config.max_quote_age, max_spread_fraction=config.max_spread_fraction, min_volume=config.min_volume)
                outcome = _build_structure_candidate(shape=shape, legs_and_quotes=[("BUY", pe, la)], forward_resolution=forward_resolution, config=config, expiry=expiry, timestamp=timestamp, rationale=f"single-leg long ATM put, strike {strike_paise}")
            else:
                if ce is None or pe is None:
                    last_reason = f"strike {strike_paise} missing {'CE' if ce is None else 'PE'} quote"
                    continue
                la_ce = assess_leg_liquidity(ce, as_of=as_of, max_quote_age=config.max_quote_age, max_spread_fraction=config.max_spread_fraction, min_volume=config.min_volume)
                la_pe = assess_leg_liquidity(pe, as_of=as_of, max_quote_age=config.max_quote_age, max_spread_fraction=config.max_spread_fraction, min_volume=config.min_volume)
                outcome = _build_structure_candidate(shape=shape, legs_and_quotes=[("BUY", ce, la_ce), ("BUY", pe, la_pe)], forward_resolution=forward_resolution, config=config, expiry=expiry, timestamp=timestamp, rationale=f"long ATM straddle, strike {strike_paise}")
            if outcome.candidate is not None:
                built = outcome
                break
            last_reason = outcome.rejected_reason or last_reason
        outcomes.append(built if built is not None else CandidateGenerationOutcome(shape=shape, candidate=None, rejected_reason=last_reason))
    return tuple(outcomes)
