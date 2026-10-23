from __future__ import annotations

import dataclasses
import datetime as dt
from decimal import Decimal
from typing import Literal, get_args

from py_lets_be_rational.exceptions import VolatilityValueException, BelowIntrinsicException, AboveMaximumException

from .availability import session_close_utc
from .bars import DerivativeBar

PricingModel = Literal["BLACK_76"]
_PRICING_MODELS: tuple[str, ...] = get_args(PricingModel)

EuropeanOptionType = Literal["CE", "PE"]

IVStatus = Literal["OK", "UNAVAILABLE_NOT_TRADED", "UNAVAILABLE_BEFORE_KNOWN", "UNSUPPORTED_EXERCISE_STYLE",
                  "EXPIRED_OR_ZERO_TIME", "BELOW_INTRINSIC", "ABOVE_MAXIMUM", "INVALID_INPUT"]


def time_to_expiry_years(valuation_at: dt.datetime, expiry_at: dt.datetime) -> Decimal:
    if valuation_at.tzinfo is None or expiry_at.tzinfo is None:
        raise ValueError("valuation_at and expiry_at must both be timezone-aware")
    if expiry_at <= valuation_at:
        return Decimal(0)
    seconds = Decimal((expiry_at - valuation_at).total_seconds())
    return seconds / Decimal(365 * 24 * 3600)


def _flag(option_type: EuropeanOptionType) -> str:
    if option_type not in get_args(EuropeanOptionType):
        raise ValueError(f"option_type must be one of {get_args(EuropeanOptionType)} (Black-76 prices European options only), got {option_type!r}")
    return "c" if option_type == "CE" else "p"


def _validate_common(forward: Decimal, strike: Decimal, time_to_expiry_years_: Decimal, volatility: Decimal | None = None) -> None:
    if forward <= 0:
        raise ValueError(f"forward must be positive, got {forward!r}")
    if strike <= 0:
        raise ValueError(f"strike must be positive, got {strike!r}")
    if time_to_expiry_years_ < 0:
        raise ValueError(f"time_to_expiry_years must not be negative, got {time_to_expiry_years_!r}")
    if volatility is not None and volatility <= 0:
        raise ValueError(f"volatility must be positive, got {volatility!r}")


@dataclasses.dataclass(frozen=True, slots=True)
class ValuationInputs:

    option_type: EuropeanOptionType
    forward: Decimal
    strike: Decimal
    valuation_at: dt.datetime
    expiry_at: dt.datetime
    risk_free_rate: Decimal
    model: PricingModel = "BLACK_76"

    def __post_init__(self) -> None:
        if self.model not in _PRICING_MODELS:
            raise ValueError(f"model must be one of {_PRICING_MODELS}, got {self.model!r}")
        _flag(self.option_type)
        if self.valuation_at.tzinfo is None or self.expiry_at.tzinfo is None:
            raise ValueError("valuation_at and expiry_at must both be timezone-aware")
        _validate_common(self.forward, self.strike, time_to_expiry_years(self.valuation_at, self.expiry_at))

    @property
    def t(self) -> Decimal:
        return time_to_expiry_years(self.valuation_at, self.expiry_at)


@dataclasses.dataclass(frozen=True, slots=True)
class ImpliedVolatilityResult:

    status: IVStatus
    iv: Decimal | None
    market_price: Decimal | None
    forward: Decimal | None
    strike: Decimal | None
    time_to_expiry_years: Decimal | None
    risk_free_rate: Decimal | None
    model: PricingModel
    detail: str | None = None

    def __post_init__(self) -> None:
        if (self.status == "OK") != (self.iv is not None):
            raise ValueError(f"iv must be set iff status is OK (status={self.status!r}, iv={self.iv!r})")


@dataclasses.dataclass(frozen=True, slots=True)
class Greeks:

    delta: Decimal
    gamma: Decimal
    theta_per_day: Decimal
    vega_per_vol_point: Decimal
    rho_per_rate_point: Decimal
    model: PricingModel


def black76_price(inputs: ValuationInputs, volatility: Decimal) -> Decimal:
    _validate_common(inputs.forward, inputs.strike, inputs.t, volatility)
    if inputs.t == 0:
        raise ValueError("time_to_expiry_years is 0 (at or after expiry) - Black-76 has no model time value here; price intrinsic value explicitly instead of calling this function")
    from vollib.black import black as _black76
    price = _black76(_flag(inputs.option_type), float(inputs.forward), float(inputs.strike), float(inputs.t), float(inputs.risk_free_rate), float(volatility))
    return Decimal(repr(float(price)))


def black76_greeks(inputs: ValuationInputs, volatility: Decimal) -> Greeks:
    _validate_common(inputs.forward, inputs.strike, inputs.t, volatility)
    if inputs.t == 0:
        raise ValueError("time_to_expiry_years is 0 (at or after expiry) - Greeks are undefined/degenerate here; this function refuses rather than return a number")
    from vollib.black.greeks import analytical as g
    flag, F, K, t, r, sigma = _flag(inputs.option_type), float(inputs.forward), float(inputs.strike), float(inputs.t), float(inputs.risk_free_rate), float(volatility)
    def d(x: float) -> Decimal:
        return Decimal(repr(float(x)))

    return Greeks(
        delta=d(g.delta(flag, F, K, t, r, sigma)), gamma=d(g.gamma(flag, F, K, t, r, sigma)),
        theta_per_day=d(g.theta(flag, F, K, t, r, sigma)), vega_per_vol_point=d(g.vega(flag, F, K, t, r, sigma)),
        rho_per_rate_point=d(g.rho(flag, F, K, t, r, sigma)), model=inputs.model,
    )


def black76_implied_volatility(inputs: ValuationInputs, market_price: Decimal) -> ImpliedVolatilityResult:
    common = dict(market_price=market_price, forward=inputs.forward, strike=inputs.strike, time_to_expiry_years=inputs.t,
                 risk_free_rate=inputs.risk_free_rate, model=inputs.model)
    try:
        _validate_common(inputs.forward, inputs.strike, inputs.t)
    except ValueError as exc:
        return ImpliedVolatilityResult(status="INVALID_INPUT", iv=None, detail=str(exc), **common)
    if market_price <= 0:
        return ImpliedVolatilityResult(status="INVALID_INPUT", iv=None, detail="market_price must be positive", **common)
    if inputs.t == 0:
        return ImpliedVolatilityResult(status="EXPIRED_OR_ZERO_TIME", iv=None, detail="valuation_at is at or after expiry_at - no model time value", **common)
    from vollib.black.implied_volatility import implied_volatility as _iv76
    try:
        iv = _iv76(float(market_price), float(inputs.forward), float(inputs.strike), float(inputs.risk_free_rate), float(inputs.t), _flag(inputs.option_type))
    except BelowIntrinsicException as exc:
        return ImpliedVolatilityResult(status="BELOW_INTRINSIC", iv=None, detail=str(exc), **common)
    except AboveMaximumException as exc:
        return ImpliedVolatilityResult(status="ABOVE_MAXIMUM", iv=None, detail=str(exc), **common)
    except VolatilityValueException as exc:
        return ImpliedVolatilityResult(status="INVALID_INPUT", iv=None, detail=f"{type(exc).__name__}: {exc}", **common)
    return ImpliedVolatilityResult(status="OK", iv=Decimal(repr(float(iv))), **common)


def implied_volatility_from_bar(bar: DerivativeBar, *, forward: Decimal, risk_free_rate: Decimal, valuation_at: dt.datetime | None = None) -> ImpliedVolatilityResult:
    common = dict(forward=forward, strike=Decimal(bar.contract.strike_paise) / 100 if bar.contract.strike_paise is not None else None,
                 risk_free_rate=risk_free_rate, model="BLACK_76", time_to_expiry_years=None, market_price=None)
    if bar.contract.option_type not in get_args(EuropeanOptionType):
        return ImpliedVolatilityResult(status="UNSUPPORTED_EXERCISE_STYLE", iv=None, detail=f"option_type={bar.contract.option_type!r} is not European (CE/PE)", **common)
    if not bar.is_traded:
        return ImpliedVolatilityResult(status="UNAVAILABLE_NOT_TRADED", iv=None, detail=f"trade_state={bar.trade_state!r}: no observed market price", **common)
    v_at = valuation_at if valuation_at is not None else bar.available_at
    if v_at.tzinfo is None:
        return ImpliedVolatilityResult(status="INVALID_INPUT", iv=None, detail="valuation_at must be timezone-aware", **common)
    if v_at < bar.available_at:
        return ImpliedVolatilityResult(status="UNAVAILABLE_BEFORE_KNOWN", iv=None, detail=f"valuation_at ({v_at}) precedes this bar's available_at ({bar.available_at}) - no look-ahead", **common)
    expiry_at = session_close_utc(bar.contract.expiry)
    inputs = ValuationInputs(option_type=bar.contract.option_type, forward=forward, strike=common["strike"], valuation_at=v_at, expiry_at=expiry_at, risk_free_rate=risk_free_rate)
    assert bar.close is not None
    return black76_implied_volatility(inputs, bar.close)
