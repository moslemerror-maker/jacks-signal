from __future__ import annotations

import dataclasses
import math
from decimal import Decimal
from statistics import NormalDist
from typing import Callable, Mapping, Protocol, runtime_checkable

from .structures import OptionStructure, structure_entry_cash_flow, structure_expiry_payoff, structure_expiry_pnl

_STD_NORMAL = NormalDist()


@runtime_checkable
class SettlementDistribution(Protocol):

    def probability_above(self, level: Decimal) -> Decimal: ...
    def probability_below(self, level: Decimal) -> Decimal: ...
    def probability_between(self, lo: Decimal, hi: Decimal) -> Decimal: ...
    def expected_value(self, payoff: Callable[[Decimal], Decimal]) -> Decimal:
        pass


@dataclasses.dataclass(frozen=True, slots=True)
class Black76RiskNeutralDistribution:

    forward: Decimal
    volatility: Decimal
    time_to_expiry_years: Decimal

    def __post_init__(self) -> None:
        if self.forward <= 0:
            raise ValueError(f"forward must be positive, got {self.forward!r}")
        if self.volatility <= 0:
            raise ValueError(f"volatility must be positive, got {self.volatility!r}")
        if self.time_to_expiry_years <= 0:
            raise ValueError(f"time_to_expiry_years must be positive, got {self.time_to_expiry_years!r} - at/after expiry there is no distribution left to model")

    def _mu_sigma(self) -> tuple[float, float]:
        forward, vol, t = float(self.forward), float(self.volatility), float(self.time_to_expiry_years)
        sigma = vol * math.sqrt(t)
        mu = math.log(forward) - 0.5 * vol * vol * t
        return mu, sigma

    def probability_above(self, level: Decimal) -> Decimal:
        if level <= 0:
            return Decimal(1)
        mu, sigma = self._mu_sigma()
        z = (math.log(float(level)) - mu) / sigma
        return Decimal(repr(1.0 - _STD_NORMAL.cdf(z)))

    def probability_below(self, level: Decimal) -> Decimal:
        return Decimal(1) - self.probability_above(level)

    def probability_between(self, lo: Decimal, hi: Decimal) -> Decimal:
        if hi < lo:
            raise ValueError(f"hi ({hi}) must be >= lo ({lo})")
        return self.probability_above(lo) - self.probability_above(hi)

    def expected_value(self, payoff: Callable[[Decimal], Decimal]) -> Decimal:
        mu, sigma = self._mu_sigma()
        lo, hi, n = -10.0, 10.0, 1000
        h = (hi - lo) / n

        def f(z: float) -> float:
            forward_t = math.exp(mu + sigma * z)
            return float(payoff(Decimal(repr(forward_t)))) * _STD_NORMAL.pdf(z)

        total = f(lo) + f(hi)
        for i in range(1, n):
            z = lo + i * h
            total += (4.0 if i % 2 else 2.0) * f(z)
        return Decimal(repr(total * h / 3.0))


def _profitable_regions(structure: OptionStructure) -> list[tuple[Decimal, Decimal | None]]:
    strikes = sorted({Decimal(leg.contract.strike_paise) / 100 for leg in structure.legs})  # type: ignore[arg-type]
    xs = sorted({Decimal(0)} | {k for k in strikes if k > 0})
    ys = [structure_expiry_pnl(structure, x) for x in xs]
    regions: list[tuple[Decimal, Decimal | None]] = []
    for i in range(len(xs) - 1):
        x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
        if y0 > 0 and y1 > 0:
            regions.append((x0, x1))
        elif y0 <= 0 and y1 <= 0:
            continue
        else:
            root = x0 + (x1 - x0) * (-y0) / (y1 - y0)
            regions.append((x0, root) if y0 > 0 else (root, x1))
    x_last, y_last = xs[-1], ys[-1]
    probe_gap = Decimal(1000)
    y_probe = structure_expiry_pnl(structure, x_last + probe_gap)
    slope = (y_probe - y_last) / probe_gap
    if slope == 0:
        if y_last > 0:
            regions.append((x_last, None))
    elif slope > 0:
        regions.append((x_last, None) if y_last > 0 else (x_last + (-y_last) / slope, None))
    else:
        if y_last > 0:
            regions.append((x_last, x_last + (-y_last) / slope))
    return regions


def probability_positive_net_pnl_by_expiry(structure: OptionStructure, distribution: SettlementDistribution) -> Decimal:
    total = Decimal(0)
    for lo, hi in _profitable_regions(structure):
        total += distribution.probability_above(lo) if hi is None else distribution.probability_between(lo, hi)
    return total


def structure_expected_payoff(structure: OptionStructure, distribution: SettlementDistribution) -> Decimal:
    return distribution.expected_value(lambda s: structure_expiry_payoff(structure, s))


def structure_expected_value(
    structure: OptionStructure, distribution: SettlementDistribution, risk_free_rate: Decimal, time_to_expiry_years: Decimal,
) -> Decimal:
    if risk_free_rate < 0:
        raise ValueError(f"risk_free_rate must not be negative, got {risk_free_rate!r}")
    if time_to_expiry_years <= 0:
        raise ValueError(f"time_to_expiry_years must be positive, got {time_to_expiry_years!r}")
    discount = Decimal(repr(math.exp(-float(risk_free_rate) * float(time_to_expiry_years))))
    expected_payoff = structure_expected_payoff(structure, distribution)
    return discount * expected_payoff + structure_entry_cash_flow(structure)


def probability_metrics_from_distribution(structure: OptionStructure, distribution: SettlementDistribution) -> Mapping[str, Decimal]:
    return {"POSITIVE_NET_PNL_BY_EXPIRY": probability_positive_net_pnl_by_expiry(structure, distribution)}
