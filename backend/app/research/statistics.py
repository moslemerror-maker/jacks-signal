from __future__ import annotations

import dataclasses
import math
import statistics as pystats
from typing import Sequence

_NORMAL = pystats.NormalDist()


@dataclasses.dataclass(frozen=True)
class MeanTestResult:

    method: str
    n: int
    mean: float
    standard_error: float
    test_statistic: float
    p_value: float
    confidence_interval: tuple[float, float]
    confidence_level: float


def _z_and_two_sided_p(mean: float, se: float) -> tuple[float, float]:
    if se == 0.0:
        z = math.inf if mean > 0 else (-math.inf if mean < 0 else 0.0)
        p_value = 0.0 if mean != 0 else 1.0
        return z, p_value
    z = mean / se
    p_value = 2.0 * (1.0 - _NORMAL.cdf(abs(z)))
    return z, p_value


def _confidence_interval(
    mean: float, se: float, confidence_level: float
) -> tuple[float, float]:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
    alpha = 1.0 - confidence_level
    z_crit = _NORMAL.inv_cdf(1.0 - alpha / 2.0)
    return (mean - z_crit * se, mean + z_crit * se)


def _population_variance(values: Sequence[float]) -> float:
    n = len(values)
    mean = pystats.fmean(values)
    return sum((v - mean) ** 2 for v in values) / n


def _newey_west_long_run_variance(values: Sequence[float], max_lag: int) -> float:
    n = len(values)
    mean = pystats.fmean(values)
    deviations = [v - mean for v in values]
    gamma_0 = sum(d * d for d in deviations) / n
    long_run_variance = gamma_0
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1)
        gamma_lag = sum(deviations[i] * deviations[i - lag] for i in range(lag, n)) / n
        long_run_variance += 2.0 * weight * gamma_lag
    return long_run_variance


def newey_west_mean_test(
    values: Sequence[float], max_lag: int, confidence_level: float = 0.95
) -> MeanTestResult:
    n = len(values)
    if n < 2:
        raise ValueError("need at least 2 observations")
    if max_lag < 0 or max_lag >= n:
        raise ValueError(f"max_lag must be in [0, {n - 1}], got {max_lag}")

    mean = pystats.fmean(values)
    long_run_variance = _newey_west_long_run_variance(values, max_lag)
    se = math.sqrt(long_run_variance / n)
    z, p_value = _z_and_two_sided_p(mean, se)
    ci = _confidence_interval(mean, se, confidence_level)

    return MeanTestResult(
        method="newey_west",
        n=n,
        mean=mean,
        standard_error=se,
        test_statistic=z,
        p_value=p_value,
        confidence_interval=ci,
        confidence_level=confidence_level,
    )


def non_overlapping_subsample_mean_test(
    values: Sequence[float], stride: int, confidence_level: float = 0.95
) -> MeanTestResult:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    subsample = list(values[::stride])
    n = len(subsample)
    if n < 2:
        raise ValueError(
            f"non-overlapping subsample has only {n} observation(s) with "
            f"stride={stride} on {len(values)} input values - not enough to test"
        )
    mean = pystats.fmean(subsample)
    variance = _population_variance(subsample)
    se = math.sqrt(variance / n)
    z, p_value = _z_and_two_sided_p(mean, se)
    ci = _confidence_interval(mean, se, confidence_level)

    return MeanTestResult(
        method="non_overlapping_subsample",
        n=n,
        mean=mean,
        standard_error=se,
        test_statistic=z,
        p_value=p_value,
        confidence_interval=ci,
        confidence_level=confidence_level,
    )


def naive_iid_mean_test(
    values: Sequence[float], confidence_level: float = 0.95
) -> MeanTestResult:
    n = len(values)
    if n < 2:
        raise ValueError("need at least 2 observations")
    mean = pystats.fmean(values)
    variance = _population_variance(values)
    se = math.sqrt(variance / n)
    z, p_value = _z_and_two_sided_p(mean, se)
    ci = _confidence_interval(mean, se, confidence_level)

    return MeanTestResult(
        method="naive_iid",
        n=n,
        mean=mean,
        standard_error=se,
        test_statistic=z,
        p_value=p_value,
        confidence_interval=ci,
        confidence_level=confidence_level,
    )
