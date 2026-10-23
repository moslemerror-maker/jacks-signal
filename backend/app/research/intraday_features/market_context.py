from __future__ import annotations

from typing import Mapping, Sequence

from .bar_math import mean


def universe_average_return(returns_at_t: Mapping[str, float]) -> float | None:
    if not returns_at_t:
        return None
    return mean(list(returns_at_t.values()))


def universe_breadth(returns_at_t: Mapping[str, float]) -> float | None:
    if not returns_at_t:
        return None
    return sum(1 for v in returns_at_t.values() if v > 0) / len(returns_at_t)


def relative_strength_vs_universe(stock_return: float, universe_avg_return: float | None) -> float | None:
    if universe_avg_return is None:
        return None
    return stock_return - universe_avg_return


def sector_relative_strength(stock_return: float, sector_peer_returns: Sequence[float]) -> float | None:
    peer_mean = mean(list(sector_peer_returns))
    if peer_mean is None:
        return None
    return stock_return - peer_mean


def relative_strength_acceleration(rs_now: float | None, rs_earlier: float | None) -> float | None:
    if rs_now is None or rs_earlier is None:
        return None
    return rs_now - rs_earlier
