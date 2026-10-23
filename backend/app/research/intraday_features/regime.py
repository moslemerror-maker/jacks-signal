from __future__ import annotations

from typing import Sequence

from .bar_math import Bar
from .price_action import range_position, return_over_bars
from .volatility_volume import realized_volatility, volatility_expansion_ratio

TREND_RETURN_THRESHOLD = 1.0
EXPANSION_RATIO_THRESHOLD = 1.3
COMPRESSION_RATIO_THRESHOLD = 0.7
RANGE_POSITION_EXTREME = 0.15

REGIMES = ("TREND_UP", "TREND_DOWN", "HIGH_VOL_EXPANSION", "LOW_VOL_COMPRESSION", "RANGE")


def intraday_regime(bars_up_to_t: Sequence[Bar], *, momentum_n_bars: int = 15, vol_n_bars: int = 15, expansion_short_n: int = 10, expansion_long_n: int = 60) -> str | None:
    vol = realized_volatility(bars_up_to_t, vol_n_bars)
    ret = return_over_bars(bars_up_to_t, momentum_n_bars)
    expansion = volatility_expansion_ratio(bars_up_to_t, expansion_short_n, expansion_long_n)
    if vol is None or ret is None or expansion is None:
        return None

    if expansion >= EXPANSION_RATIO_THRESHOLD:
        return "HIGH_VOL_EXPANSION"
    if expansion <= COMPRESSION_RATIO_THRESHOLD:
        return "LOW_VOL_COMPRESSION"

    normalized_return = ret / vol if vol else 0.0
    if normalized_return >= TREND_RETURN_THRESHOLD:
        return "TREND_UP"
    if normalized_return <= -TREND_RETURN_THRESHOLD:
        return "TREND_DOWN"

    return "RANGE"
