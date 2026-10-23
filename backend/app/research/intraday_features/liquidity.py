from __future__ import annotations

import statistics
from typing import Sequence


def rolling_median_daily_traded_value(daily_values: Sequence[float], *, window: int = 20) -> list[float | None]:
    if window < 1:
        raise ValueError("window must be >= 1")
    out: list[float | None] = []
    for i in range(len(daily_values)):
        if i < window:
            out.append(None)
        else:
            out.append(statistics.median(daily_values[i - window:i]))
    return out
