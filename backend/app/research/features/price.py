from __future__ import annotations

from typing import Sequence

from backend.app.research.features.types import Bar


def compute_return_n_day(bars: Sequence[Bar], index: int, n: int) -> float | None:
    if index - n < 0:
        return None
    prior_close = bars[index - n].close
    if prior_close == 0:
        return None
    return bars[index].close / prior_close - 1.0


def _true_range(bars: Sequence[Bar], i: int) -> float:
    bar = bars[i]
    prev_close = bars[i - 1].close
    return max(
        bar.high - bar.low,
        abs(bar.high - prev_close),
        abs(bar.low - prev_close),
    )


def compute_atr(bars: Sequence[Bar], index: int, window: int) -> float | None:
    if index < window:
        return None
    trs = [_true_range(bars, i) for i in range(index - window + 1, index + 1)]
    return sum(trs) / window
