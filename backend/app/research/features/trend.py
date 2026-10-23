from __future__ import annotations

from typing import Sequence

from backend.app.research.features.types import Bar


def compute_sma(bars: Sequence[Bar], index: int, window: int) -> float | None:
    if index - window + 1 < 0:
        return None
    window_bars = bars[index - window + 1 : index + 1]
    return sum(b.close for b in window_bars) / window


def compute_close_vs_sma(bars: Sequence[Bar], index: int, window: int) -> float | None:
    sma = compute_sma(bars, index, window)
    if not sma:
        return None
    return bars[index].close / sma - 1.0
