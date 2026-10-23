from __future__ import annotations

from typing import Sequence

from .bar_math import Bar


def relative_strength_index(bars_up_to_t: Sequence[Bar], period: int) -> float | None:
    if len(bars_up_to_t) < period + 1:
        return None
    window = bars_up_to_t[-period - 1 :]
    deltas = [window[i + 1].close - window[i].close for i in range(len(window) - 1)]
    gains = [d for d in deltas if d > 0]
    losses = [-d for d in deltas if d < 0]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def rsi_slope(bars_up_to_t: Sequence[Bar], period: int, slope_bars: int) -> float | None:
    if len(bars_up_to_t) < period + 1 + slope_bars:
        return None
    now = relative_strength_index(bars_up_to_t, period)
    then = relative_strength_index(bars_up_to_t[: len(bars_up_to_t) - slope_bars], period)
    if now is None or then is None:
        return None
    return now - then


def _ema_series(values: Sequence[float], span: int) -> list[float]:
    alpha = 2.0 / (span + 1.0)
    out = [float(values[0])]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def ema(bars_up_to_t: Sequence[Bar], span: int) -> float | None:
    if len(bars_up_to_t) < span:
        return None
    closes = [b.close for b in bars_up_to_t]
    return _ema_series(closes, span)[-1]


def macd_histogram(bars_up_to_t: Sequence[Bar], fast: int = 12, slow: int = 26, signal: int = 9) -> float | None:
    if len(bars_up_to_t) < slow + signal:
        return None
    closes = [b.close for b in bars_up_to_t]
    ema_fast = _ema_series(closes, fast)
    ema_slow = _ema_series(closes, slow)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = _ema_series(macd_line, signal)
    return macd_line[-1] - signal_line[-1]
