from __future__ import annotations

import dataclasses
from typing import Sequence

from .bar_math import Bar

DEFAULT_THRESHOLD = 0.5

UP, DOWN, NEUTRAL = "UP", "DOWN", "NEUTRAL"


def _session_bounded_future_window(current_trading_date, future_bars: Sequence[Bar], horizon_bars: int) -> list[Bar] | None:
    if len(future_bars) < horizon_bars:
        return None
    window = list(future_bars[:horizon_bars])
    if any(b.trading_date != current_trading_date for b in window):
        return None
    return window


@dataclasses.dataclass(frozen=True)
class ForwardMovementLabel:
    horizon_bars: int
    reference_price: float
    future_close: float
    move: float
    vol_normalizer: float
    normalized_move: float
    threshold: float
    label_class: str


def forward_movement_label(
    bars_up_to_t: Sequence[Bar], future_bars: Sequence[Bar], *, horizon_bars: int, vol_normalizer: float | None, threshold: float = DEFAULT_THRESHOLD,
) -> ForwardMovementLabel | None:
    if not bars_up_to_t or not vol_normalizer:
        return None
    current = bars_up_to_t[-1]
    window = _session_bounded_future_window(current.trading_date, future_bars, horizon_bars)
    if window is None:
        return None
    reference_price = current.close
    future_close = window[-1].close
    move = (future_close / reference_price) - 1.0
    normalized_move = move / vol_normalizer
    if normalized_move >= threshold:
        label_class = UP
    elif normalized_move <= -threshold:
        label_class = DOWN
    else:
        label_class = NEUTRAL
    return ForwardMovementLabel(horizon_bars, reference_price, future_close, move, vol_normalizer, normalized_move, threshold, label_class)


@dataclasses.dataclass(frozen=True)
class ExcursionLabel:
    horizon_bars: int
    reference_price: float
    max_upside_excursion: float
    max_downside_excursion: float


def excursion_label(bars_up_to_t: Sequence[Bar], future_bars: Sequence[Bar], *, horizon_bars: int) -> ExcursionLabel | None:
    if not bars_up_to_t:
        return None
    current = bars_up_to_t[-1]
    window = _session_bounded_future_window(current.trading_date, future_bars, horizon_bars)
    if window is None:
        return None
    reference_price = current.close
    max_high = max(b.high for b in window)
    min_low = min(b.low for b in window)
    return ExcursionLabel(
        horizon_bars=horizon_bars, reference_price=reference_price,
        max_upside_excursion=max(0.0, (max_high - reference_price) / reference_price),
        max_downside_excursion=max(0.0, (reference_price - min_low) / reference_price),
    )


@dataclasses.dataclass(frozen=True)
class EntryPathResult:
    entry_offset_bars: int
    horizon_bars: int
    entry_price: float
    gross_return: float
    max_upside_excursion: float
    max_downside_excursion: float


def entry_path_result(
    bars_up_to_t: Sequence[Bar], future_bars: Sequence[Bar], *, entry_offset_bars: int, horizon_bars: int,
) -> EntryPathResult | None:
    if not bars_up_to_t or entry_offset_bars < 0 or horizon_bars < 1:
        return None
    current = bars_up_to_t[-1]
    if entry_offset_bars == 0:
        entry_price = current.close
    else:
        if len(future_bars) < entry_offset_bars:
            return None
        entry_bar = future_bars[entry_offset_bars - 1]
        if entry_bar.trading_date != current.trading_date:
            return None
        entry_price = entry_bar.open
    window_start = entry_offset_bars
    if len(future_bars) < window_start + horizon_bars:
        return None
    window = future_bars[window_start:window_start + horizon_bars]
    if any(b.trading_date != current.trading_date for b in window):
        return None
    future_close = window[-1].close
    max_high = max(b.high for b in window)
    min_low = min(b.low for b in window)
    return EntryPathResult(
        entry_offset_bars=entry_offset_bars, horizon_bars=horizon_bars, entry_price=entry_price,
        gross_return=(future_close / entry_price) - 1.0,
        max_upside_excursion=max(0.0, (max_high - entry_price) / entry_price),
        max_downside_excursion=max(0.0, (entry_price - min_low) / entry_price),
    )


@dataclasses.dataclass(frozen=True)
class NextBarEntryPathResult:
    entry_price_basis: str
    horizon_bars: int
    entry_price: float
    gross_return: float
    max_upside_excursion: float
    max_downside_excursion: float


def next_bar_entry_path_result(
    bars_up_to_t: Sequence[Bar], future_bars: Sequence[Bar], *, horizon_bars: int, price_basis: str,
) -> NextBarEntryPathResult | None:
    if price_basis not in ("close", "vwap_proxy"):
        raise ValueError(f"price_basis must be 'close' or 'vwap_proxy', got {price_basis!r}")
    if not bars_up_to_t or horizon_bars < 1 or len(future_bars) < 1:
        return None
    current = bars_up_to_t[-1]
    entry_bar = future_bars[0]
    if entry_bar.trading_date != current.trading_date:
        return None
    entry_price = entry_bar.close if price_basis == "close" else (entry_bar.high + entry_bar.low + entry_bar.close) / 3.0
    if len(future_bars) < 1 + horizon_bars:
        return None
    window = future_bars[1:1 + horizon_bars]
    if any(b.trading_date != current.trading_date for b in window):
        return None
    future_close = window[-1].close
    max_high = max(b.high for b in window)
    min_low = min(b.low for b in window)
    return NextBarEntryPathResult(
        entry_price_basis=price_basis, horizon_bars=horizon_bars, entry_price=entry_price,
        gross_return=(future_close / entry_price) - 1.0,
        max_upside_excursion=max(0.0, (max_high - entry_price) / entry_price),
        max_downside_excursion=max(0.0, (entry_price - min_low) / entry_price),
    )
