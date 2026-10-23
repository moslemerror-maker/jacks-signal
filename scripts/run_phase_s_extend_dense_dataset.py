from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PHASE_R_DENSE = ROOT / "data" / "derived" / "phase_r_cross_sectional" / "dense_symbol_features.parquet"
OUT = ROOT / "data" / "derived" / "phase_s_confirmation" / "dense_symbol_features_extended.parquet"

RSI_PERIODS = (9, 14)
RSI_SLOPE_PERIOD, RSI_SLOPE_BARS = 14, 5
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
EMA_SPANS = (9, 20, 50)
EMA20_SLOPE_BARS = 3
REALIZED_VOL_WINDOW = 20


def compute_extension_for_symbol(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars[["bar_start_utc"]].copy()
    close, open_, high, low = bars["close"], bars["open"], bars["high"], bars["low"]
    day = bars["trading_date"]

    delta = close.diff()
    gains, losses = delta.clip(lower=0.0), (-delta).clip(lower=0.0)
    for period in RSI_PERIODS:
        avg_gain = gains.rolling(period).mean()
        avg_loss = losses.rolling(period).mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = avg_gain / avg_loss
            rsi = 100.0 - 100.0 / (1.0 + rs)
        flat_case = np.where(avg_gain > 0, 100.0, 50.0)
        rsi = pd.Series(np.where(avg_loss != 0, rsi, flat_case), index=bars.index)
        rsi[gains.rolling(period).count() < period] = np.nan
        out[f"rsi_{period}"] = rsi
    out["rsi_14_slope_5"] = out["rsi_14"] - out["rsi_14"].shift(RSI_SLOPE_BARS)

    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    hist = macd_line - signal_line
    hist.iloc[: MACD_SLOW + MACD_SIGNAL - 1] = np.nan
    out["macd_histogram_12_26_9"] = hist

    for span in EMA_SPANS:
        e = close.ewm(span=span, adjust=False).mean()
        e.iloc[: span - 1] = np.nan
        out[f"ema_{span}"] = e
    out["ema_20_slope_3"] = out["ema_20"] - out["ema_20"].shift(EMA20_SLOPE_BARS)

    one_bar_return = close.pct_change(1)
    out["realized_volatility_20"] = one_bar_return.rolling(REALIZED_VOL_WINDOW).std(ddof=1)
    out["distance_from_ema20_normalized"] = (close - out["ema_20"]) / out["realized_volatility_20"]

    day_high = high.groupby(day).transform("max")
    day_low = low.groupby(day).transform("min")
    day_close_last = close.groupby(day).transform("last")
    per_day = pd.DataFrame({"day": day, "day_high": day_high, "day_low": day_low, "day_close_last": day_close_last}).drop_duplicates("day").set_index("day")
    prev = per_day.shift(1)
    out["previous_day_high"] = day.map(prev["day_high"])
    out["previous_day_low"] = day.map(prev["day_low"])
    out["previous_day_close"] = day.map(prev["day_close_last"])

    out["session_high_so_far"] = high.groupby(day).cummax()
    out["session_low_so_far"] = low.groupby(day).cummin()

    first_open_per_day = open_.groupby(day).transform("first")
    out["distance_from_session_open"] = close / first_open_per_day - 1.0

    direction = pd.Series(np.sign(close - open_), index=bars.index).astype(int)
    out["candle_direction"] = direction
    run_id = (direction != direction.shift(1)).cumsum()
    count_in_run = direction.groupby(run_id).cumcount() + 1
    out["consecutive_same_direction_count"] = count_in_run.where(direction != 0, 0)

    return out


def _nearest_level(current_price: pd.Series, candidates: list[pd.Series], *, above: bool) -> pd.Series:
    stacked = pd.concat(candidates, axis=1)
    if above:
        masked = stacked.where(stacked.gt(current_price, axis=0))
        return masked.min(axis=1)
    masked = stacked.where(stacked.lt(current_price, axis=0))
    return masked.max(axis=1)


def extend_dataset() -> None:
    dense = pd.read_parquet(PHASE_R_DENSE)
    print(f"loaded Phase R dense dataset: {len(dense)} rows, {dense['security_id'].nunique()} symbols")

    t0 = time.monotonic()
    per_symbol_results = []
    for i, sid in enumerate(dense["security_id"].unique(), 1):
        sym_dense = dense[dense["security_id"] == sid].sort_values("bar_start_utc")
        bars = sym_dense[["bar_start_utc", "trading_date", "open", "high", "low", "close", "volume"]].copy()
        ext = compute_extension_for_symbol(bars)
        ext["security_id"] = sid
        per_symbol_results.append(ext)
        if i % 50 == 0:
            print(f"  {i} symbols processed, {time.monotonic() - t0:.0f}s elapsed")

    extension = pd.concat(per_symbol_results, ignore_index=True)
    merged = dense.merge(extension, on=["security_id", "bar_start_utc"], how="left")

    below_candidates = [merged["previous_day_low"], merged["session_low_so_far"], merged["opening_range_low"]]
    above_candidates = [merged["previous_day_high"], merged["session_high_so_far"], merged["opening_range_high"]]
    merged["nearest_resistance"] = _nearest_level(merged["close"], above_candidates, above=True)
    merged["nearest_support"] = _nearest_level(merged["close"], below_candidates, above=False)
    merged["distance_to_nearest_resistance"] = merged["nearest_resistance"] / merged["close"] - 1.0
    merged["distance_to_nearest_support"] = 1.0 - merged["nearest_support"] / merged["close"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT, index=False)
    print(f"wrote {len(merged)} rows, {len(merged.columns)} columns -> {OUT.relative_to(ROOT)} ({time.monotonic() - t0:.0f}s total)")


if __name__ == "__main__":
    extend_dataset()
