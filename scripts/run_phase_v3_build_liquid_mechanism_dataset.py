"""Phase V Part 3: builds the research dataset for the liquid-only continuation/expansion mechanism research.

REUSES Phase S's full dense parquet (ALL 298 symbols, ALL sessions, 5-minute grid - NOT restricted to Phase
R's extreme-mover event population, unlike Phase T/U) as the base: every column already there (returns,
relative_volume_20, RSI/MACD/EMA, VWAP, previous-day/session levels, nearest support/resistance,
gross_return_Xm/direction_label_Xm) is read verbatim, never recomputed. This script adds ONLY the columns that
do not already exist, every one a VECTORIZED reproduction of an already-existing pure function in
`backend.app.research.intraday_features`, proven bit-identical to it on synthetic data (see
backend/tests/test_phase_v3_build_liquid_mechanism_dataset.py) before being trusted here - the same discipline
every phase since Q has used.

NEW COLUMNS:
  - realized_volatility_10, realized_volatility_60, volatility_expansion_ratio, compression_then_expansion_flag
    (volatility_volume.py / mover_mechanics.py, unchanged formulas)
  - candle_body_ratio, minutes_since_open (price_action.py, unchanged formulas)
  - directional_pressure_10 (mover_mechanics.py, unchanged formula)
  - meaningful_level_up/down, breakout_up_meaningful/down_meaningful (mover_mechanics.recent_breakout_up/down,
    unchanged formula, applied to a level = max/min of previous_day_high/low and opening_range_high/low -
    both already PIT-safe existing columns)
  - normalized_move_{h}, direction_label_pit_{h} for h in (5,15,30,60): labels.py's own
    `forward_movement_label`, vol-normalized by realized_volatility_20 (already PIT-safe, already a column),
    threshold=0.5 (the function's own documented, un-tuned starting point) - the STATISTICAL target for this
    phase. Deliberately distinct from the ALREADY-EXISTING `direction_label_{h}m` (Phase K's fixed-20bps
    convention, kept as-is, read not recomputed) - this phase's own pre-registration explains why both exist
    and which is used for what.
  - rolling_liquidity_group: Phase U's EXACT pure reference function (`rolling_median_daily_traded_value`),
    applied here across the FULL universe/every session (not just Phase R's event population) - the same
    formula, a broader application.

PIT SAFETY: every new column depends only on bars_up_to_t (same symbol, at or before row T) except
normalized_move_h/direction_label_pit_h, which are LABELS (Phase G's documented, deliberate exemption - look
forward from T by construction, exactly like the already-existing gross_return_Xm/direction_label_Xm columns
this same dataset already carries).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from backend.app.config import load_dotenv
from backend.app.db.session import get_session
from backend.app.research.intraday_features.liquidity import rolling_median_daily_traded_value
from scripts.run_phase_u_build_execution_liquidity import compute_rolling_liquidity_classification

ROOT = Path(__file__).resolve().parents[1]
PHASE_S_FILE = ROOT / "data" / "derived" / "phase_s_confirmation" / "dense_symbol_features_extended.parquet"
OUT = ROOT / "data" / "derived" / "phase_v3_liquid_mechanisms" / "dense_liquid_mechanism_dataset.parquet"

HORIZONS = (5, 15, 30, 60)
BREAKOUT_WITHIN_BARS = 5
DIRECTIONAL_PRESSURE_LOOKBACK = 10
VOL_SHORT_N, VOL_LONG_N = 10, 60
COMPRESSION_THRESHOLD, EXPANSION_THRESHOLD = 0.7, 1.3
FORWARD_LABEL_THRESHOLD = 0.5
GRID_MINUTES = 5
LIQUIDITY_WINDOW_DAYS = 20


OPENING_RANGE_BARS = 15


def compute_new_columns_for_symbol(bars: pd.DataFrame) -> pd.DataFrame:
    close, open_, high, low, volume, day = bars["close"], bars["open"], bars["high"], bars["low"], bars["volume"], bars["trading_date"]
    day_group = bars.groupby(day)

    day_high, day_low = high.groupby(day).transform("max"), low.groupby(day).transform("min")
    prev = pd.DataFrame({"day_high": day_high, "day_low": day_low, "day": day}).drop_duplicates("day").sort_values("day")
    prev["previous_day_high"], prev["previous_day_low"] = prev["day_high"].shift(1), prev["day_low"].shift(1)
    prev = prev.set_index("day")
    previous_day_high = day.map(prev["previous_day_high"])
    previous_day_low = day.map(prev["previous_day_low"])

    or_high_per_day = high.groupby(day).transform(lambda s: s.iloc[:OPENING_RANGE_BARS].max())
    or_low_per_day = low.groupby(day).transform(lambda s: s.iloc[:OPENING_RANGE_BARS].min())
    have_full_opening_range = day_group.cumcount() >= (OPENING_RANGE_BARS - 1)
    opening_range_high = or_high_per_day.where(have_full_opening_range)
    opening_range_low = or_low_per_day.where(have_full_opening_range)

    meaningful_level_up = pd.concat([previous_day_high, opening_range_high], axis=1).max(axis=1)
    meaningful_level_down = pd.concat([previous_day_low, opening_range_low], axis=1).min(axis=1)
    close_then = close.shift(BREAKOUT_WITHIN_BARS)
    undefined_up = meaningful_level_up.isna() | close_then.isna()
    undefined_down = meaningful_level_down.isna() | close_then.isna()
    breakout_up_meaningful = ((close > meaningful_level_up) & (close_then <= meaningful_level_up)).mask(undefined_up)
    breakout_down_meaningful = ((close < meaningful_level_down) & (close_then >= meaningful_level_down)).mask(undefined_down)

    ret_1bar = close.pct_change(1)
    rv10 = ret_1bar.rolling(VOL_SHORT_N).std()
    rv60 = ret_1bar.rolling(VOL_LONG_N).std()
    vol_expansion_ratio = rv10 / rv60
    ratio_then = vol_expansion_ratio.shift(DIRECTIONAL_PRESSURE_LOOKBACK)
    compression_then_expansion_flag = ((ratio_then <= COMPRESSION_THRESHOLD) & (vol_expansion_ratio >= EXPANSION_THRESHOLD)).where(ratio_then.notna() & vol_expansion_ratio.notna())

    candle_body_ratio = (close - open_).abs() / (high - low).replace(0.0, np.nan)
    candle_body_ratio = candle_body_ratio.fillna(0.0)
    candle_dir = np.sign(close - open_)
    directional_pressure_10 = (candle_dir * volume).rolling(DIRECTIONAL_PRESSURE_LOOKBACK).sum()

    ist = bars["bar_start_utc"].dt.tz_convert("Asia/Kolkata")
    minutes_since_open = (ist.dt.hour - 9) * 60 + (ist.dt.minute - 15)

    new_columns: dict[str, pd.Series] = {
        "realized_volatility_10": rv10, "realized_volatility_60": rv60, "volatility_expansion_ratio": vol_expansion_ratio,
        "compression_then_expansion_flag": compression_then_expansion_flag.astype("boolean"),
        "candle_body_ratio": candle_body_ratio, "directional_pressure_10": directional_pressure_10,
        "minutes_since_open": minutes_since_open,
        "breakout_up_meaningful": breakout_up_meaningful.astype("boolean"), "breakout_down_meaningful": breakout_down_meaningful.astype("boolean"),
    }

    rv20 = ret_1bar.rolling(20).std()
    for h in HORIZONS:
        future_close = close.shift(-h)
        window_ok = day.shift(-h) == day
        move = (future_close / close) - 1.0
        normalized_move = (move / rv20).where(window_ok & rv20.notna() & (rv20 != 0))
        label = pd.Series(pd.array([None] * len(bars), dtype="object"), index=bars.index)
        label = label.mask(normalized_move >= FORWARD_LABEL_THRESHOLD, "UP")
        label = label.mask(normalized_move <= -FORWARD_LABEL_THRESHOLD, "DOWN")
        label = label.mask(normalized_move.notna() & (normalized_move.abs() < FORWARD_LABEL_THRESHOLD), "NEUTRAL")
        new_columns[f"normalized_move_{h}"] = normalized_move
        new_columns[f"direction_label_pit_{h}"] = label

    return pd.concat([bars, pd.DataFrame(new_columns, index=bars.index)], axis=1)


def subsample_to_grid(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["bar_start_utc"].dt.minute % GRID_MINUTES == 0].copy()


def daily_traded_value_for_symbol(bars: pd.DataFrame) -> pd.DataFrame:
    daily = bars.assign(traded_value=bars["volume"] * bars["close"]).groupby("trading_date", as_index=False)["traded_value"].sum()
    return daily


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    load_dotenv()
    base = pd.read_parquet(PHASE_S_FILE)
    print(f"loaded Phase S dense parquet: {len(base)} rows, {base['security_id'].nunique()} symbols")
    security_ids = sorted(base["security_id"].unique().tolist())

    session = get_session()
    t0 = time.monotonic()
    merged_results = []
    daily_tables = []
    try:
        for i, sid in enumerate(security_ids, 1):
            rows = session.execute(
                text(
                    """
                    SELECT bar_start_utc, trading_date, open, high, low, close, volume
                    FROM intraday_equity_bars WHERE security_id = :sid AND timeframe = '1m' ORDER BY bar_start_utc ASC
                    """
                ),
                {"sid": sid},
            ).fetchall()
            if not rows:
                continue
            bars = pd.DataFrame(rows, columns=["bar_start_utc", "trading_date", "open", "high", "low", "close", "volume"])
            for c in ("open", "high", "low", "close", "volume"):
                bars[c] = bars[c].astype(float)

            computed = compute_new_columns_for_symbol(bars)
            dense_new = subsample_to_grid(computed)
            new_cols = [c for c in dense_new.columns if c not in ("open", "high", "low", "close", "volume")]
            base_sym = base[base["security_id"] == sid]
            merged = base_sym.merge(dense_new[new_cols], on=["bar_start_utc", "trading_date"], how="left", validate="one_to_one")

            daily = daily_traded_value_for_symbol(bars)
            daily["security_id"] = sid
            daily_tables.append(daily)
            merged_results.append(merged)

            if i % 25 == 0:
                print(f"  {i}/{len(security_ids)} symbols processed, {time.monotonic() - t0:.0f}s elapsed")
    finally:
        session.close()

    full = pd.concat(merged_results, ignore_index=True)
    daily_all = pd.concat(daily_tables, ignore_index=True)
    print(f"computing rolling liquidity classification over {len(daily_all)} symbol-days, {time.monotonic() - t0:.0f}s elapsed")
    liquidity = compute_rolling_liquidity_classification(daily_all)

    full = full.merge(
        liquidity[["security_id", "trading_date", "rolling_median_daily_traded_value_20d", "rolling_liquidity_group"]],
        on=["security_id", "trading_date"], how="left",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(args.out, index=False)
    n_liquid = full["rolling_liquidity_group"].isin(["mid_liquidity", "high_liquidity"]).sum()
    print(f"wrote {len(full)} rows, {full.shape[1]} columns, {n_liquid} mid/high-liquidity rows -> {args.out.relative_to(ROOT)} ({time.monotonic() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
