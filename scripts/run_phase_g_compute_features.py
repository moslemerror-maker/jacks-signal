"""Phase G: compute the full intraday_features feature set + 5/15/30-minute labels over the
deterministic representative sample selected by run_phase_g_sample_selection.py.

SCOPE CONTROL ("not a gigantic feature table blindly", per the governing instruction): this does
NOT compute a row for every minute of the full 1-year history of all 24 sampled symbols (that
would be ~2.2M rows of pure-Python feature computation). Instead it samples a fixed, deterministic
set of TRADING DATES (every Nth distinct trading_date present in the acquired data, spanning the
whole year - see `DATE_SAMPLE_STRIDE`) and computes a row for EVERY 1-minute bar within each
sampled date, for every one of the 24 symbols. This keeps full intraday (1-minute) granularity
within each sampled day - which the time-of-day, VWAP and 5/15/30-min-horizon feature families
need - while controlling total row count (~24 symbols x ~25 dates x ~360 bars =~ 216,000 rows).

Using the SAME sampled dates (and hence the same shared bar_start_utc instants) for every symbol
is deliberate: the Family 9/10 cross-sectional functions (`market_context.py`) require multiple
symbols' independently-computed, PIT-safe values at an identical T - that only holds if every
symbol is evaluated at the same clock instants.

TWO PASSES:
  Pass 1 (single-stock): for every symbol, load its FULL persisted bar history once (needed so
    lookback-hungry features - e.g. the 10-prior-session time-of-day baselines - have real prior
    data even though only a subset of T's are emitted), then for each sampled-date bar compute
    every single-stock feature family (1-8, 11) plus the 5/15/30-minute forward-movement and
    excursion labels.
  Pass 2 (cross-sectional): group the Pass-1 output by the shared `bar_start_utc`, compute the
    Family 9/10 cross-sectional/relative-strength values from the ALREADY-COMPUTED, ALREADY-PIT-
    SAFE per-symbol 1-minute returns at that same T, and merge them back onto every row.

Zero new Dhan calls - reads only from the already-persisted `intraday_equity_bars` table via the
existing `select_bars_as_of` point-in-time primitive (S1.4), reused as-is.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.config import load_dotenv
from backend.app.db.session import get_session
from backend.app.research.intraday_features import bar_math, labels, market_context, price_action, regime, volatility_volume, vwap_levels
from backend.app.research.intraday_point_in_time import select_bars_as_of

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_FILE = ROOT / "data" / "derived" / "phase_g_feature_foundation" / "sample_universe.json"
OUT_DIR = ROOT / "data" / "derived" / "phase_g_feature_foundation"

FAR_FUTURE_UTC = dt.datetime(2035, 1, 1, tzinfo=dt.timezone.utc)
HORIZONS = (5, 15, 30)
VOL_N_BARS = 20
LABEL_THRESHOLD = labels.DEFAULT_THRESHOLD
DATE_SAMPLE_STRIDE = 10

HISTORY_WINDOW_BARS = 5000


def load_sample() -> list[dict[str, Any]]:
    return json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))["sample"]


def load_all_bars(session, security_id: int) -> list[bar_math.Bar]:
    views = select_bars_as_of(session, security_id=security_id, timeframe="1m", as_of_utc=FAR_FUTURE_UTC)
    return [bar_math.from_canonical(v) for v in views]


def pick_sampled_dates(all_bars_by_symbol: dict[int, list[bar_math.Bar]]) -> list[dt.date]:
    dates = sorted({b.trading_date for bars in all_bars_by_symbol.values() for b in bars})
    return dates[::DATE_SAMPLE_STRIDE]


def _collect_support_resistance_levels(bars_up_to_t: list[bar_math.Bar]) -> list[float]:
    levels: list[float] = []
    prev_hlc = vwap_levels.previous_day_high_low_close(bars_up_to_t)
    if prev_hlc is not None:
        levels.extend(prev_hlc)
    sess_hl = vwap_levels.session_high_low_so_far(bars_up_to_t)
    if sess_hl is not None:
        levels.extend(sess_hl)
    or_hl = vwap_levels.opening_range_high_low(bars_up_to_t)
    if or_hl is not None:
        levels.extend(or_hl)
    swing_highs, swing_lows = vwap_levels.rolling_swing_points(bars_up_to_t)
    levels.extend(swing_highs)
    levels.extend(swing_lows)
    return levels


def compute_single_stock_row(bars_up_to_t: list[bar_math.Bar], future_bars: list[bar_math.Bar]) -> dict[str, Any]:
    cur = bars_up_to_t[-1]
    row: dict[str, Any] = {}

    row["return_1m"] = price_action.return_over_bars(bars_up_to_t, 1)
    row["return_3m"] = price_action.return_over_bars(bars_up_to_t, 3)
    row["return_5m"] = price_action.return_over_bars(bars_up_to_t, 5)
    row["return_15m"] = price_action.return_over_bars(bars_up_to_t, 15)
    row["return_30m"] = price_action.return_over_bars(bars_up_to_t, 30)
    row["distance_from_session_open"] = price_action.distance_from_session_open(bars_up_to_t)
    row["distance_from_prev_close"] = price_action.distance_from_prev_close(bars_up_to_t)
    row["range_position_20"] = price_action.range_position(bars_up_to_t, 20)
    row["range_expansion_20"] = price_action.range_expansion(bars_up_to_t, 20)
    row["momentum_acceleration_10"] = price_action.momentum_acceleration(bars_up_to_t, 10)

    row["candle_body_ratio"] = price_action.candle_body_ratio(cur)
    row["candle_upper_wick_ratio"] = price_action.candle_upper_wick_ratio(cur)
    row["candle_lower_wick_ratio"] = price_action.candle_lower_wick_ratio(cur)
    row["candle_direction"] = price_action.candle_direction(cur)
    row["consecutive_same_direction_count"] = price_action.consecutive_same_direction_count(bars_up_to_t)
    row["is_bullish_engulfing"] = price_action.is_bullish_engulfing(bars_up_to_t)
    row["is_bearish_engulfing"] = price_action.is_bearish_engulfing(bars_up_to_t)

    row["minutes_since_open"] = price_action.minutes_since_open(bars_up_to_t)
    row["time_bucket"] = price_action.time_bucket(bars_up_to_t)

    row["overnight_gap"] = price_action.overnight_gap(bars_up_to_t)

    row["realized_volatility_20"] = volatility_volume.realized_volatility(bars_up_to_t, VOL_N_BARS)
    row["average_true_range_14"] = volatility_volume.average_true_range(bars_up_to_t, 14)
    row["volatility_expansion_ratio"] = volatility_volume.volatility_expansion_ratio(bars_up_to_t)
    row["volatility_normalized_return_5_20"] = volatility_volume.volatility_normalized_return(bars_up_to_t, 5, VOL_N_BARS)
    row["realized_vol_vs_tod_baseline"] = volatility_volume.realized_vol_vs_time_of_day_baseline(bars_up_to_t)

    row["relative_volume_20"] = volatility_volume.relative_volume(bars_up_to_t, 20)
    row["volume_acceleration_10"] = volatility_volume.volume_acceleration(bars_up_to_t, 10)
    row["volume_vs_tod_baseline"] = volatility_volume.volume_vs_time_of_day_baseline(bars_up_to_t)
    row["price_volume_correlation_20"] = volatility_volume.price_volume_correlation(bars_up_to_t, 20)

    row["session_vwap"] = vwap_levels.session_vwap(bars_up_to_t)
    row["distance_from_vwap"] = vwap_levels.distance_from_vwap(bars_up_to_t)
    row["vwap_slope_10"] = vwap_levels.vwap_slope(bars_up_to_t, 10)
    row["vwap_cross"] = vwap_levels.vwap_cross(bars_up_to_t)
    row["vwap_deviation_normalized"] = vwap_levels.vwap_deviation_normalized(bars_up_to_t, VOL_N_BARS)

    levels = _collect_support_resistance_levels(bars_up_to_t)
    row["distance_to_nearest_resistance"] = vwap_levels.nearest_level_distance(cur.close, levels, above=True)
    row["distance_to_nearest_support"] = vwap_levels.nearest_level_distance(cur.close, levels, above=False)

    row["intraday_regime"] = regime.intraday_regime(bars_up_to_t)

    vol_normalizer = row["realized_volatility_20"]
    for h in HORIZONS:
        lbl = labels.forward_movement_label(bars_up_to_t, future_bars, horizon_bars=h, vol_normalizer=vol_normalizer, threshold=LABEL_THRESHOLD)
        row[f"label_class_{h}m"] = lbl.label_class if lbl else None
        row[f"label_normalized_move_{h}m"] = lbl.normalized_move if lbl else None
        exc = labels.excursion_label(bars_up_to_t, future_bars, horizon_bars=h)
        row[f"mfe_{h}m"] = exc.max_upside_excursion if exc else None
        row[f"mae_{h}m"] = exc.max_downside_excursion if exc else None

    return row


def run_pass_1(session, sample: list[dict[str, Any]]) -> pd.DataFrame:
    all_bars_by_symbol: dict[int, list[bar_math.Bar]] = {}
    for entry in sample:
        all_bars_by_symbol[entry["security_id"]] = load_all_bars(session, entry["security_id"])

    sampled_dates = set(pick_sampled_dates(all_bars_by_symbol))
    print(f"sampled {len(sampled_dates)} trading dates out of the full acquired range")

    records: list[dict[str, Any]] = []
    for entry in sample:
        bars = all_bars_by_symbol[entry["security_id"]]
        t0 = time.monotonic()
        row_count = 0
        for i, b in enumerate(bars):
            if b.trading_date not in sampled_dates:
                continue
            lo = max(0, i - HISTORY_WINDOW_BARS + 1)
            bars_up_to_t = bars[lo : i + 1]
            future_bars = bars[i + 1 : i + 1 + max(HORIZONS)]
            row = compute_single_stock_row(bars_up_to_t, future_bars)
            row.update(
                symbol=entry["symbol"], security_id=entry["security_id"], sector=entry["sector"], tier=entry["tier"],
                bar_start_utc=b.bar_start_utc, trading_date=b.trading_date, close=b.close, volume=b.volume,
            )
            records.append(row)
            row_count += 1
        print(f"  {entry['symbol']:<16} {row_count} rows in {time.monotonic() - t0:.1f}s")

    return pd.DataFrame.from_records(records)


def run_pass_2_cross_sectional(df: pd.DataFrame, sample: list[dict[str, Any]]) -> pd.DataFrame:
    sector_by_symbol = {e["symbol"]: e["sector"] for e in sample}

    cross_sectional = (
        df.groupby("bar_start_utc")["return_1m"]
        .agg(universe_average_return="mean", universe_breadth=lambda s: (s > 0).mean())
        .reset_index()
    )
    df = df.merge(cross_sectional, on="bar_start_utc", how="left")
    df["relative_strength_vs_universe"] = df["return_1m"] - df["universe_average_return"]

    sector_group = df.groupby(["bar_start_utc", "sector"])["return_1m"].agg(["sum", "count"]).reset_index()
    sector_group = sector_group.rename(columns={"sum": "_sector_return_sum", "count": "_sector_count"})
    df = df.merge(sector_group, on=["bar_start_utc", "sector"], how="left")
    peer_count = df["_sector_count"] - 1
    peer_sum = df["_sector_return_sum"] - df["return_1m"]
    df["sector_relative_strength"] = (df["return_1m"] - peer_sum / peer_count).where(peer_count > 0)
    df = df.drop(columns=["_sector_return_sum", "_sector_count"])

    df = df.sort_values(["symbol", "bar_start_utc"]).reset_index(drop=True)
    grp = df.groupby(["symbol", "trading_date"])
    rs_5_ago = grp["relative_strength_vs_universe"].shift(5)
    t_5_ago = grp["bar_start_utc"].shift(5)
    exactly_5_min_apart = (df["bar_start_utc"] - t_5_ago) == pd.Timedelta(minutes=5)
    df["relative_strength_acceleration"] = (df["relative_strength_vs_universe"] - rs_5_ago).where(exactly_5_min_apart)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "feature_label_dataset.parquet")
    args = parser.parse_args()

    load_dotenv()
    sample = load_sample()

    session = get_session()
    try:
        df = run_pass_1(session, sample)
    finally:
        session.close()

    print(f"pass 1 complete: {len(df)} rows, {len(df.columns)} columns")
    df = run_pass_2_cross_sectional(df, sample)
    print(f"pass 2 complete: {len(df)} rows, {len(df.columns)} columns")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {args.out.relative_to(ROOT)}")

    summary = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "row_count": len(df),
        "column_count": len(df.columns),
        "symbols": sorted(df["symbol"].unique().tolist()),
        "sampled_trading_dates": sorted(str(d) for d in df["trading_date"].unique().tolist()),
        "date_sample_stride": DATE_SAMPLE_STRIDE,
        "horizons_minutes": list(HORIZONS),
        "label_threshold": LABEL_THRESHOLD,
        "vol_n_bars": VOL_N_BARS,
    }
    summary_path = OUT_DIR / "feature_label_dataset.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"wrote {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
