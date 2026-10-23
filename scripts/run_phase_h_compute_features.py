"""Phase H, H1/H3: full-universe (298-symbol) feature/label computation, at coarser within-day
sampling than Phase G, plus new "mover" diagnostics (H3) computed alongside - NOT replacing the
original Phase G label definitions.

SCOPE (approved): all 298 universe symbols with >=1 persisted bar (only `011NSETEST`, a zero-bar
NSE test security, is excluded - see run_phase_h_audit.py's audit.json for the full documented
exclusion list; the 14 sparse/newly-listed symbols are NOT excluded, they simply contribute fewer
or zero rows on sampled dates before their listing). ~45 sampled trading dates (vs Phase G's 25),
giving H7's chronological 3-way split enough independent dates to be meaningful.

WITHIN-DAY SAMPLING STRIDE (the approved cost/coverage trade-off): a feature/label row is emitted
only every 5th bar of each sampled day (local bar-of-day index 0, 5, 10, ...) - but every feature
is still computed from the COMPLETE, un-strided 1-minute bar history via `bars_up_to_t`, and every
label's future window is the complete, un-strided next-N-minutes via `future_bars`. Only the number
of DECISION POINTS T sampled per day is reduced, not the resolution of the data those decisions see.

HORIZONS: 5, 15, 30 AND a newly-added 60-minute horizon (approved this phase) - `labels.py` itself
needed no changes, since `horizon_bars` was already a required, non-defaulted argument.

MOVER DIAGNOSTICS (H3, additive, never replacing labels.py's existing fields): for each horizon,
`mover_future_abs_return_Xm` (unsigned close-to-close move, magnitude only) and
`mover_future_range_pct_Xm` (total high/low range actually traveled during the window, relative to
the reference price - the size of the move regardless of the label's threshold or direction). These
answer H3's "does this predict magnitude, not just direction" question and are computed directly
here from `future_bars`, deliberately kept out of labels.py to keep its PIT-exemption boundary the
one, single, already-audited place forward-looking computation happens.

REUSE, NOT A NEW FRAMEWORK: every feature/label value comes from the existing
`backend/app/research/intraday_features/` package, unmodified. This script only adds new
orchestration (full-universe scope, coarser sampling, the two mover columns).
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
UNIVERSE_FILE = ROOT / "data" / "derived" / "phase_d_research_dataset" / "universe.json"
AUDIT_FILE = ROOT / "data" / "derived" / "phase_h_full_universe_validation" / "audit.json"
OUT_DIR = ROOT / "data" / "derived" / "phase_h_full_universe_validation"

FAR_FUTURE_UTC = dt.datetime(2035, 1, 1, tzinfo=dt.timezone.utc)
HORIZONS = (5, 15, 30, 60)
VOL_N_BARS = 20
LABEL_THRESHOLD = labels.DEFAULT_THRESHOLD
DATE_SAMPLE_STRIDE = 5
WITHIN_DAY_STRIDE = 5
HISTORY_WINDOW_BARS = 5000


def load_universe_symbols() -> list[dict[str, Any]]:
    universe = json.loads(UNIVERSE_FILE.read_text(encoding="utf-8"))["universe"]
    audit = json.loads(AUDIT_FILE.read_text(encoding="utf-8"))
    excluded_ids = {e["security_id"] for e in audit["excluded_symbols"]}
    return [e for e in universe if e["security_id"] not in excluded_ids]


def load_all_bars(session, security_id: int) -> list[bar_math.Bar]:
    views = select_bars_as_of(session, security_id=security_id, timeframe="1m", as_of_utc=FAR_FUTURE_UTC)
    return [bar_math.from_canonical(v) for v in views]


def pick_sampled_dates(all_bars_by_symbol: dict[int, list[bar_math.Bar]]) -> list[dt.date]:
    dates = sorted({b.trading_date for bars in all_bars_by_symbol.values() for b in bars})
    return dates[::DATE_SAMPLE_STRIDE]


def _day_blocks(bars: list[bar_math.Bar]) -> list[tuple[int, int, dt.date]]:
    blocks: list[tuple[int, int, dt.date]] = []
    start = 0
    for i in range(1, len(bars) + 1):
        if i == len(bars) or bars[i].trading_date != bars[start].trading_date:
            blocks.append((start, i - start, bars[start].trading_date))
            start = i
    return blocks


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


def mover_diagnostics(reference_price: float, reference_trading_date: dt.date, future_bars: list[bar_math.Bar], horizon_bars: int) -> dict[str, float | None]:
    window = labels._session_bounded_future_window(reference_trading_date, future_bars, horizon_bars)
    if window is None:
        return {"future_abs_return": None, "future_range_pct": None}
    future_close = window[-1].close
    max_high = max(b.high for b in window)
    min_low = min(b.low for b in window)
    return {
        "future_abs_return": abs(future_close - reference_price) / reference_price,
        "future_range_pct": (max_high - min_low) / reference_price,
    }


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
        mover = mover_diagnostics(cur.close, cur.trading_date, future_bars, h)
        row[f"mover_future_abs_return_{h}m"] = mover["future_abs_return"]
        row[f"mover_future_range_pct_{h}m"] = mover["future_range_pct"]

    return row


def run_pass_1(session, sample: list[dict[str, Any]]) -> pd.DataFrame:
    all_bars_by_symbol: dict[int, list[bar_math.Bar]] = {}
    t0 = time.monotonic()
    for entry in sample:
        all_bars_by_symbol[entry["security_id"]] = load_all_bars(session, entry["security_id"])
    print(f"loaded {sum(len(b) for b in all_bars_by_symbol.values())} total bars for {len(sample)} symbols in {time.monotonic() - t0:.1f}s")

    sampled_dates = set(pick_sampled_dates(all_bars_by_symbol))
    print(f"sampled {len(sampled_dates)} trading dates out of the full acquired range")

    records: list[dict[str, Any]] = []
    for entry in sample:
        bars = all_bars_by_symbol[entry["security_id"]]
        t0 = time.monotonic()
        row_count = 0
        for start, length, trading_date in _day_blocks(bars):
            if trading_date not in sampled_dates:
                continue
            for local_i in range(0, length, WITHIN_DAY_STRIDE):
                i = start + local_i
                b = bars[i]
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
    sample = load_universe_symbols()
    print(f"computing over {len(sample)} universe symbols (see audit.json for the excluded/sparse breakdown)")

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
        "row_count": len(df), "column_count": len(df.columns),
        "symbols_with_rows": sorted(df["symbol"].unique().tolist()),
        "n_symbols_with_rows": int(df["symbol"].nunique()),
        "sampled_trading_dates": sorted(str(d) for d in df["trading_date"].unique().tolist()),
        "date_sample_stride": DATE_SAMPLE_STRIDE, "within_day_stride": WITHIN_DAY_STRIDE,
        "horizons_minutes": list(HORIZONS), "label_threshold": LABEL_THRESHOLD, "vol_n_bars": VOL_N_BARS,
    }
    summary_path = OUT_DIR / "feature_label_dataset.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"wrote {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
