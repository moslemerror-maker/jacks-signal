"""Phase T: builds the event-anchored, 1-minute-resolution entry/path dataset used for the entry-timing,
MFE/MAE, and risk-structure research questions.

WHY A NEW BUILD, NOT A REUSE OF THE PHASE R/S DENSE PARQUET: that parquet is subsampled to the 5-minute
cross-sectional grid (Phase R design freeze) - the 1-minute bars BETWEEN grid points are discarded after the
per-symbol feature computation. Phase T needs those discarded minutes back (1/2/3-minute delayed entries,
1/2/3/5/10-minute holding horizons, minute-resolution MFE/MAE), so this script goes back to the same raw
`intraday_equity_bars` table Phase R itself started from, using the identical per-symbol query pattern
(scripts/run_phase_r_build_dense_dataset.py::main), and re-attaches the result to Phase R/S's FROZEN, UNCHANGED
event population (the 4 core buckets from Phase S's extended parquet) - it does not redefine that population,
target, or ranking in any way.

EVERY quantity below is a VECTORIZED reproduction of the pure function
`backend.app.research.intraday_features.labels.entry_path_result`, proven bit-identical to it on synthetic
data in backend/tests/test_phase_t_build_event_paths.py before being trusted here - the same discipline Phase
Q/R/S used for RSI/MACD/EMA/support-resistance.

PIT SAFETY: entry_price/gross_return/mfe/mae for entry_offset_bars=k, horizon_bars=h at row i depend only on
rows i..i+k+h (all at or after the signal row i) - by construction this is a LABEL-side quantity, exempt from
the bars_up_to_t-only contract in the same documented way excursion_label/forward_movement_label already are
(see labels.py's module docstring). The signal itself (bucket membership, cs_rank_15m, IS_LONG_BUCKET) is read
verbatim from Phase S's already-PIT-verified parquet, never recomputed differently here.
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

ROOT = Path(__file__).resolve().parents[1]
PHASE_S_FILE = ROOT / "data" / "derived" / "phase_s_confirmation" / "dense_symbol_features_extended.parquet"
OUT = ROOT / "data" / "derived" / "phase_t_risk_structure" / "event_paths.parquet"

ENTRY_OFFSETS = (0, 1, 2, 3, 5)
HORIZONS = (1, 2, 3, 5, 10, 15, 30, 60)

BUCKET_COLUMNS = ("top_1pct_rank15m", "top_2pct_rank15m", "bottom_1pct_rank15m", "bottom_2pct_rank15m")
CONTEXT_COLUMNS = (
    "distance_from_ema20_normalized", "ema_9", "ema_20", "ema_20_slope_3",
    "realized_volatility_20", "relative_volume_20", "distance_from_vwap", "consecutive_same_direction_count",
)


def compute_paths_for_symbol(bars: pd.DataFrame) -> pd.DataFrame:
    close, open_, high, low, day = bars["close"], bars["open"], bars["high"], bars["low"], bars["trading_date"]

    fwd_max = {h: pd.Series(high[::-1].rolling(h, min_periods=h).max().to_numpy()[::-1], index=bars.index) for h in HORIZONS}
    fwd_min = {h: pd.Series(low[::-1].rolling(h, min_periods=h).min().to_numpy()[::-1], index=bars.index) for h in HORIZONS}

    new_columns: dict[str, pd.Series] = {}
    for k in ENTRY_OFFSETS:
        if k == 0:
            entry_price = close
            entry_ok = pd.Series(True, index=bars.index)
        else:
            entry_price = open_.shift(-k)
            entry_ok = day.shift(-k) == day
        new_columns[f"entry_price_o{k}"] = entry_price.where(entry_ok)

        for h in HORIZONS:
            window_ok = entry_ok & (day.shift(-(k + h)) == day)
            future_close = close.shift(-(k + h))
            max_high_w = fwd_max[h].shift(-(k + 1))
            min_low_w = fwd_min[h].shift(-(k + 1))
            gross = (future_close / entry_price) - 1.0
            mfe = ((max_high_w - entry_price) / entry_price).clip(lower=0.0)
            mae = ((entry_price - min_low_w) / entry_price).clip(lower=0.0)
            new_columns[f"gross_return_o{k}_h{h}"] = gross.where(window_ok)
            new_columns[f"mfe_o{k}_h{h}"] = mfe.where(window_ok)
            new_columns[f"mae_o{k}_h{h}"] = mae.where(window_ok)
    return pd.concat([bars, pd.DataFrame(new_columns, index=bars.index)], axis=1)


def load_event_population() -> pd.DataFrame:
    df = pd.read_parquet(PHASE_S_FILE)
    df["cs_rank_15m"] = df.groupby("bar_start_utc")["return_15m"].rank(pct=True)
    masks = {
        "top_1pct_rank15m": df["cs_rank_15m"] >= 0.99,
        "top_2pct_rank15m": df["cs_rank_15m"] >= 0.98,
        "bottom_1pct_rank15m": df["cs_rank_15m"] <= 0.01,
        "bottom_2pct_rank15m": df["cs_rank_15m"] <= 0.02,
    }
    for name, mask in masks.items():
        df[name] = mask.fillna(False)
    event_mask = masks["top_2pct_rank15m"] | masks["bottom_2pct_rank15m"]
    cols = ["security_id", "symbol", "bar_start_utc", "trading_date", *BUCKET_COLUMNS, *CONTEXT_COLUMNS]
    return df.loc[event_mask, cols].copy()


def compute_liquidity_tertiles(session, security_ids: list[int]) -> pd.DataFrame:
    rows = session.execute(
        text(
            """
            SELECT security_id, trading_date, SUM(volume * close) AS traded_value
            FROM intraday_equity_bars WHERE timeframe = '1m'
            GROUP BY security_id, trading_date
            """
        )
    ).fetchall()
    daily = pd.DataFrame(rows, columns=["security_id", "trading_date", "traded_value"])
    daily["traded_value"] = daily["traded_value"].astype(float)
    per_symbol = daily.groupby("security_id")["traded_value"].median().rename("median_daily_traded_value")
    per_symbol = per_symbol.reindex(security_ids)
    tertile = pd.qcut(per_symbol, 3, labels=["low_liquidity", "mid_liquidity", "high_liquidity"])
    return pd.DataFrame({"security_id": per_symbol.index, "median_daily_traded_value": per_symbol.to_numpy(), "liquidity_group": tertile.to_numpy()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    load_dotenv()
    events = load_event_population()
    print(f"loaded frozen event population: {len(events)} rows, {events['security_id'].nunique()} symbols")
    security_ids = sorted(events["security_id"].unique().tolist())

    session = get_session()
    t0 = time.monotonic()
    liquidity = compute_liquidity_tertiles(session, security_ids)
    print(f"computed liquidity tertiles for {len(liquidity)} symbols, {time.monotonic() - t0:.0f}s elapsed")

    results = []
    try:
        for i, sid in enumerate(security_ids, 1):
            sym_events = events[events["security_id"] == sid]
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
            paths = compute_paths_for_symbol(bars)
            merged = sym_events.merge(paths, on=["bar_start_utc", "trading_date"], how="left", validate="one_to_one")
            results.append(merged)
            if i % 25 == 0:
                print(f"  {i}/{len(security_ids)} symbols processed, {time.monotonic() - t0:.0f}s elapsed")
    finally:
        session.close()

    full = pd.concat(results, ignore_index=True)
    full = full.merge(liquidity, on="security_id", how="left")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(args.out, index=False)
    print(f"wrote {len(full)} rows, {full.shape[1]} columns, {full['security_id'].nunique()} symbols -> {args.out.relative_to(ROOT)} ({time.monotonic() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
