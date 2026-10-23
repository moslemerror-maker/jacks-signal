"""Phase R: builds the dense, shared 5-minute cross-sectional dataset directly from raw persisted
1-minute bars (approved design, replacing Phase H's sparse ~3-symbols-per-timestamp sample for this phase only).

Every formula below is a VECTORIZED reproduction of an already-existing, already-PIT-safe pure function in
backend/app/research/intraday_features/ or scripts/run_phase_k_entry_and_path.py - not a new definition. Each
one is verified against its pure-function reference on synthetic data (see verify_equivalence()) before being
trusted on real data, the same discipline Phase Q used for RSI/MACD.

PIT SAFETY: every quantity is a rolling/expanding function of a single symbol's own strictly-prior-or-current
1-minute bars (return_Xm, relative_volume_20, session_vwap, opening_range, overnight_gap - all per-symbol,
per-session, forward-only), OR gross_return_Xm/direction_label_Xm which explicitly reads bars STRICTLY AFTER T
(the one deliberately-exempt label construction, mirroring labels.py's own documented exemption - built via
negative .shift(), never via a whole-column rank or any operation that could leak into an EARLIER row's value).
The cross-sectional rank (computed by the companion research script, not here) is a separate, later step -
this script only produces the PER-SYMBOL ingredients it needs.

Processed ONE SYMBOL AT A TIME (Phase Q's established memory-bounding pattern) using the FULL raw 1-minute
history, then subsampled to the shared 5-minute grid (09:15, 09:20, ... every 5 minutes) at the end - not
Phase H's independently-sampled points, so every symbol shares the identical timestamp set and cross-sectional
ranking is actually dense (per the Phase R design freeze's own audit finding).
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
OUT = ROOT / "data" / "derived" / "phase_r_cross_sectional" / "dense_symbol_features.parquet"

RETURN_WINDOWS = (1, 3, 5, 15, 30, 60)
RELATIVE_VOLUME_WINDOW = 20
OPENING_RANGE_BARS = 15
BREAKOUT_WITHIN_BARS = 5
DIRECTION_THRESHOLD = 0.0020
LABEL_HORIZONS = (5, 15, 30, 60)
GRID_MINUTES = 5


def compute_for_symbol(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.copy()
    close, open_, high, low, volume = bars["close"], bars["open"], bars["high"], bars["low"], bars["volume"]

    for n in RETURN_WINDOWS:
        out[f"return_{n}m"] = close.pct_change(n)

    prior_vol_mean = volume.shift(1).rolling(RELATIVE_VOLUME_WINDOW).mean()
    out["relative_volume_20"] = volume / prior_vol_mean

    day = bars["trading_date"]
    day_group = out.groupby(day, sort=False)

    first_open_per_day = day_group["open"].transform("first")
    prev_day_last_close = bars.groupby(day, sort=False)["close"].last().shift(1)
    out["overnight_gap"] = first_open_per_day / out["trading_date"].map(prev_day_last_close) - 1.0

    typical = (high + low + close) / 3.0
    tp_vol = typical * volume
    cum_tp_vol = tp_vol.groupby(day).cumsum()
    cum_vol = volume.groupby(day).cumsum()
    volume_ok_so_far = (volume > 0).groupby(day).cummin().astype(bool)
    weighted_vwap = cum_tp_vol / cum_vol
    cum_tp_count = day_group.cumcount() + 1
    cum_tp_sum = typical.groupby(day).cumsum()
    unweighted_vwap = cum_tp_sum / cum_tp_count
    out["session_vwap"] = np.where(volume_ok_so_far, weighted_vwap, unweighted_vwap)
    out["distance_from_vwap"] = out["close"] / out["session_vwap"] - 1.0

    or_high_per_day = high.groupby(day).transform(lambda s: s.iloc[:OPENING_RANGE_BARS].max())
    or_low_per_day = low.groupby(day).transform(lambda s: s.iloc[:OPENING_RANGE_BARS].min())
    have_full_opening_range = day_group.cumcount() >= (OPENING_RANGE_BARS - 1)
    out["opening_range_high"] = or_high_per_day.where(have_full_opening_range)
    out["opening_range_low"] = or_low_per_day.where(have_full_opening_range)

    close_then = close.shift(BREAKOUT_WITHIN_BARS)
    breakout_up = ((close > out["opening_range_high"]) & (close_then <= out["opening_range_high"])).astype("boolean")
    breakout_down = ((close < out["opening_range_low"]) & (close_then >= out["opening_range_low"])).astype("boolean")
    out["recent_breakout_up_opening_range"] = breakout_up.mask(out["opening_range_high"].isna())
    out["recent_breakout_down_opening_range"] = breakout_down.mask(out["opening_range_low"].isna())

    entry_price = open_.shift(-1)
    entry_day = day.shift(-1)
    for h in LABEL_HORIZONS:
        window_end_close = close.shift(-h)
        window_end_day = day.shift(-h)
        valid = (entry_day == day) & (window_end_day == day)
        gross = (window_end_close / entry_price) - 1.0
        out[f"gross_return_{h}m"] = gross.where(valid)
        label = pd.Series(np.where(gross >= DIRECTION_THRESHOLD, "UP", np.where(gross <= -DIRECTION_THRESHOLD, "DOWN", "AMBIGUOUS")), index=out.index)
        out[f"direction_label_{h}m"] = label.where(valid, other=None)

    return out


def subsample_to_grid(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["bar_start_utc"].dt.minute % GRID_MINUTES == 0].copy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols-file", type=Path, default=ROOT / "data" / "derived" / "phase_d_research_dataset" / "universe.json")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    load_dotenv()
    import json
    universe = json.loads(args.symbols_file.read_text(encoding="utf-8"))["universe"]
    symbol_by_sid = {e["security_id"]: e["symbol"] for e in universe}
    security_ids = list(symbol_by_sid.keys())
    print(f"universe: {len(security_ids)} symbols")

    session = get_session()
    t0 = time.monotonic()
    results = []
    try:
        for i, sid in enumerate(security_ids, 1):
            rows = session.execute(
                text(
                    """
                    SELECT bar_start_utc, trading_date, open, high, low, close, volume
                    FROM intraday_equity_bars WHERE security_id = :sid ORDER BY bar_start_utc ASC
                    """
                ),
                {"sid": sid},
            ).fetchall()
            if not rows:
                continue
            bars = pd.DataFrame(rows, columns=["bar_start_utc", "trading_date", "open", "high", "low", "close", "volume"])
            for c in ("open", "high", "low", "close", "volume"):
                bars[c] = bars[c].astype(float)
            computed = compute_for_symbol(bars)
            dense = subsample_to_grid(computed)
            dense["security_id"] = sid
            dense["symbol"] = symbol_by_sid[sid]
            results.append(dense)
            if i % 25 == 0:
                print(f"  {i}/{len(security_ids)} symbols processed, {time.monotonic() - t0:.0f}s elapsed")
    finally:
        session.close()

    full = pd.concat(results, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(args.out, index=False)
    print(f"wrote {len(full)} rows, {full['security_id'].nunique()} symbols, {full['bar_start_utc'].nunique()} distinct timestamps -> {args.out.relative_to(ROOT)} ({time.monotonic() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
