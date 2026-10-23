"""Phase Q: computes RSI (9, 14) + RSI(14) 5-bar slope + MACD(12,26,9) histogram onto the frozen Phase H/J/K/L
research dataset.

Reads the FULL 1-minute bar history per symbol from `intraday_equity_bars` (not just the 5-minute-sampled rows
in the Phase H dataset) via `select_bars_as_of`-equivalent bulk pulls, so the rolling/EMA windows see genuine
1-minute granularity - matching exactly what `relative_strength_index`/`macd_histogram` would compute if called
bar-by-bar via the pure-function contract in backend/app/research/intraday_features/momentum_oscillators.py.

VECTORIZED BULK COMPUTATION == THE PURE-FUNCTION DEFINITION, not an approximation:
  - `relative_strength_index` is a FIXED trailing window -> pandas `.rolling(period).mean()` on gains/losses
    over the full per-symbol series gives, at every row, exactly the same value the pure function would return
    if called with `bars_up_to_t` truncated at that row (proven in test_momentum_oscillators.py).
  - `macd_histogram`'s EMA is recursive over the full prior series -> pandas `.ewm(span=..., adjust=False)
    .mean()` computed once over the full per-symbol series gives, at every row, the identical recursive formula
    evaluated incrementally forward - the same value `_ema_series` would produce if re-run fresh on
    `bars_up_to_t` truncated at that row (also proven in test_momentum_oscillators.py).

Processed ONE SYMBOL AT A TIME (not one giant multi-symbol frame) to keep peak memory bounded - ~90K 1-minute
bars per symbol rather than ~27M across all 298 symbols at once.

Does NOT modify the frozen Phase H/J/K/L datasets on disk - writes a NEW, separate parquet file that Phase Q's
own research script joins in. Zero Dhan API calls (reads only already-persisted `intraday_equity_bars`).
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
OUT = ROOT / "data" / "derived" / "phase_q_direction_discovery" / "oscillators.parquet"

RSI_PERIODS = (9, 14)
RSI_SLOPE_PERIOD, RSI_SLOPE_BARS = 14, 5
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
MIN_BARS_NEEDED = MACD_SLOW + MACD_SIGNAL


def compute_for_symbol(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars[["bar_start_utc"]].copy()
    close = bars["close"]
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

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
    hist.iloc[: MIN_BARS_NEEDED - 1] = pd.NA
    out["macd_histogram_12_26_9"] = hist

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols-file", type=Path, default=ROOT / "data" / "derived" / "phase_d_research_dataset" / "universe.json")
    ap.add_argument("--max-date", default="2026-09-23", help="inclusive upper bound on trading_date pulled - matches the frozen Phase H dataset's own max date, never Phase P's newer live-collected days")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    load_dotenv()
    import json
    universe = json.loads(args.symbols_file.read_text(encoding="utf-8"))["universe"]
    security_ids = [e["security_id"] for e in universe]
    print(f"universe: {len(security_ids)} symbols, pulling bars through {args.max_date}")

    session = get_session()
    t0 = time.monotonic()
    results = []
    try:
        for i, sid in enumerate(security_ids, 1):
            rows = session.execute(
                text(
                    """
                    SELECT bar_start_utc, close FROM intraday_equity_bars
                    WHERE security_id = :sid AND trading_date <= :max_date
                    ORDER BY bar_start_utc ASC
                    """
                ),
                {"sid": sid, "max_date": args.max_date},
            ).fetchall()
            if not rows:
                continue
            bars = pd.DataFrame(rows, columns=["bar_start_utc", "close"])
            bars["close"] = bars["close"].astype(float)
            computed = compute_for_symbol(bars)
            computed["security_id"] = sid
            results.append(computed)
            if i % 50 == 0:
                print(f"  {i}/{len(security_ids)} symbols processed, {time.monotonic() - t0:.0f}s elapsed")
    finally:
        session.close()

    full = pd.concat(results, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(args.out, index=False)
    print(f"wrote {len(full)} rows, {full['security_id'].nunique()} symbols -> {args.out.relative_to(ROOT)} ({time.monotonic() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
