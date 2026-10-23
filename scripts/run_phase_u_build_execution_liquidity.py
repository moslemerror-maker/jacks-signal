"""Phase U: builds the two new execution-price variants (next-bar close, next-bar VWAP proxy) and the
PIT-safe rolling liquidity classification, merged onto Phase T's EXACT frozen event population.

WHY A SEPARATE FILE, NOT AN EDIT TO PHASE T'S OUTPUT: "Do not alter Phase R/T historical results" - this reads
Phase T's build script's `load_event_population()` (unchanged, imported) and Phase T's own event_paths.parquet
is never opened or modified here. A second, independent pass over the raw 1-minute bars is made because the
two new entry-price variants need the next bar's OWN close/high/low (not just its open, which is all Phase T
stored), and the liquidity classification needs each symbol's FULL daily aggregate history, something Phase T
never computed.

PIT SAFETY:
  - entry_price_close1 / entry_price_vwap1 and their gross_return/mfe/mae: vectorized reproduction of
    `labels.py::next_bar_entry_path_result`, proven bit-identical to it on synthetic data (see
    backend/tests/test_phase_u_build_execution_liquidity.py) before being trusted here.
  - rolling_median_daily_traded_value_20d: vectorized reproduction of
    `liquidity.py::rolling_median_daily_traded_value` (the trailing 20 days STRICTLY BEFORE the event's own
    day - same-day volume never included), also proven equivalent before use.
  - avg_1m_volume_20: the exact `volume.shift(1).rolling(20).mean()` formula Phase R's
    `relative_volume_20` already uses (current bar excluded) - not a new formula, reused verbatim.
  - rolling_liquidity_group: a CROSS-SECTIONAL percentile split of rolling_median_daily_traded_value_20d
    AT EACH DATE, using `.rank(pct=True)` (Phase R/S/T's own established percentile-split convention, not
    pandas qcut, which can raise on duplicate bin edges) - never uses a future date's cross-section.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from backend.app.config import load_dotenv
from backend.app.db.session import get_session
from scripts.run_phase_t_build_event_paths import load_event_population

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "derived" / "phase_u_execution_liquidity" / "event_paths_extended.parquet"

HORIZONS = (15, 60)
AVG_VOLUME_WINDOW = 20
LIQUIDITY_WINDOW_DAYS = 20


def compute_execution_variants_for_symbol(bars: pd.DataFrame) -> pd.DataFrame:
    close, high, low, open_, volume, day = bars["close"], bars["high"], bars["low"], bars["open"], bars["volume"], bars["trading_date"]

    entry_ok = day.shift(-1) == day
    entry_price = {"close1": close.shift(-1).where(entry_ok), "vwap1": ((high + low + close) / 3.0).shift(-1).where(entry_ok)}

    fwd_max = {h: pd.Series(high[::-1].rolling(h, min_periods=h).max().to_numpy()[::-1], index=bars.index) for h in HORIZONS}
    fwd_min = {h: pd.Series(low[::-1].rolling(h, min_periods=h).min().to_numpy()[::-1], index=bars.index) for h in HORIZONS}

    new_columns: dict[str, pd.Series] = {
        "entry_price_close1": entry_price["close1"], "entry_price_vwap1": entry_price["vwap1"],
        "avg_1m_volume_20": volume.shift(1).rolling(AVG_VOLUME_WINDOW).mean(),
    }
    for basis in ("close1", "vwap1"):
        ep = entry_price[basis]
        for h in HORIZONS:
            window_ok = entry_ok & (day.shift(-(1 + h)) == day)
            future_close = close.shift(-(1 + h))
            max_high_w = fwd_max[h].shift(-2)
            min_low_w = fwd_min[h].shift(-2)
            gross = (future_close / ep) - 1.0
            mfe = ((max_high_w - ep) / ep).clip(lower=0.0)
            mae = ((ep - min_low_w) / ep).clip(lower=0.0)
            new_columns[f"gross_return_{basis}_h{h}"] = gross.where(window_ok)
            new_columns[f"mfe_{basis}_h{h}"] = mfe.where(window_ok)
            new_columns[f"mae_{basis}_h{h}"] = mae.where(window_ok)
    return pd.concat([bars, pd.DataFrame(new_columns, index=bars.index)], axis=1)


def daily_traded_value_for_symbol(bars: pd.DataFrame) -> pd.DataFrame:
    daily = bars.assign(traded_value=bars["volume"] * bars["close"]).groupby("trading_date", as_index=False)["traded_value"].sum()
    return daily


def compute_rolling_liquidity_classification(daily_by_symbol: pd.DataFrame) -> pd.DataFrame:
    daily_by_symbol = daily_by_symbol.sort_values(["security_id", "trading_date"]).reset_index(drop=True)
    daily_by_symbol["rolling_median_daily_traded_value_20d"] = daily_by_symbol.groupby("security_id")["traded_value"].transform(
        lambda s: s.shift(1).rolling(LIQUIDITY_WINDOW_DAYS).median()
    )
    pct_rank = daily_by_symbol.groupby("trading_date")["rolling_median_daily_traded_value_20d"].rank(pct=True)
    daily_by_symbol["rolling_liquidity_group"] = pd.Series(
        pd.NA, index=daily_by_symbol.index, dtype="object"
    )
    defined = pct_rank.notna()
    daily_by_symbol.loc[defined & (pct_rank <= 1 / 3), "rolling_liquidity_group"] = "low_liquidity"
    daily_by_symbol.loc[defined & (pct_rank > 1 / 3) & (pct_rank <= 2 / 3), "rolling_liquidity_group"] = "mid_liquidity"
    daily_by_symbol.loc[defined & (pct_rank > 2 / 3), "rolling_liquidity_group"] = "high_liquidity"
    return daily_by_symbol


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
    merged_results = []
    daily_tables = []
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
            variants = compute_execution_variants_for_symbol(bars)
            merged = sym_events.merge(variants, on=["bar_start_utc", "trading_date"], how="left", validate="one_to_one")
            merged_results.append(merged)

            daily = daily_traded_value_for_symbol(bars)
            daily["security_id"] = sid
            daily_tables.append(daily)

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
    n_defined = full["rolling_liquidity_group"].notna().sum()
    print(f"wrote {len(full)} rows, {full.shape[1]} columns, {n_defined} with a defined liquidity group -> {args.out.relative_to(ROOT)} ({time.monotonic() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
