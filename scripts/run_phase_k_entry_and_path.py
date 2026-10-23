"""Phase K, Parts 3 & 6: realistic next-bar-open entry + descriptive intraday path (MFE/MAE) dataset,
for the economic/tradability analysis. Built the same way Phase I's I3 next-bar-entry dataset was
built (same script pattern, same PIT discipline) - a separate, additive dataset, never modifying
labels.py or any Phase G/H/I/J artifact.

EXACT TIMESTAMP SEMANTICS, stated explicitly per the phase instruction:
  SIGNAL TIMESTAMP      T  = bars_up_to_t[-1].bar_start_utc (the last closed bar the signal saw)
  DECISION TIMESTAMP    T  = the same instant
  EARLIEST TRADABLE BAR    = the bar immediately after T
  ENTRY PRICE              = that bar's OPEN (never T's own close - the exact bug Phase I's I3 was
                              built to avoid, reused unchanged here)
  TARGET MEASUREMENT WINDOW = `horizon_bars` bars STARTING AT the entry bar (bars T+1 .. T+horizon) -
                              the trader is measured over the full stated horizon from when they
                              could actually be in the trade, not from T itself.
  SESSION-BOUNDARY / MISSING-BAR RULE: undefined if the entry bar or any bar in the window falls
  outside T's own trading_date, or if fewer than `horizon_bars` bars exist after entry - identical
  in spirit to labels.py and Phase I's I3.

PATH FIELDS (all LONG-referenced, i.e. "if you bought at entry_price"; a short interpretation is the
mirror image - mfe_short = mae_long, mae_short = mfe_long, no separate columns needed):
  gross_return_Xm    = (close of the horizon's last bar / entry_price) - 1  (signed)
  mfe_Xm             = (max high across the window / entry_price) - 1      (>= 0)
  mae_Xm             = (entry_price - min low across the window) / entry_price  (>= 0)
  bars_to_mfe_Xm     = 1-indexed bar offset from entry where the window's max high occurred
  bars_to_mae_Xm     = 1-indexed bar offset from entry where the window's min low occurred
None of these use future information beyond the row's own stated horizon window - MFE/MAE are
label-like diagnostics (exempt from the PIT-features rule the same way labels.py's own MFE/MAE are),
never fed back into any feature.

Zero new Dhan calls - reads only the already-persisted `intraday_equity_bars` table via the existing
`select_bars_as_of` point-in-time primitive, exactly as every prior phase's compute scripts did.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import time
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.config import load_dotenv
from backend.app.db.session import get_session
from backend.app.research.intraday_features import bar_math
from backend.app.research.intraday_point_in_time import select_bars_as_of

ROOT = Path(__file__).resolve().parents[1]
PHASE_H_DATASET = ROOT / "data" / "derived" / "phase_h_full_universe_validation" / "feature_label_dataset.parquet"
OUT_DIR = ROOT / "data" / "derived" / "phase_k_economic_validation"

FAR_FUTURE_UTC = dt.datetime(2035, 1, 1, tzinfo=dt.timezone.utc)
HORIZONS = (5, 15, 30, 60)


@dataclasses.dataclass(frozen=True)
class EntryAndPath:
    entry_price: float
    gross_return: float
    mfe: float
    mae: float
    bars_to_mfe: int
    bars_to_mae: int


def entry_and_path(bars: list[bar_math.Bar], t_index: int, *, horizon_bars: int) -> EntryAndPath | None:
    if t_index + horizon_bars >= len(bars):
        return None
    t_date = bars[t_index].trading_date
    entry_bar = bars[t_index + 1]
    window = bars[t_index + 1 : t_index + 1 + horizon_bars]
    if entry_bar.trading_date != t_date or any(b.trading_date != t_date for b in window):
        return None

    entry_price = entry_bar.open
    gross_return = (window[-1].close / entry_price) - 1.0

    best_high, best_high_offset = window[0].high, 1
    worst_low, worst_low_offset = window[0].low, 1
    for offset, b in enumerate(window, start=1):
        if b.high > best_high:
            best_high, best_high_offset = b.high, offset
        if b.low < worst_low:
            worst_low, worst_low_offset = b.low, offset

    return EntryAndPath(
        entry_price=entry_price, gross_return=gross_return,
        mfe=(best_high / entry_price) - 1.0, mae=(entry_price - worst_low) / entry_price,
        bars_to_mfe=best_high_offset, bars_to_mae=worst_low_offset,
    )


def load_all_bars(session, security_id: int) -> list[bar_math.Bar]:
    views = select_bars_as_of(session, security_id=security_id, timeframe="1m", as_of_utc=FAR_FUTURE_UTC)
    return [bar_math.from_canonical(v) for v in views]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "entry_and_path_dataset.parquet")
    args = parser.parse_args()

    load_dotenv()
    base = pd.read_parquet(PHASE_H_DATASET, columns=["symbol", "security_id", "bar_start_utc"])
    print(f"loaded {len(base)} rows from the frozen Phase H dataset (read-only)")

    session = get_session()
    try:
        symbols = sorted(base["security_id"].unique().tolist())
        records: list[dict[str, Any]] = []
        for sid in symbols:
            t0 = time.monotonic()
            bars = load_all_bars(session, sid)
            index_by_ts = {b.bar_start_utc: i for i, b in enumerate(bars)}
            rows = base[base["security_id"] == sid]
            symbol_name = rows["symbol"].iloc[0]
            n_rows = 0
            for _, row in rows.iterrows():
                t_index = index_by_ts.get(row["bar_start_utc"])
                if t_index is None:
                    continue
                rec: dict[str, Any] = {"symbol": symbol_name, "security_id": sid, "bar_start_utc": row["bar_start_utc"]}
                for h in HORIZONS:
                    ep = entry_and_path(bars, t_index, horizon_bars=h)
                    rec[f"entry_price_{h}m"] = ep.entry_price if ep else None
                    rec[f"gross_return_{h}m"] = ep.gross_return if ep else None
                    rec[f"path_mfe_{h}m"] = ep.mfe if ep else None
                    rec[f"path_mae_{h}m"] = ep.mae if ep else None
                    rec[f"bars_to_mfe_{h}m"] = ep.bars_to_mfe if ep else None
                    rec[f"bars_to_mae_{h}m"] = ep.bars_to_mae if ep else None
                records.append(rec)
                n_rows += 1
            print(f"  {symbol_name:<16} {n_rows} rows in {time.monotonic() - t0:.1f}s")
    finally:
        session.close()

    out_df = pd.DataFrame.from_records(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"wrote {args.out.relative_to(ROOT)} ({len(out_df)} rows, {len(out_df.columns)} columns)")


if __name__ == "__main__":
    main()
