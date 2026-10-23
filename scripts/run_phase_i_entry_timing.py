"""Phase I, I3: entry-timing comparison - same-bar-close "entry" vs. a realistic next-bar-open entry.

THE PROBLEM THIS FIXES: every label in labels.py (Phase G, frozen, NOT modified here) uses
`reference_price = bars_up_to_t[-1].close` - i.e. the SAME bar's close is used as both the last
piece of information the signal saw AND the hypothetical entry price. A real system cannot buy at a
price that is only known once the bar has already closed - by the time a decision could be acted on,
only the NEXT bar's open is actually executable. This script builds a SEPARATE, NEW, clearly-labeled
label variant using that realistic entry price, WITHOUT touching labels.py or any Phase G/H artifact.

EXACT TIMESTAMP SEMANTICS (per the phase instruction, stated explicitly, not left implicit):
  SIGNAL TIMESTAMP       T           = bars_up_to_t[-1].bar_start_utc (unchanged from Phase G/H)
  DECISION TIMESTAMP     T           = the same instant - the decision to act is made immediately
  EARLIEST EXECUTABLE ENTRY          = the OPEN of the bar immediately following T (T+1's open) -
                                        the earliest price a real order could plausibly fill at
  OUTCOME MEASUREMENT WINDOW         = from that entry price to the close of the bar `horizon_bars`
                                        minutes AFTER THE ENTRY bar (not after T) - i.e. the trader
                                        gets the full stated horizon of exposure STARTING FROM WHEN
                                        THEY COULD ACTUALLY BE IN THE TRADE, not from T itself.
  SESSION-BOUNDARY / MISSING-BAR RULE: identical in spirit to labels.py - undefined if the entry bar
  or any bar in the outcome window falls on a different trading_date than T, or if fewer than
  `horizon_bars` bars exist after entry.

NOT A REPLACEMENT: this is an ADDITIVE, side-by-side comparison dataset - the original Phase G/H
labels remain exactly as computed. Every column produced here is prefixed `nextbar_` to make that
unambiguous in the merged analysis.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.config import load_dotenv
from backend.app.db.session import get_session
from backend.app.research.intraday_features import bar_math
from backend.app.research.intraday_point_in_time import select_bars_as_of

ROOT = Path(__file__).resolve().parents[1]
DATASET_FILE = ROOT / "data" / "derived" / "phase_h_full_universe_validation" / "feature_label_dataset.parquet"
OUT_DIR = ROOT / "data" / "derived" / "phase_i_economic_validation"

FAR_FUTURE_UTC = dt.datetime(2035, 1, 1, tzinfo=dt.timezone.utc)
HORIZONS = (5, 15, 30, 60)


@dataclasses.dataclass(frozen=True)
class NextBarEntryLabel:
    entry_price: float
    exit_price: float
    move: float
    label_class: str

    UP = "UP"
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"


def next_bar_entry_label(bars: list[bar_math.Bar], t_index: int, *, horizon_bars: int, vol_normalizer: float | None, threshold: float) -> NextBarEntryLabel | None:
    if not vol_normalizer or t_index + horizon_bars >= len(bars):
        return None
    t_date = bars[t_index].trading_date
    entry_bar = bars[t_index + 1]
    window = bars[t_index + 1 : t_index + 1 + horizon_bars]
    if entry_bar.trading_date != t_date or any(b.trading_date != t_date for b in window):
        return None

    entry_price = entry_bar.open
    exit_price = window[-1].close
    move = (exit_price / entry_price) - 1.0
    normalized_move = move / vol_normalizer
    if normalized_move >= threshold:
        label_class = NextBarEntryLabel.UP
    elif normalized_move <= -threshold:
        label_class = NextBarEntryLabel.DOWN
    else:
        label_class = NextBarEntryLabel.NEUTRAL
    return NextBarEntryLabel(entry_price=entry_price, exit_price=exit_price, move=move, label_class=label_class)


def load_all_bars(session, security_id: int) -> list[bar_math.Bar]:
    views = select_bars_as_of(session, security_id=security_id, timeframe="1m", as_of_utc=FAR_FUTURE_UTC)
    return [bar_math.from_canonical(v) for v in views]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "next_bar_entry_dataset.parquet")
    args = parser.parse_args()

    load_dotenv()
    base = pd.read_parquet(DATASET_FILE, columns=["symbol", "security_id", "bar_start_utc", "realized_volatility_20"])
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
                vol_normalizer = row["realized_volatility_20"]
                rec: dict[str, Any] = {"symbol": symbol_name, "security_id": sid, "bar_start_utc": row["bar_start_utc"]}
                for h in HORIZONS:
                    lbl = next_bar_entry_label(bars, t_index, horizon_bars=h, vol_normalizer=vol_normalizer, threshold=0.5)
                    rec[f"nextbar_entry_price_{h}m"] = lbl.entry_price if lbl else None
                    rec[f"nextbar_normalized_move_{h}m"] = (lbl.move / vol_normalizer) if lbl and vol_normalizer else None
                    rec[f"nextbar_label_class_{h}m"] = lbl.label_class if lbl else None
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
