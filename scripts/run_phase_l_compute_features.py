"""Phase L, Step 1: compute the two genuinely NEW candidate features - G5 (directional_pressure) and
G6 (opening_range_position) - across the same 298-symbol, frozen Phase H sample every prior phase's
compute scripts have used. G1, G2, G3, G4, G7, G8, G9, G10 all reuse columns already persisted in the
Phase H/J/K datasets and need no new computation at all - only G5/G6 require reloading per-symbol bar
history, done here in one lean, additive pass.

Both functions are CHEAP relative to Phase J's own compute pass: `directional_pressure` only ever
looks at the trailing 10 bars, and `opening_range_position` only needs today's first 15 bars (via
`today_bars`, an early-terminating reverse scan) - neither touches the expensive full-history swing-
point/level-collection logic that dominated Phase J's runtime, so no HISTORY_WINDOW_BARS trimming is
needed here.

Zero new Dhan calls - reads only the already-persisted `intraday_equity_bars` table via the existing
`select_bars_as_of` point-in-time primitive.
"""
from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.config import load_dotenv
from backend.app.db.session import get_session
from backend.app.research.intraday_features import bar_math, mover_mechanics
from backend.app.research.intraday_point_in_time import select_bars_as_of

ROOT = Path(__file__).resolve().parents[1]
PHASE_H_DATASET = ROOT / "data" / "derived" / "phase_h_full_universe_validation" / "feature_label_dataset.parquet"
OUT_DIR = ROOT / "data" / "derived" / "phase_l_direction_discovery"

FAR_FUTURE_UTC = dt.datetime(2035, 1, 1, tzinfo=dt.timezone.utc)


def compute_g5_g6_row(bars_up_to_t: list[bar_math.Bar]) -> dict[str, Any]:
    return {
        "g5_directional_pressure_10": mover_mechanics.directional_pressure(bars_up_to_t, lookback_bars=10),
        "g6_opening_range_position": mover_mechanics.opening_range_position(bars_up_to_t, n_bars=15),
    }


def load_all_bars(session, security_id: int) -> list[bar_math.Bar]:
    views = select_bars_as_of(session, security_id=security_id, timeframe="1m", as_of_utc=FAR_FUTURE_UTC)
    return [bar_math.from_canonical(v) for v in views]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "g5_g6_dataset.parquet")
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
                bars_up_to_t = bars[: t_index + 1]
                rec = compute_g5_g6_row(bars_up_to_t)
                rec.update(symbol=symbol_name, security_id=sid, bar_start_utc=row["bar_start_utc"])
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
