"""Phase J, Step 2: compute the new mover_mechanics features (compression->expansion, opening-range
breakout events, support/resistance touch-count) across the same 298-symbol universe and sampled
rows the frozen Phase H dataset already covers. Additive only - joins onto the frozen dataset by
(security_id, bar_start_utc), exactly like Phase I's I3 next-bar-entry dataset.

LEVELS USED (kept deliberately narrow, not "every possible level", per the phase's anti-data-mining
instruction): opening-range high/low (the single most canonical, well-defined intraday breakout
level) for the breakout-event features, and the nearest support/resistance level (the SAME
already-collected candidate set `nearest_level_distance` uses in the existing pipeline - prior-day
H/L/C, session H/L so far, opening range H/L, confirmed swing points) for the touch-count features.

Zero new Dhan calls - reads only from the already-persisted `intraday_equity_bars` table via the
existing `select_bars_as_of` point-in-time primitive, exactly as Phase H/I did.
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
from backend.app.research.intraday_features import bar_math, mover_mechanics, vwap_levels
from backend.app.research.intraday_point_in_time import select_bars_as_of

ROOT = Path(__file__).resolve().parents[1]
PHASE_H_DATASET = ROOT / "data" / "derived" / "phase_h_full_universe_validation" / "feature_label_dataset.parquet"
OUT_DIR = ROOT / "data" / "derived" / "phase_j_mover_detection"

FAR_FUTURE_UTC = dt.datetime(2035, 1, 1, tzinfo=dt.timezone.utc)

HISTORY_WINDOW_BARS = 5000
LEVEL_WINDOW_BARS = 750


def _nearest_levels(bars_up_to_t: list[bar_math.Bar]) -> tuple[float | None, float | None]:
    current_price = bars_up_to_t[-1].close
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

    above = [lv for lv in levels if lv > current_price]
    below = [lv for lv in levels if lv < current_price]
    resistance = min(above, key=lambda lv: abs(lv - current_price)) if above else None
    support = min(below, key=lambda lv: abs(lv - current_price)) if below else None
    return support, resistance


def compute_mover_mechanics_row(bars_up_to_t: list[bar_math.Bar]) -> dict[str, Any]:
    level_window = bars_up_to_t[-LEVEL_WINDOW_BARS:] if len(bars_up_to_t) > LEVEL_WINDOW_BARS else bars_up_to_t
    or_hl = vwap_levels.opening_range_high_low(level_window)
    or_high, or_low = or_hl if or_hl is not None else (None, None)
    support, resistance = _nearest_levels(level_window)

    return {
        "compression_then_expansion": mover_mechanics.compression_then_expansion(bars_up_to_t),
        "recent_breakout_up_opening_range": mover_mechanics.recent_breakout_up(bars_up_to_t, or_high, within_bars=5),
        "recent_breakout_down_opening_range": mover_mechanics.recent_breakout_down(bars_up_to_t, or_low, within_bars=5),
        "level_touch_count_support": mover_mechanics.level_touch_count(bars_up_to_t, support),
        "level_touch_count_resistance": mover_mechanics.level_touch_count(bars_up_to_t, resistance),
    }


def load_all_bars(session, security_id: int) -> list[bar_math.Bar]:
    views = select_bars_as_of(session, security_id=security_id, timeframe="1m", as_of_utc=FAR_FUTURE_UTC)
    return [bar_math.from_canonical(v) for v in views]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "mover_mechanics_dataset.parquet")
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
                lo = max(0, t_index - HISTORY_WINDOW_BARS + 1)
                bars_up_to_t = bars[lo : t_index + 1]
                rec = compute_mover_mechanics_row(bars_up_to_t)
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
