"""Phase V Part 6: builds the true MFE/MAE opportunity dataset, reusing Part 4's state-feature library as the
base and Phase T's own already-proven path-excursion function for the genuinely new piece this phase needs.

WHY NO NEW PURE FUNCTION, NO NEW EQUIVALENCE TEST: `compute_paths_for_symbol` (scripts/run_phase_t_build_
event_paths.py) already computes, from raw 1-minute bars, entry_price_o{k} and gross_return/mfe/mae_o{k}_h{h}
for every (entry_offset, horizon) in ENTRY_OFFSETS x HORIZONS - imported and called UNCHANGED here. It was
proven bit-identical to labels.py::entry_path_result on synthetic data in Phase T and has not been modified
since. This script's only new work is: (a) re-querying raw bars for the full liquid universe (not just Phase
R's extreme-mover population, which is all that function had ever been applied to before), and (b) merging
offset=1 (next-bar-open, the project's standard executable entry) results onto Part 4's dense state-feature
parquet.

ENTRY CONVENTION: offset=1 (open of the bar immediately after the state is observed) - the EXECUTABLE
timestamp, never the same-bar close the state itself was measured at (the INFORMATION timestamp). This is
Phase T's own established distinction, reused, not redefined.

MFE/MAE ARE DIRECTION-AGNOSTIC AT THIS STAGE (upside/downside excursion, exactly entry_path_result's own
contract) - LONG vs SHORT interpretation (favorable=upside vs favorable=downside) is applied downstream in
the statistics script, per state, never baked into this build.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from backend.app.config import load_dotenv
from backend.app.db.session import get_session
from scripts.run_phase_t_build_event_paths import ENTRY_OFFSETS, HORIZONS, compute_paths_for_symbol
from scripts.run_phase_v4_strategy_discovery import add_feature_columns, load_dataset as load_part4_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "derived" / "phase_v6_opportunity" / "dense_opportunity_dataset.parquet"

ENTRY_OFFSET_USED = 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    load_dotenv()
    liquid = load_part4_dataset()
    liquid = add_feature_columns(liquid)
    print(f"loaded liquid population via Part 4's pipeline: {len(liquid)} rows, {liquid['security_id'].nunique()} symbols")
    security_ids = sorted(liquid["security_id"].unique().tolist())

    session = get_session()
    t0 = time.monotonic()
    merged_results = []
    try:
        for i, sid in enumerate(security_ids, 1):
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
            keep_cols = ["bar_start_utc", "trading_date", f"entry_price_o{ENTRY_OFFSET_USED}"] + [
                f"{metric}_o{ENTRY_OFFSET_USED}_h{h}" for metric in ("gross_return", "mfe", "mae") for h in HORIZONS
            ]
            sym_base = liquid[liquid["security_id"] == sid]
            merged = sym_base.merge(paths[keep_cols], on=["bar_start_utc", "trading_date"], how="left", validate="one_to_one")
            merged_results.append(merged)

            if i % 25 == 0:
                print(f"  {i}/{len(security_ids)} symbols processed, {time.monotonic() - t0:.0f}s elapsed")
    finally:
        session.close()

    full = pd.concat(merged_results, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(args.out, index=False)
    n_defined = full[f"mfe_o{ENTRY_OFFSET_USED}_h15"].notna().sum()
    print(f"wrote {len(full)} rows, {full.shape[1]} columns, {n_defined} with defined h15 MFE/MAE -> {args.out.relative_to(ROOT)} ({time.monotonic() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
