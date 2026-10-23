from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

from sqlalchemy import text

from backend.app.data.canonical.intraday_loader import upsert_intraday_bars_batched
from backend.app.data.normalizers import dhan_intraday_equity as norm
from backend.app.data.raw.store import RawStore
from backend.app.db.session import get_session

ROOT = Path(__file__).resolve().parents[1]
PHASE_D_REPORT = ROOT / "data" / "derived" / "phase_d_research_dataset" / "report.json"
OUT = ROOT / "data" / "derived" / "phase_f_full_persistence" / "report.json"
BATCH_SIZE = 500


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume-log", default=None, help="write incremental per-symbol progress here for crash visibility")
    args = ap.parse_args(argv)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if args.resume_log:
        Path(args.resume_log).parent.mkdir(parents=True, exist_ok=True)

    store = RawStore(ROOT / "data" / "raw" / "providers")
    phase_d = json.loads(PHASE_D_REPORT.read_text())
    per_symbol_plan = phase_d["per_symbol"]
    print(f"symbols in Phase D report: {len(per_symbol_plan)}")

    session = get_session()
    started_at = dt.datetime.now(dt.timezone.utc)
    t0 = time.monotonic()

    results: dict[str, Any] = {}
    totals = {
        "symbols_processed": 0, "symbols_skipped_already_persisted": 0, "symbols_failed": 0,
        "total_expected_canonical_bars": 0, "total_persisted_upserts": 0, "total_rejected_rows": 0,
        "batches_executed_estimate": 0,
    }

    for i, (symbol, plan) in enumerate(per_symbol_plan.items(), 1):
        sec_id = plan["security_id"]
        expected_bars = plan["normalized_bar_count"]
        totals["total_expected_canonical_bars"] += expected_bars
        totals["total_rejected_rows"] += plan["rejected_row_count"]

        try:
            existing = session.execute(
                text("SELECT COUNT(*) FROM intraday_equity_bars WHERE security_id = :sid AND timeframe = '1m'"), {"sid": sec_id}
            ).scalar_one()

            if existing == expected_bars and expected_bars > 0:
                totals["symbols_skipped_already_persisted"] += 1
                results[symbol] = {"security_id": sec_id, "status": "SKIPPED_ALREADY_PERSISTED", "expected_bars": expected_bars, "db_rows": existing}
                continue

            all_bars = []
            for c in plan["chunks"]:
                if c["outcome"] not in ("SUCCESS", "SKIPPED_EXISTING"):
                    continue
                bars, rejected = norm.normalize_dhan_intraday_artifact(store, source_key=c["source_key"], trading_symbol=symbol)
                all_bars.extend(bars)

            t_start = time.monotonic()
            upserted = upsert_intraday_bars_batched(session, all_bars, batch_size=BATCH_SIZE)
            session.commit()
            elapsed = time.monotonic() - t_start

            post_count = session.execute(
                text("SELECT COUNT(*) FROM intraday_equity_bars WHERE security_id = :sid AND timeframe = '1m'"), {"sid": sec_id}
            ).scalar_one()
            first_last = session.execute(
                text("SELECT MIN(bar_start_utc), MAX(bar_start_utc) FROM intraday_equity_bars WHERE security_id = :sid AND timeframe = '1m'"),
                {"sid": sec_id},
            ).one()

            totals["symbols_processed"] += 1
            totals["total_persisted_upserts"] += upserted
            totals["batches_executed_estimate"] += -(-len(all_bars) // BATCH_SIZE)

            results[symbol] = {
                "security_id": sec_id, "status": "PERSISTED", "canonical_bars": len(all_bars), "upserted": upserted,
                "db_rows_after": post_count, "elapsed_seconds": round(elapsed, 3),
                "first_timestamp_utc": first_last[0].isoformat() if first_last[0] else None,
                "last_timestamp_utc": first_last[1].isoformat() if first_last[1] else None,
            }
        except Exception as exc:
            session.rollback()
            totals["symbols_failed"] += 1
            results[symbol] = {"security_id": sec_id, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
            print(f"[{i}/{len(per_symbol_plan)}] FAILED {symbol}: {type(exc).__name__}: {exc}")
            continue

        if i % 10 == 0 or i == len(per_symbol_plan):
            elapsed_so_far = time.monotonic() - t0
            rate = totals["total_persisted_upserts"] / elapsed_so_far if elapsed_so_far > 0 else 0
            print(f"[{i}/{len(per_symbol_plan)}] {symbol}: {results[symbol]['status']} "
                  f"| cumulative persisted={totals['total_persisted_upserts']:,} | {elapsed_so_far/60:.1f} min elapsed | ~{rate:.0f} bars/sec")

        if args.resume_log:
            Path(args.resume_log).write_text(json.dumps({"last_index": i, "last_symbol": symbol, "totals": totals}, indent=2))

    total_elapsed = time.monotonic() - t0
    ended_at = dt.datetime.now(dt.timezone.utc)
    session.close()

    report = {
        "started_at_utc": started_at.isoformat(), "ended_at_utc": ended_at.isoformat(),
        "total_elapsed_seconds": round(total_elapsed, 2), "total_elapsed_minutes": round(total_elapsed / 60, 2),
        "batch_size": BATCH_SIZE, "totals": totals, "per_symbol": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(totals, indent=2))
    print(f"total elapsed: {total_elapsed/60:.1f} minutes")
    print("report:", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
