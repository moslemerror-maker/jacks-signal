from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

from backend.app.config import load_dotenv
from backend.app.data.backfill.chunk_planner import plan_chunks
from backend.app.data.backfill.executor import ChunkOutcome, ExecutorConfig, execute_plan
from backend.app.data.calendar import equity_trading_calendar as cal
from backend.app.data.normalizers import dhan_intraday_equity as norm
from backend.app.data.raw.store import RawStore
from backend.app.data.sources import dhan_equity_historical as deh
from backend.app.data.sources.provider_credentials import DhanCredentials, MissingCredentialsError

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_FILE = ROOT / "data" / "derived" / "phase_d_research_dataset" / "universe.json"
OUT_DIR = ROOT / "data" / "derived" / "phase_d_research_dataset"

TO_DATE = dt.date(2026, 9, 25)
FROM_DATE = TO_DATE - dt.timedelta(days=365)
RECENT_SESSION_GRACE_DAYS = 3


def run(store: RawStore, adapter, securities: list[tuple[int, str]], *, pace_seconds: float = 1.0) -> dict[str, Any]:
    chunks = plan_chunks(securities, overall_from=FROM_DATE, overall_to=TO_DATE, chunk_days=90)

    holidays = cal.load_equity_holiday_dates(ROOT / "data" / "raw" / "nse_calendar" / "holiday_master.json")
    expected_sessions = cal.expected_trading_days(FROM_DATE, TO_DATE, holidays)
    expected_set = set(expected_sessions)
    recent_cutoff = TO_DATE - dt.timedelta(days=RECENT_SESSION_GRACE_DAYS)

    t0 = time.monotonic()
    plan_result = execute_plan(store, adapter, chunks, config=ExecutorConfig(pace_seconds=pace_seconds))
    elapsed_total = time.monotonic() - t0

    by_security: dict[int, dict[str, Any]] = {}
    for r in plan_result.results:
        sec_id = r.chunk.security_id
        entry = by_security.setdefault(sec_id, {
            "trading_symbol": r.chunk.trading_symbol, "chunks": [], "all_dates": set(), "shape_counts": {},
            "total_normalized_bars": 0, "total_rejected_rows": 0, "reconciliation_clean": True, "discrepancies": [],
        })
        chunk_info = {
            "from_date": r.chunk.from_date.isoformat(), "to_date": r.chunk.to_date.isoformat(),
            "outcome": r.outcome.value, "attempts": r.attempts, "elapsed_seconds": round(r.elapsed_seconds, 3),
            "raw_bar_count": r.bar_count, "source_key": r.source_key, "detail": r.detail,
        }
        entry["chunks"].append(chunk_info)

        if r.outcome in (ChunkOutcome.SUCCESS, ChunkOutcome.SKIPPED_EXISTING):
            bars, rejected = norm.normalize_dhan_intraday_artifact(store, source_key=r.source_key, trading_symbol=r.chunk.trading_symbol)
            reconciliation = norm.reconcile(store, source_key=r.source_key, bars=bars, rejected=rejected)
            sessions = norm.session_boundary_reports(bars)
            for s in sessions:
                entry["all_dates"].add(s.trading_date)
                entry["shape_counts"][s.session_shape.value] = entry["shape_counts"].get(s.session_shape.value, 0) + 1
            entry["total_normalized_bars"] += len(bars)
            entry["total_rejected_rows"] += len(rejected)
            if reconciliation.discrepancies:
                entry["reconciliation_clean"] = False
                entry["discrepancies"].extend(reconciliation.discrepancies)

    per_symbol: dict[str, Any] = {}
    total_normalized = total_rejected = 0
    total_missing_trailing = total_missing_gap = 0
    shape_totals: dict[str, int] = {}
    recon_failures = []

    for sec_id, entry in by_security.items():
        missing = sorted(expected_set - entry["all_dates"])
        trailing = [d for d in missing if d >= recent_cutoff]
        gap = [d for d in missing if d < recent_cutoff]
        total_missing_trailing += len(trailing)
        total_missing_gap += len(gap)
        total_normalized += entry["total_normalized_bars"]
        total_rejected += entry["total_rejected_rows"]
        for shape, count in entry["shape_counts"].items():
            shape_totals[shape] = shape_totals.get(shape, 0) + count
        if not entry["reconciliation_clean"]:
            recon_failures.append({"security_id": sec_id, "symbol": entry["trading_symbol"], "discrepancies": entry["discrepancies"]})

        per_symbol[entry["trading_symbol"]] = {
            "security_id": sec_id, "chunks": entry["chunks"],
            "sessions_acquired": len(entry["all_dates"]), "expected_sessions": len(expected_sessions),
            "missing_sessions_trailing_provider_availability": [d.isoformat() for d in trailing],
            "missing_sessions_gap": [d.isoformat() for d in gap],
            "session_shape_counts": entry["shape_counts"],
            "normalized_bar_count": entry["total_normalized_bars"], "rejected_row_count": entry["total_rejected_rows"],
            "reconciliation_clean": entry["reconciliation_clean"],
        }

    return {
        "requested_from": FROM_DATE.isoformat(), "requested_to": TO_DATE.isoformat(),
        "expected_trading_sessions": len(expected_sessions), "recent_session_grace_days": RECENT_SESSION_GRACE_DAYS,
        "elapsed_total_seconds": round(elapsed_total, 3),
        "plan_summary": plan_result.summary, "stopped_early": plan_result.stopped_early, "stop_reason": plan_result.stop_reason,
        "per_symbol": per_symbol,
        "totals": {
            "symbols_planned": len(securities), "total_chunks": len(chunks),
            "total_normalized_bars": total_normalized, "total_rejected_rows": total_rejected,
            "session_shape_totals": shape_totals,
            "total_missing_sessions_trailing_provider_availability": total_missing_trailing,
            "total_missing_sessions_gap": total_missing_gap,
            "reconciliation_failure_count": len(recon_failures),
        },
        "reconciliation_failures": recon_failures,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report.json")
    args = ap.parse_args(argv)

    load_dotenv()
    try:
        creds = DhanCredentials.from_env()
    except MissingCredentialsError as exc:
        print(f"NOT CONFIGURED: {exc}")
        return 2

    universe_data = json.loads(UNIVERSE_FILE.read_text())
    universe = universe_data["universe"]
    securities = [(u["security_id"], u["symbol"]) for u in universe]
    print(f"universe: {len(securities)} securities, window {FROM_DATE} -> {TO_DATE}")

    store = RawStore(ROOT / "data" / "raw" / "providers")
    adapter = deh.DhanEquityHistoricalAdapter(creds)

    report = run(store, adapter, securities)

    out_path = OUT_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))

    print(json.dumps(report["plan_summary"], indent=2))
    print(json.dumps(report["totals"], indent=2))
    print(f"stopped_early={report['stopped_early']} reason={report['stop_reason']}")
    print("report:", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
