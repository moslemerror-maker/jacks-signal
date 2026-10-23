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
from backend.app.data.canonical.intraday_loader import upsert_intraday_bars
from backend.app.data.normalizers import dhan_intraday_equity as norm
from backend.app.data.raw.store import RawStore
from backend.app.data.sources import dhan_equity_historical as deh
from backend.app.data.sources.provider_credentials import DhanCredentials, MissingCredentialsError
from backend.app.db.session import get_session

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_FILE = ROOT / "data" / "derived" / "phase_b_capacity_validation" / "universe.json"
OUT_DIR = ROOT / "data" / "derived" / "phase_b_capacity_validation"

FROM_DATE = dt.date(2026, 8, 26)
TO_DATE = dt.date(2026, 9, 25)


def run(store: RawStore, adapter, securities: list[tuple[int, str]], *, persist_session=None, pace_seconds: float = 1.0) -> dict[str, Any]:
    chunks = plan_chunks(securities, overall_from=FROM_DATE, overall_to=TO_DATE, chunk_days=90)

    holidays = cal.load_equity_holiday_dates(ROOT / "data" / "raw" / "nse_calendar" / "holiday_master.json")
    expected_sessions = cal.expected_trading_days(FROM_DATE, TO_DATE, holidays)

    t0 = time.monotonic()
    plan_result = execute_plan(store, adapter, chunks, config=ExecutorConfig(pace_seconds=pace_seconds))
    elapsed_total = time.monotonic() - t0

    per_symbol: dict[str, Any] = {}
    anomalies: list[dict[str, Any]] = []
    total_normalized = total_persisted = total_rejected = 0

    for r in plan_result.results:
        sym = r.chunk.trading_symbol
        entry: dict[str, Any] = {
            "security_id": r.chunk.security_id, "outcome": r.outcome.value, "attempts": r.attempts,
            "elapsed_seconds": round(r.elapsed_seconds, 3), "raw_bar_count": r.bar_count, "source_key": r.source_key, "detail": r.detail,
        }
        if r.outcome in (ChunkOutcome.SUCCESS, ChunkOutcome.SKIPPED_EXISTING):
            try:
                bars, rejected = norm.normalize_dhan_intraday_artifact(store, source_key=r.source_key, trading_symbol=sym)
                reconciliation = norm.reconcile(store, source_key=r.source_key, bars=bars, rejected=rejected)
                sessions = norm.session_boundary_reports(bars)
                shape_counts: dict[str, int] = {}
                for s in sessions:
                    shape_counts[s.session_shape.value] = shape_counts.get(s.session_shape.value, 0) + 1

                entry.update({
                    "normalized_bar_count": len(bars), "rejected_row_count": len(rejected),
                    "sessions_acquired": len(sessions), "expected_sessions": len(expected_sessions),
                    "session_count_matches_calendar": len(sessions) == len(expected_sessions),
                    "session_shape_counts": shape_counts,
                    "reconciliation_clean": reconciliation.discrepancies == (),
                    "reconciliation_discrepancies": list(reconciliation.discrepancies),
                })
                total_normalized += len(bars)
                total_rejected += len(rejected)

                is_anomalous = (
                    len(rejected) > 0 or reconciliation.discrepancies != ()
                    or len(sessions) != len(expected_sessions)
                    or shape_counts.get("UNRECOGNIZED_SHAPE", 0) > 0
                )
                if is_anomalous:
                    anomalies.append({"symbol": sym, "security_id": r.chunk.security_id, "reason": entry})

                if persist_session is not None and bars:
                    persisted = upsert_intraday_bars(persist_session, bars)
                    persist_session.commit()
                    total_persisted += persisted
                    entry["persisted_count"] = persisted
            except Exception as exc:
                if persist_session is not None:
                    persist_session.rollback()
                entry["processing_error"] = f"{type(exc).__name__}: {exc}"
                anomalies.append({"symbol": sym, "security_id": r.chunk.security_id, "reason": entry})

        per_symbol[sym] = entry

    return {
        "requested_from": FROM_DATE.isoformat(), "requested_to": TO_DATE.isoformat(),
        "expected_trading_sessions": len(expected_sessions),
        "elapsed_total_seconds": round(elapsed_total, 3),
        "plan_summary": plan_result.summary, "stopped_early": plan_result.stopped_early, "stop_reason": plan_result.stop_reason,
        "per_symbol": per_symbol, "anomalies": anomalies,
        "totals": {"symbols_planned": len(securities), "total_normalized_bars": total_normalized, "total_rejected_rows": total_rejected, "total_persisted_bars": total_persisted},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-persist", action="store_true")
    ap.add_argument("--out", default=None, help="override the output report filename (for the idempotency re-run)")
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
    print(f"universe: {len(securities)} securities")

    store = RawStore(ROOT / "data" / "raw" / "providers")
    adapter = deh.DhanEquityHistoricalAdapter(creds)
    persist_session = None if args.no_persist else get_session()

    try:
        report = run(store, adapter, securities, persist_session=persist_session)
    finally:
        if persist_session is not None:
            persist_session.close()

    out_path = OUT_DIR / (args.out or "report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))

    print(json.dumps(report["plan_summary"], indent=2))
    print(json.dumps(report["totals"], indent=2))
    print(f"anomalous symbols: {len(report['anomalies'])}")
    print(f"stopped_early={report['stopped_early']} reason={report['stop_reason']}")
    print("report:", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
