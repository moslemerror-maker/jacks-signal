from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

from backend.app.data.normalizers import dhan_intraday_equity as norm
from backend.app.data.raw.store import RawStore
from backend.app.data.sources import dhan_equity_historical as deh

ROOT = Path(__file__).resolve().parents[1]
S1_2_REPORT = ROOT / "data" / "derived" / "s1_2_pilot_backfill" / "report.json"
OUT = ROOT / "data" / "derived" / "s1_3_normalization_pilot" / "report.json"


def _asdict(x: Any) -> Any:
    if dataclasses.is_dataclass(x):
        return {k: _asdict(v) for k, v in dataclasses.asdict(x).items()}
    if isinstance(x, (list, tuple)):
        return [_asdict(v) for v in x]
    return x


def run(store: RawStore, resolved: dict[str, int], *, from_date, to_date) -> dict[str, Any]:
    report: dict[str, Any] = {"requested_from": from_date.isoformat(), "requested_to": to_date.isoformat(), "symbols": {}, "summary": {}}
    total_raw = total_normalized = total_rejected = 0
    all_clean = True

    for symbol, security_id in resolved.items():
        source_key = deh.source_key_for(security_id, from_date, to_date)
        bars, rejected = norm.normalize_dhan_intraday_artifact(store, source_key=source_key, trading_symbol=symbol)
        reconciliation = norm.reconcile(store, source_key=source_key, bars=bars, rejected=rejected)
        sessions = norm.session_boundary_reports(bars)
        incomplete_sessions = [s for s in sessions if not s.is_complete_360]

        total_raw += reconciliation.raw_row_count
        total_normalized += len(bars)
        total_rejected += len(rejected)
        if rejected or reconciliation.discrepancies or incomplete_sessions:
            all_clean = False

        report["symbols"][symbol] = {
            "security_id": security_id, "source_key": source_key,
            "raw_row_count": reconciliation.raw_row_count, "normalized_row_count": len(bars), "rejected_row_count": len(rejected),
            "rejected_rows": [_asdict(r) for r in rejected],
            "reconciliation": _asdict(reconciliation),
            "session_count": len(sessions), "complete_360_sessions": len(sessions) - len(incomplete_sessions),
            "incomplete_sessions": [_asdict(s) for s in incomplete_sessions],
            "earliest_bar_ist": bars[0].bar_start_ist.isoformat() if bars else None,
            "latest_bar_ist": bars[-1].bar_start_ist.isoformat() if bars else None,
        }

    report["summary"] = {
        "instruments_normalized": len(resolved), "total_raw_rows": total_raw, "total_normalized_rows": total_normalized,
        "total_rejected_rows": total_rejected, "expected_bars_baseline": len(resolved) * 22 * 360, "all_clean": all_clean,
    }
    return report


def main(argv=None) -> int:
    if not S1_2_REPORT.exists():
        print(f"NOT AVAILABLE: {S1_2_REPORT} does not exist - run S1.2's pilot backfill first. Nothing was normalized.")
        return 2
    s1_2 = json.loads(S1_2_REPORT.read_text())
    resolved = s1_2["resolved"]
    from datetime import date
    from_date, to_date = date.fromisoformat(s1_2["requested_from"]), date.fromisoformat(s1_2["requested_to"])

    store = RawStore(ROOT / "data" / "raw" / "providers")
    report = run(store, resolved, from_date=from_date, to_date=to_date)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))

    for symbol, s in report["symbols"].items():
        status = "CLEAN" if not s["rejected_rows"] and not s["reconciliation"]["discrepancies"] else "ISSUES"
        print(f"  {symbol:10} raw={s['raw_row_count']:<6} normalized={s['normalized_row_count']:<6} rejected={s['rejected_row_count']:<3} "
              f"sessions={s['session_count']} complete360={s['complete_360_sessions']}  [{status}]")
    print(json.dumps(report["summary"], indent=2))
    print("report:", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
