from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

from backend.app.data.canonical.intraday_loader import upsert_intraday_bars
from backend.app.data.normalizers import dhan_intraday_equity as norm
from backend.app.data.raw.store import RawStore
from backend.app.data.sources import dhan_equity_historical as deh
from backend.app.db.session import get_session

ROOT = Path(__file__).resolve().parents[1]
S1_2_REPORT = ROOT / "data" / "derived" / "s1_2_pilot_backfill" / "report.json"
OUT = ROOT / "data" / "derived" / "s1_4_persistence_pilot" / "report.json"


def run(session, store: RawStore, resolved: dict[str, int], *, from_date: date, to_date: date) -> dict[str, Any]:
    report: dict[str, Any] = {"requested_from": from_date.isoformat(), "requested_to": to_date.isoformat(), "symbols": {}}
    total_normalized = total_persisted = total_rejected = 0

    for symbol, security_id in resolved.items():
        source_key = deh.source_key_for(security_id, from_date, to_date)
        bars, rejected = norm.normalize_dhan_intraday_artifact(store, source_key=source_key, trading_symbol=symbol)
        persisted = upsert_intraday_bars(session, bars)
        total_normalized += len(bars)
        total_persisted += persisted
        total_rejected += len(rejected)
        report["symbols"][symbol] = {
            "security_id": security_id, "normalized_row_count": len(bars), "persisted_row_count": persisted, "rejected_row_count": len(rejected),
        }

    report["summary"] = {
        "instruments": len(resolved), "total_normalized_rows": total_normalized, "total_persisted_rows": total_persisted,
        "total_rejected_rows": total_rejected, "expected_baseline": len(resolved) * 22 * 360,
    }
    return report


def main(argv=None) -> int:
    if not S1_2_REPORT.exists():
        print(f"NOT AVAILABLE: {S1_2_REPORT} does not exist - run S1.2's pilot backfill first. Nothing was persisted.")
        return 2
    s1_2 = json.loads(S1_2_REPORT.read_text())
    resolved = s1_2["resolved"]
    from_date, to_date = date.fromisoformat(s1_2["requested_from"]), date.fromisoformat(s1_2["requested_to"])

    store = RawStore(ROOT / "data" / "raw" / "providers")
    session = get_session()
    try:
        report = run(session, store, resolved, from_date=from_date, to_date=to_date)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))

    for symbol, s in report["symbols"].items():
        print(f"  {symbol:10} normalized={s['normalized_row_count']:<6} persisted={s['persisted_row_count']:<6} rejected={s['rejected_row_count']}")
    print(json.dumps(report["summary"], indent=2))
    print("report:", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
