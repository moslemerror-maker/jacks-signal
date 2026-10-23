from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from backend.app.config import load_dotenv
from backend.app.data.raw.adapter import IngestStatus, ingest
from backend.app.data.raw.store import RawStore
from backend.app.data.sources import dhan_equity_historical as deh
from backend.app.data.sources import dhan_security_master as dsm
from backend.app.data.sources.provider_credentials import DhanCredentials, MissingCredentialsError

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "derived" / "s1_2_pilot_backfill" / "report.json"

FROM_DATE = dt.date(2026, 8, 26)
TO_DATE = dt.date(2026, 9, 25)
INTERVAL = 1

PILOT_SYMBOLS = (
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY",
    "SBIN", "ITC", "LT", "AXISBANK", "IDEA",
)


def run(
    credentials: DhanCredentials, *, store: RawStore | None, symbols: tuple[str, ...] = PILOT_SYMBOLS,
    master: Any = None, adapter: Any = None, pace_seconds: float = 1.0,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "requested_from": FROM_DATE.isoformat(), "requested_to": TO_DATE.isoformat(), "interval_minutes": INTERVAL,
        "pilot_symbols_requested": list(symbols), "resolved": {}, "resolution_failures": [], "acquisitions": [], "failures": [],
    }

    if master is None:
        master = dsm.fetch_nse_mainboard_equity_master()
    try:
        resolved = dsm.resolve_security_ids(list(symbols), master)
    except dsm.SymbolNotFoundError as exc:
        resolved = dsm.resolve_security_ids([s for s in symbols if s not in exc.symbols], master) if len(exc.symbols) < len(symbols) else {}
        report["resolution_failures"] = exc.symbols
    report["resolved"] = resolved

    if adapter is None:
        adapter = deh.DhanEquityHistoricalAdapter(credentials)
    sizes: list[int] = []
    for i, (symbol, security_id) in enumerate(resolved.items()):
        if i > 0 and pace_seconds > 0:
            time.sleep(pace_seconds)
        source_key = deh.source_key_for(security_id, FROM_DATE, TO_DATE, INTERVAL)
        acquired_at = dt.datetime.now(dt.timezone.utc)
        try:
            if store is not None:
                result = ingest(store, adapter, deh.DATASET, source_key, now=acquired_at)
                status, sha256, size = result.status.value, result.sha256, (store.latest("dhan", deh.DATASET, source_key).size if result.sha256 else None)
            else:
                fetched = adapter.fetch(deh.DATASET, source_key)
                if fetched is None:
                    status, sha256, size = "NOT_AVAILABLE", None, None
                else:
                    adapter.validate(deh.DATASET, fetched.content)
                    status, sha256, size = "FETCHED_NOT_STORED", hashlib.sha256(fetched.content).hexdigest(), len(fetched.content)
            entry = {
                "symbol": symbol, "security_id": security_id, "requested_from": FROM_DATE.isoformat(), "requested_to": TO_DATE.isoformat(),
                "interval_minutes": INTERVAL, "source_key": source_key, "status": status, "sha256": sha256, "raw_response_size_bytes": size,
                "acquired_at_utc": acquired_at.isoformat(), "provider_response_status": "success" if status not in ("QUARANTINED", "NOT_AVAILABLE") else status,
            }
            if size is not None:
                sizes.append(size)
            if status == "QUARANTINED":
                report["failures"].append({**entry, "reason": result.reason})
            else:
                report["acquisitions"].append(entry)
        except Exception as exc:
            report["failures"].append({
                "symbol": symbol, "security_id": security_id, "source_key": source_key, "status": "ERROR",
                "reason": f"{type(exc).__name__}: {exc}", "acquired_at_utc": acquired_at.isoformat(),
            })

    report["summary"] = {
        "instruments_requested": len(symbols), "resolution_failures": len(report["resolution_failures"]),
        "successful_acquisitions": len(report["acquisitions"]), "failed_acquisitions": len(report["failures"]),
        "total_raw_artifacts": len(sizes), "total_storage_bytes": sum(sizes),
        "average_artifact_size_bytes": (sum(sizes) / len(sizes)) if sizes else None,
        "largest_artifact_bytes": max(sizes) if sizes else None, "smallest_artifact_bytes": min(sizes) if sizes else None,
    }
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-store", action="store_true", help="fetch and validate only; do not write to RawStore")
    args = ap.parse_args(argv)
    load_dotenv()
    try:
        creds = DhanCredentials.from_env()
    except MissingCredentialsError as exc:
        print(f"NOT CONFIGURED: {exc}")
        return 2

    store = None if args.no_store else RawStore(ROOT / "data" / "raw" / "providers")
    report = run(creds, store=store)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))

    for a in report["acquisitions"]:
        print(f"  OK  {a['symbol']:10} sec_id={a['security_id']:<7} status={a['status']:<14} size={a['raw_response_size_bytes']}B sha256={a['sha256'][:12]}...")
    for f in report["failures"]:
        print(f"  FAIL {f['symbol']:10} sec_id={f.get('security_id')}  {f['status']}: {f['reason']}")
    if report["resolution_failures"]:
        print(f"  UNRESOLVED symbols (not in security master): {report['resolution_failures']}")
    print(json.dumps(report["summary"], indent=2))
    print("report:", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
