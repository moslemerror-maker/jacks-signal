from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

from backend.app.data.raw.artifact import check_component
from backend.app.data.raw.store import RawStore
from backend.app.data.sources import dhan_client as dc
from backend.app.data.sources.provider_credentials import DhanCredentials, MissingCredentialsError

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "derived" / "o1_provider_verification" / "dhan.json"
STRIKES = ("ATM-3", "ATM", "ATM+3")


def _store(store: RawStore | None, key: str, body: Any) -> None:
    if store is None:
        return
    store.put(provider="dhan", dataset="rolling_option", source_key=check_component("k", key), content=json.dumps(body, default=str).encode(),
             source_url=f"{dc.BASE}/charts/rollingoption", http_status=200 if isinstance(body, dict) and body.get("status") == "success" else 0,
             response_headers={}, fetcher_version="run_o1_verify_dhan/2", media_type="application/json", license_status="UNCLEAR")


def verify(
    sdk_client: Any, *, security_id: int, exchange_segment: str, instrument: str = dc.NIFTY_INSTRUMENT, store: RawStore | None = None,
    from_date: str, to_date: str, strikes: tuple[str, ...] = STRIKES, expiry_code: dc.ExpiryCode = 1, expiry_flag: str = "WEEK",
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "security_id": security_id, "exchange_segment": exchange_segment,
        "expiry_code": expiry_code, "expiry_flag": expiry_flag, "requested_window": {"from": from_date, "to": to_date}, "strikes": {},
    }
    for strike in strikes:
        for cp in ("CALL", "PUT"):
            response = dc.expired_options_data_via_sdk(
                sdk_client, security_id=security_id, exchange_segment=exchange_segment, instrument_type=instrument,
                expiry_flag=expiry_flag, expiry_code=expiry_code, strike=strike, option_type=cp,
                required_data=list(dc.REQUIRED_DATA), from_date=from_date, to_date=to_date, interval=1,
            )
            key = f"{strike}_{cp}_{from_date}".replace("-", "").replace("+", "p")
            _store(store, key, response)
            side = "ce" if cp == "CALL" else "pe"
            leg = ((response or {}).get("data") or {}).get("data", {}).get(side) or {}
            close = leg.get("close") or []
            report["strikes"][f"{strike}_{cp}"] = {
                "status": (response or {}).get("status"), "rows": len(close),
                "has_open": bool(leg.get("open")), "has_high": bool(leg.get("high")), "has_low": bool(leg.get("low")),
                "has_close": bool(close), "has_volume": bool(leg.get("volume")), "has_oi": bool(leg.get("oi")),
                "has_iv": bool(leg.get("iv")), "has_strike_series": bool(leg.get("strike")), "has_spot": bool(leg.get("spot")),
                "distinct_strike_values_returned": sorted(set(leg.get("strike", []))) if leg.get("strike") else None,
            }
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--security-id", type=int, default=dc.NIFTY_SECURITY_ID)
    ap.add_argument("--exchange-segment", default=dc.NIFTY_EXCHANGE_SEGMENT)
    ap.add_argument("--expiry-code", type=int, default=1, choices=(1, 2, 3), help="1=Near Expiry, 2=Next Expiry, 3=Far Expiry")
    ap.add_argument("--days-back", type=int, default=30, help="Dhan documents up to 30 days per call")
    ap.add_argument("--no-store", action="store_true")
    args = ap.parse_args(argv)
    try:
        creds = DhanCredentials.from_env()
    except MissingCredentialsError as exc:
        print(f"NOT CONFIGURED: {exc}")
        return 2

    from dhanhq import DhanContext, dhanhq

    sdk_client = dhanhq(DhanContext(creds.client_id, creds.access_token))
    to_date = dt.date.today()
    from_date = to_date - dt.timedelta(days=args.days_back)
    store = None if args.no_store else RawStore(ROOT / "data" / "raw" / "providers")
    report = verify(sdk_client, security_id=args.security_id, exchange_segment=args.exchange_segment, store=store,
                    from_date=from_date.isoformat(), to_date=to_date.isoformat(), expiry_code=args.expiry_code)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, default=str))
    for k, v in report["strikes"].items():
        print(f"  {k:12} status {v['status']} rows {v['rows']} OHLC {v['has_open'] and v['has_close']} vol {v['has_volume']} oi {v['has_oi']} iv {v['has_iv']}")
    print("report:", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
