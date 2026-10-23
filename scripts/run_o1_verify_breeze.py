from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

from backend.app.data.raw.artifact import check_component
from backend.app.data.raw.store import RawStore
from backend.app.data.sources.breeze_client import BreezeClient, BreezePackageNotInstalled, HistoricalOptionsRequest
from backend.app.data.sources.provider_credentials import BreezeCredentials, MissingCredentialsError

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "derived" / "o1_provider_verification" / "breeze.json"


def _iso_midnight_utc(d: dt.date) -> str:
    return dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).strftime("%Y-%m-%dT06:00:00.000Z")


def verify(client: BreezeClient, *, stock_code: str, expiry: dt.date, spot: float, strike_step: int, lookback_days: int, store: RawStore | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "stock_code": stock_code, "expiry": expiry.isoformat(), "spot": spot, "strikes": {}}
    strikes = [round(spot / strike_step) * strike_step + k * strike_step for k in (-1, 0, 1)]
    to_date = min(expiry, dt.date.today())
    from_date = to_date - dt.timedelta(days=lookback_days)
    for strike in strikes:
        for right in ("call", "put"):
            for interval in ("1day", "5minute"):
                req = HistoricalOptionsRequest(stock_code=stock_code, expiry_date=_iso_midnight_utc(expiry), strike_price=str(strike), right=right,
                                               interval=interval, from_date=_iso_midnight_utc(from_date), to_date=_iso_midnight_utc(to_date))
                try:
                    result = client.get_option_historical(req)
                except Exception as exc:
                    report["strikes"].setdefault(strike, {})[f"{right}_{interval}"] = {"error": str(exc)}
                    continue
                rows = result.get("Success") or []
                if store is not None:
                    key = check_component("k", f"{stock_code}_{expiry:%Y%m%d}_{strike}_{right}_{interval}")
                    store.put(provider="breeze", dataset="historical_options", source_key=key, content=json.dumps(result, default=str).encode(),
                             source_url="https://api.icicidirect.com/breezeapi/api/v1/historicalcharts", http_status=200 if result.get("Error") is None else 0,
                             response_headers={}, fetcher_version="run_o1_verify_breeze/1", media_type="application/json", license_status="UNCLEAR")
                sample = rows[0] if rows else {}
                report["strikes"].setdefault(strike, {})[f"{right}_{interval}"] = {
                    "error_field": result.get("Error"), "rows": len(rows), "has_ohlc": all(k in sample for k in ("open", "high", "low", "close")),
                    "has_volume": "volume" in sample, "has_open_interest": "open_interest" in sample and sample.get("open_interest") is not None,
                }
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock-code", default="NIFTY")
    ap.add_argument("--expiry", required=True, help="YYYY-MM-DD")
    ap.add_argument("--spot", type=float, required=True, help="an approximate spot level to centre the strike ladder on")
    ap.add_argument("--strike-step", type=int, default=100)
    ap.add_argument("--lookback-days", type=int, default=30)
    ap.add_argument("--no-store", action="store_true")
    args = ap.parse_args(argv)
    try:
        creds = BreezeCredentials.from_env()
    except MissingCredentialsError as exc:
        print(f"NOT CONFIGURED: {exc}")
        return 2
    try:
        client = BreezeClient(creds)
    except BreezePackageNotInstalled as exc:
        print(f"NOT CONFIGURED: {exc}")
        return 2
    store = None if args.no_store else RawStore(ROOT / "data" / "raw" / "providers")
    report = verify(client, stock_code=args.stock_code, expiry=dt.date.fromisoformat(args.expiry), spot=args.spot, strike_step=args.strike_step,
                    lookback_days=args.lookback_days, store=store)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, default=str))
    for strike, by_leg in report["strikes"].items():
        for leg, v in by_leg.items():
            print(f"  {strike} {leg:14} {v}")
    print("report:", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
