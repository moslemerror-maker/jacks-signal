from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

from dhanhq import DhanContext, dhanhq

from backend.app.data.raw.artifact import check_component
from backend.app.data.raw.store import RawStore
from backend.app.data.sources.provider_credentials import DhanCredentials, MissingCredentialsError

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "derived" / "o1_provider_verification" / "dhan_optionchain_sdk.json"
NIFTY_SECURITY_ID = 13
NIFTY_SEGMENT = dhanhq.INDEX
GREEK_FIELDS = ("delta", "theta", "gamma", "vega")
LEG_FIELDS = ("last_price", "top_bid_price", "top_ask_price", "volume", "oi", "implied_volatility")


def _store(store: RawStore | None, dataset: str, key: str, body: Any, source_url: str) -> None:
    if store is None:
        return
    store.put(provider="dhan", dataset=dataset, source_key=check_component("k", key), content=json.dumps(body, default=str).encode(),
             source_url=source_url, http_status=200 if isinstance(body, dict) and body.get("status") == "success" else 0, response_headers={},
             fetcher_version="run_o1_verify_dhan_optionchain_sdk/1", media_type="application/json", license_status="UNCLEAR")


def summarize(chain: dict) -> dict[str, Any]:
    outer = chain.get("data") or {}
    data = outer.get("data") or outer
    oc = data.get("oc") or {}
    per_strike = {}
    for strike, legs in oc.items():
        row: dict[str, Any] = {}
        for side in ("ce", "pe"):
            leg = legs.get(side) or {}
            row[side] = {f: (f in leg) for f in LEG_FIELDS} | {"greeks": {g: (g in (leg.get("greeks") or {})) for g in GREEK_FIELDS}}
        per_strike[strike] = row
    any_leg = next(iter(next(iter(oc.values()), {}).values()), {}) if oc else {}
    return {
        "status": chain.get("status"), "underlying_last_price_present": "last_price" in data, "strikes_returned": len(oc),
        "ce_present": sum(1 for v in oc.values() if "ce" in v), "pe_present": sum(1 for v in oc.values() if "pe" in v),
        "fields_present_on_a_sample_leg": sorted(any_leg.keys()) if any_leg else None, "per_strike_field_presence_sample": dict(list(per_strike.items())[:2]),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expiry")
    ap.add_argument("--no-store", action="store_true")
    args = ap.parse_args(argv)
    try:
        creds = DhanCredentials.from_env()
    except MissingCredentialsError as exc:
        print(f"NOT CONFIGURED: {exc}")
        return 2

    dhan = dhanhq(DhanContext(creds.client_id, creds.access_token))
    store = None if args.no_store else RawStore(ROOT / "data" / "raw" / "providers")

    report: dict[str, Any] = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "sdk_package": "dhanhq", "sdk_version": "2.2.0",
                              "security_id": NIFTY_SECURITY_ID, "segment": NIFTY_SEGMENT}

    expiry = args.expiry
    if not expiry:
        el = dhan.expiry_list(NIFTY_SECURITY_ID, NIFTY_SEGMENT)
        _store(store, "optionchain_expirylist", f"NIFTY_{dt.date.today():%Y%m%d}_sdk", el, "https://api.dhan.co/v2/optionchain/expirylist")
        dates = sorted((el or {}).get("data") or [])
        report["expiry_list_status"] = (el or {}).get("status")
        report["expiry_list_dates"] = dates
        if not dates:
            report["error"] = "expiry_list returned no dates; cannot proceed"
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(report, indent=1, default=str))
            print(json.dumps(report, indent=1, default=str))
            return 1
        expiry = dates[0]
    report["expiry_used"] = expiry
    report["expiry_selection"] = "nearest returned by expiry_list()" if not args.expiry else "explicit --expiry"

    print("Non-secret request about to be sent via the official SDK's option_chain():")
    print(json.dumps({"under_security_id": NIFTY_SECURITY_ID, "under_exchange_segment": NIFTY_SEGMENT, "expiry": expiry}, indent=1))

    chain = dhan.option_chain(NIFTY_SECURITY_ID, NIFTY_SEGMENT, expiry)
    _store(store, "optionchain", f"NIFTY_{expiry.replace('-', '')}_sdk", chain, "https://api.dhan.co/v2/optionchain")

    report["option_chain_raw_status_field"] = (chain or {}).get("status")
    report["option_chain_error"] = {k: chain.get(k) for k in ("errorType", "errorCode", "errorMessage") if isinstance(chain, dict) and k in chain} or None
    report["summary"] = summarize(chain or {})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))
    print("report:", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
