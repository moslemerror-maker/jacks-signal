from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

from backend.app.data.raw.artifact import check_component
from backend.app.data.raw.store import RawStore
from backend.app.data.sources.credential_preflight import dhan_credential_preflight
from backend.app.data.sources.dhan_live_quote import parse_dhan_optionchain
from backend.app.data.sources.dhan_optionchain_client import (
    NIFTY_SECURITY_ID, NIFTY_SEGMENT, DhanOptionChainClient, DhanOptionChainError,
)
from backend.app.data.sources.provider_credentials import DhanCredentials, MissingCredentialsError

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "derived" / "o2h_verification" / "dhan_optionchain_wrapper.json"

_LEG_FIELDS = ("last_price", "top_bid_price", "top_ask_price", "volume", "oi", "implied_volatility")
_GREEK_FIELDS = ("delta", "theta", "gamma", "vega")


def summarize_chain(chain: dict) -> dict[str, Any]:
    outer = chain.get("data") or {}
    data = outer.get("data") if isinstance(outer.get("data"), dict) else outer
    oc = data.get("oc") or {}
    ce_count = sum(1 for v in oc.values() if isinstance(v, dict) and "ce" in v)
    pe_count = sum(1 for v in oc.values() if isinstance(v, dict) and "pe" in v)
    any_leg = None
    for v in oc.values():
        if isinstance(v, dict) and "ce" in v:
            any_leg = v["ce"]
            break
    return {
        "status": chain.get("status"),
        "underlying_last_price_present": "last_price" in data,
        "strikes_returned": len(oc),
        "ce_legs_present": ce_count,
        "pe_legs_present": pe_count,
        "sample_leg_fields_present": sorted(f for f in _LEG_FIELDS if any_leg and f in any_leg) if any_leg else None,
        "sample_leg_greeks_present": sorted(g for g in _GREEK_FIELDS if any_leg and "greeks" in any_leg and g in any_leg["greeks"]) if any_leg else None,
        "raw_top_level_keys": sorted(chain.keys()),
        "raw_data_level_keys": sorted(data.keys()) if isinstance(data, dict) else None,
    }


def summarize_live_quotes(quotes: list) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    for q in quotes:
        by_status[q.status] = by_status.get(q.status, 0) + 1
    provider_iv_present = sum(1 for q in quotes if q.provider_reported_iv is not None)
    provider_greeks_present = sum(1 for q in quotes if q.provider_reported_greeks is not None)
    underlying_prices = {q.underlying_last_price for q in quotes if q.underlying_last_price is not None}
    return {
        "total_live_quotes_parsed": len(quotes),
        "status_breakdown": by_status,
        "provider_reported_iv_present_count": provider_iv_present,
        "provider_reported_greeks_present_count": provider_greeks_present,
        "distinct_underlying_last_price_values_seen": sorted(str(v) for v in underlying_prices),
    }


def main() -> int:
    report: dict[str, Any] = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "wrapper": "DhanOptionChainClient.fetch_chain + parse_dhan_optionchain"}
    now = dt.datetime.now(dt.timezone.utc)
    store = RawStore(ROOT / "data" / "raw" / "providers")

    preflight = dhan_credential_preflight()
    report["credential_preflight_status"] = preflight.status
    if preflight.status != "OK":
        report["credential_preflight_detail"] = preflight.detail
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=1))
        print(json.dumps(report, indent=1))
        return 2

    try:
        creds = DhanCredentials.from_env()
    except MissingCredentialsError as exc:
        report["error"] = f"MissingCredentialsError: {exc}"
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=1))
        print(json.dumps(report, indent=1))
        return 2

    client = DhanOptionChainClient(creds)

    try:
        result = client.fetch_chain(NIFTY_SECURITY_ID, NIFTY_SEGMENT)
    except DhanOptionChainError as exc:
        report["authentication_and_call_status"] = "FAILED"
        report["error"] = str(exc)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=1, default=str))
        print(json.dumps(report, indent=1, default=str))
        return 1

    report["authentication_and_call_status"] = "SUCCESS"
    report["expiry_used"] = result.expiry_used
    report["expiry_selection"] = result.expiry_selection
    report["expiry_list_dates"] = DhanOptionChainClient._expiry_dates(result.expiry_list_response)
    report["summary"] = summarize_chain(result.option_chain_response)

    key_date = now.strftime("%Y%m%dT%H%M%SZ")
    store.put(provider="dhan", dataset="optionchain_expirylist", source_key=check_component("k", f"NIFTY_{key_date}"),
             content=json.dumps(result.expiry_list_response, default=str).encode(), source_url="https://api.dhan.co/v2/optionchain/expirylist",
             http_status=200, response_headers={}, fetcher_version="run_o2h_verify_dhan_optionchain_wrapper/1", media_type="application/json", license_status="UNCLEAR")
    store.put(provider="dhan", dataset="optionchain", source_key=check_component("k", f"NIFTY_{result.expiry_used.replace('-', '')}_{key_date}"),
             content=json.dumps(result.option_chain_response, default=str).encode(), source_url="https://api.dhan.co/v2/optionchain",
             http_status=200, response_headers={}, fetcher_version="run_o2h_verify_dhan_optionchain_wrapper/1", media_type="application/json", license_status="UNCLEAR")

    try:
        quotes = parse_dhan_optionchain(
            result.option_chain_response, underlying="NIFTY", expiry=dt.date.fromisoformat(result.expiry_used),
            observed_at=now, source_row_hash=None, provenance_class="PRODUCTION_RESEARCH_DATASET",
        )
        report["live_quote_parsing_status"] = "SUCCESS"
        report["live_quote_summary"] = summarize_live_quotes(quotes)
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic script; any parsing failure must be visible, not swallowed
        report["live_quote_parsing_status"] = "FAILED"
        report["live_quote_parsing_error"] = f"{type(exc).__name__}: {exc}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))
    print("report:", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
