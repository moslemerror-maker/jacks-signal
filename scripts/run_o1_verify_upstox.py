from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

from backend.app.data.raw.artifact import check_component
from backend.app.data.raw.store import RawStore
from backend.app.data.sources.provider_credentials import MissingCredentialsError, UpstoxCredentials
from backend.app.data.sources.upstox_client import ApiResponse, UpstoxClient, expired_candle_url, expired_option_contracts_url, expiries_url

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "derived" / "o1_provider_verification" / "upstox.json"
YEARS = (2024, 2025, 2026)
MAX_EXPIRIES_PER_YEAR = 3
PLUS_REQUIRED_ERROR_CODES = {"UDAPI1149"}
NSE_CROSS_CHECK_EXPIRIES = {"2025-09-02": "udiff", "2025-09-30": "udiff", "2026-01-06": "udiff", "2026-09-15": "udiff"}


def _redact_headers(headers: dict[str, str | None]) -> dict[str, str | None]:
    return {k: v for k, v in headers.items() if "auth" not in k.lower() and "token" not in k.lower()}


def _store_response(store: RawStore | None, dataset: str, key: str, resp: ApiResponse, url: str) -> None:
    if store is None:
        return
    store.put(provider="upstox", dataset=dataset, source_key=check_component("source_key", key), content=resp.content, source_url=url,
             http_status=resp.status_code, response_headers=_redact_headers(resp.headers), fetcher_version="run_o1_verify_upstox/2",
             media_type=resp.headers.get("Content-Type"), license_status="UNCLEAR")


def rate_limit_headers(*responses: ApiResponse) -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    for r in responses:
        for k, v in r.headers.items():
            if "rate" in k.lower() or "limit" in k.lower():
                found[k] = v
    return found


def is_plan_upgrade_error(resp: ApiResponse) -> bool:
    if resp.status_code == 403:
        return True
    if isinstance(resp.json, dict):
        codes = {e.get("errorCode") for e in resp.json.get("errors", []) if isinstance(e, dict)}
        if codes & PLUS_REQUIRED_ERROR_CODES:
            return True
    return False


def contract_stats(contracts: list[dict[str, Any]]) -> dict[str, Any]:
    ce = [c for c in contracts if c.get("instrument_type") == "CE"]
    pe = [c for c in contracts if c.get("instrument_type") == "PE"]
    strikes = sorted({c["strike_price"] for c in contracts if "strike_price" in c})
    spacing = sorted({round(b - a, 4) for a, b in zip(strikes, strikes[1:])}) if len(strikes) > 1 else []
    keyed = Counter((c.get("strike_price"), c.get("instrument_type")) for c in contracts)
    duplicates = {f"{k[1]} {k[0]}": n for k, n in keyed.items() if n > 1}
    return {
        "rows": len(contracts), "CE": len(ce), "PE": len(pe), "unique_strikes": len(strikes),
        "lowest_strike": strikes[0] if strikes else None, "highest_strike": strikes[-1] if strikes else None, "strike_intervals": spacing,
        "lot_sizes": dict(Counter(c.get("lot_size") for c in contracts)), "distinct_lot_sizes": len({c.get("lot_size") for c in contracts}),
        "weekly": dict(Counter(c.get("weekly") for c in contracts)), "duplicate_strike_type_pairs": duplicates,
    }


def pick_itm_atm_otm(contracts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ce = sorted((c for c in contracts if c.get("instrument_type") == "CE"), key=lambda c: c.get("strike_price", 0))
    if not ce:
        return []
    return [ce[0], ce[len(ce) // 2], ce[-1]]


def classify_candles(candles: list[list[Any]]) -> dict[str, Any]:
    from scripts.run_o1_0b_sample_gate import classify_trade_state
    states = Counter()
    timestamps = [c[0] for c in candles if c]
    for c in candles:
        if len(c) < 6:
            states["MALFORMED_ROW"] += 1
            continue
        vol = c[5]
        try:
            _, state = classify_trade_state(int(vol), Decimal(str(c[1])), Decimal(str(c[2])), Decimal(str(c[3])), Decimal(str(c[4])))
        except Exception:
            state = "UNPARSEABLE"
        states[state] += 1
    return {"rows": len(candles), "trade_states": dict(states), "duplicate_timestamps": len(timestamps) - len(set(timestamps)),
            "has_oi_field": bool(candles) and len(candles[0]) >= 7}


def reconcile_with_nse(expiry: str, contracts: list[dict[str, Any]], tested_candles: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    family = NSE_CROSS_CHECK_EXPIRIES.get(expiry)
    if family is None:
        return None
    from scripts.run_o1_0b_sample_gate import ContractKey, load, strike_to_paise, udiff_bars
    expiry_date = dt.date.fromisoformat(expiry)
    got = load(family, expiry_date)
    if got is None:
        return {"available": False, "reason": "local NSE sample file not present (data/raw/nse_fo_sample) - run scripts/run_o1_0b_sample_gate.py first"}
    nse = {b.key: b for b in udiff_bars(*got) if b.key.underlying == "NIFTY" and b.key.dtype == "IDO" and b.key.expiry == expiry_date}
    up_ce = sum(1 for c in contracts if c.get("instrument_type") == "CE")
    up_pe = sum(1 for c in contracts if c.get("instrument_type") == "PE")
    nse_ce = sum(1 for k in nse if k.cp == "CE")
    nse_pe = sum(1 for k in nse if k.cp == "PE")
    out: dict[str, Any] = {"available": True, "contract_count": {
        "upstox_CE": up_ce, "upstox_PE": up_pe, "nse_CE": nse_ce, "nse_PE": nse_pe,
        "classification": "MATCH" if (up_ce, up_pe) == (nse_ce, nse_pe) else "UNEXPLAINED",
    }, "contracts": {}}
    for ck, per_contract in tested_candles.items():
        strike_paise, exact = strike_to_paise(str(per_contract.get("strike", "")))
        key = ContractKey("NIFTY", "IDO", expiry_date, strike_paise if strike_paise is not None else -1, per_contract.get("type"))
        nse_bar = nse.get(key)
        if nse_bar is None:
            out["contracts"][ck] = {"classification": "MISSING", "detail": "no NSE contract at this exact strike/type on this date"}
            continue
        lot_up = per_contract.get("lot_size")
        lot_cls = "MATCH" if lot_up == nse_bar.lot else ("MISSING" if lot_up is None else "UNEXPLAINED")
        out["contracts"][ck] = {"strike_identity": "MATCH" if exact else "UNEXPLAINED", "lot": {"upstox": lot_up, "nse": nse_bar.lot, "classification": lot_cls}}
    return out


def verify(client: UpstoxClient, *, instrument_key: str, underlying: str, store: RawStore | None = None, years: tuple[int, ...] = YEARS) -> dict[str, Any]:
    report: dict[str, Any] = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "instrument_key": instrument_key, "underlying": underlying,
                              "pit_note": "available_at is OUR fetch time only; Upstox documents no publication timestamp and none is invented",
                              "years": {}}
    exp = client.expiries(instrument_key)
    _store_response(store, "expiries", check_component("k", underlying), exp, expiries_url(instrument_key))
    report["expiries_status"] = exp.status_code
    report["rate_limit_headers_observed"] = rate_limit_headers(exp)
    if is_plan_upgrade_error(exp):
        report["plan_upgrade_required"] = {"status_code": exp.status_code, "body": exp.json, "detail": "stopped before testing any contracts/candles - see docs/38"}
        return report

    all_expiries = sorted((exp.json or {}).get("data", [])) if exp.json else []
    report["all_expiries_returned"] = all_expiries
    report["earliest_expiry_returned"] = all_expiries[0] if all_expiries else None
    report["latest_expiry_returned"] = all_expiries[-1] if all_expiries else None

    stability_checked = False
    all_responses = [exp]
    for year in years:
        in_year = [e for e in all_expiries if e.startswith(str(year))]
        picked = sorted({in_year[0], in_year[len(in_year) // 2], in_year[-1]}) if in_year else []
        picked = picked[:MAX_EXPIRIES_PER_YEAR]
        year_report: dict[str, Any] = {"expiries_available_in_year": in_year, "tested": {}}
        for expiry in picked:
            oc = client.expired_option_contracts(instrument_key, expiry)
            all_responses.append(oc)
            key = check_component("k", f"{underlying}_{expiry.replace('-', '')}")
            _store_response(store, "option_contracts", key, oc, expired_option_contracts_url(instrument_key, expiry))
            contracts = oc.json.get("data", []) if isinstance(oc.json, dict) else []
            entry: dict[str, Any] = {"status": oc.status_code, **contract_stats(contracts), "candles": {}}

            if not stability_checked:
                oc2 = client.expired_option_contracts(instrument_key, expiry)
                _store_response(store, "option_contracts", check_component("k", f"{key}_repeat"), oc2, expired_option_contracts_url(instrument_key, expiry))
                ids1 = {c.get("instrument_key") for c in contracts}
                ids2 = {c.get("instrument_key") for c in (oc2.json.get("data", []) if isinstance(oc2.json, dict) else [])}
                entry["identifier_stability_same_session"] = {"identical": ids1 == ids2, "only_in_first_call": len(ids1 - ids2), "only_in_second_call": len(ids2 - ids1)}
                stability_checked = True

            for c in pick_itm_atm_otm(contracts):
                ck = c.get("instrument_key")
                if not ck:
                    continue
                per_contract: dict[str, Any] = {"strike": c.get("strike_price"), "type": c.get("instrument_type"), "lot_size": c.get("lot_size")}
                for interval in ("day", "5minute"):
                    cand = client.expired_candles(ck, interval, expiry, expiry)
                    all_responses.append(cand)
                    _store_response(store, "candles", check_component("k", f"{key}_{c.get('instrument_type')}_{interval}"), cand, expired_candle_url(ck, interval, expiry, expiry))
                    candles = (cand.json or {}).get("data", {}).get("candles", []) if isinstance(cand.json, dict) else []
                    per_contract[interval] = {"status": cand.status_code, "queryable": cand.status_code == 200 and bool(candles), **classify_candles(candles)}
                entry["candles"][ck] = per_contract
            entry["nse_cross_check"] = reconcile_with_nse(expiry, contracts, entry["candles"])
            year_report["tested"][expiry] = entry
        report["years"][year] = year_report
    report["rate_limit_headers_observed"] = {**report["rate_limit_headers_observed"], **rate_limit_headers(*all_responses)}
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument-key", default="NSE_INDEX|Nifty 50")
    ap.add_argument("--underlying", default="NIFTY")
    ap.add_argument("--no-store", action="store_true", help="skip writing raw responses to RawStore")
    args = ap.parse_args(argv)
    try:
        creds = UpstoxCredentials.from_env()
    except MissingCredentialsError as exc:
        print(f"NOT CONFIGURED: {exc}")
        return 2
    store = None if args.no_store else RawStore(ROOT / "data" / "raw" / "providers")
    report = verify(UpstoxClient(creds), instrument_key=args.instrument_key, underlying=args.underlying, store=store)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, default=str))
    if "plan_upgrade_required" in report:
        print("STOPPED: Upstox reports a plan/subscription upgrade is required.")
        print(json.dumps(report["plan_upgrade_required"], indent=1, default=str))
        print("report:", OUT)
        return 3
    print(f"earliest expiry returned: {report['earliest_expiry_returned']} | latest: {report['latest_expiry_returned']}")
    for year, y in report["years"].items():
        print(f"  {year}: {len(y['expiries_available_in_year'])} expiries in the Get Expiries response, {len(y['tested'])} tested")
    if report["rate_limit_headers_observed"]:
        print("rate-limit-related headers observed:", report["rate_limit_headers_observed"])
    print("report:", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
