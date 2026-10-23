from __future__ import annotations

import datetime as dt
import json
import sys
from decimal import Decimal
from pathlib import Path

from backend.app.data.sources.credential_preflight import dhan_credential_preflight
from backend.app.options.candidate_generator import CandidateGenerationConfig, generate_candidates
from backend.app.options.decision_policy import DecisionPolicyThresholds, decide_for_underlying
from backend.app.options.expiry_eligibility import select_eligible_expiry
from backend.app.options.forward import resolve_forward_spot_proxy
from backend.app.options.probability import Black76RiskNeutralDistribution

UTC = dt.timezone.utc
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "derived" / "o2h_verification" / "final_live_preflight.json"


def distribution_factory(forward_resolution):
    def factory(candidate):
        ok = [r.iv for r in candidate.iv_context if r.status == "OK" and r.iv is not None]
        if not ok or not candidate.iv_context:
            return None
        avg_iv = sum(ok, Decimal(0)) / len(ok)
        t = candidate.iv_context[0].time_to_expiry_years
        if t is None or t <= 0:
            return None
        return Black76RiskNeutralDistribution(forward=forward_resolution.forward, volatility=avg_iv, time_to_expiry_years=t)
    return factory


def main() -> int:
    report: dict = {"generated_at": dt.datetime.now(UTC).isoformat(), "live_api_calls_made": 0}

    preflight = dhan_credential_preflight()
    report["credentials_configured"] = "YES" if preflight.status != "MISSING" else "NO"
    report["credential_preflight_status"] = preflight.status
    if preflight.status != "OK":
        report["provider_client_initialization"] = "FAILURE"
        report["stop_reason"] = f"credential preflight = {preflight.status}: {preflight.detail}"
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=1))
        print(json.dumps(report, indent=1))
        return 2

    from backend.app.data.sources.dhan_live_chain_source import DhanLiveChainSource
    from backend.app.data.sources.dhan_optionchain_client import NIFTY_SECURITY_ID, NIFTY_SEGMENT, DhanOptionChainClient, DhanOptionChainError
    from backend.app.data.sources.provider_credentials import DhanCredentials, MissingCredentialsError

    try:
        creds = DhanCredentials.from_env()
        client = DhanOptionChainClient(creds)
        source = DhanLiveChainSource(client=client, security_id=NIFTY_SECURITY_ID, segment=NIFTY_SEGMENT, underlying="NIFTY")
        report["provider_client_initialization"] = "SUCCESS"
    except MissingCredentialsError as exc:
        report["provider_client_initialization"] = "FAILURE"
        report["stop_reason"] = str(exc)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=1))
        print(json.dumps(report, indent=1))
        return 2

    now = dt.datetime.now(UTC)
    try:
        snapshot = source.fetch_snapshot(as_of=now)
        report["live_api_calls_made"] = 2
        report["api_call_status"] = "SUCCESS"
    except DhanOptionChainError as exc:
        report["live_api_calls_made"] = "unknown (failed partway - see error)"
        report["api_call_status"] = "FAILURE"
        report["error"] = str(exc)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=1, default=str))
        print(json.dumps(report, indent=1, default=str))
        return 1

    usable = [q for q in snapshot.quotes if q.is_usable]
    no_quote = [q for q in snapshot.quotes if q.status == "NO_QUOTE"]
    anomalous = [q for q in snapshot.quotes if q.status not in ("QUOTED", "NO_QUOTE")]
    with_bid_ask = [q for q in snapshot.quotes if q.bid is not None and q.ask is not None]
    with_iv = [q for q in snapshot.quotes if q.provider_reported_iv is not None]
    with_greeks = [q for q in snapshot.quotes if q.provider_reported_greeks is not None]
    underlying_prices = sorted({q.underlying_last_price for q in snapshot.quotes if q.underlying_last_price is not None})
    contract_keys = [(q.contract.strike_paise, q.contract.option_type) for q in snapshot.quotes]
    duplicates = len(contract_keys) - len(set(contract_keys))

    report["selected_expiry"] = sorted({q.contract.expiry for q in snapshot.quotes})
    report["option_leg_counts"] = {
        "total_received": len(snapshot.quotes), "normalized_successfully": len(snapshot.quotes),
        "usable_quoted": len(usable), "no_quote": len(no_quote), "anomalous": len(anomalous),
        "with_bid_ask": len(with_bid_ask), "with_provider_iv": len(with_iv), "with_provider_greeks": len(with_greeks),
    }
    report["underlying_last_price_values_seen"] = [str(p) for p in underlying_prices]
    report["duplicate_contract_count"] = duplicates
    report["snapshot_observed_at"] = snapshot.observed_at.isoformat()
    report["raw_response_sha256"] = snapshot.raw_response_sha256

    independent_eligible = select_eligible_expiry(snapshot.expiry_list_dates, as_of=now)
    used_expiry = snapshot.quotes[0].contract.expiry if snapshot.quotes else None
    report["expiry_list_dates"] = [d.isoformat() for d in snapshot.expiry_list_dates]
    report["expiry_used_by_source"] = used_expiry.isoformat() if used_expiry else None
    report["independent_eligible_expiry_recheck"] = independent_eligible.isoformat()
    report["expiry_safety_confirmed"] = (used_expiry == independent_eligible)

    futures_quotes = [q for q in snapshot.quotes if q.contract.derivative_class in ("IDF", "STF")]
    report["futures_quote_present_in_snapshot"] = len(futures_quotes) > 0
    if not underlying_prices:
        report["forward_resolution_mode"] = "UNRESOLVABLE - no underlying_last_price in snapshot"
        forward_resolution = None
    else:
        forward_resolution = resolve_forward_spot_proxy(underlying_prices[0])
        report["forward_resolution_mode"] = forward_resolution.source
        report["forward_value"] = str(forward_resolution.forward)
        report["forward_labeled_as_proxy_explicitly"] = forward_resolution.source == "SPOT_PROXY_UNVERIFIED"

    if used_expiry is not None and forward_resolution is not None:
        config = CandidateGenerationConfig(lots=1, lot_size=75, lot_size_source="CALLER_SUPPLIED",
                                           risk_free_rate=Decimal("0.065"), max_quote_age=dt.timedelta(minutes=10),
                                           max_spread_fraction=Decimal("0.15"), min_volume=None)
        option_quotes = [q for q in snapshot.quotes if q.contract.is_option]
        outcomes = generate_candidates(option_quotes, expiry=used_expiry, forward_resolution=forward_resolution, config=config,
                                       timestamp=snapshot.observed_at, as_of=snapshot.observed_at)
        thresholds = DecisionPolicyThresholds(min_probability_positive_net_pnl=Decimal("0.55"), require_verified_forward=False)
        results = decide_for_underlying(outcomes, distribution_factory=distribution_factory(forward_resolution), timestamp=snapshot.observed_at,
                                        underlying="NIFTY", thresholds=thresholds, model_version="preflight-preview", strategy_version="preflight-preview")
        report["decision_preview"] = {
            "note": "READ-ONLY PREVIEW - nothing below was persisted to any table",
            "candidates_considered": [
                {"shape": o.shape, "formed": o.candidate is not None, "rejected_reason": o.rejected_reason}
                for o in outcomes
            ],
            "per_shape_decisions": [
                {"shape": o.shape, "status": r.status, "reasons": list(r.reasons)}
                for o, r in zip(outcomes, results)
            ],
        }
        import backend.app.options.decision_policy as dp
        report["predictive_probability_not_established"] = dp.PREDICTIVE_PROBABILITY_STATUS != "ESTABLISHED"
        report["trade_blocked"] = all(r.status != "TRADE" for r in results)
    else:
        report["decision_preview"] = "SKIPPED - no usable expiry/forward to preview against"

    report["order_execution_api_called"] = False
    report["banknifty_api_called"] = False
    report["rollingoption_called"] = False
    report["long_running_session_started"] = False
    report["db_session_opened"] = False
    report["db_state_changed"] = "NO - no database module was imported or used anywhere in this script"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))
    print("report:", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
