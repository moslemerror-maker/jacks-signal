"""O2-H Item 7: explicit CLI entry point for the live paper decision orchestrator.

PAPER ONLY, ALWAYS. There is no order/position/execution API anywhere in this project's options code - this script
cannot place a real trade no matter what flags are passed. NIFTY is the only supported underlying (O2-H Item 7
scope; BANKNIFTY's Dhan security id remains unverified and is deliberately not guessed here).

NEVER AUTO-INVOKED ON IMPORT. Nothing runs unless this script is executed directly - importing
`backend.app.options.live_session` or `backend.app.data.sources.dhan_live_chain_source` anywhere else in the
codebase has zero side effects and contacts nothing.

--mode selects the data source explicitly and STRUCTURALLY, not just via a flag the orchestrator itself branches on:
  - `synthetic`: builds a small, fixed, illustrative in-memory snapshot (`SyntheticLiveChainSource`) - never imports
    or constructs anything Dhan-related. For manual smoke-verification of the plumbing only; the real automated
    test suite (`backend/tests/test_options_live_session.py`) is the actual correctness proof, not this script.
  - `live`: runs a real credential pre-flight check, then constructs the real `DhanOptionChainClient` +
    `DhanLiveChainSource` - the ONLY place in this script that can reach Dhan.

    Examples (none of these were run during O2-H Item 7's implementation):

    # one synthetic cycle
    .venv\\Scripts\\python.exe scripts\\run_o2h_live_session.py --mode synthetic --cycles 1

    # one bounded synthetic session (3 cycles, 5s apart, for manual plumbing smoke-verification)
    .venv\\Scripts\\python.exe scripts\\run_o2h_live_session.py --mode synthetic --cycles 3 --interval-seconds 5

    # one live paper cycle (requires --confirm-live; makes exactly one real Dhan call)
    .venv\\Scripts\\python.exe scripts\\run_o2h_live_session.py --mode live --cycles 1 --confirm-live

    # repeated live paper session (requires --confirm-live; Ctrl+C for graceful shutdown)
    .venv\\Scripts\\python.exe scripts\\run_o2h_live_session.py --mode live --confirm-live
"""
from __future__ import annotations

import argparse
import datetime as dt
import signal
import sys
from decimal import Decimal

from backend.app.data.sources.credential_preflight import dhan_credential_preflight
from backend.app.db.session import get_session
from backend.app.options.candidate_generator import CandidateGenerationConfig
from backend.app.options.contracts import ContractIdentity
from backend.app.options.decision_policy import DecisionPolicyThresholds
from backend.app.options.live_quote import LiveQuote
from backend.app.options.live_session import (
    ChainSnapshot, DEFAULT_POLLING_INTERVAL, OrchestratorConfig, RunnerConfig, SyntheticLiveChainSource, run_session,
)

UTC = dt.timezone.utc


def _synthetic_source() -> SyntheticLiveChainSource:
    expiry = dt.date.today() + dt.timedelta(days=7)

    def factory(as_of: dt.datetime) -> ChainSnapshot:
        def q(strike, cp, last, bid, ask):
            return LiveQuote.from_raw(
                contract=ContractIdentity("NIFTY", "IDO", expiry, strike * 100, cp), observed_at=as_of,
                last_price=Decimal(last), bid=Decimal(bid), ask=Decimal(ask), volume=500, open_interest=1000,
                underlying_last_price=Decimal("24500"), provider_reported_iv=None, provider_reported_greeks=None,
                source="synthetic", source_row_hash=None, provenance_class="DEVELOPMENT_FIXTURE",
            )
        quotes = (q(24500, "CE", "140", "138", "142"), q(24500, "PE", "135", "133", "137"),
                 q(24400, "CE", "190", "188", "192"), q(24400, "PE", "95", "93", "97"),
                 q(24600, "CE", "100", "98", "102"), q(24600, "PE", "185", "183", "187"))
        return ChainSnapshot(source="synthetic", underlying="NIFTY", observed_at=as_of, expiry_list_dates=(expiry,),
                             quotes=quotes, raw_response_sha256=None)

    return SyntheticLiveChainSource(factory=factory)


def _live_source(args: argparse.Namespace):
    preflight = dhan_credential_preflight()
    if preflight.status != "OK":
        print(f"REFUSING TO START LIVE MODE: credential pre-flight = {preflight.status}: {preflight.detail}", file=sys.stderr)
        sys.exit(2)
    if not args.confirm_live:
        print("REFUSING TO START LIVE MODE: --mode live requires --confirm-live (this makes real Dhan API calls)", file=sys.stderr)
        sys.exit(2)

    from backend.app.data.sources.dhan_live_chain_source import DhanLiveChainSource
    from backend.app.data.sources.dhan_optionchain_client import NIFTY_SECURITY_ID, NIFTY_SEGMENT, DhanOptionChainClient
    from backend.app.data.sources.provider_credentials import DhanCredentials

    creds = DhanCredentials.from_env()
    client = DhanOptionChainClient(creds)
    return DhanLiveChainSource(client=client, security_id=NIFTY_SECURITY_ID, segment=NIFTY_SEGMENT, underlying="NIFTY")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("synthetic", "live"), required=True)
    ap.add_argument("--cycles", type=int, default=None, help="bounded session (omit for an unbounded live session, Ctrl+C to stop)")
    ap.add_argument("--interval-seconds", type=float, default=DEFAULT_POLLING_INTERVAL.total_seconds())
    ap.add_argument("--min-probability", type=str, default="0.55", help="min risk-neutral P(net P&L>0) - irrelevant while PREDICTIVE_PROBABILITY_STATUS is NOT_ESTABLISHED (TRADE stays unreachable regardless)")
    ap.add_argument("--require-verified-forward", action="store_true", help="block TRADE whenever only the spot-proxy forward is available (default: allowed, but always flagged)")
    ap.add_argument("--model-version", default="o2h-item7-v1")
    ap.add_argument("--strategy-version", default="o2h-item7-v1")
    ap.add_argument("--confirm-live", action="store_true", help="required alongside --mode live - makes this script's live-call intent explicit at the command line, not just in code")
    args = ap.parse_args(argv)

    print(f"AEVORYN O2-H live paper orchestrator - mode={args.mode.upper()} - PAPER ONLY, no order/execution API exists")

    source = _synthetic_source() if args.mode == "synthetic" else _live_source(args)

    config = OrchestratorConfig(
        mode="SYNTHETIC" if args.mode == "synthetic" else "LIVE_PAPER", underlying="NIFTY", source=source,
        candidate_config=CandidateGenerationConfig(lots=1, lot_size=75, lot_size_source="CALLER_SUPPLIED",
                                                   risk_free_rate=Decimal("0.065"), max_quote_age=dt.timedelta(minutes=10),
                                                   max_spread_fraction=Decimal("0.15"), min_volume=None),
        decision_thresholds=DecisionPolicyThresholds(min_probability_positive_net_pnl=Decimal(args.min_probability),
                                                      require_verified_forward=args.require_verified_forward),
        model_version=args.model_version, strategy_version=args.strategy_version,
    )
    runner_config = RunnerConfig(polling_interval=dt.timedelta(seconds=args.interval_seconds), max_cycles=args.cycles)

    stop = {"requested": False}
    def _handle_signal(signum, frame):
        print(f"\nshutdown requested (signal {signum}) - finishing the current cycle, then stopping gracefully")
        stop["requested"] = True
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    def on_stop() -> bool:
        return stop["requested"]

    results = run_session(session_factory=get_session, config=config, runner_config=runner_config, stop_flag=on_stop)

    for r in results:
        print(f"cycle {r.cycle_started_at.isoformat()} -> outcome={r.outcome} status={r.status} "
             f"recommendation_id={r.recommendation_id} duplicate={r.was_duplicate} paper_entry={r.paper_entry_event_id} error={r.error}")
    print(f"session complete: {len(results)} cycle(s) run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
