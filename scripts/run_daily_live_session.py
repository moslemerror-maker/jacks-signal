"""Stage A: the automatic daily NIFTY options live-session entry point - what Windows Task Scheduler invokes at
09:15 IST (see docs/58 for the exact scheduler setup). PAPER ONLY, LIVE NIFTY ONLY - same structural guarantee as
`run_o2h_live_session.py`: there is no order/execution API anywhere in this project's options code, so this script
cannot place a real trade regardless of flags.

DIFFERENCE FROM `run_o2h_live_session.py`: that script is the general-purpose, manually-invoked CLI (bounded/
unbounded, synthetic/live, `--confirm-live` gated) used throughout O2-H's development and both live drills -
UNCHANGED by this stage, still the right tool for a manually-supervised session. THIS script is the automatic,
unattended entry point: it adds trading-day validation, DB/credential preflight, a session-state record, a
duplicate-session guard, and a genuine wall-clock 15:15 stop (`RunnerConfig.stop_at`) - nothing here duplicates
`run_o2h_live_session.py`'s CLI surface; it always runs in live mode, always confirmed, always NIFTY.

STAGE B.1 ROOT-CAUSE FIX (2026-09-24): the automatic 09:15 Task Scheduler run on 2026-09-24 crashed with exit code 1
and left zero trace anywhere (no SESSION_FAILED event, no recommendation, no traceback). Reproduced exactly this
stage: Task Scheduler's registered action (`python.exe scripts\run_daily_live_session.py --confirm-live`,
`WorkingDirectory=D:/Projects/trading-system`) never sets `PYTHONPATH` - and Python's default `sys.path[0]` for a
directly-invoked script is the SCRIPT'S OWN DIRECTORY (`scripts/`), never the project root or the CWD. Every prior
manual invocation of this project's scripts (this entire engagement) was run with `PYTHONPATH=.` explicitly set,
masking this - Task Scheduler is the first caller that doesn't. The fix: a `sys.path` bootstrap, first thing in this
file, before any `backend.app` import is attempted - the standard, minimal pattern for a script that must run
regardless of PYTHONPATH/CWD. `scripts/run_api.py` had the identical latent defect (also fixed this stage);
`scripts/run_dashboard.py` has no `backend.app` import and was never affected.

STAGE B.1 DEFENSE IN DEPTH: beyond the specific fix above, `main()`'s entire body is now wrapped in one outer
handler that NO exception occurring before `run_daily_session_workflow()` can escape silently again, regardless of
its cause. On any such exception: (1) a full traceback is appended to a persistent, append-only file
(`data/run/daily_live_session_startup_failures.log`) - guaranteed even if Task Scheduler captures no stdout/stderr
and even if the database itself is unreachable; (2) a best-effort `SESSION_FAILED` event is attempted (only if the
database is actually reachable - failure to log there never masks the original exception); (3) the process exits
non-zero. This NEVER creates a recommendation, a PAPER_ENTRY, or converts a startup crash into WAIT/NO_TRADE - a
crash is not a trading decision, and nothing in this handler touches decision/candidate/probability logic at all.

EXIT CODES: 0 = nothing further to do today (not a trading day, or already COMPLETED) - NOT an error. 2 = a genuine
failure at any stage (DB unreachable, holiday calendar unavailable, credential preflight failed, already FAILED
today without --force, an uncaught exception during the live session itself, or - new this stage - ANY uncaught
exception anywhere in startup, now always logged per the paragraph above rather than silently producing exit 1).

    .venv\\Scripts\\python.exe scripts\\run_daily_live_session.py --confirm-live [--holiday-file PATH] [--force]
"""
from __future__ import annotations

import sys
import traceback
import datetime as dt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402
import signal  # noqa: E402
from decimal import Decimal  # noqa: E402

UTC = dt.timezone.utc
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOLIDAY_FILE = ROOT / "data" / "raw" / "nse_calendar" / "holiday_master.json"
DEFAULT_STOP_FILE = ROOT / "data" / "run" / "daily_live_session.stop"
STARTUP_FAILURE_LOG = ROOT / "data" / "run" / "daily_live_session_startup_failures.log"


def _write_startup_failure_log(phase: str, exc: BaseException) -> None:
    try:
        STARTUP_FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with STARTUP_FAILURE_LOG.open("a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"timestamp={dt.datetime.now(UTC).isoformat()} script=run_daily_live_session.py phase={phase!r}\n")
            f.write(f"exception_type={type(exc).__name__}\n")
            f.write(f"exception_message={exc}\n")
            f.write("traceback:\n")
            f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    except Exception:
        pass


def _try_log_session_failed(phase: str, exc: BaseException) -> None:
    try:
        from backend.app.db.session import get_session
        from backend.app.options.session_scheduler import check_db_reachable, log_session_state

        db = get_session()
        try:
            if check_db_reachable(db):
                log_session_state(db, state="FAILED", detail=f"startup failure during {phase}: {type(exc).__name__}: {exc}",
                                  event_timestamp=dt.datetime.now(UTC), severity="ERROR")
                db.commit()
        finally:
            db.close()
    except Exception:
        pass


def _live_source(underlying: str):
    from backend.app.data.sources.dhan_live_chain_source import DhanLiveChainSource
    from backend.app.data.sources.dhan_optionchain_client import NIFTY_SECURITY_ID, NIFTY_SEGMENT, DhanOptionChainClient
    from backend.app.data.sources.provider_credentials import DhanCredentials

    creds = DhanCredentials.from_env()
    client = DhanOptionChainClient(creds)
    return DhanLiveChainSource(client=client, security_id=NIFTY_SECURITY_ID, segment=NIFTY_SEGMENT, underlying=underlying)


def main(argv=None) -> int:
    phase = "backend imports"
    db = None
    try:
        from backend.app.data.sources.credential_preflight import dhan_credential_preflight
        from backend.app.db.session import get_session
        from backend.app.options.candidate_generator import CandidateGenerationConfig
        from backend.app.options.decision_policy import DecisionPolicyThresholds
        from backend.app.options.live_session import DEFAULT_POLLING_INTERVAL, OrchestratorConfig, RunnerConfig
        from backend.app.options.session_scheduler import (
            check_db_reachable, log_session_state, run_daily_session_workflow, windows_safe_stop_flag,
        )
        from backend.app.options.session_schedule import ist_today, session_stop_utc
        from backend.app.options.trading_calendar import HolidayCalendarUnavailable, load_fo_holiday_dates

        phase = "argument parsing"
        ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        ap.add_argument("--holiday-file", type=Path, default=DEFAULT_HOLIDAY_FILE, help="cached FO-segment holiday-master JSON (see backend/app/data/sources/nse_calendar.py)")
        ap.add_argument("--stop-file", type=Path, default=DEFAULT_STOP_FILE, help="create this file to request a graceful stop - the reliable, Windows-safe mechanism (SIGTERM is confirmed unreliable here)")
        ap.add_argument("--force", action="store_true", help="start even if today's session already reached a terminal state (COMPLETED/FAILED) - human override only, never used by the scheduled trigger itself")
        ap.add_argument("--interval-seconds", type=float, default=DEFAULT_POLLING_INTERVAL.total_seconds())
        ap.add_argument("--max-consecutive-fetch-failures", type=int, default=6, dest="max_consecutive_fetch_failures",
                         help="Phase 1 reliability fix: stop the session (logging SESSION_FAILED) after this many "
                              "consecutive provider-fetch failures, instead of retrying identically all day "
                              "(the 2026-09-28 incident). 6 x 5min default = 30 minutes before giving up for today.")
        ap.add_argument("--min-probability", type=str, default="0.55", help="irrelevant while PREDICTIVE_PROBABILITY_STATUS is NOT_ESTABLISHED - TRADE stays unreachable regardless")
        ap.add_argument("--require-verified-forward", action="store_true")
        ap.add_argument("--model-version", default="stage-a-v1")
        ap.add_argument("--strategy-version", default="stage-a-v1")
        ap.add_argument("--confirm-live", action="store_true", required=True, help="required - makes this script's live-call intent explicit, mirroring run_o2h_live_session.py's own safety convention")
        args = ap.parse_args(argv)

        now = dt.datetime.now(UTC)
        today = ist_today(now)

        phase = "DB preflight"
        try:
            db = get_session()
        except Exception as exc:
            print(f"REFUSING TO START: could not open a database session: {exc}", file=sys.stderr)
            return 2
        if not check_db_reachable(db):
            db.close()
            db = None
            print("REFUSING TO START: database is not reachable - this failure cannot be logged to the database "
                  f"(no working DB); see {STARTUP_FAILURE_LOG} / this exit code / Task Scheduler's own task history.", file=sys.stderr)
            _write_startup_failure_log(phase, RuntimeError("database not reachable (check_db_reachable returned False)"))
            return 2

        try:
            phase = "holiday calendar load"
            try:
                holiday_dates = load_fo_holiday_dates(args.holiday_file)
            except HolidayCalendarUnavailable as exc:
                log_session_state(db, state="FAILED", detail=f"holiday calendar unavailable: {exc}", event_timestamp=now, severity="ERROR")
                db.commit()
                print(f"REFUSING TO START: {exc}", file=sys.stderr)
                return 2

            phase = "credential preflight"
            preflight = dhan_credential_preflight()
            if preflight.status != "OK":
                log_session_state(db, state="FAILED", detail=f"credential preflight = {preflight.status}: {preflight.detail}", event_timestamp=now, severity="ERROR")
                db.commit()
                print(f"REFUSING TO START: credential preflight = {preflight.status}: {preflight.detail}", file=sys.stderr)
                return 2

            phase = "live source / config construction"
            source = _live_source("NIFTY")
            config = OrchestratorConfig(
                mode="LIVE_PAPER", underlying="NIFTY", source=source,
                candidate_config=CandidateGenerationConfig(lots=1, lot_size=75, lot_size_source="CALLER_SUPPLIED",
                                                           risk_free_rate=Decimal("0.065"), max_quote_age=dt.timedelta(minutes=10),
                                                           max_spread_fraction=Decimal("0.15"), min_volume=None),
                decision_thresholds=DecisionPolicyThresholds(min_probability_positive_net_pnl=Decimal(args.min_probability),
                                                              require_verified_forward=args.require_verified_forward),
                model_version=args.model_version, strategy_version=args.strategy_version,
            )
            runner_config = RunnerConfig(polling_interval=dt.timedelta(seconds=args.interval_seconds), max_cycles=None,
                                         stop_at=session_stop_utc(today),
                                         max_consecutive_fetch_failures=args.max_consecutive_fetch_failures)

            phase = "signal handler registration"
            stop = {"requested": False}
            def _handle_signal(signum, frame):
                print(f"\nshutdown requested (signal {signum}) - finishing the current cycle, then stopping gracefully")
                stop["requested"] = True
            signal.signal(signal.SIGINT, _handle_signal)
            signal.signal(signal.SIGTERM, _handle_signal)

            phase = "stop-file setup"
            args.stop_file.parent.mkdir(parents=True, exist_ok=True)
            args.stop_file.unlink(missing_ok=True)
            stop_flag = windows_safe_stop_flag(signal_state=stop, stop_file=args.stop_file)

            phase = "run_daily_session_workflow"
            outcome = run_daily_session_workflow(
                db=db, session_factory=get_session, today=today, holiday_dates=holiday_dates, config=config,
                runner_config=runner_config, stop_flag=stop_flag, force=args.force,
            )
            print(f"workflow outcome: {outcome.outcome} ({outcome.detail})")
            if outcome.outcome == "NOT_TRADING_DAY":
                return 0
            if outcome.outcome == "REFUSE_ALREADY_COMPLETED":
                return 0
            if outcome.outcome == "REFUSE_ALREADY_FAILED":
                return 2
            if outcome.outcome == "FAILED":
                print(f"SESSION FAILED: {outcome.detail}", file=sys.stderr)
                return 2
            return 0
        finally:
            if db is not None:
                db.close()
    except Exception as exc:
        _write_startup_failure_log(phase, exc)
        _try_log_session_failed(phase, exc)
        print(f"STARTUP FAILURE during {phase!r}: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"Full traceback written to: {STARTUP_FAILURE_LOG}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
