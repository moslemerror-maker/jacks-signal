"""Options Pipeline Reliability Phase (Phase 1): the "simple reliable alert" the strategic review called for -
deliberately NOT a monitoring platform. Prints a one-line OK, or one line per concern, to stdout; exits non-zero
only when at least one concern is found, so it can be wrapped by anything (a cron-style scheduled task, a manual
check, CI) that distinguishes exit codes without needing to parse output.

READ-ONLY. Makes no Dhan call, starts no session, writes nothing except (optionally) appending to its own tiny
log file - never touches `option_recommendations`/`option_session_events`/the scheduler.

REGISTERING THIS AS ITS OWN SCHEDULED TASK IS A SEPARATE, MANUAL, NOT-YET-APPROVED STEP. This script exists and is
tested; nothing in this phase registers it with Windows Task Scheduler (see the Phase 1 report's "Scheduler state"
section for exactly what manual/elevated action that would require).

    .venv\\Scripts\\python.exe scripts\\run_options_pipeline_healthcheck.py [--holiday-file PATH] [--task-name NAME] [--log-file PATH]
"""
from __future__ import annotations

import sys
import datetime as dt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402

UTC = dt.timezone.utc
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOLIDAY_FILE = ROOT / "data" / "raw" / "nse_calendar" / "holiday_master.json"
DEFAULT_TASK_NAME = "AEVORYN-Daily-Live-Session"
DEFAULT_LOG_FILE = ROOT / "data" / "run" / "options_pipeline_healthcheck.log"


def main(argv: list[str] | None = None) -> int:
    from backend.app.db.session import get_session
    from backend.app.options.pipeline_health import health_concerns, pipeline_health_snapshot
    from backend.app.options.session_scheduler import check_db_reachable
    from backend.app.options.session_schedule import ist_today
    from backend.app.options.trading_calendar import HolidayCalendarUnavailable, load_fo_holiday_dates

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holiday-file", type=Path, default=DEFAULT_HOLIDAY_FILE)
    ap.add_argument("--task-name", default=DEFAULT_TASK_NAME, help="Windows Scheduled Task name to check, best-effort (set to '' to skip)")
    ap.add_argument("--underlying", default="NIFTY")
    ap.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    args = ap.parse_args(argv)

    now = dt.datetime.now(UTC)

    try:
        db = get_session()
    except Exception as exc:
        print(f"HEALTHCHECK COULD NOT RUN: database session could not be opened: {exc}", file=sys.stderr)
        return 2
    try:
        if not check_db_reachable(db):
            print("HEALTHCHECK COULD NOT RUN: database is not reachable", file=sys.stderr)
            return 2
        try:
            holiday_dates = load_fo_holiday_dates(args.holiday_file)
        except HolidayCalendarUnavailable as exc:
            print(f"HEALTHCHECK COULD NOT RUN: holiday calendar unavailable: {exc}", file=sys.stderr)
            return 2

        today = ist_today(now)
        snapshot = pipeline_health_snapshot(
            db, underlying=args.underlying, today=today, now=now, holiday_dates=holiday_dates,
            task_name=args.task_name or None,
        )
    finally:
        db.close()

    concerns = health_concerns(snapshot)
    lines = [f"[{now.isoformat()}] options pipeline healthcheck for {args.underlying} on {today.isoformat()}"]
    if not concerns:
        lines.append(f"  OK - session_state={snapshot.session_state}, is_trading_day={snapshot.is_trading_day}, "
                      f"cycles_today={snapshot.cycle_count_today}, scheduled_task_state={snapshot.scheduled_task_state!r}")
    else:
        for c in concerns:
            lines.append(f"  CONCERN: {c}")

    output = "\n".join(lines)
    print(output)
    try:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        with args.log_file.open("a", encoding="utf-8") as f:
            f.write(output + "\n")
    except Exception:
        pass

    return 1 if concerns else 0


if __name__ == "__main__":
    sys.exit(main())
