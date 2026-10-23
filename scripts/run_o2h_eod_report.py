"""O2-H Item 9: explicit CLI entry point for the end-of-day paper-session report.

READ-ONLY, OFFLINE. This script never contacts Dhan and never requires credentials - `eod_report.py`'s
`build_eod_report` only `SELECT`s from the already-persisted O2-H tables. It works even if Dhan is completely
offline, and produces the same report content for the same DB state regardless of when it is run (the one
deliberately time-varying field, `generated_at`, is called out as metadata in the report itself).

NEVER AUTO-INVOKED. Nothing runs unless this script is executed directly.

    # today's report (NIFTY), printed to stdout
    .venv\\Scripts\\python.exe scripts\\run_o2h_eod_report.py --date 2026-09-23

    # a specific date, written to a file
    .venv\\Scripts\\python.exe scripts\\run_o2h_eod_report.py --date 2026-09-23 --output reports\\2026-09-23-nifty-eod.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from backend.app.db.session import get_session
from backend.app.options.eod_report import build_eod_report, render_markdown


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="session date, YYYY-MM-DD (IST calendar day)")
    ap.add_argument("--underlying", default="NIFTY", choices=("NIFTY",), help="NIFTY only in O2-H")
    ap.add_argument("--output", default=None, help="write the report here instead of printing to stdout")
    args = ap.parse_args(argv)

    session_date = dt.date.fromisoformat(args.date)
    db = get_session()
    try:
        report = build_eod_report(db, underlying=args.underlying, session_date=session_date)
    finally:
        db.close()

    markdown = render_markdown(report)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        print(f"report written to {out_path}")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
