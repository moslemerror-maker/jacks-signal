from __future__ import annotations

import datetime as dt
import json
import sys
import time

from .data import LAB_DIR, download_all, load_m5, universe
from .features import IST, day_records
from .gates import fmt0, fmt1, gate0, gate1

RESULTS = LAB_DIR / "results"


def _opt(args, name, default):
    return type(default)(args[args.index(name) + 1]) if name in args and args.index(name) + 1 < len(args) else default


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv and not argv[0].startswith("-") else "all"
    years = max(1, min(3, _opt(argv, "--years", 2)))
    now = dt.datetime.now(IST)
    if now.weekday() < 5 and dt.time(9, 0) <= now.time() <= dt.time(15, 35) and "--force" not in argv:
        print("Market hours: Stock Lab would compete with Signal Desk for Dhan's data limits. Run after 3:35 PM "
              "(or add --force).")
        return 1
    from signal_desk.config import Settings
    from signal_desk.dhan_api import DhanApi
    s = Settings.load()
    api = DhanApi(s.dhan_client_id, s.dhan_access_token)
    today = now.date()
    end = today - dt.timedelta(days=1)
    start = end - dt.timedelta(days=int(365.25 * years) + 30)
    stocks = universe(today)
    print(f"Stock Lab: {len(stocks)} F&O stocks, {start} .. {end} ({years} year(s) of 5-min candles)")
    t0 = time.time()
    if cmd in ("all", "download"):
        download_all(api, stocks, start, end, progress=lambda m: print("..", m, flush=True))
        print(f".. download done in {(time.time() - t0) / 60:.1f} min")
        if cmd == "download":
            return 0
    days = []
    for i, st in enumerate(stocks):
        if i % 25 == 0:
            print(f".. building day records {i}/{len(stocks)}", flush=True)
        rows = load_m5(api, st.security_id, start, end, refresh=False)
        days.extend(day_records(st.symbol, rows))
    print(f".. {len(days)} stock-days")
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%d %b %Y %H:%M")
    if cmd in ("all", "gate0"):
        r0 = gate0(days)
        txt = fmt0(r0)
        print("\n" + txt)
        (RESULTS / "gate0.txt").write_text(f"Run {stamp}\n\n{txt}\n", encoding="utf-8")
        (RESULTS / "gate0.json").write_text(json.dumps(r0, default=str, indent=1), encoding="utf-8")
    if cmd in ("all", "gate1"):
        r1 = gate1(days)
        txt = fmt1(r1)
        print("\n" + txt)
        (RESULTS / "gate1.txt").write_text(f"Run {stamp}\n\n{txt}\n", encoding="utf-8")
        (RESULTS / "gate1.json").write_text(json.dumps(r1, default=str, indent=1), encoding="utf-8")
    print(f"\nSaved to {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
