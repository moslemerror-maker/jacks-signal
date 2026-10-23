from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

from signal_desk.clock import IST

from . import analysis as A

LAB_DIR = Path(__file__).resolve().parent
RESULTS = LAB_DIR / "results"


def _opt(args, name, default):
    return type(default)(args[args.index(name) + 1]) if name in args and args.index(name) + 1 < len(args) else default


def build(data: dict, today: dt.date, years: int) -> dict:
    by: dict[dt.date, int] = {}
    for c in data["nifty"]:
        by[c.start.date()] = by.get(c.start.date(), 0) + 1
    all_days = sorted(by)
    since = today - dt.timedelta(days=int(365.25 * years))
    tested = [d for d in all_days if since <= d < today and by[d] >= 20]
    half = len(tested) // 2
    r = A.run(data, tested, set(tested[:half]), all_days)
    r.update(ran_at=dt.datetime.now(IST).strftime("%d %b %Y %H:%M"), sessions=len(tested),
             older=[str(tested[0]), str(tested[half - 1]), half], newer=[str(tested[half]), str(tested[-1]), len(tested) - half])
    return r


def fmt(r: dict) -> str:
    L = ["EXPIRY LAB - do the live Nifty breakout signals behave differently on weekly expiry day?", f"run {r['ran_at']}", "",
         f"Sessions: {r['sessions']} ({r['expiry_days_tested']} expiry days)   older {r['older'][0]} .. {r['older'][1]} "
         f"({r['older'][2]})   newer {r['newer'][0]} .. {r['newer'][1]} ({r['newer'][2]})",
         "Expiry: Thursday till 28 Aug 2025, Tuesday from 2 Sep 2025, previous trading day if a holiday.",
         f"PASS MARK (fixed before running): expiry-minus-other avg R same sign in both years, |gap| >= {A.MIN_GAP} R, "
         f"expiry n >= {A.MIN_N} each year, two-sided day-shuffle p <= {A.MAX_P}.", "",
         "Live Signal Desk rules (13:00 cutoff + trend filter), TAKE signals only, after ₹60 costs",
         f"{'':8}{'group':18}{'n':>5}{'win%':>7}{'avgR':>7}{'PF':>6}{'rupees':>10}{'maxDD':>9}"]
    for half in ("older", "newer"):
        h = r["halves"][half]
        for k in ("all", "expiry", "day_before", "other", "expiry_calls", "expiry_puts", "other_calls", "other_puts"):
            x = h[k]
            L.append(f"{half:8}{k:18}{x['n']:>5}{str(x['win_rate']):>7}{str(x['avg_r']):>7}{str(x['pf']):>6}"
                     f"{x['rupees']:>10,}{x['max_dd']:>9,}")
        L.append(f"{half:8}expiry-minus-other gap = {h['gap']} R   two-sided day-shuffle p = {h['shuffle_p']}")
    L += ["", "By expiry era (information only):"]
    for era in ("thursday_era", "tuesday_era"):
        for g in ("expiry", "other"):
            x = r[era][g]
            L.append(f"  {era:14}{g:8} n={x['n']:>4}  win%={x['win_rate']}  avgR={x['avg_r']}  PF={x['pf']}  ₹{x['rupees']:,}")
    L += ["", "VERDICT: " + r["verdict"]] + ["  " + w for w in r["why"]]
    L += ["", "notes: OI checks skipped (no history). Rupees = index points x 0.5 delta x lot - ₹60 (theta not modelled).",
          "       Signal Desk buys NEXT week's contract on expiry day; the index-points estimate is the same either way.",
          "       Only the expiry vs other rows decide the verdict; day_before, calls/puts and eras are information."]
    return "\n".join(L)


def main(argv: list[str]) -> int:
    from range_lab.data import IDS, load, update
    years = max(1, min(4, _opt(argv, "--years", 2)))
    now = dt.datetime.now(IST)
    today = now.date()
    if argv[:1] == ["report"]:
        data = {k: load(k) for k in IDS}
    else:
        if now.weekday() < 5 and dt.time(9, 0) <= now.time() <= dt.time(15, 35) and "--force" not in argv:
            print("Market hours: run after 3:35 PM (or add --force).")
            return 1
        from signal_desk.config import Settings
        from signal_desk.dhan_api import DhanApi
        s = Settings.load()
        api = DhanApi(s.dhan_client_id, s.dhan_access_token)
        data = update(api, today - dt.timedelta(days=int(365.25 * years) + 30), today - dt.timedelta(days=1))
    if len(data["nifty"]) < 25 * 60:
        print(f"Only {len(data['nifty'])} Nifty candles - not enough history.")
        return 1
    r = build(data, today, years)
    RESULTS.mkdir(parents=True, exist_ok=True)
    text = fmt(r)
    (RESULTS / "report.txt").write_text(text, encoding="utf-8")
    (RESULTS / "report.json").write_text(json.dumps(r, default=str, indent=1), encoding="utf-8")
    print(text)
    print(f"\nSaved to {RESULTS / 'report.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
