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
    since = today - dt.timedelta(days=int(365.25 * years))
    days = [d for d in A.build_days(data["nifty"], data.get("vix", [])) if since <= d.date < today]
    half = len(days) // 2
    older = {d.date for d in days[:half]}
    r = A.run(days, older)
    r.update(ran_at=dt.datetime.now(IST).strftime("%d %b %Y %H:%M"), sessions=len(days),
             older=[str(days[0].date), str(days[half - 1].date), half] if days else None,
             newer=[str(days[half].date), str(days[-1].date), len(days) - half] if days else None)
    return r


def fmt(r: dict) -> str:
    L = ["MOMENTUM LAB - does Nifty's first half-hour (+ the 14:30-15:00 move) predict the last half-hour?",
         f"run {r['ran_at']}", "",
         f"Sessions: {r['sessions']}   older {r['older'][0]} .. {r['older'][1]} ({r['older'][2]})"
         f"   newer {r['newer'][0]} .. {r['newer'][1]} ({r['newer'][2]})", "",
         "PRIMARY RULE: if (prev close -> 9:45) and (14:30 -> 15:00) moved the same way, buy CE (up) / PE (down) at 15:00,",
         "exit 15:30. Rupees = points x 0.5 x 65 - ₹60.",
         f"PASS MARK (fixed before running): both years n >= {A.MIN_N}, avg ₹ > 0, PF >= {A.MIN_PF}, hit > {A.MIN_HIT}%; "
         f"random-direction p <= {A.MAX_P}.", "",
         f"{'':8}{'group':22}{'n':>5}{'hit%':>7}{'avg pts':>9}{'avg ₹':>8}{'PF':>6}{'total ₹':>10}{'maxDD':>9}"]
    keys = ["primary", "first_only", "primary_exit_1515", "primary_high_vix", "primary_low_vix", "primary_expiry",
            "primary_non_expiry"]
    for half in ("older", "newer"):
        h = r["halves"][half]
        for k in keys:
            if k not in h:
                continue
            x = h[k]
            L.append(f"{half:8}{k:22}{x['n']:>5}{str(x['hit']):>7}{str(x['avg_pts']):>9}{str(x['avg_rs']):>8}"
                     f"{str(x['pf']):>6}{x['rupees']:>10,}{x['max_dd']:>9,}")
        L.append(f"{half:8}random-direction p: primary {h['primary_flip_p']}, first-only {h['first_only_flip_p']}"
                 f"   | avg size of last half-hour move: {h['avg_abs_last30_pts']} pts")
    L += ["", "VERDICT: " + r["verdict"]] + ["  " + w for w in r["why"]]
    L += ["", f"notes: high/low VIX split at the older-year median VIX at 15:00 ({r['vix_cut']}).",
          "       expiry days approximated by weekday (Thu till Aug 2025, Tue from Sep 2025).",
          "       Only the PRIMARY rows decide the verdict; all other rows are information.",
          "       Exit at the 15:30 close is slightly optimistic - in practice exit by ~15:25."]
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
