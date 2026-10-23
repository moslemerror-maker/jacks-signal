from __future__ import annotations

import datetime as dt
import json
import sys

from signal_desk.clock import IST

from . import analysis as A
from .data import IDS, LAB_DIR, load, update

RESULTS = LAB_DIR / "results"


def _opt(args, name, default):
    return type(default)(args[args.index(name) + 1]) if name in args and args.index(name) + 1 < len(args) else default


def build(data: dict, today: dt.date, years: int) -> dict:
    days = A.days_from(data["nifty"])
    since = today - dt.timedelta(days=int(365.25 * years))
    tested = [d.date for d in days if d.ratio is not None and since <= d.date < today]
    half = len(tested) // 2
    older = set(tested[:half])
    q1, q2 = A.assign(days, older)
    tset = set(tested)
    g0 = {"older": A.gate0([d for d in days if d.date in older]),
          "newer": A.gate0([d for d in days if d.date in tset and d.date not in older])}
    g1 = A.gate1(data, days, tested, older)
    return {"ran_at": dt.datetime.now(IST).strftime("%d %b %Y %H:%M"), "sessions": len(tested),
            "older": [str(tested[0]), str(tested[half - 1]), half] if tested else None,
            "newer": [str(tested[half]), str(tested[-1]), len(tested) - half] if tested else None,
            "cutoffs": {"narrow_below": round(q1, 3), "wide_from": round(q2, 3)}, "gate0": g0, "gate1": g1}


def fmt(r: dict) -> str:
    L = ["RANGE LAB - is the live Nifty breakout better after a NARROW previous day?", f"run {r['ran_at']}", "",
         f"Sessions tested: {r['sessions']}   older half {r['older'][0]} .. {r['older'][1]} ({r['older'][2]})"
         f"   newer half {r['newer'][0]} .. {r['newer'][1]} ({r['newer'][2]})",
         f"Measure: yesterday's range / avg of the 10 days before it.  narrow < {r['cutoffs']['narrow_below']}"
         f"   wide >= {r['cutoffs']['wide_from']}   (cut-offs from the older half only)", "",
         f"PASS MARK (fixed before running): in BOTH halves narrow n >= {A.MIN_N}, narrow avgR beats wide by >= "
         f"{A.MIN_GAP_R}, narrow PF >= {A.MIN_PF}; and day-shuffle p <= {A.MAX_P}.", "",
         "GATE 0 - raw days, no rules (expansion = today's range / 10-day avg; trend share = |close-open| / range)",
         f"{'':8}{'bucket':8}{'days':>6}{'expansion':>11}{'trend share':>13}{'trend days %':>14}"]
    for half in ("older", "newer"):
        for b in A.BUCKETS:
            x = r["gate0"][half][b]
            L.append(f"{half:8}{b:8}{x['days']:>6}{str(x['expansion']):>11}{str(x['trend_share']):>13}{str(x['trend_days_pct']):>14}")
    L += ["", "GATE 1 - live Signal Desk rules (13:00 cutoff + trend filter), TAKE signals only, after ₹60 costs",
          f"{'':8}{'group':10}{'n':>5}{'puts':>6}{'win%':>7}{'avgR':>7}{'PF':>6}{'rupees':>10}{'maxDD':>9}"]
    for half in ("older", "newer"):
        h = r["gate1"]["halves"][half]
        for k in ("all",) + A.BUCKETS + ("nr7", "inside"):
            x = h[k]
            L.append(f"{half:8}{k:10}{x['n']:>5}{str(x.get('puts', '')):>6}{str(x['win_rate']):>7}{str(x['avg_r']):>7}"
                     f"{str(x['pf']):>6}{x['rupees']:>10,}{x['max_dd']:>9,}")
        L.append(f"{half:8}day-shuffle p (narrow vs wide) = {h['shuffle_p']}")
    L += ["", "VERDICT: " + r["gate1"]["verdict"]] + ["  " + w for w in r["gate1"]["why"]]
    L += ["", "notes: OI checks skipped (no history). Rupees = index points x 0.5 delta x lot - ₹60 (theta not modelled).",
          "       nr7 / inside rows are for information only - the decision uses the narrow/wide buckets.",
          "       ~1 year per half gives few trades per bucket: a weak result means 'unproven', not 'useless'."]
    return "\n".join(L)


def main(argv: list[str]) -> int:
    years = max(1, min(4, _opt(argv, "--years", 2)))
    now = dt.datetime.now(IST)
    today = now.date()
    if argv[:1] == ["report"]:
        data = {k: load(k) for k in IDS}
        if not data["nifty"]:
            print("No cached data yet - run without 'report' first.")
            return 1
    else:
        if now.weekday() < 5 and dt.time(9, 0) <= now.time() <= dt.time(15, 35) and "--force" not in argv:
            print("Market hours: Range Lab would compete with Signal Desk for Dhan's data limits. Run after 3:35 PM "
                  "(or add --force).")
            return 1
        from signal_desk.config import Settings
        from signal_desk.dhan_api import DhanApi
        s = Settings.load()
        api = DhanApi(s.dhan_client_id, s.dhan_access_token)
        start = today - dt.timedelta(days=int(365.25 * years) + 30)
        data = update(api, start, today - dt.timedelta(days=1))
    if len(data["nifty"]) < 25 * 60:
        print(f"Only {len(data['nifty'])} Nifty candles - not enough history (Dhan token expired or no data access?).")
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
