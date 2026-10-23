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


def compute(fetch, data, tested, expiry, older) -> dict:
    from signal_desk.research import VARIANTS, replay
    book = A.Book(fetch=fetch)
    rs = [x for x in replay(VARIANTS[0], data["nifty"], data["banknifty"], data["vix"], tested) if x.signal.action == "TAKE"]
    day_c: dict = {}
    for c in data["nifty"]:
        day_c.setdefault(c.start.date(), []).append(c)
    tr = A.trades(book, rs, day_c, expiry)
    s = A.summarise(tr, older)
    s["trades"] = [dict(t, date=str(t["date"])) for t in tr]
    return s


def main(argv: list[str]) -> int:
    from expiry_0dte_lab.__main__ import WORKERS, download
    from expiry_0dte_lab.data import cached_day, is_cached
    from expiry_lab.analysis import expiry_days
    from range_lab.data import IDS, load, update
    years = max(1, min(3, _opt(argv, "--years", 2)))
    now = dt.datetime.now(IST)
    today = now.date()
    cache_only = argv[:1] == ["report"]
    say = (lambda m: None) if cache_only else (lambda m: print(m, flush=True))
    if cache_only:
        data = {k: load(k) for k in IDS}
    else:
        if now.weekday() < 5 and dt.time(9, 0) <= now.time() <= dt.time(15, 35) and "--force" not in argv:
            print("Market hours: Premium Lab would compete with Signal Desk for Dhan's limits. Run after 3:35 PM.")
            return 1
        import signal_desk.dhan_api as DA
        from signal_desk.config import Settings
        DA.TIMEOUT = 90
        s = Settings.load()
        data = update(DA.DhanApi(s.dhan_client_id, s.dhan_access_token),
                      today - dt.timedelta(days=int(365.25 * years) + 30), today - dt.timedelta(days=1))
    by: dict = {}
    for c in data["nifty"]:
        by[c.start.date()] = by.get(c.start.date(), 0) + 1
    since = today - dt.timedelta(days=int(365.25 * years))
    tested = [d for d in sorted(by) if since <= d < today and by[d] >= 20]
    if len(tested) < 100:
        print(f"Only {len(tested)} sessions in the Nifty cache - not enough.")
        return 1
    expiry = expiry_days(sorted(by))
    half = len(tested) // 2
    older = set(tested[:half])
    if not cache_only:
        for rnd in range(1, 9):
            misses: set = set()

            def record(code, typ, off, day):
                if not is_cached(code, typ, off, day):
                    misses.add((code, typ, off, day))
                return cached_day(code, typ, off, day)
            compute(record, data, tested, expiry, older)
            if not misses:
                break
            say(f".. round {rnd}: {len(misses)} pieces to download ({WORKERS} at a time)")
            download(sorted(misses, key=lambda m: (m[3], m[0], m[1], m[2])), s, say)
    r = compute(cached_day, data, tested, expiry, older)
    r.update(ran_at=now.strftime("%d %b %Y %H:%M"), sessions=len(tested),
             older=[str(tested[0]), str(tested[half - 1]), half], newer=[str(tested[half]), str(tested[-1]), len(tested) - half])
    RESULTS.mkdir(parents=True, exist_ok=True)
    text = fmt(r)
    (RESULTS / "report.txt").write_text(text, encoding="utf-8")
    (RESULTS / "report.json").write_text(json.dumps(r, default=str, indent=1), encoding="utf-8")
    print(text)
    print(f"\nSaved to {RESULTS / 'report.txt'}")
    return 0


def fmt(r: dict) -> str:
    L = ["PREMIUM LAB - Signal Desk's live Nifty rules with REAL option premiums (Dhan expired-options data)",
         f"run {r['ran_at']}", "",
         f"Sessions: {r['sessions']}   older {r['older'][0]} .. {r['older'][1]} ({r['older'][2]})   "
         f"newer {r['newer'][0]} .. {r['newer'][1]} ({r['newer'][2]})",
         "Contract as alerted: nearest weekly (next week's on expiry day); strike CE at/below, PE at/above the entry.",
         "Exits from the index rules: half at T1, rest at stop / T2 / cost / 15:15. Lot 65, ₹60 per trade.",
         f"PASS MARK (fixed before running): REAL avg ₹ > 0 and PF >= {A.MIN_PF} in both years; >= 80% of signals priced.", "",
         f"{'':8}{'group':12}{'n':>5}{'win%':>7}{'avg ₹':>8}{'PF':>6}{'total ₹':>10}{'maxDD':>10}"]
    for half in ("older", "newer"):
        h = r["halves"][half]
        for k in ("real", "estimate", "calls", "puts", "grade_A", "grade_B", "expiry_day", "before_11", "from_11"):
            x = h[k]
            label = {"real": "REAL", "estimate": "estimate*"}.get(k, k)
            L.append(f"{half:8}{label:12}{x['n']:>5}{str(x['win']):>7}{str(x['avg']):>8}{str(x['pf']):>6}"
                     f"{x['total']:>10,}{x['max_dd']:>10,}")
        L.append(f"{half:8}signals priced: {h['priced']}/{h['signals']}")
    L += ["", "VERDICT: " + r["verdict"]] + ["  " + w for w in r["why"]]
    L += ["", "* estimate = the old method used by every earlier lab (index points x 0.5 x lot - ₹60), same trades.",
          "  calls/puts/grades/expiry/time rows are REAL premiums, information only.",
          "notes: 5-min candle closes, no extra bid-ask spread beyond the ₹60 cost. OI checks skipped (no history)."]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
