from __future__ import annotations

import datetime as dt
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from signal_desk.clock import IST

from . import analysis as A
from .data import LAB_DIR, cached_day, fetch_day, is_cached

RESULTS = LAB_DIR / "results"
PROBE_CODES = (1, 2)
WORKERS = 4


def _opt(args, name, default):
    return type(default)(args[args.index(name) + 1]) if name in args and args.index(name) + 1 < len(args) else default


def main(argv: list[str]) -> int:
    from expiry_lab.analysis import expiry_days
    from range_lab.data import IDS, load, update
    years = max(1, min(3, _opt(argv, "--years", 2)))
    now = dt.datetime.now(IST)
    today = now.date()
    cache_only = argv[:1] == ["report"]
    if cache_only:
        data = {k: load(k) for k in IDS}
    else:
        if now.weekday() < 5 and dt.time(9, 0) <= now.time() <= dt.time(15, 35) and "--force" not in argv:
            print("Market hours: run after 3:35 PM (or add --force).")
            return 1
        import signal_desk.dhan_api as DA
        from signal_desk.config import Settings
        DA.TIMEOUT = 90
        s = Settings.load()
        api = DA.DhanApi(s.dhan_client_id, s.dhan_access_token)
        data = update(api, today - dt.timedelta(days=int(365.25 * years) + 30), today - dt.timedelta(days=1))
    say = (lambda m: None) if cache_only else (lambda m: print(m, flush=True))
    by: dict[dt.date, int] = {}
    for c in data["nifty"]:
        by[c.start.date()] = by.get(c.start.date(), 0) + 1
    all_days = sorted(by)
    start = today - dt.timedelta(days=int(365.25 * years))
    exp = sorted(d for d in expiry_days(all_days) if start <= d < today and by[d] >= 20)
    if len(exp) < 20:
        print(f"Only {len(exp)} expiry days in the Nifty cache - not enough.")
        return 1

    if not cache_only:
        for rnd in range(1, 9):
            misses: set = set()

            def record(code, typ, off, day):
                if not is_cached(code, typ, off, day):
                    misses.add((code, typ, off, day))
                return cached_day(code, typ, off, day)
            compute(record, data, exp, now, quiet=True)
            if not misses:
                break
            say(f".. round {rnd}: {len(misses)} pieces to download ({WORKERS} at a time)")
            download(sorted(misses, key=lambda m: (m[3], m[0], m[1], m[2])), s, say)
    res = compute(cached_day, data, exp, now, quiet=False, say=say)
    if res is None:
        print("Could not identify the expiring contract from Dhan's data - see the ! lines above.")
        return 1
    RESULTS.mkdir(parents=True, exist_ok=True)
    text = fmt(res)
    (RESULTS / "report.txt").write_text(text, encoding="utf-8")
    (RESULTS / "report.json").write_text(json.dumps(res, default=str, indent=1), encoding="utf-8")
    print(text)
    print(f"\nSaved to {RESULTS / 'report.txt'}")
    return 0


def compute(fetch, data, exp, now, quiet=True, say=lambda m: None) -> dict | None:
    from expiry_lab.analysis import SWITCH as sw
    from signal_desk.research import VARIANTS, replay
    book = A.Book(fetch=fetch)
    codes = A.detect_codes(book, exp[:6], PROBE_CODES)
    if codes["zero"] is None:
        return None
    if not quiet:
        say(f".. expiring-week code = {codes['zero']}, next-week code = {codes['next']}  "
            f"({codes['median_atm_straddle_1515']})")
    half = len(exp) // 2
    older = set(exp[:half])
    res: dict = {"ran_at": now.strftime("%d %b %Y %H:%M"), "expiry_days": len(exp),
                 "older": [str(exp[0]), str(exp[half - 1]), half], "newer": [str(exp[half]), str(exp[-1]), len(exp) - half],
                 "codes": codes}
    q1 = {d: A.straddle(book, codes["zero"], d) for d in exp}
    h1 = {}
    for name, cond in (("older", lambda d: d in older), ("newer", lambda d: d not in older)):
        h1[name] = A.pf_stats([v for d, v in q1.items() if cond(d) and v is not None])
    p1 = A.signflip_p([v for v in q1.values() if v is not None])
    res["q1"] = {"halves": h1, "missing_days": sum(1 for v in q1.values() if v is None), "p": p1,
                 "eras": {"thursday": A.pf_stats([v for d, v in q1.items() if v is not None and d < sw]),
                          "tuesday": A.pf_stats([v for d, v in q1.items() if v is not None and d >= sw])}}
    res["q1"]["verdict"], res["q1"]["why"] = A.verdict_q1(h1, p1)
    rs = replay(VARIANTS[0], data["nifty"], data["banknifty"], data["vix"], exp)
    day_c: dict[dt.date, list] = {}
    for c in data["nifty"]:
        day_c.setdefault(c.start.date(), []).append(c)
    sigs = [(x.signal, [c for c in day_c[x.signal.time.date()] if c.start >= x.signal.time])
            for x in rs if x.signal.action == "TAKE"]
    trades = A.q2_trades(book, codes, sigs)
    h2, diffs_all = {}, []
    for name, cond in (("older", lambda d: d in older), ("newer", lambda d: d not in older)):
        pairs = [t for t in trades if cond(t["date"]) and t["zero"] is not None and t["next"] is not None]
        diffs = [t["zero"] - t["next"] for t in pairs]
        diffs_all += diffs
        h2[name] = {"signals": sum(1 for t in trades if cond(t["date"])), "pairs": len(pairs),
                    "zero": A.pf_stats([t["zero"] for t in pairs]), "next": A.pf_stats([t["next"] for t in pairs]),
                    "diff_avg": round(sum(diffs) / len(diffs)) if diffs else None}
    p2 = A.signflip_p(diffs_all)
    res["q2"] = {"halves": h2, "p": p2, "trades": [dict(t, date=str(t["date"])) for t in trades]}
    res["q2"]["verdict"], res["q2"]["why"] = A.verdict_q2(h2, p2)
    return res


def download(jobs: list, settings, say) -> None:
    import signal_desk.dhan_api as DA
    local = threading.local()
    done = {"n": 0, "t0": time.monotonic()}
    lock = threading.Lock()

    def work(job):
        if not hasattr(local, "api"):
            local.api = DA.DhanApi(settings.dhan_client_id, settings.dhan_access_token)
        fetch_day(local.api, *job, say=say)
        with lock:
            done["n"] += 1
            if done["n"] % 20 == 0 or done["n"] == len(jobs):
                el = time.monotonic() - done["t0"]
                left = (len(jobs) - done["n"]) * el / done["n"]
                say(f"   .. {done['n']}/{len(jobs)} done, {int(el // 60)} min so far, ~{int(left // 60)} min left this round")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for f in [ex.submit(work, j) for j in jobs]:
            f.result()


def fmt(r: dict) -> str:
    L = ["EXPIRY 0DTE LAB - real Nifty weekly option premiums on expiry days (Dhan expired-options data)",
         f"run {r['ran_at']}", "",
         f"Expiry days: {r['expiry_days']}   older {r['older'][0]} .. {r['older'][1]} ({r['older'][2]})   "
         f"newer {r['newer'][0]} .. {r['newer'][1]} ({r['newer'][2]})",
         f"Dhan expiryCode: expiring week = {r['codes']['zero']}, next week = {r['codes']['next']}   "
         f"(median ATM straddle at 15:15 by code: {r['codes']['median_atm_straddle_1515']})", "",
         "Q1 - buy ATM CE + PE of the EXPIRING week at 9:20, sell 15:15 (lot 65, ₹60 per option)",
         f"PASS MARK (fixed before running): both years n >= {A.Q1_MIN_N} and PF >= {A.Q1_BUY_PF} (buyers' edge) or "
         f"<= {A.Q1_SELL_PF} (sellers' edge); sign-flip p <= {A.MAX_P}.",
         f"{'':8}{'n':>5}{'win%':>7}{'avg ₹':>8}{'PF':>6}{'total ₹':>10}{'maxDD':>10}"]
    for k in ("older", "newer"):
        x = r["q1"]["halves"][k]
        L.append(f"{k:8}{x['n']:>5}{str(x['win']):>7}{str(x['avg']):>8}{str(x['pf']):>6}{x['total']:>10,}{x['max_dd']:>10,}")
    for k in ("thursday", "tuesday"):
        x = r["q1"]["eras"][k]
        L.append(f"  (info) {k} era: n={x['n']} win={x['win']}% avg=₹{x['avg']} PF={x['pf']} total=₹{x['total']:,}")
    L.append(f"  days skipped (strike data missing): {r['q1']['missing_days']}")
    L += ["VERDICT Q1: " + r["q1"]["verdict"]] + ["  " + w for w in r["q1"]["why"]]
    L += ["", "Q2 - Signal Desk expiry-day TAKE signals: the expiring (0DTE) ATM option vs next week's ATM option",
          "(same entry/exit times from the live rules; half at T1, rest at stop / T2 / cost / 15:15; ₹60 per trade)"]
    for k in ("older", "newer"):
        x = r["q2"]["halves"][k]
        L.append(f"{k:8}signals={x['signals']} paired={x['pairs']}   0DTE: avg ₹{x['zero']['avg']} PF {x['zero']['pf']} "
                 f"total ₹{x['zero']['total']:,}   next week: avg ₹{x['next']['avg']} PF {x['next']['pf']} "
                 f"total ₹{x['next']['total']:,}")
    L += ["VERDICT Q2: " + r["q2"]["verdict"]] + ["  " + w for w in r["q2"]["why"]]
    L += ["", "notes: real premiums from 5-min candle closes (no bid-ask spread beyond the ₹60 cost - 0DTE spreads",
          "       are usually tight in ATM Nifty). Lot 65 for every year for comparability. OI checks skipped in Q2."]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
