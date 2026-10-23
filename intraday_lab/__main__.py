"""Intraday Lab - research only (never writes into signal_desk/ or stock_lab/, never alerts, never places orders).

    python -m intraday_lab            download/update 1-min candles (first run ~30-60 min) + explore idea B
    python -m intraday_lab b          explore idea B from the cache
    python -m intraday_lab vault b    ONE-TIME test of idea B's candidates on the locked last 6 months
    python -m intraday_lab news-check fetch 3 days of NSE announcements once (connection test)
    python -m intraday_lab news       download NSE announcements (free) for the whole period
    python -m intraday_lab a          explore idea A (needs candles + announcements)
    python -m intraday_lab vault a    ONE-TIME vault test of idea A's candidates
    python -m intraday_lab cd         explore ideas C and D (stock_lab 5-min + range_lab Nifty caches, read-only)
    python -m intraday_lab vault cd   ONE-TIME vault test of idea C/D candidates
Options: --force (allow running during market hours).  Results: intraday_lab/results/."""
from __future__ import annotations

import datetime as dt
import json
import pickle
import sys

from . import idea_a as A
from . import idea_b as B
from . import idea_cd as CD
from . import news as N
from .data import IST, LAB_DIR, M1, download_all, read_bars, read_turnover, universe

RESULTS = LAB_DIR / "results"
DATA_FROM = dt.date(2024, 10, 23)
VAULT_FROM = dt.date(2026, 4, 23)


def load_records(stocks, say) -> list[dict]:
    recs = []
    for i, s in enumerate(stocks):
        if i % 25 == 0:
            say(f".. reading {i}/{len(stocks)}")
        recs.extend(stock_records(s))
    return recs


def stock_records(s) -> list[dict]:
    src = M1 / f"{s.security_id}.csv.gz"
    if not src.exists():
        return []
    pk = M1 / f"{s.security_id}.b{B.VERSION}.pkl"
    if pk.exists() and pk.stat().st_mtime >= src.stat().st_mtime:
        try:
            return pickle.loads(pk.read_bytes())
        except Exception:
            pass
    out = B.stock_days(s.security_id, read_bars(s.security_id), read_turnover(s.security_id))
    for r in out:
        r["symbol"] = s.symbol
    pk.write_bytes(pickle.dumps(out))
    return out


def fmt_b(r: dict, title: str) -> str:
    L = [title, f"run {r['ran_at']}   stocks with data: {r['stocks']}   eligible stock-days: {r['eligible_days']}", "",
         f"Exploration halves: older {r['halves']['older'][0]} .. {r['halves']['older'][1]} ({r['halves']['older'][2]} days)"
         f"   newer {r['halves']['newer'][0]} .. {r['halves']['newer'][1]} ({r['halves']['newer'][2]} days)",
         f"VAULT (locked, not used here): {VAULT_FROM} onwards.", "",
         "Rule: at 9:17 rank eligible F&O stocks by change from yesterday's close; top 3 gainers + top 3 losers;",
         "FOLLOW = buy gainers / short losers, FADE = the opposite; entry 9:17 open; ₹1 lakh per trade; real costs.",
         f"PASS (fixed before running): both halves net > 0, PF >= {B.MIN_PF}, better than random; random-direction p <= {B.MAX_P}.", "",
         f"{'exit':7}{'mode':16}| {'OLDER n':>7}{'hit%':>6}{'gross%':>8}{'net%':>7}{'PF':>6}{'₹/trade':>8} | "
         f"{'NEWER n':>7}{'hit%':>6}{'gross%':>8}{'net%':>7}{'PF':>6}{'₹/trade':>8} | {'rand net% o/n':>14} {'p':>7}  verdict"]
    for x in r["rows"]:
        o, n = x["older"], x["newer"]
        L.append(f"{x['exit']:7}{x['mode']:16}| {o['n']:>7}{str(o['hit']):>6}{str(o['gross']):>8}{str(o['net']):>7}"
                 f"{str(o['pf']):>6}{str(o['rs_per_trade']):>8} | {n['n']:>7}{str(n['hit']):>6}{str(n['gross']):>8}"
                 f"{str(n['net']):>7}{str(n['pf']):>6}{str(n['rs_per_trade']):>8} | "
                 f"{str(x['random']['older']):>6}/{str(x['random']['newer']):<7} {str(x['p']):>7}  {x['verdict']}")
    L += ["", "VAULT CANDIDATES: " + (", ".join(r["candidates"]) if r["candidates"] else "none - idea B fails exploration")]
    L += ["", "notes: T03/T05/T10 = target and stop 0.3/0.5/1.0%, else out 10:15; X930/X945/X1015 = time exits.",
          "       gross% = before costs (Gate 0), net% = after Dhan charges + 0.05% slippage per side (Gate 1).",
          "       Universe = today's F&O list (stocks that left F&O are missing - slight survivorship bias)."]
    return "\n".join(L)


def cd_records(stocks, say) -> list[dict]:
    recs = []
    for i, s in enumerate(stocks):
        if i % 25 == 0:
            say(f".. reading 5-min {i}/{len(stocks)}")
        src = CD.M5 / f"{s.security_id}.csv.gz"
        if not src.exists():
            continue
        pk = M1 / f"{s.security_id}.cd{CD.VERSION}.pkl"
        if pk.exists() and pk.stat().st_mtime >= src.stat().st_mtime:
            try:
                recs.extend(pickle.loads(pk.read_bytes()))
                continue
            except Exception:
                pass
        out = CD.stock_days(s.security_id, s.symbol, CD.read_m5(s.security_id))
        M1.mkdir(parents=True, exist_ok=True)
        pk.write_bytes(pickle.dumps(out))
        recs.extend(out)
    return recs


def run_cd(cmd, stocks, now, say) -> int:
    from range_lab.data import load as load_index
    nifty = CD.nifty_945(load_index("nifty"))
    recs = cd_records(stocks, say)
    days = sorted({r["date"] for r in recs if r["eligible"]})
    explore_days = [d for d in days if d < VAULT_FROM]
    if len(explore_days) < 100 or not nifty:
        print("Not enough data (needs stock_lab 5-min cache and range_lab Nifty cache).")
        return 1
    RESULTS.mkdir(parents=True, exist_ok=True)
    if cmd == ["cd"]:
        r = CD.explore([x for x in recs if x["date"] < VAULT_FROM], explore_days, nifty)
        r.update(ran_at=now.strftime("%d %b %Y %H:%M"), stocks=len({x["sid"] for x in recs}),
                 eligible_days=sum(1 for x in recs if x["eligible"] and x["date"] < VAULT_FROM))
        text = fmt_b(r, "INTRADAY LAB - IDEAS C and D (EXPLORATION, vault untouched)")
        text = text.replace("Rule: at 9:17 rank eligible F&O stocks by change from yesterday's close; top 3 gainers + top 3 losers;\n"
                            "FOLLOW = buy gainers / short losers, FADE = the opposite; entry 9:17 open;",
                            "C = top-20 'stocks in play' (first 5-min volume vs 14-day avg), 5-min opening range breakout in the\n"
                            "first candle's direction, stop 10% of ATR, out 15:15.  D = on Nifty -0.3%/+0.3% mornings, the 3\n"
                            "strongest / weakest stocks at 9:45, stop 1%, out 15:15 (1130/1300 = earlier exits, info).")
        text = text.replace(f"random-direction p <= {B.MAX_P}.", f"random-direction p <= {CD.MAX_P} (one primary version each).")
        text = text.replace("VAULT CANDIDATES: none - idea B fails exploration", "VAULT CANDIDATES: none - ideas C and D fail exploration")
        fut = ["", "futures-cost estimate (info: ₹5 lakh stock-futures position, flat ₹20 brokerage, 0.05% slippage per side):"]
        for x in r["rows"]:
            fut.append(f"  {x['mode']:7}{x['exit']:6} net% older {x['older']['fut_net']}  newer {x['newer']['fut_net']}")
        text = text.replace("\nnotes:", "\n".join(fut) + "\n\nnotes:", 1)
        (RESULTS / "idea_cd.txt").write_text(text, encoding="utf-8")
        (RESULTS / "idea_cd.json").write_text(json.dumps(r, default=str, indent=1), encoding="utf-8")
        print(text)
        return 0
    prev = json.loads((RESULTS / "idea_cd.json").read_text(encoding="utf-8")) if (RESULTS / "idea_cd.json").exists() else {}
    cands = prev.get("candidates") or []
    if not cands:
        print("No vault candidates from ideas C/D - the vault stays locked.")
        return 1
    if (RESULTS / "idea_cd_vault.txt").exists():
        print("The vault was already used for ideas C/D - see results/idea_cd_vault.txt.")
        return 1
    vault_days = {d for d in days if d >= VAULT_FROM}
    lines = [f"INTRADAY LAB - IDEAS C/D VAULT TEST ({min(vault_days)} .. {max(vault_days)}, {len(vault_days)} days)",
             "PASS: avg net % > 0 and PF >= 1.1", ""]
    for c in cands:
        mode, key = c.split("/")
        real = (CD.trades_d(recs, vault_days, nifty, key) if mode.endswith("D") else CD.trades_c(recs, vault_days, key))[0]
        s_ = CD._stats(real)
        ok = (s_["net"] or -9) > 0 and (s_["pf"] or 0) >= 1.1
        lines.append(f"{c:14} n={s_['n']} hit={s_['hit']}% net={s_['net']}% PF={s_['pf']} ₹/trade={s_['rs_per_trade']}"
                     f"  ->  {'PASS' if ok else 'FAIL'}")
    text = "\n".join(lines)
    (RESULTS / "idea_cd_vault.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


def run_a(cmd, stocks, now, say) -> int:
    events = N.load_events()
    if len(events) < 1000:
        print(f"Only {len(events)} NSE announcements cached - run 'python -m intraday_lab news' first.")
        return 1
    recs = load_records(stocks, say)
    days = sorted({r["date"] for r in recs if r["eligible"]})
    explore_days = [d for d in days if d < VAULT_FROM]
    RESULTS.mkdir(parents=True, exist_ok=True)
    if cmd == ["a"]:
        r = A.explore([x for x in recs if x["date"] < VAULT_FROM], events, explore_days)
        r.update(ran_at=now.strftime("%d %b %Y %H:%M"), stocks=len({x["sid"] for x in recs}),
                 eligible_days=sum(1 for x in recs if x["eligible"] and x["date"] < VAULT_FROM))
        text = fmt_b(r, "INTRADAY LAB - IDEA A: 9:17 movers split by fresh NSE announcements (EXPLORATION, vault untouched)")
        text = text.replace("none - idea B fails exploration", "none - idea A fails exploration")
        text = text.replace("Rule: at 9:17 rank eligible F&O stocks by change from yesterday's close; top 3 gainers + top 3 losers;\n"
                            "FOLLOW = buy gainers / short losers, FADE = the opposite;",
                            f"Rule: at 9:17 top 10 gainers + top 10 losers; NEWS (material NSE announcement since yesterday 15:30) -> "
                            f"follow, NO NEWS -> fade.\nNews movers per day: {r['news_movers_per_day']}.")
        (RESULTS / "idea_a.txt").write_text(text, encoding="utf-8")
        (RESULTS / "idea_a.json").write_text(json.dumps(r, default=str, indent=1), encoding="utf-8")
        print(text)
        return 0
    prev = json.loads((RESULTS / "idea_a.json").read_text(encoding="utf-8")) if (RESULTS / "idea_a.json").exists() else {}
    cands = prev.get("candidates") or []
    if not cands:
        print("No vault candidates from idea A exploration - the vault stays locked.")
        return 1
    if (RESULTS / "idea_a_vault.txt").exists():
        print("The vault was already used for idea A - see results/idea_a_vault.txt.")
        return 1
    vault_days = {d for d in days if d >= VAULT_FROM}
    cls = A.classify([x for x in recs if x["date"] >= VAULT_FROM - dt.timedelta(days=7)], events)
    lines = [f"INTRADAY LAB - IDEA A VAULT TEST ({min(vault_days)} .. {max(vault_days)}, {len(vault_days)} days)",
             "PASS: avg net % > 0 and PF >= 1.1", ""]
    for c in cands:
        ek, legmode = c.split("/")
        leg, mode = legmode.split("-")
        n, g, _ = A.leg_trades(cls, vault_days, leg, mode, ek)
        s_ = B._stats(n, g)
        ok = (s_["net"] or -9) > 0 and (s_["pf"] or 0) >= 1.1
        lines.append(f"{c:22} n={s_['n']} hit={s_['hit']}% net={s_['net']}% PF={s_['pf']} ₹/trade={s_['rs_per_trade']}"
                     f"  ->  {'PASS' if ok else 'FAIL'}")
    text = "\n".join(lines)
    (RESULTS / "idea_a_vault.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


def main(argv: list[str]) -> int:
    now = dt.datetime.now(IST)
    today = now.date()
    say = lambda m: print(m, flush=True)
    cmd = [a for a in argv if not a.startswith("--")]
    stocks = universe(today)
    if not cmd:
        if now.weekday() < 5 and dt.time(9, 0) <= now.time() <= dt.time(15, 35) and "--force" not in argv:
            print("Market hours: run after 3:35 PM (or add --force).")
            return 1
        from signal_desk.config import Settings
        from signal_desk.dhan_api import DhanApi
        s = Settings.load()
        download_all(DhanApi(s.dhan_client_id, s.dhan_access_token), stocks, DATA_FROM, today - dt.timedelta(days=1), say)
        cmd = ["b"]
    if cmd == ["news-check"]:
        return 0 if N.check(say) else 1
    if cmd == ["news"]:
        if not N.check(say):
            return 1
        N.download(DATA_FROM - dt.timedelta(days=7), today - dt.timedelta(days=1), say)
        return 0
    if cmd in (["cd"], ["vault", "cd"]):
        return run_cd(cmd, stocks, now, say)
    if cmd in (["a"], ["vault", "a"]):
        return run_a(cmd, stocks, now, say)
    if cmd == ["b"] or cmd == ["vault", "b"]:
        recs = load_records(stocks, say)
        days = sorted({r["date"] for r in recs if r["eligible"]})
        explore_days = [d for d in days if d < VAULT_FROM]
        if len(explore_days) < 100:
            print(f"Only {len(explore_days)} exploration days with data - download first.")
            return 1
        RESULTS.mkdir(parents=True, exist_ok=True)
        if cmd == ["b"]:
            r = B.explore([x for x in recs if x["date"] < VAULT_FROM], explore_days)
            r.update(ran_at=now.strftime("%d %b %Y %H:%M"), stocks=len({x["sid"] for x in recs}),
                     eligible_days=sum(1 for x in recs if x["eligible"] and x["date"] < VAULT_FROM))
            text = fmt_b(r, "INTRADAY LAB - IDEA B: the 9:17 movers, short holds (EXPLORATION, vault untouched)")
            (RESULTS / "idea_b.txt").write_text(text, encoding="utf-8")
            (RESULTS / "idea_b.json").write_text(json.dumps(r, default=str, indent=1), encoding="utf-8")
            print(text)
            return 0
        prev = json.loads((RESULTS / "idea_b.json").read_text(encoding="utf-8")) if (RESULTS / "idea_b.json").exists() else {}
        cands = prev.get("candidates") or []
        if not cands:
            print("No vault candidates from exploration - the vault stays locked (nothing to test).")
            return 1
        if (RESULTS / "idea_b_vault.txt").exists():
            print("The vault was already used for idea B - it is a one-time test. See results/idea_b_vault.txt.")
            return 1
        vault_days = {d for d in days if d >= VAULT_FROM}
        ev = B.evaluate([x for x in recs if x["date"] >= VAULT_FROM], vault_days)
        lines = [f"INTRADAY LAB - IDEA B VAULT TEST ({min(vault_days)} .. {max(vault_days)}, {len(vault_days)} days), run "
                 f"{now:%d %b %Y %H:%M}", "PASS: avg net % > 0 and PF >= 1.1", ""]
        for c in cands:
            ek, mode = c.split("/")
            s_ = ev[(ek, mode)]["stats"]
            ok = (s_["net"] or -9) > 0 and (s_["pf"] or 0) >= 1.1
            lines.append(f"{c:14} n={s_['n']} hit={s_['hit']}% net={s_['net']}% PF={s_['pf']} ₹/trade={s_['rs_per_trade']}"
                         f"  random net={ev[(ek, 'random')]['stats']['net']}%  ->  {'PASS' if ok else 'FAIL'}")
        text = "\n".join(lines)
        (RESULTS / "idea_b_vault.txt").write_text(text, encoding="utf-8")
        print(text)
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
