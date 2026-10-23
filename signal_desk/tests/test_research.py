import datetime as dt
import random

from signal_desk import backtest as BT
from signal_desk import research as RS
from signal_desk.clock import IST

NIFTY, BNF, VIX = 13, 25, 21


class HistDhan:

    def __init__(self, first: dt.date, last: dt.date):
        self.calls, self.ranges = 0, []
        self.data = {}
        for sid, base, vol in ((NIFTY, 22000.0, 9.0), (BNF, 50000.0, 22.0), (VIX, 14.0, 0.05)):
            rnd = random.Random(sid)
            px, rows, d = base, [], first
            while d <= last:
                if d.weekday() < 5:
                    drift = rnd.choice([-1, 0, 0, 1]) * vol * 0.35
                    t = dt.datetime.combine(d, dt.time(9, 15), IST)
                    px += rnd.gauss(0, vol * 4)
                    for _ in range(75):
                        o = px
                        c = o + drift / 3 + rnd.gauss(0, vol / 1.7)
                        hi, lo = max(o, c) + abs(rnd.gauss(0, vol / 3)), min(o, c) - abs(rnd.gauss(0, vol / 3))
                        rows.append((t, o, hi, lo, c))
                        px, t = c, t + dt.timedelta(minutes=5)
                d += dt.timedelta(days=1)
            self.data[sid] = rows

    def intraday(self, sid, seg, inst, a, b, interval=15):
        self.calls += 1
        self.ranges.append((a, b))
        assert (b - a).days <= 90, "Dhan allows ~90 days per call"
        rows = []
        for r in self.data[int(sid)]:
            if not (a <= r[0].date() < b):
                continue
            m = (r[0].hour * 60 + r[0].minute - 555) // interval * interval + 555
            slot = r[0].replace(hour=m // 60, minute=m % 60)
            if rows and rows[-1][0] == slot:
                x = rows[-1]
                rows[-1] = (slot, x[1], max(x[2], r[2]), min(x[3], r[3]), r[4])
            else:
                rows.append((slot, r[1], r[2], r[3], r[4]))
        return {"open": [r[1] for r in rows], "high": [r[2] for r in rows], "low": [r[3] for r in rows],
                "close": [r[4] for r in rows], "volume": [0] * len(rows), "timestamp": [r[0].timestamp() for r in rows]}


TODAY = dt.date(2026, 10, 9)


def _api():
    return HistDhan(TODAY - dt.timedelta(days=420), TODAY - dt.timedelta(days=1))


def test_fetch_range_chunks_and_dedupes():
    api = _api()
    got = RS.fetch_range(api, NIFTY, TODAY - dt.timedelta(days=300), TODAY - dt.timedelta(days=1))
    assert api.calls >= 4
    starts = [c.start for c in got]
    assert starts == sorted(set(starts))
    days = {c.start.date() for c in got}
    assert min(days) >= TODAY - dt.timedelta(days=300) and max(days) <= TODAY - dt.timedelta(days=1)


def test_live_variant_matches_backtest():
    api = _api()
    bt = BT.run(api, NIFTY, BNF, VIX, TODAY, days=40, budget=1e9)
    n = RS.fetch_range(api, NIFTY, TODAY - dt.timedelta(days=80), TODAY - dt.timedelta(days=1))
    b = RS.fetch_range(api, BNF, TODAY - dt.timedelta(days=80), TODAY - dt.timedelta(days=1))
    v = RS.fetch_range(api, VIX, TODAY - dt.timedelta(days=80), TODAY - dt.timedelta(days=1))
    sessions = sorted({c.start.date() for c in n})[-40:]
    rs = RS.replay({x.key: x for x in RS.VARIANTS}["old1430"], n, b, v, sessions)
    mine = [(x.signal.id, x.signal.grade, x.outcome, x.rupees) for x in rs]
    theirs = [(f"NIFTY-{t['date'].replace('-', '')}-{t['time'].replace(':', '')}-{t['side']}", t["grade"], t["outcome"], t["rupees"])
              for t in bt["trades"]]
    assert mine == theirs and len(mine) > 5


def test_variants_change_behaviour():
    api = _api()
    end = TODAY - dt.timedelta(days=1)
    n = RS.fetch_range(api, NIFTY, TODAY - dt.timedelta(days=120), end)
    b = RS.fetch_range(api, BNF, TODAY - dt.timedelta(days=120), end)
    v = RS.fetch_range(api, VIX, TODAY - dt.timedelta(days=120), end)
    sessions = sorted({c.start.date() for c in n})
    V = {x.key: x for x in RS.VARIANTS}
    live = RS.replay(V["live"], n, b, v, sessions)
    first = RS.replay(V["first"], n, b, v, sessions)
    live = RS.replay(V["old1430"], n, b, v, sessions)
    per_day = {}
    for x in first:
        per_day[x.signal.time.date()] = per_day.get(x.signal.time.date(), 0) + 1
    assert per_day and max(per_day.values()) == 1
    assert len(first) < len(live)
    late = RS.replay(V["late"], n, b, v, sessions)
    assert all(x.signal.action == "SKIP" for x in late if x.signal.time.time() < dt.time(10, 0))
    cut = RS.replay(V["live"], n, b, v, sessions)
    assert all(x.signal.action == "SKIP" for x in cut if x.signal.time.time() > dt.time(13, 0))
    t1 = RS.replay(V["t1_1r"], n, b, v, sessions)
    assert all(abs(abs(x.signal.t1 - x.signal.entry) - x.signal.risk_pts) < 0.2 for x in t1)
    m30 = RS.replay(V["min30"], n, b, v, sessions)
    assert all(x.signal.risk_pts >= 29.9 for x in m30)
    nof = RS.replay(V["nofilt"], n, b, v, sessions)
    assert all(not any(c.name in ("Bank Nifty", "VIX") for c in x.signal.checks) for x in nof)
    tr = RS.replay(V["live"], n, b, v, sessions)
    assert all(any(c.name == "Trend" for c in x.signal.checks) for x in tr if x.signal.time.time() <= dt.time(13, 0))
    assert any(c.name == "Trend" and c.status == "fail" for x in tr for c in x.signal.checks)
    atr_v = RS.replay(V["atr"], n, b, v, sessions)
    assert all(20 - 0.1 <= x.signal.risk_pts <= 80.1 for x in atr_v)
    one = RS.replay(V["one"], n, b, v, sessions)
    taken = {}
    for x in one:
        if x.signal.action == "TAKE":
            taken[x.signal.time.date()] = taken.get(x.signal.time.date(), 0) + 1
    assert not taken or max(taken.values()) == 1


def test_run_report_shape_and_split():
    api = _api()
    res = RS.run(api, NIFTY, BNF, VIX, TODAY, days=200, seen=60)
    assert res["status"] == "done" and res["seen"]["n"] == 60 and res["unseen"]["n"] == res["sessions"] - 60
    assert res["unseen"]["to"] < res["seen"]["from"]
    assert [r["key"] for r in res["rows"]] == [v.key for v in RS.VARIANTS]
    assert res["rows"][0]["verdict"] == "baseline"
    tf = {r["key"]: r for r in res["rows"]}
    for k in ("tf25", "tf60"):
        assert tf[k]["tf"] in (25, 60) and tf[k]["unseen"]["take"]["n"] > 0, k
        assert tf[k]["unseen"]["take"]["per_day"] <= 2
    for r in res["rows"]:
        for p in ("seen", "unseen"):
            assert set(r[p]) == {"take", "a_only", "calls", "puts"}
            assert r[p]["calls"]["n"] + r[p]["puts"]["n"] == r[p]["take"]["n"]
            assert abs(r[p]["calls"]["rupees"] + r[p]["puts"]["rupees"] - r[p]["take"]["rupees"]) <= 2
    txt = RS.format_table(res)
    assert "UNSEEN" in txt and "Live rules" in txt
