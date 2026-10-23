import datetime as dt
import random

from signal_desk import stock_research as SR
from signal_desk.clock import IST
from signal_desk.instruments import Stock

TODAY = dt.date(2026, 10, 9)


class FakeMarket:

    def __init__(self, n_stocks=8):
        self.calls = 0
        self.data = {}
        first, last = TODAY - dt.timedelta(days=420), TODAY - dt.timedelta(days=1)
        specs = [(13, 22000.0, 8.0, 0)] + [(1000 + i, 150.0 + 90 * i, 0.9 + 0.25 * i, 1) for i in range(n_stocks)]
        for sid, base, vol, is_stock in specs:
            rnd = random.Random(sid)
            px, rows, d = base, [], first
            while d <= last:
                if d.weekday() < 5:
                    drift = rnd.choice([-1, 0, 0, 1]) * vol * 0.5
                    px += rnd.gauss(0, vol * 2)
                    t = dt.datetime.combine(d, dt.time(9, 15), IST)
                    for k in range(25):
                        o = px
                        c = o + drift + rnd.gauss(0, vol)
                        hi, lo = max(o, c) + abs(rnd.gauss(0, vol / 3)), min(o, c) - abs(rnd.gauss(0, vol / 3))
                        v = (abs(rnd.gauss(0, 1)) + 0.5) * 2e5 * (6 if (is_stock and rnd.random() < 0.04) else 1)
                        rows.append((t, o, hi, lo, c, v if is_stock else 0))
                        px, t = c, t + dt.timedelta(minutes=15)
                d += dt.timedelta(days=1)
            self.data[sid] = rows

    def intraday(self, sid, seg, inst, a, b, interval=15):
        self.calls += 1
        assert (b - a).days <= 90
        rows = [r for r in self.data[int(sid)] if a <= r[0].date() < b]
        return {"open": [r[1] for r in rows], "high": [r[2] for r in rows], "low": [r[3] for r in rows],
                "close": [r[4] for r in rows], "volume": [r[5] for r in rows], "timestamp": [r[0].timestamp() for r in rows]}


def test_charges_are_realistic():
    c = SR.charges(300.0, 303.0, 100, "LONG")
    assert 50 < c < 75
    assert SR.charges(300.0, 297.0, 100, "SHORT") > 0


def test_premarket_uses_only_previous_days():
    days = [SR.DayStat(100 + i, 102 + i, 99 + i, 101.8 + i, 80e7) for i in range(30)]
    a = SR.premarket_score(days[:25])
    days2 = days[:25] + [SR.DayStat(0, 0, 0, 0, 0)]
    assert SR.premarket_score(days2[:25]) == a


def test_run_caps_cache_and_shape(tmp_path, monkeypatch=None):
    SR.CACHE = tmp_path / "stocks15"
    m = FakeMarket()
    stocks = [Stock(f"S{i}", 1000 + i, 1) for i in range(8)]
    res = SR.run(m, stocks, 13, TODAY, days=200, seen=60)
    assert res["status"] == "done" and res["sessions"] == 200 and res["events"] > 0
    keys = [r["key"] for r in res["rows"]]
    assert keys == [v.key for v in SR.SVARIANTS]
    for r in res["rows"]:
        for p in ("seen", "unseen"):
            if r["key"] != "monday" and r[p]["n"]:
                assert r[p]["per_day"] <= 2
    f = next(r for r in res["rows"] if r["key"] == "funnel")
    by_day = {}
    for t in f["trades"]:
        by_day.setdefault(t["date"], []).append(t)
        assert "10:00" <= t["time"] <= "13:00"
    assert all(len(v) <= 2 for v in by_day.values())
    assert all(len({t["time"] for t in v}) == len(v) for v in by_day.values())
    assert all(len({t["symbol"] for t in v}) == len(v) for v in by_day.values())
    mon = next(r for r in res["rows"] if r["key"] == "monday")
    assert (mon["unseen"]["n"] + mon["seen"]["n"]) >= (f["unseen"]["n"] + f["seen"]["n"])
    calls = m.calls
    SR.run(m, stocks, 13, TODAY, days=200, seen=60)
    assert m.calls - calls <= 12
    assert "UNSEEN" in SR.format_table(res)
