import datetime as dt
import pathlib
import random
import tempfile

from stock_lab import data as D
from stock_lab import features as F
from stock_lab import gates as G
from stock_lab.costs import charges

IST = F.IST


class Fake:

    def __init__(self, n=30, days=320, end=dt.date(2026, 10, 16)):
        self.calls, self.data = 0, {}
        d, ds = end, []
        while len(ds) < days:
            if d.weekday() < 5:
                ds.append(d)
            d -= dt.timedelta(days=1)
        ds.reverse()
        for sid in range(1, n + 1):
            rnd = random.Random(sid)
            px, rows = 200.0 + 40 * sid, []
            for day in ds:
                inplay = rnd.random() < 0.08
                drift = rnd.choice([-1, 1]) * (0.0012 if inplay else 0.0)
                px *= 1 + rnd.gauss(0, 0.01) + (drift * 5)
                t = dt.datetime.combine(day, dt.time(9, 15), IST)
                for k in range(75):
                    o = px
                    c = o * (1 + drift + rnd.gauss(0, 0.002))
                    h, l = max(o, c) * (1 + abs(rnd.gauss(0, 0.001))), min(o, c) * (1 - abs(rnd.gauss(0, 0.001)))
                    v = 4e5 * (1 + abs(rnd.gauss(0, .5))) * (8 if (inplay and k == 0) else 1)
                    rows.append((int(t.timestamp()), o, h, l, c, v))
                    px, t = c, t + dt.timedelta(minutes=5)
            self.data[sid] = rows

    def intraday(self, sid, seg, inst, a, b, interval=5):
        self.calls += 1
        assert interval == 5 and (b - a).days <= 90
        lo = dt.datetime.combine(a, dt.time(0), IST).timestamp()
        hi = dt.datetime.combine(b, dt.time(0), IST).timestamp()
        rows = [r for r in self.data[int(sid)] if lo <= r[0] < hi]
        return {"timestamp": [r[0] for r in rows], "open": [r[1] for r in rows], "high": [r[2] for r in rows],
                "low": [r[3] for r in rows], "close": [r[4] for r in rows], "volume": [r[5] for r in rows]}


def _bars(spec):
    out = []
    for hhmm, o, h, l, c in spec:
        t = dt.time(*map(int, hhmm.split(":")))
        out.append((0, o, h, l, c, 1, t))
    return out


def test_simulate_long_target_stop_and_ties():
    after = _bars([("09:20", 100, 100.5, 99.8, 100.4), ("09:25", 100.4, 101.0, 100.3, 100.9), ("09:30", 100.9, 102.2, 100.8, 102.1),
                   ("14:55", 102, 102.1, 101.9, 102.0), ("15:00", 102, 103, 101, 102.5)])
    tr = F._simulate("LONG", 100.5, 99.5, after)
    assert tr.filled and tr.entry == 100.5 and abs(tr.risk_pct - 0.995) < 0.01
    assert tr.hold_out == "15:00" and abs(tr.hold_pct - (102.0 - 100.5) / 100.5 * 100) < 1e-6
    assert tr.r2_out == "15:00"
    tie = _bars([("09:20", 100, 100.6, 99.4, 100)])
    assert F._simulate("LONG", 100.5, 99.5, tie).hold_out == "SL"
    late = _bars([("10:30", 100, 101, 100, 101)])
    assert not F._simulate("LONG", 100.5, 99.5, late).filled
    wide = _bars([("09:20", 100, 101, 99, 100)])
    assert not F._simulate("LONG", 100.5, 97.0, wide).filled
    tiny = F._simulate("LONG", 100.5, 100.3, _bars([("09:20", 100.4, 100.6, 100.4, 100.5)]))
    assert tiny.filled and abs(tiny.risk_pct - 0.5) < 1e-6


def test_costs_rupee_lakh():
    c = charges(500, 505, 200, "LONG")
    assert 150 < c < 200


def test_cache_download_and_pipeline():
    D.M5 = pathlib.Path(tempfile.mkdtemp()) / "m5"
    f = Fake()
    start, end = dt.date(2025, 7, 1), dt.date(2026, 10, 16)
    rows = D.load_m5(f, 1, start, end)
    calls = f.calls
    assert len(rows) > 300 * 70 and calls >= 5
    rows2 = D.load_m5(f, 1, start, end, refresh=False)
    assert f.calls == calls and len(rows2) == len(rows)
    days = []
    for sid in range(1, 31):
        days.extend(F.day_records(f"S{sid}", D.load_m5(f, sid, start, end)))
    assert days and all(d.date.weekday() < 5 for d in days)
    first_dates = sorted({d.date for d in days})
    all_dates = sorted({dt.datetime.fromtimestamp(r[0], IST).date() for r in rows})
    assert first_dates[0] > all_dates[14]
    r0 = G.gate0(days)
    assert [r["key"] for r in r0["rows"]] == [k for k, *_ in G.G0_RULES] and r0["rv_buckets"]
    rv = next(r for r in r0["rows"] if r["key"] == "rv5")
    rnd = next(r for r in r0["rows"] if r["key"] == "random")
    assert rv["older"]["follow_avg"] > rnd["older"]["follow_avg"]
    r1 = G.gate1(days)
    assert [r["key"] for r in r1["rows"]] == [v.key for v in G.G1_VARIANTS]
    main = r1["rows"][0]
    assert main["older"]["n"] > 0 and main["older"]["per_day"] <= 3
    assert "GATE 0" in G.fmt0(r0) and "GATE 1" in G.fmt1(r1)
