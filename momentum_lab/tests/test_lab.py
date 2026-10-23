import datetime as dt

from signal_desk.candles import Candle
from signal_desk.clock import IST

from momentum_lab import analysis as A
from momentum_lab.__main__ import build, fmt
from range_lab.tests.test_lab import _series


def _day(d, closes):
    out, px = [], 100.0
    for k in range(25):
        t = dt.datetime.combine(d, dt.time(9, 15), IST) + dt.timedelta(minutes=15 * k)
        px = closes.get(t.time(), px)
        out.append(Candle(t, px, px + 1, px - 1, px, 0, 15))
    return out


def test_day_fields_use_candle_end_times():
    d0, d1 = dt.date(2026, 10, 19), dt.date(2026, 10, 20)
    cs = _day(d0, {}) + _day(d1, {dt.time(9, 30): 101, dt.time(14, 15): 102, dt.time(14, 45): 103, dt.time(15, 15): 105})
    days = A.build_days(cs)
    assert len(days) == 1
    x = days[0]
    assert (x.prev_close, x.c945, x.c1430, x.c1500, x.c1530) == (100, 101, 102, 103, 105)
    assert x.r1 > 0 and x.rpen > 0 and x.move == 2
    assert A.primary_trades(days) == [(x, 1)]


def test_disagreeing_signs_mean_no_trade():
    d0, d1 = dt.date(2026, 10, 19), dt.date(2026, 10, 20)
    cs = _day(d0, {}) + _day(d1, {dt.time(9, 30): 101, dt.time(14, 15): 102, dt.time(14, 45): 101.5})
    assert A.primary_trades(A.build_days(cs)) == []


def test_rupees_and_verdict():
    assert A.rupees(10) == 10 * 0.5 * 65 - 60
    good = {"n": 60, "hit": 56.0, "avg_pts": 6, "avg_rs": 135, "pf": 1.4}
    r = {"halves": {"older": {"primary": good}, "newer": {"primary": good}}, "primary_flip_p_pooled": 0.03}
    assert A.verdict(r)[0].startswith("PASS")
    r["primary_flip_p_pooled"] = 0.4
    assert A.verdict(r)[0].startswith("PROMISING")
    r["halves"]["newer"]["primary"] = dict(good, avg_rs=-5)
    assert A.verdict(r)[0].startswith("FAIL")


def test_end_to_end_report():
    n = _series(200, seed=4)
    vix = [Candle(c.start, 14, 14.5, 13.5, 14 + (c.start.day % 3), 0, 15) for c in n]
    r = build({"nifty": n, "vix": vix}, n[-1].start.date() + dt.timedelta(days=1), 1)
    t = fmt(r)
    assert "PASS MARK" in t and "VERDICT" in t and r["sessions"] > 100
