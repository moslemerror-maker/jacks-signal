import datetime as dt

from signal_desk.candles import Candle

from expiry_lab import analysis as A
from expiry_lab.__main__ import build, fmt
from range_lab.tests.test_lab import _series


def _weekdays(a, b, skip=()):
    out, d = [], a
    while d <= b:
        if d.weekday() < 5 and d not in skip:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def test_expiry_thursday_then_tuesday_and_holiday_shift():
    days = _weekdays(dt.date(2025, 8, 18), dt.date(2025, 9, 12), skip={dt.date(2025, 8, 27)})
    e = A.expiry_days(days)
    assert dt.date(2025, 8, 21) in e and dt.date(2025, 8, 28) in e
    assert dt.date(2025, 9, 2) in e and dt.date(2025, 9, 9) in e
    assert dt.date(2025, 9, 4) not in e
    days2 = _weekdays(dt.date(2024, 10, 14), dt.date(2024, 10, 25), skip={dt.date(2024, 10, 24)})
    assert dt.date(2024, 10, 23) in A.expiry_days(days2)
    days3 = _weekdays(dt.date(2026, 3, 30), dt.date(2026, 4, 3), skip={dt.date(2026, 3, 31)})
    assert dt.date(2026, 3, 30) in A.expiry_days(days3)


def test_groups_day_before():
    days = _weekdays(dt.date(2026, 10, 12), dt.date(2026, 10, 23))
    g = A.groups(days)
    assert g[dt.date(2026, 10, 13)] == "expiry" and g[dt.date(2026, 10, 12)] == "day_before"
    assert g[dt.date(2026, 10, 15)] == "other" and g[dt.date(2026, 10, 19)] == "day_before"


def test_verdict_rules():
    def h(en, ea, oa):
        return {"expiry": {"n": en, "avg_r": ea, "pf": 1}, "other": {"n": 100, "avg_r": oa, "pf": 1}, "gap": round(ea - oa, 3)}
    r = {"halves": {"older": h(30, 0.5, 0.1), "newer": h(25, 0.4, 0.1)}, "shuffle_p_pooled": 0.02}
    assert A.verdict(r)[0].startswith("PASS - expiry-day signals are consistently BETTER")
    r["halves"]["newer"] = h(25, -0.2, 0.1)
    assert A.verdict(r)[0].startswith("FAIL")
    r = {"halves": {"older": h(30, -0.3, 0.1), "newer": h(25, 0.0, 0.1)}, "shuffle_p_pooled": 0.02}
    assert A.verdict(r)[0].startswith("PROMISING") and "WORSE" in A.verdict(r)[0]


def test_end_to_end():
    n = _series(150, seed=6)
    data = {"nifty": n, "banknifty": [Candle(c.start, c.open * 2, c.high * 2, c.low * 2, c.close * 2, 0, 15) for c in n],
            "vix": [Candle(c.start, 14, 14.2, 13.9, 14.1, 0, 15) for c in n]}
    r = build(data, n[-1].start.date() + dt.timedelta(days=1), 1)
    assert "VERDICT" in fmt(r) and r["expiry_days_tested"] > 10
