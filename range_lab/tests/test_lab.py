import datetime as dt
import random
import tempfile
from pathlib import Path

from signal_desk.candles import Candle
from signal_desk.clock import IST

import range_lab.data as D
from range_lab import analysis as A
from range_lab.__main__ import build, fmt


def _series(n_days=90, seed=3, base=22000.0, start=dt.date(2026, 3, 2)):
    rnd = random.Random(seed)
    out, d, px = [], start, base
    while len(out) < n_days * 25:
        if d.weekday() < 5:
            vol = rnd.choice((6, 10, 18))
            drift = rnd.choice((-1, 0, 1)) * vol * 0.4
            for k in range(25):
                t = dt.datetime.combine(d, dt.time(9, 15), IST) + dt.timedelta(minutes=15 * k)
                o = px
                c = o + drift + rnd.gauss(0, vol)
                out.append(Candle(t, o, max(o, c) + abs(rnd.gauss(0, vol / 2)), min(o, c) - abs(rnd.gauss(0, vol / 2)), c, 0, 15))
                px = c
        d += dt.timedelta(days=1)
    return out


def test_ratio_and_flags():
    days = A.days_from(_series(30))
    x = days[A.LOOK + 1]
    avg = sum(b.rng for b in days[0:A.LOOK]) / A.LOOK
    assert abs(x.ratio - days[A.LOOK].rng / avg) < 1e-9
    assert all(d.ratio is None for d in days[:A.LOOK + 1])


def test_cutoffs_come_from_older_half_only():
    days = A.days_from(_series(60))
    have = [d for d in days if d.ratio is not None]
    older = {d.date for d in have[: len(have) // 2]}
    q1, q2 = A.assign(days, older)
    assert (q1, q2) == A.terciles([d.ratio for d in have if d.date in older])
    assert {d.bucket for d in have} <= set(A.BUCKETS)


def test_build_and_report_end_to_end():
    n = _series(140, seed=5)
    data = {"nifty": n, "banknifty": [Candle(c.start, c.open * 2, c.high * 2, c.low * 2, c.close * 2, 0, 15) for c in n],
            "vix": [Candle(c.start, 14, 14.2, 13.9, 14.1, 0, 15) for c in n]}
    r = build(data, n[-1].start.date() + dt.timedelta(days=1), 1)
    assert r["sessions"] > 50 and r["gate1"]["verdict"].split(" ")[0] in ("PASS", "PROMISING", "FAIL")
    txt = fmt(r)
    assert "PASS MARK" in txt and "GATE 0" in txt and "VERDICT" in txt


def test_verdict_rules():
    def h(nn, na, npf, wa):
        return {"narrow": {"n": nn, "avg_r": na, "pf": npf}, "wide": {"n": 20, "avg_r": wa, "pf": 0.9}}
    g = {"halves": {"older": h(20, 0.3, 1.5, 0.0), "newer": h(18, 0.25, 1.4, 0.05)}, "shuffle_p_pooled": 0.04}
    assert A.verdict(g)[0].startswith("PASS")
    g["shuffle_p_pooled"] = 0.3
    assert A.verdict(g)[0].startswith("PROMISING")
    g["halves"]["newer"] = h(18, 0.0, 0.9, 0.1)
    assert A.verdict(g)[0].startswith("FAIL")


def test_cache_fetches_only_missing():
    calls = []

    class Api:
        def intraday(self, sid, seg, ins, a, b, tf):
            calls.append((sid, a))
            cs = [c for c in _series(20) if a <= c.start.date() < b]
            return {"open": [c.open for c in cs], "high": [c.high for c in cs], "low": [c.low for c in cs],
                    "close": [c.close for c in cs], "volume": [0] * len(cs), "timestamp": [c.start.timestamp() for c in cs]}
    old = D.CACHE
    D.CACHE = Path(tempfile.mkdtemp())
    try:
        a, b = dt.date(2026, 3, 2), dt.date(2026, 3, 20)
        out = D.update(Api(), a, b, say=lambda m: None)
        first = len(calls)
        assert first > 0 and len(out["nifty"]) == len(D.load("nifty")) > 0
        D.update(Api(), a, b, say=lambda m: None)
        assert len(calls) - first <= 3
    finally:
        D.CACHE = old
