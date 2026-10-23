import datetime as dt
import io
import contextlib
import tempfile
from pathlib import Path

from signal_desk.candles import Candle
from signal_desk.clock import IST, at
from signal_desk.signals import Signal

import expiry_0dte_lab.data as D
from expiry_0dte_lab.analysis import Book
from expiry_0dte_lab.tests.test_lab import DAY, FakeApi, _spot
from premium_lab import analysis as A


def test_strike_rule_matches_signal_desk():
    assert A.strike_for("LONG", 22849) == 22800 and A.strike_for("LONG", 22850) == 22850
    assert A.strike_for("SHORT", 22801) == 22850 and A.strike_for("SHORT", 22800) == 22800


def test_real_pnl_uses_the_alerted_strike_and_half_at_t1():
    D.CACHE = Path(tempfile.mkdtemp())
    api = FakeApi()
    book = Book(fetch=lambda c, typ, off, day: D.fetch_day(api, c, typ, off, day, say=lambda m: None))
    t0 = at(DAY, "10:00")
    e = _spot(t0)
    sig = Signal("x", "NIFTY", "NIFTY", "LONG", t0, e - 5, e, e - 30, e + 15, e + 600)
    v_full = A.real_pnl(book, 2, sig, None, at(DAY, "12:00"))
    v_half = A.real_pnl(book, 2, sig, at(DAY, "11:00"), at(DAY, "12:00"))
    k = A.strike_for("LONG", e)
    p0, p1, p2 = (book.price(2, "CALL", k, t) for t in (t0, at(DAY, "11:00"), at(DAY, "12:00")))
    assert abs(v_full - ((p2 - p0) * 65 - 60)) < 1e-6
    assert abs(v_half - ((0.5 * (p1 - p0) + 0.5 * (p2 - p0)) * 65 - 60)) < 1e-6


def test_verdicts():
    def h(avg, pf, priced=50):
        st = {"n": priced, "win": 50, "avg": avg, "pf": pf, "total": avg * priced, "max_dd": 0}
        return {"signals": 50, "priced": priced, "real": st, "estimate": st}
    assert A.verdict({"halves": {"older": h(100, 1.3), "newer": h(50, 1.12)}})[0].startswith("SURVIVES")
    assert A.verdict({"halves": {"older": h(100, 1.3), "newer": h(-50, 0.9)}})[0].startswith("MIXED")
    assert A.verdict({"halves": {"older": h(-1, 0.99), "newer": h(-50, 0.9)}})[0].startswith("FAILS")
    assert A.verdict({"halves": {"older": h(100, 1.3, 30), "newer": h(50, 1.2)}})[0].startswith("INCOMPLETE")


def test_end_to_end_with_fake_dhan():
    import premium_lab.__main__ as M
    import range_lab.data as RD
    import signal_desk.dhan_api as DA
    from range_lab.tests.test_lab import _series
    from signal_desk.config import Settings
    D.CACHE = Path(tempfile.mkdtemp())
    M.RESULTS = Path(tempfile.mkdtemp())
    n = _series(130, seed=2)
    syn = {"nifty": n, "banknifty": [Candle(c.start, c.open * 2, c.high * 2, c.low * 2, c.close * 2, 0, 15) for c in n],
           "vix": [Candle(c.start, 14, 14.2, 13.9, 14.1, 0, 15) for c in n]}
    old = (RD.update, RD.load, DA.DhanApi, Settings.load)
    fake = FakeApi()
    RD.update, RD.load = (lambda api, a, b: syn), (lambda k: syn[k])
    DA.DhanApi = lambda *a, **k: fake
    Settings.load = classmethod(lambda cls: cls(dhan_client_id="1", dhan_access_token="x"))
    try:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            assert M.main(["--years", "1", "--force"]) == 0
        calls = fake.calls
        txt = out.getvalue()
        assert "VERDICT" in txt and "REAL" in txt and calls > 0
        with contextlib.redirect_stdout(io.StringIO()):
            assert M.main(["report", "--years", "1"]) == 0
        assert fake.calls == calls
    finally:
        RD.update, RD.load, DA.DhanApi, Settings.load = old
