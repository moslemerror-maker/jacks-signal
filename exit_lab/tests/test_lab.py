import contextlib
import datetime as dt
import io
import tempfile
from pathlib import Path

from signal_desk.candles import Candle
from signal_desk.clock import IST, at
from signal_desk.signals import Signal

import expiry_0dte_lab.data as D
from exit_lab import analysis as A

DAY = dt.date(2026, 10, 13)


class PathBook:
    def __init__(self, t0, path):
        self.t0, self.path = t0, sorted(path.items())

    def price(self, code, typ, k, when):
        m = (when - self.t0).total_seconds() / 60
        v = None
        for mm, p in self.path:
            if mm <= m:
                v = p
        return v


def _sig(r=40):
    t0 = at(DAY, "10:00")
    return Signal("x", "NIFTY", "NIFTY", "LONG", t0, 22000, 22010, 22010 - r, 22010 + 1.5 * r, 22010 + 2.5 * r), t0


def test_premium_stop_hit_first():
    sig, t0 = _sig(40)
    v = A.premium_exit(PathBook(t0, {0: 200, 15: 185, 30: 179, 45: 260}), 1, sig)
    assert v == (179 - 200) * 65 - 60


def test_premium_half_at_t1_then_back_to_entry():
    sig, t0 = _sig(40)
    v = A.premium_exit(PathBook(t0, {0: 200, 20: 232, 40: 199}), 1, sig)
    assert v == (0.5 * 32 + 0.5 * (-1)) * 65 - 60


def test_premium_t2_and_squareoff():
    sig, t0 = _sig(40)
    assert A.premium_exit(PathBook(t0, {0: 200, 10: 255}), 1, sig) == 55 * 65 - 60
    v = A.premium_exit(PathBook(t0, {0: 200, 10: 210}), 1, sig)
    assert v == 10 * 65 - 60


def test_verdict():
    def h(pi, pp, pf, n=100):
        st = lambda a, f: {"n": n, "win": 50, "avg": a, "pf": f, "total": a * n, "max_dd": 0}
        return {"signals": 100, "paired": n, "index": st(pi, 1.0), "premium": st(pp, pf), "diff_avg": pp - pi}
    s = {"halves": {"older": h(10, 120, 1.2), "newer": h(50, 150, 1.3)}, "p": 0.03}
    assert A.verdict(s)[0].startswith("PASS")
    s["p"] = 0.4
    assert A.verdict(s)[0].startswith("FAIL")
    s = {"halves": {"older": h(10, 120, 1.2), "newer": h(50, 40, 1.3)}, "p": 0.03}
    assert A.verdict(s)[0].startswith("FAIL")
    s = {"halves": {"older": h(10, 120, 1.2, n=50), "newer": h(50, 150, 1.3)}, "p": 0.03}
    assert A.verdict(s)[0].startswith("INCOMPLETE")


def test_end_to_end_with_fake_dhan():
    import exit_lab.__main__ as M
    import range_lab.data as RD
    import signal_desk.dhan_api as DA
    from expiry_0dte_lab.tests.test_lab import FakeApi
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
        txt = out.getvalue()
        rounds = [l for l in txt.splitlines() if l.startswith(".. round")]
        assert "VERDICT" in txt and "PREMIUM" in txt and len(rounds) <= 6, rounds
        calls = fake.calls
        with contextlib.redirect_stdout(io.StringIO()):
            assert M.main(["report", "--years", "1"]) == 0
        assert fake.calls == calls
    finally:
        RD.update, RD.load, DA.DhanApi, Settings.load = old
