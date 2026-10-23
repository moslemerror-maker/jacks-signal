import datetime as dt
import tempfile
from pathlib import Path

from signal_desk.candles import Candle
from signal_desk.clock import IST, at
from signal_desk.signals import Signal

import expiry_0dte_lab.data as D
from expiry_0dte_lab import analysis as A

DAY = dt.date(2026, 10, 13)


def _spot(t):
    m = (t.hour * 60 + t.minute) - (9 * 60 + 15)
    return 22000 + m * 0.5


class FakeApi:
    def __init__(self):
        self.calls = 0

    def post(self, ep, body):
        self.calls += 1
        if not body["expiryCode"]:
            from signal_desk.dhan_api import DhanError
            raise DhanError("/charts/rollingoption: HTTP 400 DH-905 expiryCode is required")
        a, b = dt.date.fromisoformat(body["fromDate"]), dt.date.fromisoformat(body["toDate"])
        off = body["strike"]
        k_off = 0 if off == "ATM" else int(off[3:])
        ts, cl, sk, sp = [], [], [], []
        d = a
        while d < b:
            if d.weekday() < 5:
                for i in range(74):
                    t = dt.datetime.combine(d, dt.time(9, 15), IST) + dt.timedelta(minutes=5 * i)
                    s = _spot(t + dt.timedelta(minutes=5))
                    k = round(s / 50) * 50 + 50 * k_off
                    intr = max(s - k, 0) if body["drvOptionType"] == "CALL" else max(k - s, 0)
                    tv = 40 * (1 - i / 74) if body["expiryCode"] == 1 else 100 * (body["expiryCode"] - 1)
                    ts.append(t.timestamp()); cl.append(intr + tv); sk.append(k); sp.append(s)
            d += dt.timedelta(days=1)
        side = {"timestamp": ts, "close": cl, "strike": sk, "spot": sp}
        return {"data": {"ce": side, "pe": None} if body["drvOptionType"] == "CALL" else {"ce": None, "pe": side}}


def _book(api):
    D.CACHE = Path(tempfile.mkdtemp())
    return A.Book(fetch=lambda c, typ, off, day: D.fetch_day(api, c, typ, off, day, say=lambda m: None))


def test_parse_and_cache_reuse():
    api = FakeApi()
    old = D.CACHE
    D.CACHE = Path(tempfile.mkdtemp())
    try:
        s1 = D.fetch_day(api, 1, "CALL", "ATM", DAY - dt.timedelta(days=1), say=lambda m: None)
        n = api.calls
        s2 = D.fetch_day(api, 1, "CALL", "ATM", DAY - dt.timedelta(days=1), say=lambda m: None)
        assert s1 == s2 and api.calls == n and len(s1) == 74
        assert D.cached_day(1, "CALL", "ATM", DAY - dt.timedelta(days=1)) == s1
        assert D.cached_day(1, "PUT", "ATM", DAY) == {}
    finally:
        D.CACHE = old
    assert D.parse({"data": {"ce": None, "pe": None}}, "CALL") == {} and D.parse(None, "PUT") == {}


def test_detect_codes_and_straddle_keeps_strike():
    api = FakeApi()
    book = _book(api)
    days = [DAY - dt.timedelta(days=7), DAY - dt.timedelta(days=6), DAY - dt.timedelta(days=5), DAY]
    codes = A.detect_codes(book, days, (1, 2))
    assert codes["zero"] == 1 and codes["next"] == 2
    k0 = book.atm_strike(1, at(DAY, "09:20"))
    assert k0 == round(_spot(at(DAY, "09:20")) / 50) * 50
    n0 = api.calls
    pnl = A.straddle(book, 1, DAY)
    assert api.calls - n0 <= 4
    ce0 = book.price(1, "CALL", k0, at(DAY, "09:20"))
    assert book.price(0, "CALL", k0, at(DAY, "09:20")) is None
    assert pnl is not None and ce0 is not None and pnl > 0


def test_exit_times_and_option_trade_half_at_t1():
    book = _book(FakeApi())
    t0 = at(DAY, "10:00")
    e = _spot(t0)
    sig = Signal("x", "NIFTY", "NIFTY", "LONG", t0, e - 5, e, e - 30, e + 15, e + 600)
    after = []
    for i in range(21):
        st = t0 + dt.timedelta(minutes=15 * i)
        s = _spot(st + dt.timedelta(minutes=15))
        after.append(Candle(st, s - 3, s + 1, s - 4, s, 0, 15))
    t1, tx = A.exit_times(sig, after, at(DAY, "15:15"))
    assert t1 is not None and t1 < tx <= at(DAY, "15:15")
    z = A.option_trade(book, 1, sig, t1, tx)
    n = A.option_trade(book, 2, sig, t1, tx)
    assert z is not None and n is not None
    stop_sig = Signal("y", "NIFTY", "NIFTY", "SHORT", t0, e + 5, e, e + 20, e - 30, e - 60)
    t1b, txb = A.exit_times(stop_sig, after, at(DAY, "15:15"))
    assert t1b is None and txb < at(DAY, "11:00")


def test_verdicts():
    good = {"n": 50, "win": 55, "avg": 500, "pf": 1.5, "total": 25000, "max_dd": -5000}
    bad = dict(good, pf=0.6, avg=-400)
    assert A.verdict_q1({"older": good, "newer": good}, 0.01)[0].startswith("PASS (BUYERS")
    assert A.verdict_q1({"older": bad, "newer": bad}, 0.01)[0].startswith("PASS (SELLERS")
    assert A.verdict_q1({"older": good, "newer": bad}, 0.01)[0].startswith("FAIL")
    assert A.verdict_q1({"older": dict(good, n=10), "newer": good}, 0.01)[0].startswith("INCONCLUSIVE")
    st = {"avg": 1, "pf": 1}
    h = {"older": {"pairs": 20, "zero": st, "next": st, "diff_avg": -300},
         "newer": {"pairs": 20, "zero": st, "next": st, "diff_avg": -200}}
    assert A.verdict_q2(h, 0.02)[0].startswith("0DTE WORSE")
    h["newer"]["diff_avg"] = 100
    assert A.verdict_q2(h, 0.02)[0].startswith("NO CLEAR")
    assert A.signflip_p([1, 2, 3]) is None and 0 <= A.signflip_p([5, 6, 7, 8, 9, 10]) <= 0.1


def test_server_hiccups_are_retried_and_never_cached():
    import gzip, json
    from signal_desk.dhan_api import DhanError
    D.CACHE = Path(tempfile.mkdtemp())
    D.time.sleep = lambda s: None
    good = FakeApi()

    class Flaky:
        n = 0

        def post(self, ep, body):
            Flaky.n += 1
            if Flaky.n <= 2:
                raise DhanError("/charts/rollingoption: HTTP 504  <!DOCTYPE HTML>")
            return good.post(ep, body)
    day = DAY - dt.timedelta(days=1)
    assert len(D.fetch_day(Flaky(), 1, "CALL", "ATM", day, say=lambda m: None)) == 74 and Flaky.n == 3
    p = D._day_path(1, "PUT", "ATM", day)
    p.write_bytes(gzip.compress(json.dumps({"error": "/charts/rollingoption: HTTP 504 <!DOCTYPE"}).encode()))
    assert D.cached_day(1, "PUT", "ATM", day) == {}
    assert len(D.fetch_day(good, 1, "PUT", "ATM", day, say=lambda m: None)) == 74
    assert len(D.cached_day(1, "PUT", "ATM", day)) == 74
