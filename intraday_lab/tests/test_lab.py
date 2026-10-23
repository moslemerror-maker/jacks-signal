import datetime as dt
import random
import tempfile
from pathlib import Path

import intraday_lab.data as D
from intraday_lab import idea_b as B

IST = D.IST


def _ts(d, h, m):
    return int(dt.datetime.combine(d, dt.time(h, m), IST).timestamp())


def _day_bars(d, start_px, drift_per_min, vol=1000, seed=0):
    rnd = random.Random(seed)
    out, px = [], start_px
    for k in range(375):
        t = dt.datetime.combine(d, dt.time(9, 15), IST) + dt.timedelta(minutes=k)
        o = px
        c = o * (1 + drift_per_min + rnd.gauss(0, 0.0003))
        out.append((int(t.timestamp()), o, max(o, c) * 1.0002, min(o, c) * 0.9998, c, vol))
        px = c
    return out


def test_walk_target_stop_and_time():
    d = dt.date(2026, 3, 2)
    bars = [(_ts(d, 9, 17 + i), 100, 100.2, 99.9, 100.1, 1) for i in range(10)]
    assert B._walk(1, 100, bars, "T03") == 100.1
    bars[3] = (_ts(d, 9, 20), 100, 100.4, 99.9, 100.3, 1)
    assert B._walk(1, 100, bars, "T03") == 100 * 1.003
    bars[2] = (_ts(d, 9, 19), 100, 100.5, 99.6, 100, 1)
    assert B._walk(1, 100, bars, "T03") == 100 * 0.997
    assert B._walk(-1, 100, bars, "T03") == 100 * 1.003
    bars30 = [(_ts(d, 9, 17 + i), 100, 101, 99, 100 + i, 1) for i in range(15)]
    assert B._walk(1, 100, bars30, "X930") == 100 + 12


def test_costs_make_net_below_gross():
    g, n = B.gross_pct(1, 500, 502.5), B.net_pct(1, 500, 502.5)
    assert g == 0.5 and 0.3 < n < 0.45
    assert B.net_pct(-1, 500, 497.5) < 0.5


def test_prev_close_is_late_vwap_and_selection():
    days = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(60)]
    days = [d for d in days if d.weekday() < 5]
    bars = {}
    for i, d in enumerate(days):
        for b in _day_bars(d, 1000, 0.00002, seed=i):
            bars[b[0]] = b
    turn = {d: 80e7 for d in days}
    recs = B.stock_days(1, bars, turn)
    assert recs and recs[0]["eligible"] is False
    assert any(r["eligible"] for r in recs)
    r = recs[-1]
    assert set(k[1] for k in r["res"]) == set(B.EXITS) and abs(r["chg"]) < 0.05


def test_explore_finds_planted_follow_edge_and_not_random():
    rnd = random.Random(1)
    recs = []
    days = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(300)]
    days = [d for d in days if d.weekday() < 5]
    for d in days:
        for sid in range(20):
            chg = rnd.gauss(0, 0.01)
            sgn = 1 if chg > 0 else -1
            res = {}
            for side in (1, -1):
                for ek in B.EXITS:
                    mv = (0.6 if abs(chg) > 0.012 else 0.0) * sgn * side + rnd.gauss(0, 0.3)
                    res[(side, ek)] = (mv, mv - 0.12)
            recs.append({"date": d, "sid": sid, "eligible": True, "chg": chg, "res": res})
    r = B.explore(recs, days)
    verdict = {(x["exit"], x["mode"]): x["verdict"] for x in r["rows"]}
    assert verdict[("T05", "follow")] == "VAULT CANDIDATE" and verdict[("T05", "fade")] == "fail"


def test_download_trims_and_caches():
    class Api:
        calls = 0

        def intraday(self, sid, seg, ins, a, b, iv):
            Api.calls += 1
            assert iv == 1
            rows = []
            d = a
            while d < b:
                if d.weekday() < 5:
                    rows += _day_bars(d, 500, 0, vol=10)
                d += dt.timedelta(days=1)
            return {k: [r[i] for r in rows] for i, k in enumerate(("timestamp", "open", "high", "low", "close", "volume"))}
    D.M1 = Path(tempfile.mkdtemp())
    a, b = dt.date(2026, 3, 2), dt.date(2026, 3, 13)
    D.load_m1(Api(), 7, a, b)
    bars, turn = D.read_bars(7), D.read_turnover(7)
    times = {dt.datetime.fromtimestamp(k, IST).time() for k in bars}
    assert dt.time(9, 15) in times and dt.time(11, 29) in times and dt.time(15, 29) in times
    assert dt.time(12, 0) not in times and len(turn) == 10 and all(v > 0 for v in turn.values())
    n = Api.calls
    D.load_m1(Api(), 7, a, b)
    assert Api.calls == n


def test_reads_files_with_windows_line_endings():
    import gzip
    D.M1 = Path(tempfile.mkdtemp())
    (D.M1 / "5.csv.gz").write_bytes(gzip.compress(b"1729655100,852.5,861.05,852.5,859.3,600176\r\r\n1729655160,859.15,861.5,858,861.1,647484\r\r\n"))
    assert len(D.read_bars(5)) == 2


def test_news_window_and_parsing():
    from intraday_lab import idea_a as A, news as N
    IST_ = D.IST
    ev = [("ABC", dt.datetime(2026, 3, 2, 18, 5, tzinfo=IST_), True),
          ("ABC", dt.datetime(2026, 3, 2, 11, 0, tzinfo=IST_), True),
          ("XYZ", dt.datetime(2026, 3, 3, 9, 20, tzinfo=IST_), True)]
    idx = A.news_index(ev, True)
    assert A.had_news(idx, "ABC", dt.date(2026, 3, 2), dt.date(2026, 3, 3))
    assert not A.had_news(idx, "XYZ", dt.date(2026, 3, 2), dt.date(2026, 3, 3))
    assert not A.had_news(A.news_index(ev[1:2], True), "ABC", dt.date(2026, 3, 2), dt.date(2026, 3, 3))
    assert N._when({"an_dt": "02-Mar-2026 18:05:44"}) == dt.datetime(2026, 3, 2, 18, 5, 44, tzinfo=IST_)
    assert N._when({"sort_date": "2026-03-02 18:05:44"}).hour == 18
    assert N.routine("Trading Window-XBRL") and N.routine("Copy of Newspaper Publication")
    assert not N.routine("Financial Result Updates") and not N.routine("Outcome of Board Meeting")


def test_idea_a_finds_planted_news_drift_and_nonews_reversal():
    from intraday_lab import idea_a as A
    rnd = random.Random(3)
    recs, events = [], []
    days = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(320)]
    days = [d for d in days if d.weekday() < 5]
    for i, d in enumerate(days):
        for sid in range(30):
            chg = rnd.gauss(0, 0.012)
            g = 1 if chg > 0 else -1
            news = sid < 4
            if news and i:
                events.append((f"S{sid}", dt.datetime.combine(days[i - 1], dt.time(19, 0), D.IST), True))
            res = {}
            for side in (1, -1):
                for ek in B.EXITS:
                    edge = (0.8 if news else -0.5) * g * side if abs(chg) > 0.008 else 0
                    mv = edge + rnd.gauss(0, 0.3)
                    res[(side, ek)] = (mv, mv - 0.15)
            recs.append({"date": d, "sid": sid, "symbol": f"S{sid}", "eligible": True, "chg": chg, "res": res})
    r = A.explore(recs, events, days[1:])
    v = {(x["exit"], x["mode"]): x["verdict"] for x in r["rows"]}
    assert v[("T05", "news-follow")] == "VAULT CANDIDATE" and v[("T05", "nonews-fade")] == "VAULT CANDIDATE"
    assert v[("T05", "news-fade")] == "info"


def test_news_download_caches_past_weeks():
    from intraday_lab import news as N
    calls = []
    N.NEWS = Path(tempfile.mkdtemp())
    N._session = lambda: object()
    N.fetch = lambda s, a, b: calls.append(a) or [{"symbol": "ABC", "an_dt": f"{a:%d-%b-%Y} 18:00:00", "desc": "Press Release"}]
    N.time.sleep = lambda s: None
    N.download(dt.date(2026, 1, 5), dt.date(2026, 1, 25), say=lambda m: None)
    first = len(calls)
    N.download(dt.date(2026, 1, 5), dt.date(2026, 1, 25), say=lambda m: None)
    assert first == 3 and len(calls) == first
    ev = N.load_events()
    assert len(ev) == 3 and ev[0][0] == "ABC" and ev[0][2] is True


def _m5_day(d, start, drift, vol, seed):
    rnd = random.Random(seed)
    out, px = [], start
    for k in range(75):
        t = (dt.datetime.combine(d, dt.time(9, 15)) + dt.timedelta(minutes=5 * k)).time()
        o = px
        c = o * (1 + drift + rnd.gauss(0, 0.001))
        out.append((t, o, max(o, c) * 1.0005, min(o, c) * 0.9995, c, vol * (3 if k == 0 else 1)))
        px = c
    return out


def test_orb_entry_stop_and_eod():
    from intraday_lab import idea_cd as CD
    T = dt.time
    bars = [(T(9, 20), 100, 100.5, 99.9, 100.4, 1), (T(9, 25), 100.4, 101.2, 100.6, 101.0, 1)]
    bars += [(T(9, 30 + 5 * k) if 30 + 5 * k < 60 else T(10, (30 + 5 * k) - 60), 101, 101.3, 100.9, 101.1, 1) for k in range(5)]
    bars.append((T(15, 10), 101, 102, 100.95, 101.8, 1))
    g, net, fut = CD._orb(1, 101.0, 5.0, bars, False)
    assert round(g, 3) == round(100 * 0.8 / 101.0, 3) and net < g and fut < g
    stop_bars = bars[:2] + [(T(9, 30), 101, 101.1, 100.4, 100.6, 1)]
    g2, _, _ = CD._orb(1, 101.0, 5.0, stop_bars, False)
    assert round(g2, 3) == round(-100 * 0.5 / 101.0, 3)
    assert CD._orb(1, 105.0, 5.0, bars, False) is None


def test_cd_records_and_selection():
    from intraday_lab import idea_cd as CD
    days = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(45)]
    days = [d for d in days if d.weekday() < 5]
    by = {d: _m5_day(d, 1000, 0.0002, 100000, i) for i, d in enumerate(days)}
    recs = CD.stock_days(1, "ABC", by)
    assert recs and all(r["eligible"] for r in recs) and abs(recs[0]["rv"] - 1) < 1e-9
    assert ("c" in recs[0]) and ("d" in recs[0]) and recs[0]["color"] in (1, -1, 0)
    sel = {r["date"] for r in recs}
    real, rand, pairs = CD.trades_c(recs, sel)
    assert len(pairs) == sum(1 for r in recs if r["color"]) and len(rand) <= len(recs)
    nifty = {d: -0.005 for d in sel}
    real_d, _, _ = CD.trades_d(recs, sel, nifty)
    assert len(real_d) == sum(1 for r in recs if r["chg945"] > 0)


def test_read_m5_windows_files():
    import gzip
    from intraday_lab import idea_cd as CD
    tmp = Path(tempfile.mkdtemp())
    (tmp / "9.csv.gz").write_bytes(gzip.compress(b"1729655100,852.5,861.05,852.5,859.3,600176\r\r\n1729655400,859,861,858,860,6\r\r\n"))
    by = CD.read_m5(9, tmp)
    assert sum(len(v) for v in by.values()) == 2
