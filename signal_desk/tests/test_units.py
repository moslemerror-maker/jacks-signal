import datetime as dt
import json

from signal_desk import candles as C
from signal_desk import optionchain as OC
from signal_desk import stocks as S
from signal_desk import tracker
from signal_desk.clock import IST
from signal_desk.config import read_env, write_env_value
from signal_desk.dhan_api import unwrap_candles, unwrap_chain, unwrap_feed
from signal_desk.nifty import MarketCtx, NiftyRules
from signal_desk.notify import Notifier
from signal_desk.risk import RiskGuard, Trade
from signal_desk.signals import Signal

DAY = dt.date(2026, 10, 9)


def t(hhmm):
    return dt.datetime.combine(DAY, dt.time(*map(int, hhmm.split(":"))), IST)


def cndl(hhmm, o, h, l, c, v=0):
    return C.Candle(t(hhmm), o, h, l, c, v, 15)


def test_unwrap_chain_both_nestings():
    inner = {"last_price": 22500.5, "oc": {"22500.000000": {"ce": {"last_price": 100}, "pe": {"last_price": 90}}}}
    assert unwrap_chain({"status": "success", "data": {"data": inner}})["last_price"] == 22500.5
    assert unwrap_chain({"data": inner})["oc"]
    assert unwrap_chain(inner) is inner


def test_unwrap_candles_flat_and_sdk_envelope():
    flat = {"open": [1], "high": [2], "low": [0.5], "close": [1.5], "timestamp": [1791000000]}
    assert unwrap_candles(flat)["volume"] == [0]
    assert unwrap_candles({"status": "success", "data": flat})["close"] == [1.5]


def test_unwrap_feed():
    body = {"data": {"IDX_I": {"13": {"last_price": 22500}}}, "status": "success"}
    assert unwrap_feed(body)["IDX_I"]["13"]["last_price"] == 22500
    assert unwrap_feed({"data": {"data": {"NSE_EQ": {}}}}) == {"NSE_EQ": {}}


def test_epoch_to_ist_candle():
    ts = t("09:15").timestamp()
    cs = C.from_arrays({"open": [1], "high": [2], "low": [0], "close": [1], "timestamp": [ts]}, 15)
    assert cs[0].start == t("09:15") and cs[0].end == t("09:30")


def test_completed_excludes_forming_candle():
    cs = [cndl("09:15", 1, 2, 0, 1), cndl("09:30", 1, 2, 0, 1)]
    assert len(C.completed(cs, t("09:44"))) == 1
    assert len(C.completed(cs, t("09:45") + dt.timedelta(seconds=5))) == 2


def _chain(spot=22548):
    oc = {}
    for k in range(22300, 22850, 50):
        ce_oi = 5_000_000 if k == 22600 else 3_000_000 if k == 22700 else 1_000_000
        pe_oi = 6_000_000 if k == 22400 else 2_500_000 if k == 22300 else 900_000
        oc[f"{k}.000000"] = {"ce": {"last_price": max(spot - k, 0) + 60, "oi": ce_oi, "previous_oi": ce_oi - 100_000,
                                    "greeks": {"delta": 0.5}},
                             "pe": {"last_price": max(k - spot, 0) + 55, "oi": pe_oi, "previous_oi": pe_oi - 300_000,
                                    "greeks": {"delta": -0.5}}}
    return OC.parse_chain({"last_price": spot, "oc": oc}, "2026-10-13")


def test_walls_pcr_maxpain_bias():
    st = OC.stats(_chain())
    assert st.call_wall == 22600 and st.put_wall == 22400
    assert st.call_wall_2 == 22700 and st.put_wall_2 == 22300
    assert st.pcr and 0.5 < st.pcr < 2
    assert st.max_pain in {r.strike for r in _chain().rows}
    assert st.oi_bias_label.startswith("bullish")


def test_pick_strike_and_expiry():
    ch = _chain()
    assert OC.pick_strike(ch, 22548, "CE") == 22500
    assert OC.pick_strike(ch, 22548, "PE") == 22550
    ex = ["2026-10-13", "2026-10-20", "2026-10-27"]
    assert OC.choose_expiry(ex, dt.date(2026, 10, 9)) == "2026-10-13"
    assert OC.choose_expiry(ex, dt.date(2026, 10, 13)) == "2026-10-20"
    assert OC.nearest_expiry(ex, dt.date(2026, 10, 13)) == "2026-10-13"


def _ctx(candles, now, chain=None, budget=1500, blocked=None, bnf=None, vix=(14.0, 14.1)):
    ch = chain or _chain(candles[-1].close)
    return MarketCtx(now=now, nifty=candles, prev_hlc=(22300, 22100, 22231.8),
                     bnf=bnf if bnf is not None else [cndl(c.start.strftime("%H:%M"), c.open * 2.42, c.high * 2.42, c.low * 2.42, c.close * 2.42) for c in candles],
                     bnf_prev_close=53800, vix=vix[0], vix_prev_close=vix[1], oi_stats=OC.stats(ch), trade_chain=ch,
                     lot=65, budget=budget, blocked_reason=blocked)


def _range_day():
    return [cndl("09:15", 22315, 22370, 22300, 22360), cndl("09:30", 22360, 22430, 22350, 22420),
            cndl("09:45", 22420, 22500, 22410, 22490), cndl("10:00", 22490, 22534, 22450, 22470),
            cndl("10:15", 22470, 22520, 22455, 22500)]


def test_breakout_close_creates_take_signal_with_option_plan():
    cs = _range_day() + [cndl("10:30", 22505, 22552, 22520, 22548)]
    r = NiftyRules()
    out = r.on_candle_close(_ctx(cs, t("10:45")))
    assert len(out) == 1
    s = out[0]
    assert s.side == "LONG" and s.level == 22534 and s.entry == 22548
    assert s.option == "CE" and s.strike == 22500 and s.expiry == "2026-10-13"
    assert s.stop < s.entry < s.t1 < s.t2
    assert s.lots == 1 and s.risk_per_lot and s.risk_per_lot <= 1500
    assert s.action == "TAKE", [(c.name, c.status, c.detail) for c in s.checks]
    assert r.on_candle_close(_ctx(cs, t("10:45"))) == []


def test_no_signal_inside_range():
    cs = _range_day() + [cndl("10:30", 22500, 22530, 22480, 22525)]
    assert NiftyRules().on_candle_close(_ctx(cs, t("10:45"))) == []


def test_breakdown_creates_pe():
    cs = _range_day() + [cndl("10:30", 22320, 22330, 22280, 22285)]
    s = NiftyRules().on_candle_close(_ctx(cs, t("10:45"), vix=(14.5, 14.1)))[0]
    assert s.side == "SHORT" and s.option == "PE" and s.strike >= s.entry


def test_wick_warns_and_wall_fails():
    cs = _range_day() + [cndl("10:30", 22520, 22660, 22515, 22590)]
    s = NiftyRules().on_candle_close(_ctx(cs, t("10:45"), chain=_chain(22590)))[0]
    st = {c.name: c.status for c in s.checks}
    assert st["Candle"] == "warn"
    assert s.action == "SKIP"


def test_budget_and_guard_block():
    cs = _range_day() + [cndl("10:30", 22505, 22552, 22520, 22548)]
    s = NiftyRules().on_candle_close(_ctx(cs, t("10:45"), budget=300))[0]
    assert s.lots == 1 and {c.name: c.status for c in s.checks}["Worst case"] == "info" and s.action == "TAKE"
    s2 = NiftyRules().on_candle_close(_ctx(cs, t("10:45"), blocked="daily loss limit"))[0]
    assert s2.action == "SKIP"


def test_after_cutoff_is_skip():
    base = [cndl(f"{9 + (15 + 15 * i) // 60}:{(15 + 15 * i) % 60:02d}", 22400, 22450, 22390, 22420) for i in range(21)]
    last = base[-1].start + dt.timedelta(minutes=15)
    cs = base + [C.Candle(last, 22420, 22480, 22440, 22478, 0, 15)]
    s = NiftyRules().on_candle_close(_ctx(cs, last + dt.timedelta(minutes=16)))[0]
    assert {c.name: c.status for c in s.checks}["Time"] == "fail" and s.action == "SKIP"


def test_rearm_needs_a_base():
    r = NiftyRules()
    cs = _range_day() + [cndl("10:30", 22505, 22552, 22520, 22548)]
    s1 = r.on_candle_close(_ctx(cs, t("10:45")))[0]
    s1.status = "T1"
    cs2 = cs + [cndl("10:45", 22548, 22580, 22545, 22575)]
    assert r.on_candle_close(_ctx(cs2, t("11:00"))) == []
    cs3 = cs2 + [cndl("11:00", 22575, 22578, 22540, 22550), cndl("11:15", 22550, 22570, 22545, 22560),
                 cndl("11:30", 22560, 22600, 22558, 22596)]
    assert len(r.on_candle_close(_ctx(cs3, t("11:45")))) == 1


def test_approach_headsup_once_and_throttled():
    cs = _range_day()
    r = NiftyRules()
    ctx = _ctx(cs, t("10:35"))
    assert "breakout level 22,534" in r.approach(22528, ctx)
    assert r.approach(22529, ctx) is None


def test_risk_guard_limits_and_alerts():
    g = RiskGuard(1000, 1500, 2, 500, "14:30", "15:15")
    assert g.budget(t("10:00")) == 1500 and g.budget(t("10:00"), "STOCK") == 500
    tr = Trade("T1", "OPTION", "NIFTY 22500 CE", "BUY", 150, 65, t("10:00"), sl=130, target=180)
    g.trades.append(tr)
    tr.ltp = 171
    msgs = g.trade_alerts(tr)
    assert any("Move your stop to cost" in m for m in msgs)
    tr.ltp = 181
    assert any("target" in m for m in g.trade_alerts(tr))
    tr.ltp = 120
    assert any("stop" in m for m in g.trade_alerts(tr))
    tr.exit, tr.closed_at = 135, t("10:30")
    assert g.day_pnl(DAY) < -1000
    assert "warning" in g.check_limits(t("10:31"))
    assert g.check_limits(t("10:32")) is None
    assert g.budget(t("10:31")) < 500
    g.trades.append(Trade("T2", "STOCK", "ALPHA", "BUY", 100, 10, t("11:00")))
    assert "max 2 trades" in g.blocked_reason(t("11:05"))


def test_risk_guard_cutoff():
    g = RiskGuard(1000, 1500, 2, 500, "14:30", "15:15")
    assert g.blocked_reason(t("14:29")) is None
    assert "after 14:30" in g.blocked_reason(t("14:31"))


def test_tracker_first_touch_and_r(tmp_path):
    s = Signal("x", "NIFTY", "NIFTY", "LONG", t("10:45"), 22534, 22548, 22520, 22590, 22618, action="TAKE", grade="A")
    assert tracker.update(s, 22560, t("10:50")) is None
    assert tracker.update(s, 22591, t("11:00")) == "T1"
    assert tracker.update(s, 22619, t("11:10")) == "T2"
    assert tracker.result_r(s) == round(70 / 28, 2)
    s2 = Signal("y", "NIFTY", "NIFTY", "LONG", t("10:45"), 22534, 22548, 22520, 22590, 22618, action="TAKE")
    assert tracker.update(s2, 22519, t("10:50")) == "SL" and tracker.result_r(s2) == -1.0
    path = tmp_path / "journal.csv"
    assert tracker.append_journal(path, [s, s2]) == 2
    assert tracker.append_journal(path, [s, s2]) == 0


def test_stock_rvol_and_shortlist():
    hist = []
    for d in range(1, 6):
        day = DAY - dt.timedelta(days=d)
        for i in range(4):
            st = dt.datetime.combine(day, dt.time(9, 15), IST) + dt.timedelta(minutes=15 * i)
            hist.append(C.Candle(st, 100, 101, 99, 100, 10_000, 15))
    today = [cndl("09:15", 100, 101, 99.5, 100.5, 12_000), cndl("09:30", 100.5, 101.2, 100.2, 101.0, 11_000),
             cndl("09:45", 101.0, 102.4, 100.9, 102.3, 40_000)]
    prof = S.DailyProfile("ALPHA", 1, 101.5, 98.5, 100, 2, 1e6, 1e6, 102, 95, False, False)
    rules = S.StockRules()
    s = rules.evaluate("ALPHA", today, hist, prof, 0.5, t("10:01"), 500, None)
    assert s and s.side == "LONG" and {c.name: c.status for c in s.checks}["Volume"] == "pass"
    assert s.qty and s.risk_rupees <= 500 and s.action == "TAKE"
    assert rules.evaluate("ALPHA", today, hist, prof, 0.5, t("10:01"), 500, None) is None
    quotes = {1: {"ltp": 102.3, "high": 102.4, "low": 99.5}, 2: {"ltp": 50, "high": 55, "low": 49}}
    profs = {1: prof, 2: S.DailyProfile("B", 2, 51, 49, 50, 1, 1, 1, 52, 48, False, False)}
    assert S.shortlist(quotes, profs) == [1]


def test_write_env_value(tmp_path):
    p = tmp_path / ".env"
    p.write_text("A=1\n# DHAN_ACCESS_TOKEN=\nB=2\n")
    write_env_value("DHAN_ACCESS_TOKEN", "abc", p)
    write_env_value("TELEGRAM_CHAT_ID", "42", p)
    e = read_env(p)
    assert e["DHAN_ACCESS_TOKEN"] == "abc" and e["TELEGRAM_CHAT_ID"] == "42" and e["A"] == "1"


class _Resp:
    def __init__(self, code, body):
        self.status_code, self._b, self.text = code, body, json.dumps(body)

    def json(self):
        return self._b


class _Sess:
    def __init__(self, codes):
        self.codes, self.posts = list(codes), []

    def post(self, url, json=None, timeout=None):
        self.posts.append(json)
        return _Resp(self.codes.pop(0), {"ok": True})

    def get(self, url, timeout=None):
        return _Resp(200, {"ok": True, "result": [{"message": {"chat": {"id": 777, "first_name": "Moslem"}}}]})


def test_notifier_html_fallback_and_chat_id():
    s = _Sess([400, 200])
    n = Notifier("tok", "1", session=s)
    assert n.send("<b>x</b> a<b") is True and "parse_mode" not in s.posts[1]
    assert n.find_chat_id()[0] == "777"
    assert Notifier().send("hi") is False and Notifier().feed == type(Notifier().feed)(maxlen=200)


def test_daily_falls_back_to_intraday60():
    from signal_desk.dhan_api import DhanApi, DhanError

    class Api(DhanApi):
        def post(self, endpoint, payload):
            if endpoint == "/charts/historical":
                raise DhanError("HTTP 400 DH-907 no data")
            ts = [t("09:15").timestamp() - 86400 * k + 3600 * h for k in (2, 1) for h in range(6)]
            n = len(ts)
            return {"open": [100] * n, "high": [101 + i for i in range(n)], "low": [99] * n, "close": [100.5] * n,
                    "volume": [10] * n, "timestamp": ts}

    a = Api("1", "x")
    d = a.daily(1, "NSE_EQ", "EQUITY", DAY - dt.timedelta(days=30), DAY)
    assert a.daily_mode == "intraday60" and len(d["close"]) == 2 and d["volume"] == [60, 60] and d["high"][1] == 112


def test_backtest_outcome_rules():
    from signal_desk.backtest import _outcome
    s = Signal("x", "NIFTY", "NIFTY", "LONG", t("10:45"), 22534, 22548, 22520, 22590, 22618)
    sq = t("15:15")
    assert _outcome(s, [cndl("10:45", 22548, 22600, 22510, 22560)], sq)[0] == "SL"
    o, pts, t2 = _outcome(s, [cndl("10:45", 22548, 22595, 22540, 22590), cndl("11:00", 22590, 22592, 22545, 22547)], sq)
    assert o == "T1" and pts == 21.0 and not t2
    o, pts, t2 = _outcome(s, [cndl("10:45", 22548, 22595, 22540, 22590), cndl("11:00", 22590, 22620, 22585, 22610)], sq)
    assert t2 and pts == 56.0
    o, pts, _ = _outcome(s, [cndl("10:45", 22548, 22560, 22530, 22555), cndl("15:15", 22555, 22700, 22500, 22690)], sq)
    assert o == "timeout" and pts == 7.0


def test_trend_filter_calls_and_puts():
    import dataclasses
    up = _range_day() + [cndl("10:30", 22505, 22552, 22520, 22548)]
    s = NiftyRules().on_candle_close(dataclasses.replace(_ctx(up, t("10:45")), trend_filter=True))[0]
    tr = next(c for c in s.checks if c.name == "Trend")
    assert tr.status == "pass" and s.action == "TAKE" and s.option == "CE"
    dn = _range_day() + [cndl("10:30", 22320, 22330, 22280, 22285)]
    s = NiftyRules().on_candle_close(dataclasses.replace(_ctx(dn, t("10:45"), vix=(14.5, 14.1)), trend_filter=True))[0]
    assert next(c for c in s.checks if c.name == "Trend").status == "fail" and s.action == "SKIP"
    ctx = dataclasses.replace(_ctx(dn, t("10:45"), vix=(14.5, 14.1)), trend_filter=True, prev_hlc=(22450, 22300, 22400))
    s = NiftyRules().on_candle_close(ctx)[0]
    assert s.option == "PE" and next(c for c in s.checks if c.name == "Trend").status == "pass"
    s = NiftyRules().on_candle_close(_ctx(dn, t("10:45"), vix=(14.5, 14.1)))[0]
    assert not any(c.name == "Trend" for c in s.checks)


def test_aggregate_hourly_anchored_at_915():
    cs = [cndl(f"{9 + (15 + 15 * i) // 60}:{(15 + 15 * i) % 60:02d}", 100 + i, 101 + i, 99 + i, 100.5 + i) for i in range(10)]
    h = C.aggregate(cs, 60)
    assert [f"{c.start:%H:%M}" for c in h] == ["09:15", "10:15"]
    assert h[0].open == 100 and h[0].close == 103.5 and h[0].high == 104 and h[0].low == 99 and h[0].minutes == 60
    assert h[0].end == t("10:15")


def test_token_expiry_reads_only_exp_claim():
    import base64
    from signal_desk.config import token_expiry
    payload = base64.urlsafe_b64encode(json.dumps({"exp": 1760331600, "dhanClientId": "1"}).encode()).decode().rstrip("=")
    exp = token_expiry(f"aaa.{payload}.sig")
    assert exp is not None and exp.utcoffset() == dt.timedelta(hours=5, minutes=30)
    assert exp.timestamp() == 1760331600
    assert token_expiry("not-a-jwt") is None and token_expiry("") is None


def test_token_reminder_before_close(tmp_path):
    import base64
    from signal_desk.config import Settings
    from signal_desk.sim import run_day
    exp = dt.datetime(2026, 10, 9, 11, 0, tzinfo=IST)
    payload = base64.urlsafe_b64encode(json.dumps({"exp": exp.timestamp()}).encode()).decode().rstrip("=")
    s = Settings(dhan_client_id="1", dhan_access_token=f"h.{payload}.s")
    eng = run_day(state_dir=tmp_path, quiet=True, end="10:45", settings=s)
    texts = [m["text"] for m in eng.notifier.feed]
    assert sum("expires today at 11:00" in x for x in texts) == 1
    assert sum("expires in" in x for x in texts) == 1
    assert eng.snapshot()["status"]["token_expires"].endswith("11:00")


def test_paper_trades_do_not_count_and_block_keeps_grade():
    g = RiskGuard(1000, 1500, 2, 500, "14:30", "15:15")
    now = t("10:30")
    for i, paper in enumerate((True, True, True)):
        tr = Trade(id=f"T{i}", kind="STOCK", symbol="X", side="BUY", entry=100, qty=1000, opened_at=now, paper=paper)
        tr.exit, tr.closed_at = 98, now
        g.trades.append(tr)
    assert g.blocked_reason(now) is None and g.day_pnl(now.date()) == 0 and g.paper_pnl(now.date()) < -5000
    g.trades[0].paper = False
    assert "daily loss limit" in (g.blocked_reason(now) or "")
    from signal_desk.signals import Check, grade
    g2, a2 = grade([Check("Time", "pass", ""), Check("Risk guard", "fail", "limit")])
    assert (g2, a2) == ("A", "SKIP")


def _jwt(exp: dt.datetime) -> str:
    import base64
    payload = base64.urlsafe_b64encode(json.dumps({"exp": exp.timestamp()}).encode()).decode().rstrip("=")
    return f"eyJhbGciOiJIUzUxMiJ9.{payload}.c2ln"


def test_renew_token_parses_reply_and_errors():
    from signal_desk.dhan_api import DhanApi, DhanAuthError, DhanError, find_jwt
    good = _jwt(dt.datetime(2026, 10, 10, 15, 30, tzinfo=IST))

    class S:
        def __init__(self, code, body):
            self.code, self.body, self.calls = code, body, []

        def get(self, url, headers=None, timeout=None):
            self.calls.append((url, headers))
            return _Resp(self.code, self.body)

    s = S(200, {"status": "success", "data": {"accessToken": good}})
    a = DhanApi("1100", "eyJold.x.y", session=s)
    assert a.renew_token() == good
    url, h = s.calls[0]
    assert url.endswith("/v2/RenewToken") and h["dhanClientId"] == "1100" and h["access-token"] == "eyJold.x.y"
    assert find_jwt({"token": good}) == good and find_jwt("no token") is None
    for sess, exc in ((S(401, {"errorCode": "DH-901"}), DhanAuthError), (S(200, {"status": "success"}), DhanError)):
        try:
            DhanApi("1", "t", session=sess).renew_token()
            raise AssertionError("no error raised")
        except exc:
            pass


def _renew_engine(tmp_path, exp, outcomes):
    from signal_desk.config import Settings
    from signal_desk.sim import run_day
    s = Settings(dhan_client_id="1", dhan_access_token=_jwt(exp))
    saved = {}

    def patch(eng):
        eng.env_writer = lambda k, v: saved.__setitem__(k, v)
        it = iter(outcomes)

        def renew():
            o = next(it)
            if isinstance(o, Exception):
                raise o
            return o
        eng.api.renew_token = renew
    return s, saved, patch, run_day


def test_auto_renew_at_1530_saves_and_tells(tmp_path):
    exp = dt.datetime(2026, 10, 10, 9, 0, tzinfo=IST)
    new = _jwt(dt.datetime(2026, 10, 10, 15, 30, tzinfo=IST))
    s, saved, patch, run_day = _renew_engine(tmp_path, exp, [new])
    eng = run_day(state_dir=tmp_path, quiet=True, start="15:20", end="15:40", settings=s, setup=patch)
    assert saved == {"DHAN_ACCESS_TOKEN": new} and eng.s.dhan_access_token == new and eng.api.access_token == new
    texts = [m["text"] for m in eng.notifier.feed]
    assert sum("renewed automatically" in x for x in texts) == 1 and not any("failed" in x for x in texts)
    assert eng.snapshot()["status"]["token_renewed"]


def test_auto_renew_morning_when_token_ends_today(tmp_path):
    exp = dt.datetime(2026, 10, 9, 11, 28, tzinfo=IST)
    new = _jwt(dt.datetime(2026, 10, 10, 9, 0, tzinfo=IST))
    s, saved, patch, run_day = _renew_engine(tmp_path, exp, [new])
    eng = run_day(state_dir=tmp_path, quiet=True, start="08:55", end="09:10", settings=s, setup=patch)
    assert saved["DHAN_ACCESS_TOKEN"] == new
    assert not any("expires today" in m["text"] for m in eng.notifier.feed)


def test_auto_renew_failure_alerts_once(tmp_path):
    from signal_desk.dhan_api import DhanAuthError, DhanError
    exp = dt.datetime(2026, 10, 10, 9, 0, tzinfo=IST)
    s, saved, patch, run_day = _renew_engine(tmp_path, exp, [DhanError("net")] * 5)
    eng = run_day(state_dir=tmp_path, quiet=True, start="15:20", end="15:40", settings=s, setup=patch)
    fails = [m for m in eng.notifier.feed if "auto-renewal failed" in m["text"]]
    assert len(fails) == 1 and fails[0]["kind"] == "error" and saved == {}
    s, saved, patch, run_day = _renew_engine(tmp_path / "b", exp, [DhanAuthError("401")])
    eng = run_day(state_dir=tmp_path / "b", quiet=True, start="15:20", end="15:40", settings=s, setup=patch)
    assert sum("auto-renewal failed" in m["text"] for m in eng.notifier.feed) == 1


def test_no_renew_for_expired_or_far_token(tmp_path):
    calls = []
    for i, exp in enumerate((dt.datetime(2026, 10, 9, 8, 0, tzinfo=IST), dt.datetime(2026, 10, 11, 9, 0, tzinfo=IST))):
        s, saved, patch, run_day = _renew_engine(tmp_path / str(i), exp, [])

        def p2(eng, patch=patch):
            patch(eng)
            eng.api.renew_token = lambda: calls.append(1)
        run_day(state_dir=tmp_path / str(i), quiet=True, start="15:20", end="15:40", settings=s, setup=p2)
    assert calls == []


def test_prev_range_ratio_and_grade_adjust_never_changes_action():
    import dataclasses
    from signal_desk.nifty import RANGE_NARROW, RANGE_WIDE, _range_adjust, prev_range_ratio
    from signal_desk.signals import Check, Signal, WARN
    hist, d = [], dt.date(2026, 9, 1)
    k = 0
    while k < 11:
        if d.weekday() < 5:
            r = 50 if k == 10 else 100
            for i in range(25):
                st = dt.datetime.combine(d, dt.time(9, 15), IST) + dt.timedelta(minutes=15 * i)
                hist.append(C.Candle(st, 22000, 22000 + (r if i == 3 else 1), 22000 - (0 if i else 0), 22000, 0, 15))
            k += 1
        d += dt.timedelta(days=1)
    assert prev_range_ratio(hist, d) == 0.5
    assert prev_range_ratio(hist[: 25 * 10], d) is None
    def sig(g, a):
        s = Signal("x", "NIFTY", "NIFTY", "LONG", t("10:00"), 1, 2, 0, 3, 4, checks=[], grade=g, action=a)
        return s
    for ratio, g, a, want in ((0.5, "B", "TAKE", "A"), (0.5, "A", "TAKE", "A"), (0.5, "C", "SKIP", "C"),
                              (1.5, "A", "TAKE", "B"), (1.5, "B", "TAKE", "B"), (1.5, "C", "SKIP", "C"),
                              (1.0, "B", "TAKE", "B"), (1.0, "A", "TAKE", "A"), (0.5, "B", "SKIP", "A")):
        s = sig(g, a)
        _range_adjust(s, ratio)
        assert (s.grade, s.action, s.base_grade, s.prev_range) == (want, a, g, ratio), (ratio, g)
        assert s.checks[-1].name == "Prev day" and s.checks[-1].status == "info"
    s = sig("B", "TAKE")
    _range_adjust(s, None)
    assert s.grade == "B" and s.base_grade is None and not s.checks
    assert RANGE_NARROW < 1 < RANGE_WIDE


def test_journal_gets_new_columns_and_keeps_old_rows(tmp_path):
    import csv
    from signal_desk import tracker
    p = tmp_path / "journal.csv"
    old = tracker.JOURNAL_FIELDS[:-2]
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=old)
        w.writeheader()
        w.writerow({k: ("2026-10-19" if k == "date" else "12:15" if k == "time" else "x") for k in old})
    s = Signal("y", "NIFTY", "NIFTY", "SHORT", t("10:30"), 1, 2, 3, 0, -1, grade="A", action="TAKE",
               base_grade="B", prev_range=0.61)
    assert tracker.append_journal(p, [s]) == 1
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    assert len(rows) == 2 and rows[0]["date"] == "2026-10-19" and rows[0]["base_grade"] == ""
    assert rows[1]["base_grade"] == "B" and rows[1]["prev_range"] == "0.61" and rows[1]["grade"] == "A"
