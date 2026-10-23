import datetime as dt
import json
import threading
import urllib.request

from signal_desk.config import Settings
from signal_desk.sim import run_day


def test_full_day(tmp_path):
    eng = run_day(state_dir=tmp_path, quiet=True)
    kinds = [f["kind"] for f in eng.notifier.feed]
    assert "watchlist" in kinds and "eod" in kinds
    assert eng.errors == []
    nifty = eng.nifty_rules.signals
    assert any(s.action == "TAKE" for s in nifty)
    assert all(s.status != "open" for s in nifty)
    assert (tmp_path / "journal.csv").exists()
    stock = eng.stock_rules.signals
    assert any(s.symbol == "ALPHA" and s.side == "LONG" for s in stock)
    assert any(s.symbol == "BETA" and s.action == "SKIP" for s in stock)
    snap = eng.snapshot()
    json.dumps(snap, default=str)
    assert snap["nifty"]["levels"]["call_wall"] in (22600, 22650)


def test_restart_midday_does_not_resend(tmp_path):
    e1 = run_day(state_dir=tmp_path, quiet=True, end="12:00")
    n1 = len([f for f in e1.notifier.feed if f["kind"] in ("signal", "skip", "watchlist")])
    e2 = run_day(state_dir=tmp_path, quiet=True, start="12:00", end="12:10")
    resent = [f for f in e2.notifier.feed if f["kind"] in ("signal", "skip", "watchlist")]
    assert n1 > 0 and resent == []
    assert len(e2.nifty_rules.signals) == len(e1.nifty_rules.signals)
    lv = e2.snapshot()["nifty"]["levels"]
    assert lv.get("day_high") and lv.get("orh")


def test_expired_token_alerts_once_and_survives(tmp_path):
    eng = run_day(state_dir=tmp_path, quiet=True, end="10:00", fail_auth=True)
    errs = [f for f in eng.notifier.feed if f["kind"] == "error"]
    assert len(errs) == 1 and "token" in errs[0]["text"]
    assert eng.token_ok is False


def test_logged_trade_tracks_pnl_and_squareoff(tmp_path):
    eng = run_day(state_dir=tmp_path, quiet=True, end="13:00")
    tr = eng.add_trade({"kind": "OPTION", "strike": "22500", "option": "CE", "entry": "150", "qty": "65", "sl": "130", "target": "180"})
    assert tr.expiry == "2026-10-13" and tr.symbol.startswith("NIFTY 22,500 CE")
    eng2 = run_day(state_dir=tmp_path, quiet=True, start="13:00", end="15:20")
    t2 = eng2.guard.trades[0]
    assert t2.ltp is not None
    texts = " ".join(f["text"] for f in eng2.notifier.feed)
    assert "square-off" in texts or "exit open positions" in texts


def test_http_api(tmp_path):
    from signal_desk.server import serve
    eng = run_day(state_dir=tmp_path, quiet=True, end="11:00")
    httpd = serve(eng, 0)
    port = httpd.server_address[1]
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    try:
        st = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state").read())
        assert st["nifty"]["ltp"] and "signals" in st
        html = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
        assert "Signal Desk" in html
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/trade", method="POST",
                                     data=json.dumps({"kind": "STOCK", "symbol": "ALPHA", "side": "BUY", "entry": 1010, "qty": 10, "sl": 1005}).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req).read())
        assert r["ok"] and r["trade"]["security_id"] == 101
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/toggles", method="POST",
                                     data=json.dumps({"min_grade": "A"}).encode(), headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req)
        assert eng.s.min_grade == "A"
    finally:
        httpd.shutdown()


def test_backtest_runs_on_engine(tmp_path):
    eng = run_day(state_dir=tmp_path, quiet=True, end="09:10")
    eng.run_backtest(days=5, wait=True)
    b = eng.backtest
    assert b["status"] == "done", b
    assert b["sessions"] >= 3 and "take" in b and isinstance(b["trades"], list)
    json.dumps(eng.snapshot(), default=str)


def test_hourly_paper_signals_are_labelled(tmp_path):
    eng = run_day(state_dir=tmp_path, quiet=True)
    h1 = eng.h1_rules.signals
    assert h1, "expected at least one 1-hour evaluation on the 9 Oct path"
    assert all(x.id.startswith("NIFTY1H-") and "paper" in x.symbol for x in h1)
    assert all(x.time.minute == 15 for x in h1)
    feed = [m["text"] for m in eng.notifier.feed if m["text"].startswith("📝") and "1H" in m["text"]]
    assert feed and all("PAPER ONLY" in m and "1-hour close" in m for m in feed)
    eod = next(m["text"] for m in eng.notifier.feed if m["kind"] == "eod")
    assert "PAPER signals (1-hour / CE)" in eod and "not counted above" in eod
    saved = json.loads(next(tmp_path.glob("day-*.json")).read_text())
    assert any(d["id"].startswith("NIFTY1H-") for d in saved["signals"])


def test_calls_are_paper_by_default_puts_live(tmp_path):
    eng = run_day(state_dir=tmp_path, quiet=True)
    ce = [x for x in eng.nifty_rules.signals if x.side == "LONG"]
    assert ce and all(x.symbol == "NIFTY (paper CE)" for x in ce)
    assert all(m["text"].startswith("📝") for m in eng.notifier.feed if "(paper CE)" in m["text"] and m["kind"] in ("signal", "skip"))
    eod = next(m["text"] for m in eng.notifier.feed if m["kind"] == "eod")
    real = [x for x in eng._all_signals() if "paper" not in x.symbol and not x.id.startswith("NIFTY1H-")]
    assert f"Signals {len(real)} (" in eod and all(x.kind == "STOCK" or x.side == "SHORT" for x in real)
    from signal_desk.config import Settings
    s = Settings(dhan_client_id="1", dhan_access_token="x", calls_mode="live")
    eng2 = run_day(state_dir=tmp_path / "live", quiet=True, settings=s)
    assert all(x.symbol == "NIFTY" for x in eng2.nifty_rules.signals if x.side == "LONG")


def test_daily_running_message_once(tmp_path):
    eng = run_day(state_dir=tmp_path, quiet=True, end="09:40")
    hello = [m for m in eng.notifier.feed if "Signal Desk is running" in m["text"]]
    assert len(hello) == 1 and "Dhan connected" in hello[0]["text"] and "trend filter on" in hello[0]["text"]


def test_stocks_off_means_no_stock_work(tmp_path):
    from signal_desk.config import Settings
    s = Settings(dhan_client_id="1", dhan_access_token="x", stock_alerts_enabled=False)
    eng = run_day(state_dir=tmp_path, quiet=True, end="11:00", settings=s)
    assert eng.stock_rules.signals == [] and eng.profiles_status == "not started"
    assert not any(m["kind"] == "watchlist" for m in eng.notifier.feed)
    assert all(x["kind"] == "NIFTY" for x in eng.snapshot()["signals"])


def test_stocks_switched_off_midday_stops_followups(tmp_path):
    e1 = run_day(state_dir=tmp_path, quiet=True, end="10:30")
    assert e1.stock_rules.signals
    from signal_desk.config import Settings
    s = Settings(dhan_client_id="1", dhan_access_token="x", stock_alerts_enabled=False)
    e2 = run_day(state_dir=tmp_path, quiet=True, start="10:30", end="15:00", settings=s)
    assert e2.s.stock_alerts_enabled is False
    stock_syms = {x.symbol for x in e1.stock_rules.signals}
    later = [m["text"] for m in e2.notifier.feed if any(m["text"].startswith(sym + " ") for sym in stock_syms)]
    assert later == []


def test_expiry_shown_everywhere_on_expiry_day(tmp_path):
    import datetime as dt
    eng = run_day(day=dt.date(2026, 10, 13), state_dir=tmp_path, quiet=True, end="12:00")
    ch = eng.snapshot()["nifty"]["chains"]
    assert [c["role"] for c in ch] == ["oi", "trade"]
    assert "EXPIRES TODAY" in ch[0]["label"] and "Tue 20 Oct" in ch[1]["label"]
    msgs = [m["text"] for m in eng.notifier.feed if "Buy NIFTY" in m["text"]]
    assert msgs and all("📅 Expiry (NOT today's expiring contract): Tue 20 Oct" in m for m in msgs)
    sig = next(s for s in eng.nifty_rules.signals if s.option)
    assert sig.expiry == "2026-10-20"


def test_expiry_line_on_normal_day(tmp_path):
    eng = run_day(state_dir=tmp_path, quiet=True, end="12:00")
    ch = eng.snapshot()["nifty"]["chains"]
    assert len(ch) == 1 and ch[0]["role"] == "both"
    msgs = [m["text"] for m in eng.notifier.feed if "Buy NIFTY" in m["text"]]
    assert msgs and all("📅 Expiry: Tue 13 Oct (4 days left)" in m for m in msgs)


def test_nse_holiday_is_idle(tmp_path):
    import datetime as dt
    eng = run_day(day=dt.date(2026, 10, 20), state_dir=tmp_path, quiet=True, end="12:00")
    texts = [m["text"] for m in eng.notifier.feed]
    assert sum("NSE holiday today (Dussehra)" in t for t in texts) == 1
    assert "Wed 21 Oct" in next(t for t in texts if "NSE holiday" in t)
    assert not any("Signal Desk is running" in t for t in texts)
    assert eng.nifty_rules.signals == [] and eng.snapshot()["status"]["holiday"] == "Dussehra"


def test_expiry_day_0dte_option(tmp_path):
    import datetime as dt
    from signal_desk.config import Settings
    s = Settings(dhan_client_id="1", dhan_access_token="x", stock_alerts_enabled=True, expiry_day_contract="today")
    eng = run_day(day=dt.date(2026, 10, 13), state_dir=tmp_path, quiet=True, end="12:00", settings=s)
    sig = next(x for x in eng.nifty_rules.signals if x.option)
    assert sig.expiry == "2026-10-13"
    msg = next(m["text"] for m in eng.notifier.feed if "Buy NIFTY" in m["text"] and "1H" not in m["text"])
    assert "0DTE - this contract EXPIRES TODAY" in msg and "EXPIRES TODAY" in msg
    assert [c["role"] for c in eng.snapshot()["nifty"]["chains"]] == ["both"]
