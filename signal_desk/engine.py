from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import threading
import time
import traceback
from typing import Any

from . import candles as C
from . import instruments as I
from . import optionchain as OC
from . import stocks as S
from . import tracker
from .clock import IST, Clock, at, fmt_t, is_weekday
from .holidays import holiday_name, next_trading_day
from .config import STATE_DIR, Settings, autostart_status, token_expiry, write_env_value
from .dhan_api import DhanApi, DhanAuthError, DhanError
from .nifty import MarketCtx, NiftyRules, levels, prev_range_ratio
from .notify import Notifier
from .risk import RiskGuard, Trade
from .signals import Signal, expiry_label, format_signal, grade_ok

log = logging.getLogger("signal_desk.engine")

QUOTE_EVERY = 30
CHAIN_EVERY = 60
H1_PREFIX = "NIFTY1H-"
CANDLE_DELAY = 20
CANDLE_GIVEUP = 240
RENEW_AT = dt.time(15, 30)
RENEW_TRIES = 3
RENEW_GAP = 60


def _boundaries(day: dt.date) -> list[dt.datetime]:
    t = at(day, "09:30")
    out = []
    while t <= at(day, "15:30"):
        out.append(t)
        t += dt.timedelta(minutes=15)
    return out


def _is_paper(sig: Signal) -> bool:
    return sig.id.startswith(H1_PREFIX) or "paper" in sig.symbol


class Engine:
    def __init__(self, settings: Settings, api: Any = None, clock: Any = None, notifier: Notifier | None = None,
                 instruments_loader=None, state_dir=STATE_DIR) -> None:
        self.s = settings
        self.api = api or DhanApi(settings.dhan_client_id, settings.dhan_access_token)
        self.clock = clock or Clock()
        self.notifier = notifier or Notifier(settings.telegram_bot_token, settings.telegram_chat_id)
        self.notifier.clock = self.clock
        self._load_instruments = instruments_loader or I.load
        self.env_writer = write_env_value
        self._renew = {"day": None, "fails": 0, "next": 0.0, "dead": False}
        self.token_renewed: dt.datetime | None = None
        self.state_dir = state_dir
        self.lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.started_at = self.clock.now()
        self.backtest: dict = {"status": "not run"}
        self.research: dict = {"status": "not run"}
        self.stock_research: dict = {"status": "not run"}
        try:
            self.stock_research = json.loads((self.state_dir / "stock_research.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            pass
        self.autostart: dict = {"ok": None, "text": "checking..."}
        threading.Thread(target=lambda: setattr(self, "autostart", autostart_status()), daemon=True).start()
        try:
            self.research = json.loads((self.state_dir / "research.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            pass
        self._reset_day(None)

    def _reset_day(self, day: dt.date | None) -> None:
        self.day = day
        self.inst: I.Instruments | None = None
        self.nifty_rules = NiftyRules()
        self.h1_rules = NiftyRules()
        self.stock_rules = S.StockRules()
        self.guard = RiskGuard(self.s.daily_loss_warn, self.s.daily_loss_limit, self.s.max_trades,
                               self.s.stock_risk, self.s.entry_cutoff, self.s.squareoff)
        self.nifty_hist: list[C.Candle] = []
        self.bnf_hist: list[C.Candle] = []
        self.vix_hist: list[C.Candle] = []
        self.ltp: dict[str, float] = {}
        self.day_ohlc: dict[str, dict] = {}
        self.stock_ltp: dict[int, float] = {}
        self.expiries: list[str] = []
        self.chains: dict[str, OC.Chain] = {}
        self.first_oi: dict[str, dict[float, tuple[float, float]]] = {}
        self.oi_stats: OC.ChainStats | None = None
        self.profiles: dict[int, S.DailyProfile] = {}
        self.watch: list[S.DailyProfile] = []
        self.profiles_status = "not started"
        self.processed: set[str] = set()
        self.sent: set[str] = set()
        self.errors: list[str] = []
        self.token_ok: bool | None = None
        self.last = {"quotes": None, "chain": None, "candles": None, "scan": None}
        self._next = {"quotes": 0.0, "chain": 0.0, "candles": 0.0}
        self.market_closed_today = False
        self._offhours_loaded = False

    def _state_path(self):
        return self.state_dir / f"day-{self.day:%Y-%m-%d}.json"

    def save(self) -> None:
        if not self.day:
            return
        self.state_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "signals": [s.to_json() for s in self._all_signals()],
            "trades": [t.to_json() for t in self.guard.trades],
            "sent": sorted(self.sent), "processed": sorted(self.processed),
            "warn_sent": self.guard.warn_sent, "limit_sent": self.guard.limit_sent,
            "approach": sorted(self.nifty_rules._approach_sent),
            "toggles": {"alerts_enabled": self.s.alerts_enabled, "stock_alerts_enabled": self.s.stock_alerts_enabled,
                        "min_grade": self.s.min_grade},
        }
        tmp = self._state_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        tmp.replace(self._state_path())

    def _load_day(self) -> None:
        p = self._state_path()
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for d in data.get("signals", []):
            sig = Signal.from_json(d)
            if sig.kind != "NIFTY":
                self.stock_rules.signals.append(sig)
            elif sig.id.startswith(H1_PREFIX):
                self.h1_rules.signals.append(sig)
            else:
                self.nifty_rules.signals.append(sig)
        self.guard.trades = [Trade.from_json(d) for d in data.get("trades", [])]
        self.sent = set(data.get("sent", []))
        self.processed = set(data.get("processed", []))
        self.guard.warn_sent = data.get("warn_sent", False)
        self.guard.limit_sent = data.get("limit_sent", False)
        self.nifty_rules._approach_sent = set(data.get("approach", []))
        for k, v in (data.get("toggles") or {}).items():
            if k != "stock_alerts_enabled":
                setattr(self.s, k, v)

    def _trade_expiry(self, day: dt.date) -> str | None:
        if not self.expiries:
            return None
        return OC.choose_expiry(self.expiries, day, avoid_expiry_day=self.s.expiry_day_contract != "today")

    def _all_signals(self) -> list[Signal]:
        return self.nifty_rules.signals + self.h1_rules.signals + self.stock_rules.signals

    def _mono(self) -> float:
        return self.clock.now().timestamp()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="signal-desk-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:
                self._err(f"engine error: {exc}")
                log.error(traceback.format_exc())
            self._stop.wait(5 if self.token_ok is not False else 30)

    def _err(self, msg: str) -> None:
        stamp = self.clock.now().strftime("%H:%M:%S")
        with self.lock:
            self.errors.insert(0, f"{stamp} {msg}")
            del self.errors[30:]
        log.warning(msg)

    def _alert(self, key: str | None, html: str, kind: str = "info", push: bool = True) -> None:
        if key and key in self.sent:
            return
        if key:
            self.sent.add(key)
        self.notifier.send(html, kind=kind, push=push and self.s.alerts_enabled)
        self.save()

    def _call(self, fn, *a, **kw):
        try:
            out = fn(*a, **kw)
            if self.token_ok is not True:
                self.token_ok = True
            return out
        except DhanAuthError as exc:
            if self.token_ok is not False:
                self._alert(f"token-{self.day}", f"🔑 Dhan access token rejected or expired. Generate a new one in Dhan web "
                            f"(Profile → DhanHQ Trading APIs) and paste it in Signal Desk → Settings.", kind="error")
            self.token_ok = False
            self._err(str(exc))
            return None
        except DhanError as exc:
            self._err(str(exc))
            return None

    def tick(self) -> None:
        now = self.clock.now()
        today = now.date()
        with self.lock:
            if self.day != today:
                self._reset_day(today)
                self._load_day()
            if self.inst is None:
                self.inst = self._load_instruments(today)
                for n in self.inst.notes:
                    self._err(n)
                if self.s.nifty_lot:
                    self.inst.nifty_lot = self.s.nifty_lot
        self._auto_renew(now)
        if not is_weekday(today):
            self._offhours(now)
            return
        hol = holiday_name(today, self.s.holidays_extra)
        if hol:
            if dt.time(8, 30) <= now.time() <= dt.time(15, 30):
                nxt = next_trading_day(today, self.s.holidays_extra)
                self._alert(f"holiday-{today}", f"📅 NSE holiday today ({hol}) - market closed, Signal Desk is idle. "
                            f"Next trading day: {nxt:%a %d %b}.", kind="info")
            self._offhours(now)
            return
        mono = self._mono()
        t = now.time()
        self._token_reminder(now)
        self._hello(now)
        if (self.s.stock_alerts_enabled and self.profiles_status == "not started" and dt.time(8, 0) <= t <= dt.time(15, 30)
                and self.inst.stocks):
            self.profiles_status = "building"
            threading.Thread(target=self._build_profiles, args=(today,), daemon=True).start()
        if self.s.stock_alerts_enabled and dt.time(8, 55) <= t < dt.time(15, 0) and self.watch and "premarket" not in self.sent:
            self._send_watchlist(now)
        if not (dt.time(9, 15) <= t <= dt.time(15, 35)):
            self._offhours(now)
            return
        if self.market_closed_today:
            return

        if mono >= self._next["quotes"]:
            self._next["quotes"] = mono + QUOTE_EVERY
            self._update_quotes(now)
        if mono >= self._next["chain"] and t <= dt.time(15, 30):
            self._next["chain"] = mono + CHAIN_EVERY
            self._update_chains(now)
        self._process_candles(now)
        self._live_checks(now)
        self._reminders(now)

    def _hello(self, now: dt.datetime) -> None:
        if f"hello-{now.date()}" in self.sent or self.token_ok is None or now.time() > dt.time(15, 30):
            return
        exp = token_expiry(self.s.dhan_access_token or "")
        dhan = ("✅ Dhan connected" if self.token_ok else "❌ Dhan token expired - paste a new one in Settings")
        if self.token_ok and exp:
            dhan += f" (token valid till {exp:%a %H:%M})"
        rules = (f"Nifty entries 9:45-{self.s.nifty_cutoff}, trend filter {'on' if self.s.nifty_trend else 'off'}, "
                 f"calls {self.s.calls_mode}, max {self.s.max_trades} trades, daily stop ₹{self.s.daily_loss_limit:,.0f}")
        self._alert(f"hello-{now.date()}", f"🟢 <b>Signal Desk is running</b> ({now:%a %d %b %H:%M})\n{dhan}\n{rules}\n"
                    f"Page: http://127.0.0.1:{self.s.port}", kind="info")

    def _auto_renew(self, now: dt.datetime) -> None:
        if not self.s.token_autorenew or not hasattr(self.api, "renew_token"):
            return
        st = self._renew
        if st["day"] != now.date():
            st.update(day=now.date(), fails=0, next=0.0, dead=False)
        if st["dead"] or st["fails"] >= RENEW_TRIES or self._mono() < st["next"]:
            return
        exp = token_expiry(self.s.dhan_access_token or "")
        if exp is None or exp <= now + dt.timedelta(minutes=2):
            return
        close_today = at(now.date(), "15:40")
        urgent = exp < close_today
        evening = now.time() >= RENEW_AT and exp - now < dt.timedelta(hours=23)
        if not (urgent or evening):
            return
        st["next"] = self._mono() + RENEW_GAP
        fail_msg = ("⚠️ <b>Dhan token auto-renewal failed</b> - paste a new token in Signal Desk → Settings "
                    "(Dhan web → Profile → DhanHQ APIs → generate token).")
        try:
            new = self.api.renew_token()
        except DhanAuthError as exc:
            st["dead"] = True
            self._err(f"Dhan token auto-renewal: {exc}")
            self._alert(f"renewfail-{now.date()}", fail_msg, kind="error")
            return
        except DhanError as exc:
            st["fails"] += 1
            self._err(f"Dhan token auto-renewal attempt {st['fails']}/{RENEW_TRIES}: {exc}")
            if st["fails"] >= RENEW_TRIES:
                self._alert(f"renewfail-{now.date()}", fail_msg + f" Current token still works till {exp:%a %d %b %H:%M}.",
                            kind="error")
            return
        new_exp = token_expiry(new)
        if new_exp is None or new_exp <= now:
            st["dead"] = True
            self._err("Dhan token auto-renewal: Dhan's reply did not contain a usable token")
            self._alert(f"renewfail-{now.date()}", fail_msg, kind="error")
            return
        try:
            self.env_writer("DHAN_ACCESS_TOKEN", new)
        except OSError as exc:
            self._err(f"Dhan token renewed but could not be saved to .env ({exc}) - it works until Signal Desk restarts")
        with self.lock:
            self.s.dhan_access_token = new
            self.api.set_token(new)
            self.token_renewed = now
        st["fails"] = 0
        log.info("Dhan token renewed automatically; new expiry %s", new_exp)
        self._alert(None, f"🔑 Dhan token renewed automatically - valid till {new_exp:%a %d %b %H:%M}.", kind="info")

    def _token_reminder(self, now: dt.datetime) -> None:
        exp = token_expiry(self.s.dhan_access_token or "")
        if exp is None or not (dt.time(8, 0) <= now.time() <= dt.time(15, 30)):
            return
        close = at(now.date(), "15:30")
        if exp <= now:
            return
        if exp < close:
            self._alert(f"tokexp-{now.date()}-{exp:%H%M}",
                        f"🔑 Your Dhan token expires today at {exp:%H:%M} - before the market closes. Generate a new one "
                        f"in Dhan web and paste it in Signal Desk → Settings (takes 1 minute).", kind="error")
            if (exp - now).total_seconds() <= 1800:
                self._alert(f"tokexp30-{now.date()}-{exp:%H%M}",
                            f"⏳ Dhan token expires in {int((exp - now).total_seconds() // 60)} min ({exp:%H:%M}). "
                            f"Paste a new one in Settings now or alerts will stop.", kind="error")

    def _offhours(self, now: dt.datetime) -> None:
        if self._offhours_loaded:
            return
        self._offhours_loaded = True
        self._update_quotes(now)
        self._update_chains(now)
        n = self._fetch_index(self.inst.nifty_id, now)
        if n:
            bnf = self._fetch_index(self.inst.banknifty_id, now) or []
            vix = self._fetch_index(self.inst.vix_id, now) or []
            with self.lock:
                self.nifty_hist, self.bnf_hist, self.vix_hist = n, bnf, vix
                self.last["candles"] = now

    def _update_quotes(self, now: dt.datetime) -> None:
        inst = self.inst
        req: dict[str, list[int]] = {"IDX_I": [inst.nifty_id, inst.banknifty_id, inst.vix_id]}
        sids = {s.security_id for s in self.stock_rules.signals if s.status in ("open", "T1") and s.security_id}
        sids |= {t.security_id for t in self.guard.trades if t.open and t.kind == "STOCK" and t.security_id}
        if sids:
            req["NSE_EQ"] = sorted(sids)
        data = self._call(self.api.ohlc, req)
        if not data:
            return
        idx = data.get("IDX_I") or {}
        with self.lock:
            for name, sid in (("NIFTY", inst.nifty_id), ("BANKNIFTY", inst.banknifty_id), ("VIX", inst.vix_id)):
                q = idx.get(str(sid)) or idx.get(sid)
                if q and q.get("last_price"):
                    self.ltp[name] = float(q["last_price"])
                    self.day_ohlc[name] = q.get("ohlc") or {}
            for sid_s, q in (data.get("NSE_EQ") or {}).items():
                if q and q.get("last_price"):
                    self.stock_ltp[int(sid_s)] = float(q["last_price"])
            self.last["quotes"] = now

    def _update_chains(self, now: dt.datetime) -> None:
        inst = self.inst
        if not self.expiries or now.minute % 30 == 0:
            ex = self._call(self.api.expiry_list, inst.nifty_id, "IDX_I")
            if ex:
                self.expiries = ex
        if not self.expiries:
            return
        near = OC.nearest_expiry(self.expiries, now.date())
        trade = self._trade_expiry(now.date())
        needed = [e for e in dict.fromkeys([near, trade] + [t.expiry for t in self.guard.trades if t.open and t.expiry]) if e][:3]
        for e in needed:
            raw = self._call(self.api.option_chain, inst.nifty_id, "IDX_I", e)
            if raw is None:
                continue
            ch = OC.parse_chain(raw, e, now)
            with self.lock:
                self.chains[e] = ch
                if e not in self.first_oi:
                    self.first_oi[e] = {r.strike: (r.ce.oi, r.pe.oi) for r in ch.rows}
                if e == near:
                    self.oi_stats = OC.stats(ch, self.first_oi.get(e))
                    if ch.spot:
                        self.ltp.setdefault("NIFTY", ch.spot)
                self.last["chain"] = now
        for t in self.guard.trades:
            if t.open and t.kind == "OPTION" and t.expiry in self.chains and t.strike:
                row = self.chains[t.expiry].row(t.strike)
                leg = (row.ce if t.option == "CE" else row.pe) if row else None
                if leg and leg.ltp:
                    t.ltp = leg.ltp

    def _fetch_index(self, sid: int, now: dt.datetime) -> list[C.Candle] | None:
        d = self._call(self.api.intraday, sid, "IDX_I", "INDEX", now.date() - dt.timedelta(days=25),
                       now.date() + dt.timedelta(days=1), 15)
        return C.from_arrays(d, 15) if d else None

    def _process_candles(self, now: dt.datetime) -> None:
        due = [b for b in _boundaries(now.date()) if b + dt.timedelta(seconds=CANDLE_DELAY) <= now]
        if not due:
            return
        b = due[-1]
        key = f"c-{b:%H%M}"
        if key in self.processed and not self.nifty_hist and self._mono() >= self._next["candles"]:
            self._next["candles"] = self._mono() + 15
            n = self._fetch_index(self.inst.nifty_id, now)
            if n:
                bnf = self._fetch_index(self.inst.banknifty_id, now) or []
                vix = self._fetch_index(self.inst.vix_id, now) or []
                with self.lock:
                    self.nifty_hist, self.bnf_hist, self.vix_hist = n, bnf, vix
                    self.last["candles"] = now
            return
        if key in self.processed or self._mono() < self._next["candles"]:
            return
        self._next["candles"] = self._mono() + 15
        n = self._fetch_index(self.inst.nifty_id, now)
        if n is None:
            return
        today_c = C.completed(C.for_day(n, now.date()), now)
        if not today_c and now >= at(now.date(), "09:45"):
            self.market_closed_today = True
            self._alert(f"closed-{now.date()}", "Market appears closed today (no Nifty candles) - Signal Desk is idle.", push=False)
            return
        if not today_c or today_c[-1].end < b:
            if (now - b).total_seconds() < CANDLE_GIVEUP:
                return
        bnf = self._fetch_index(self.inst.banknifty_id, now) or []
        vix = self._fetch_index(self.inst.vix_id, now) or []
        with self.lock:
            self.nifty_hist, self.bnf_hist, self.vix_hist = n, bnf, vix
            self.processed.add(key)
            self.last["candles"] = now
        ctx = self.ctx(now)
        if ctx is None:
            return
        for sig in self.nifty_rules.on_candle_close(ctx):
            self._announce(sig)
        if self.s.h1_paper and (b - at(now.date(), "09:15")).total_seconds() % 3600 == 0:
            self._h1_candle(now, ctx)
        self.save()
        if self.s.stock_alerts_enabled and self.inst.stocks and self.profiles:
            self._scan_stocks(now)

    def _h1_candle(self, now: dt.datetime, ctx15: MarketCtx) -> None:
        h1 = C.aggregate(ctx15.nifty, 60)
        if len(h1) < 2:
            return
        ctx = dataclasses.replace(ctx15, nifty=h1, bnf=C.aggregate(ctx15.bnf, 60))
        for sig in self.h1_rules.on_candle_close(ctx):
            sig.id = H1_PREFIX + sig.id[len("NIFTY-"):]
            sig.symbol = "NIFTY 1H (paper)"
            sig.note = (sig.note + " | " if sig.note else "") + "PAPER: 1-hour rules, still being validated"
            self._announce(sig)

    def _prev_close(self, hist: list[C.Candle], day: dt.date) -> float | None:
        prev = [c for c in hist if c.start.date() < day]
        return prev[-1].close if prev else None

    def ctx(self, now: dt.datetime) -> MarketCtx | None:
        day = now.date()
        today_c = C.completed(C.for_day(self.nifty_hist, day), now)
        prior_days = C.prior_days(self.nifty_hist, day)
        prev_hlc = C.daily_hlc(C.for_day(self.nifty_hist, prior_days[-1])) if prior_days else None
        trade_exp = self._trade_expiry(day)
        vix_prev = self._prev_close(self.vix_hist, day)
        vix_now = self.ltp.get("VIX") or (C.for_day(self.vix_hist, day)[-1].close if C.for_day(self.vix_hist, day) else None)
        lot = self.inst.nifty_lot or 65
        return MarketCtx(
            now=now, nifty=today_c, prev_hlc=prev_hlc,
            bnf=C.completed(C.for_day(self.bnf_hist, day), now), bnf_prev_close=self._prev_close(self.bnf_hist, day),
            vix=vix_now, vix_prev_close=vix_prev, oi_stats=self.oi_stats,
            trade_chain=self.chains.get(trade_exp) if trade_exp else None, lot=lot,
            budget=self.guard.budget(now, "OPTION"), blocked_reason=self.guard.blocked_reason(now), cutoff=self.s.nifty_cutoff,
            trend_filter=self.s.nifty_trend, prev_range_ratio=prev_range_ratio(self.nifty_hist, day),
        )

    def nifty_change_pct(self, now: dt.datetime) -> float | None:
        pc = self._prev_close(self.nifty_hist, now.date())
        ltp = self.ltp.get("NIFTY")
        return ((ltp - pc) / pc * 100) if pc and ltp else None

    def _build_profiles(self, today: dt.date) -> None:
        stocks = self.inst.stocks
        if self.s.stock_universe_max:
            stocks = stocks[: self.s.stock_universe_max]
        profiles: dict[int, S.DailyProfile] = {}
        cache = self.state_dir / f"profiles-{today:%Y-%m-%d}.json"
        if cache.exists():
            try:
                for d in json.loads(cache.read_text(encoding="utf-8")):
                    profiles[d["security_id"]] = S.DailyProfile(**d)
            except (OSError, ValueError, TypeError):
                profiles = {}
        if not profiles:
            for i, st in enumerate(stocks):
                d = self._call(self.api.daily, st.security_id, "NSE_EQ", "EQUITY", today - dt.timedelta(days=45), today)
                if self.token_ok is False:
                    self.profiles_status = "stopped: Dhan token invalid"
                    return
                if d:
                    days = [c for c in C.from_arrays(d, 1440) if c.start.date() < today]
                    p = S.daily_profile(st.symbol, st.security_id, days)
                    if p:
                        profiles[st.security_id] = p
                self.profiles_status = f"building {i + 1}/{len(stocks)}"
            try:
                self.state_dir.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps([p.to_json() for p in profiles.values()]), encoding="utf-8")
            except OSError:
                pass
        with self.lock:
            self.profiles = profiles
            self.watch = S.watchlist(list(profiles.values()), 10)
            self.profiles_status = f"ready ({len(profiles)} stocks)"

    def _send_watchlist(self, now: dt.datetime) -> None:
        lines = [f"<b>📋 Pre-market watchlist {now:%a %d %b}</b> (from yesterday's charts)"]
        for p in self.watch:
            trig = (f"buy &gt; {p.prev_high:,.2f}" if p.bias == "LONG" else f"sell &lt; {p.prev_low:,.2f}" if p.bias == "SHORT"
                    else f"&gt; {p.prev_high:,.2f} / &lt; {p.prev_low:,.2f}")
            lines.append(f"• <b>{p.symbol}</b> - {p.setup}; {trig}")
        lines.append("Triggers count only on a 15-min close after 9:30, with volume. Alerts will follow.")
        self._alert("premarket", "\n".join(lines), kind="watchlist")

    def _scan_stocks(self, now: dt.datetime) -> None:
        ids = list(self.profiles.keys())
        quotes: dict[int, dict] = {}
        for i in range(0, len(ids), 900):
            data = self._call(self.api.ohlc, {"NSE_EQ": ids[i:i + 900]})
            for sid_s, q in ((data or {}).get("NSE_EQ") or {}).items():
                o = q.get("ohlc") or {}
                quotes[int(sid_s)] = {"ltp": q.get("last_price"), "high": o.get("high"), "low": o.get("low")}
        short = S.shortlist(quotes, self.profiles)
        nchg = self.nifty_change_pct(now)
        new: list[Signal] = []
        for sid in short:
            prof = self.profiles[sid]
            d = self._call(self.api.intraday, sid, "NSE_EQ", "EQUITY", now.date() - dt.timedelta(days=8),
                           now.date() + dt.timedelta(days=1), 15)
            if not d:
                continue
            allc = C.from_arrays(d, 15)
            today_c = C.completed(C.for_day(allc, now.date()), now)
            hist = [c for c in allc if c.start.date() < now.date()]
            sig = self.stock_rules.evaluate(prof.symbol, today_c, hist, prof, nchg, now, self.guard.budget(now, "STOCK"),
                                            self.guard.blocked_reason(now), self.s.entry_cutoff)
            if sig:
                sig.security_id = sid
                new.append(sig)
        with self.lock:
            self.last["scan"] = now
        for sig in S.rank(new)[:3]:
            self._announce(sig)
        self.save()

    def _announce(self, sig: Signal) -> None:
        if sig.kind == "STOCK" and not self.s.stock_alerts_enabled:
            return
        blocked = any(c.name == "Risk guard" and c.status == "fail" for c in sig.checks)
        push = ((sig.action == "TAKE" or (blocked and sig.kind == "NIFTY")) and grade_ok(sig.grade, self.s.min_grade)) \
            or self.s.min_grade == "C"
        sig.notified = push
        ce_paper = (sig.kind == "NIFTY" and sig.side == "LONG" and not sig.id.startswith(H1_PREFIX)
                    and self.s.calls_mode == "paper")
        if ce_paper:
            sig.symbol = "NIFTY (paper CE)"
        text = format_signal(sig)
        if sig.kind == "NIFTY" and sig.expiry and self.expiries:
            near = OC.nearest_expiry(self.expiries, sig.time.date())
            if near == sig.time.date().isoformat() and str(sig.expiry)[:10] != near:
                text = text.replace("📅 <b>Expiry:", "📅 <b>Expiry (NOT today's expiring contract):", 1)
            elif str(sig.expiry)[:10] == sig.time.date().isoformat():
                text = text.replace("📅 <b>Expiry:", "⚠️ <b>0DTE - this contract EXPIRES TODAY.</b> Premium can lose 50-100% "
                                    "in minutes and our tests never covered it: 1 lot only, stop placed at once, out by 14:30.\n📅 <b>Expiry:", 1)
        if ce_paper:
            text = ("📝 <b>PAPER ONLY - call (CE) signal</b>. In a year of testing, calls broke even while puts made the "
                    "profit, so CE alerts are tracked on paper. Note it, don't trade real money.\n" + text)
        if sig.id.startswith(H1_PREFIX):
            text = text.replace("15-min close", "1-hour close")
            text = ("📝 <b>PAPER ONLY - 1-hour breakout test</b> (not validated yet: note it, don't trade real money)\n" + text)
        self._alert(f"sig-{sig.id}", text, kind="signal" if push else "skip", push=push)

    def _signal_sid(self, sig: Signal) -> int | None:
        if sig.security_id:
            return sig.security_id
        st = self.inst.by_symbol().get(sig.symbol) if self.inst else None
        return st.security_id if st else None

    def _live_checks(self, now: dt.datetime) -> None:
        nifty = self.ltp.get("NIFTY")
        msgs: list[tuple[str, str]] = []
        with self.lock:
            for sig in self._all_signals():
                if sig.time.date() != now.date() or sig.status not in ("open", "T1"):
                    continue
                if sig.kind == "STOCK" and not self.s.stock_alerts_enabled:
                    continue
                px = nifty if sig.kind == "NIFTY" else self.stock_ltp.get(self._signal_sid(sig) or -1)
                ev = tracker.update(sig, px, now) if px else None
                if ev and sig.notified:
                    word = {"T1": "🎯 hit T1", "SL": "🔻 hit stop", "T2": "🎯🎯 hit T2"}[ev]
                    msgs.append((f"ev-{sig.id}-{ev}", f"{sig.symbol} {sig.side} ({sig.time:%H:%M}) {word} - "
                                 f"{'Nifty' if sig.kind == 'NIFTY' else 'price'} {px:,.2f}"))
            for t in self.guard.trades:
                if t.open and t.kind == "STOCK" and t.security_id in self.stock_ltp:
                    t.ltp = self.stock_ltp[t.security_id]
                for m in self.guard.trade_alerts(t):
                    msgs.append((None, m))
            lim = self.guard.check_limits(now)
            if lim:
                msgs.append((None, lim))
            ctx = self.ctx(now) if nifty and self.nifty_hist else None
        for key, m in msgs:
            self._alert(key, m, kind="trade")
        if ctx and nifty:
            msg = self.nifty_rules.approach(nifty, ctx)
            if msg:
                self._alert(None, msg, kind="headsup")

    def _reminders(self, now: dt.datetime) -> None:
        d = now.date()
        open_trades = [t for t in self.guard.trades if t.open]
        if now >= at(d, self.s.exit_reminder) and open_trades:
            self._alert(f"exit-{d}", f"⏰ {self.s.exit_reminder} - exit open positions now ({', '.join(t.symbol for t in open_trades)}). "
                        f"Don't hold option buys into the close or the weekend.", kind="trade")
        if now >= at(d, self.s.squareoff) and open_trades:
            self._alert(f"sq-{d}", f"🛑 {self.s.squareoff} square-off time. Close: {', '.join(t.symbol for t in open_trades)}.", kind="trade")
        if now >= at(d, "15:31") and f"eod-{d}" not in self.sent:
            self._end_of_day(now)

    def _end_of_day(self, now: dt.datetime) -> None:
        sigs = [s for s in self._all_signals() if s.time.date() == now.date()]
        for s in sigs:
            px = self.ltp.get("NIFTY") if s.kind == "NIFTY" else self.stock_ltp.get(self._signal_sid(s) or -1)
            tracker.close_day(s, px, now)
        tracker.append_journal(self.state_dir / "journal.csv", sigs)
        real = [s for s in sigs if not _is_paper(s)]
        paper = [s for s in sigs if _is_paper(s)]
        sm = tracker.summary(real)
        pnl = self.guard.day_pnl(now.date())
        lines = [f"<b>📊 End of day {now:%a %d %b}</b>",
                 f"Signals {sm['signals']} (TAKE {sm['take']}): T1 {sm['t1']}, stop {sm['sl']}, timed out {sm['timeout']}, net {sm['net_r']:+.1f}R",
                 f"Your logged trades: {len(self.guard.real_today(now.date()))}, P&L ₹{pnl:,.0f}"
                 + (f" (+ paper ₹{self.guard.paper_pnl(now.date()):,.0f})" if self.guard.paper_pnl(now.date()) else ""),
                 *([f"PAPER signals (1-hour / CE) {len(paper)} (TAKE {sum(1 for x in paper if x.action == 'TAKE')}), "
                    f"net {tracker.summary(paper)['net_r']:+.1f}R - not counted above"] if paper else [])]
        for s in sigs:
            lines.append(f"• {s.time:%H:%M} {s.symbol} {s.side} {s.grade}/{s.action} → {s.status}"
                         + (f" ({tracker.result_r(s):+.1f}R)" if tracker.result_r(s) is not None else ""))
        lines.append("Saved to signal_desk/state/journal.csv")
        self._alert(f"eod-{now.date()}", "\n".join(lines), kind="eod")

    def add_trade(self, d: dict) -> Trade:
        now = self.clock.now()
        kind = d.get("kind", "OPTION").upper()
        qty = int(float(d["qty"]))
        entry = float(d["entry"])
        t = Trade(id=f"T{now:%H%M%S}", kind=kind, symbol="", side=d.get("side", "BUY").upper(), entry=entry, qty=qty,
                  opened_at=now, sl=float(d["sl"]) if d.get("sl") not in (None, "") else None,
                  target=float(d["target"]) if d.get("target") not in (None, "") else None, signal_id=d.get("signal_id") or None,
                  paper=bool(d.get("paper")))
        if kind == "OPTION":
            t.strike = float(d["strike"])
            t.option = d.get("option", "CE").upper()
            t.expiry = d.get("expiry") or self._trade_expiry(now.date())
            t.symbol = f"NIFTY {t.strike:,.0f} {t.option} {t.expiry or ''}".strip()
            ch = self.chains.get(t.expiry or "")
            row = ch.row(t.strike) if ch else None
            leg = (row.ce if t.option == "CE" else row.pe) if row else None
            t.ltp = leg.ltp if leg else None
        else:
            sym = d["symbol"].upper().strip()
            st = self.inst.by_symbol().get(sym) if self.inst else None
            t.symbol, t.security_id = sym, (st.security_id if st else None)
            t.ltp = self.stock_ltp.get(t.security_id or -1)
        with self.lock:
            self.guard.trades.append(t)
        self.save()
        self._next["quotes"] = 0
        self._next["chain"] = 0
        return t

    def set_trade_paper(self, trade_id: str, paper: bool) -> Trade | None:
        with self.lock:
            t = next((x for x in self.guard.trades if x.id == trade_id), None)
            if t:
                t.paper = bool(paper)
        self.save()
        return t

    def delete_trade(self, trade_id: str) -> bool:
        with self.lock:
            n = len(self.guard.trades)
            self.guard.trades = [x for x in self.guard.trades if x.id != trade_id]
            ok = len(self.guard.trades) < n
        self.save()
        return ok

    def close_trade(self, trade_id: str, exit_price: float) -> Trade | None:
        with self.lock:
            for t in self.guard.trades:
                if t.id == trade_id and t.open:
                    t.exit, t.closed_at = float(exit_price), self.clock.now()
                    self.save()
                    return t
        return None

    def update_trade(self, trade_id: str, sl: float | None, target: float | None) -> Trade | None:
        with self.lock:
            for t in self.guard.trades:
                if t.id == trade_id:
                    if sl is not None:
                        t.sl, t.sl_alerted = sl, False
                    if target is not None:
                        t.target, t.tgt_alerted = target, False
                    self.save()
                    return t
        return None

    def update_risk(self, d: dict) -> None:
        fields = {"daily_loss_warn": ("SD_DAILY_LOSS_WARN", float), "daily_loss_limit": ("SD_DAILY_LOSS_LIMIT", float),
                  "max_trades": ("SD_MAX_TRADES", int), "stock_risk": ("SD_STOCK_RISK", float),
                  "entry_cutoff": ("SD_ENTRY_CUTOFF", str), "nifty_cutoff": ("SD_NIFTY_CUTOFF", str)}
        for k, (env, typ) in fields.items():
            if d.get(k) in (None, ""):
                continue
            v = typ(d[k])
            if typ is str:
                h, m = map(int, str(v).split(":"))
                v = f"{h:02d}:{m:02d}"
            setattr(self.s, k, v)
            write_env_value(env, str(v))
        with self.lock:
            g = self.guard
            g.warn, g.limit, g.max_trades = self.s.daily_loss_warn, self.s.daily_loss_limit, self.s.max_trades
            g.stock_risk, g.cutoff = self.s.stock_risk, self.s.entry_cutoff
        self.save()

    def set_toggles(self, d: dict) -> None:
        for k in ("alerts_enabled", "stock_alerts_enabled"):
            if k in d:
                setattr(self.s, k, bool(d[k]))
        if d.get("expiry_day_contract") in ("next", "today"):
            self.s.expiry_day_contract = d["expiry_day_contract"]
            write_env_value("SD_EXPIRY_DAY_CONTRACT", d["expiry_day_contract"])
        if d.get("calls_mode") in ("live", "paper"):
            self.s.calls_mode = d["calls_mode"]
            write_env_value("SD_CALLS_MODE", d["calls_mode"])
        for k, env in (("nifty_trend", "SD_NIFTY_TREND"), ("h1_paper", "SD_H1_PAPER"), ("stock_alerts_enabled", "SD_STOCK_ALERTS")):
            if k in d:
                setattr(self.s, k, bool(d[k]))
                write_env_value(env, "1" if d[k] else "0")
        if d.get("min_grade") in ("A", "B", "C"):
            self.s.min_grade = d["min_grade"]
        self.save()

    def update_dhan_token(self, token: str, client_id: str | None = None) -> None:
        token = token.strip()
        write_env_value("DHAN_ACCESS_TOKEN", token)
        self.s.dhan_access_token = token
        if client_id:
            write_env_value("DHAN_CLIENT_ID", client_id.strip())
            self.s.dhan_client_id = client_id.strip()
            self.api.client_id = client_id.strip()
        self.api.set_token(token)
        self._renew.update(day=None, fails=0, next=0.0, dead=False)
        self.token_ok = None
        self.sent.discard(f"token-{self.day}")
        self._offhours_loaded = False

        def verify() -> None:
            if self.inst is not None:
                self._update_quotes(self.clock.now())
        threading.Thread(target=verify, daemon=True).start()
        if self.profiles_status.startswith("stopped"):
            self.profiles_status = "not started"
        self._next = {"quotes": 0.0, "chain": 0.0, "candles": 0.0}

    def run_backtest(self, days: int = 60, wait: bool = False) -> None:
        from . import backtest as BT

        if self.backtest.get("status") == "running":
            return

        def job() -> None:
            self.backtest = {"status": "running", "days": days}
            try:
                inst = self.inst or self._load_instruments(self.clock.now().date())
                res = BT.run(self.api, inst.nifty_id, inst.banknifty_id, inst.vix_id, self.clock.now().date(), days=days,
                             lot=inst.nifty_lot or 65, budget=self.s.daily_loss_limit, cutoff=self.s.nifty_cutoff,
                             squareoff=self.s.squareoff, max_per_day=self.s.max_trades, trend=self.s.nifty_trend)
                res["status"] = "done"
                res["ran_at"] = self.clock.now().strftime("%d %b %H:%M")
                self.backtest = res
                try:
                    (self.state_dir / "backtest.json").write_text(json.dumps(res, default=str), encoding="utf-8")
                except OSError:
                    pass
            except Exception as exc:
                self.backtest = {"status": f"failed: {exc}"}
        if wait:
            job()
        else:
            threading.Thread(target=job, daemon=True).start()

    def run_research(self, days: int = 250, wait: bool = False) -> None:
        import importlib

        from . import research as RS
        RS = importlib.reload(RS)

        if self.research.get("status", "").startswith(("running", "fetching", "testing")):
            return

        def job() -> None:
            self.research = {"status": "running - fetching history"}

            def say(m: str) -> None:
                self.research = {"status": f"running - {m}"}
            try:
                inst = self.inst or self._load_instruments(self.clock.now().date())
                res = RS.run(self.api, inst.nifty_id, inst.banknifty_id, inst.vix_id, self.clock.now().date(), days=days,
                             lot=inst.nifty_lot or 65, squareoff=self.s.squareoff, progress=say)
                res["ran_at"] = self.clock.now().strftime("%d %b %H:%M")
                self.research = res
                try:
                    (self.state_dir / "research.json").write_text(json.dumps(res, default=str), encoding="utf-8")
                except OSError:
                    pass
            except Exception as exc:
                self.research = {"status": f"failed: {exc}"}
        if wait:
            job()
        else:
            threading.Thread(target=job, daemon=True).start()

    def run_stock_research(self, days: int = 250, wait: bool = False, capital: float = 30000.0) -> str:
        import importlib

        from . import stock_research as SR
        SR = importlib.reload(SR)
        now = self.clock.now()
        if is_weekday(now.date()) and dt.time(9, 0) <= now.time() <= dt.time(15, 35) and not wait:
            return "Please run stock testing after 3:35 PM - during market hours Dhan's data limits are kept for live alerts."
        if str(self.stock_research.get("status", "")).startswith("running"):
            return "Already running."

        def job() -> None:
            self.stock_research = {"status": "running - starting"}

            def say(m: str) -> None:
                self.stock_research = {"status": f"running - {m}"}
            try:
                inst = self.inst or self._load_instruments(self.clock.now().date())
                res = SR.run(self.api, inst.stocks, inst.nifty_id, self.clock.now().date(), days=days, progress=say,
                             capital=capital)
                res["ran_at"] = self.clock.now().strftime("%d %b %H:%M")
                self.stock_research = res
                try:
                    (self.state_dir / "stock_research.json").write_text(json.dumps(res, default=str), encoding="utf-8")
                except OSError:
                    pass
            except Exception as exc:
                self.stock_research = {"status": f"failed: {exc}"}
        if wait:
            job()
        else:
            threading.Thread(target=job, daemon=True).start()
        return "Stock testing started - first run downloads about a year of candles for every F&O stock (10-25 min). Leave the page open."

    def save_telegram(self, bot_token: str | None, chat_id: str | None) -> None:
        if bot_token:
            write_env_value("TELEGRAM_BOT_TOKEN", bot_token.strip())
            self.s.telegram_bot_token = self.notifier.bot_token = bot_token.strip()
        if chat_id:
            write_env_value("TELEGRAM_CHAT_ID", chat_id.strip())
            self.s.telegram_chat_id = self.notifier.chat_id = chat_id.strip()

    def snapshot(self) -> dict:
        now = self.clock.now()
        with self.lock:
            ctx = self.ctx(now) if self.inst else None
            lv = levels(ctx) if ctx else {}
            near = OC.nearest_expiry(self.expiries, now.date()) if self.expiries else None
            trade_exp = self._trade_expiry(now.date())
            chain = self.chains.get(near or "")
            table = OC.chain_table(chain, chain.spot or self.ltp.get("NIFTY", 0)) if chain else []
            chains_out = []
            for role, e in (("oi", near), ("trade", trade_exp)):
                if not e or any(c["expiry"] == e for c in chains_out):
                    if e and chains_out:
                        chains_out[0]["role"] = "both"
                    continue
                ch = self.chains.get(e)
                if ch:
                    chains_out.append({"expiry": e, "role": role, "label": expiry_label(e, now.date()),
                                       "rows": OC.chain_table(ch, ch.spot or self.ltp.get("NIFTY", 0)),
                                       "atm": OC.nearest_strike(ch, ch.spot or self.ltp.get("NIFTY", 0)) if ch.rows else None})
            prev_close = lv.get("pdc")
            nifty = self.ltp.get("NIFTY")
            plan = self._plan_text(lv, ctx, now)
            sigs = sorted(self._all_signals(), key=lambda s: s.time, reverse=True)
            sigs = [s for s in sigs if s.time.date() == now.date() and (s.kind != "STOCK" or self.s.stock_alerts_enabled)]
            out = {
                "now": now.isoformat(), "day": str(self.day),
                "status": {
                    "token_ok": self.token_ok,
                    "token_expires": (lambda e: e.strftime("%a %d %b %H:%M") if e else None)(token_expiry(self.s.dhan_access_token or "")),
                    "token_expired": (lambda e: bool(e and e <= now))(token_expiry(self.s.dhan_access_token or "")),
                    "token_autorenew": self.s.token_autorenew,
                    "token_renewed": self.token_renewed.strftime("%a %H:%M") if self.token_renewed else None,
                    "autostart": self.autostart, "telegram": self.notifier.configured, "telegram_error": self.notifier.last_error,
                    "market_open": is_weekday(now.date()) and not holiday_name(now.date(), self.s.holidays_extra)
                    and dt.time(9, 15) <= now.time() <= dt.time(15, 30),
                    "holiday": holiday_name(now.date(), self.s.holidays_extra),
                    "market_closed_today": self.market_closed_today,
                    "last_quotes": fmt_t(self.last["quotes"]), "last_chain": fmt_t(self.last["chain"]),
                    "last_candles": fmt_t(self.last["candles"]), "last_scan": fmt_t(self.last["scan"]),
                    "profiles": self.profiles_status, "api_calls": getattr(self.api, "calls", 0),
                    "instruments": self.inst.source if self.inst else "-", "errors": self.errors[:8],
                },
                "nifty": {
                    "ltp": nifty, "chg": (nifty - prev_close) if nifty and prev_close else None,
                    "chg_pct": ((nifty - prev_close) / prev_close * 100) if nifty and prev_close else None,
                    "levels": lv, "near_expiry": near, "trade_expiry": trade_exp, "lot": self.inst.nifty_lot if self.inst else None,
                    "plan": plan, "chain": table, "chains": chains_out,
                    "near_label": expiry_label(near, now.date()) if near else None,
                    "trade_label": expiry_label(trade_exp, now.date()) if trade_exp else None, "atm": self.oi_stats.atm if self.oi_stats else None,
                },
                "banknifty": {"ltp": self.ltp.get("BANKNIFTY"), "prev_close": self._prev_close(self.bnf_hist, now.date())},
                "vix": {"ltp": self.ltp.get("VIX"), "prev_close": self._prev_close(self.vix_hist, now.date())},
                "signals": [s.to_json() for s in sigs],
                "summary": tracker.summary(sigs),
                "watchlist": [p.to_json() for p in self.watch],
                "trades": [t.to_json() for t in self.guard.today(now.date())],
                "risk": {
                    "pnl": round(self.guard.day_pnl(now.date()), 2), "paper_pnl": round(self.guard.paper_pnl(now.date()), 2),
                    "paper_trades": sum(1 for t in self.guard.today(now.date()) if t.paper), "budget": self.guard.budget(now, "OPTION"),
                    "stock_budget": self.guard.budget(now, "STOCK"),
                    "blocked": self.guard.blocked_reason(now), "warn": self.s.daily_loss_warn, "limit": self.s.daily_loss_limit,
                    "max_trades": self.s.max_trades, "trades": len(self.guard.real_today(now.date())),
                },
                "feed": list(self.notifier.feed)[:60],
                "backtest": self.backtest,
                "research": self.research,
                "stock_research": self.stock_research,
                "settings": self.s.public(),
            }
        return out

    def _plan_text(self, lv: dict, ctx: MarketCtx | None, now: dt.datetime) -> str:
        if not ctx or not ctx.nifty:
            return "Waiting for the first 15-min candle (9:15-9:30)."
        if ctx.blocked_reason:
            return f"No new trades: {ctx.blocked_reason}."
        if "long_trigger" not in lv:
            return "Opening range set; triggers start after the 9:30 candle." if len(ctx.nifty) < 2 else "Past the entry cutoff - manage open trades only."
        p = [f"CE if a 15-min candle closes above {lv['long_trigger']:,.1f}",
             f"PE if it closes below {lv['short_trigger']:,.1f}"]
        if lv.get("call_wall"):
            p.append(f"call wall {lv['call_wall']:,.0f}, put wall {lv.get('put_wall') or 0:,.0f}")
        return "; ".join(p) + "."
