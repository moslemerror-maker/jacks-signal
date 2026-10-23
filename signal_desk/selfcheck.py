from __future__ import annotations

import datetime as dt
import traceback

from . import candles as C
from . import instruments as I
from . import optionchain as OC
from .clock import IST
from .config import ENV_PATH, Settings
from .dhan_api import DhanApi, DhanAuthError
from .notify import Notifier


def _p(ok: bool | None, step: str, detail: str = "") -> bool:
    tag = "PASS" if ok else ("WARN" if ok is None else "FAIL")
    print(f"[{tag}] {step}" + (f" - {detail}" if detail else ""))
    return bool(ok)


def run(send_telegram: bool = False) -> int:
    s = Settings.load()
    today = dt.datetime.now(IST).date()
    fails = 0
    print(f"Signal Desk self-check  ({dt.datetime.now(IST):%Y-%m-%d %H:%M} IST)\n.env: {ENV_PATH} ({'found' if ENV_PATH.exists() else 'MISSING'})")
    fails += not _p(bool(s.dhan_client_id), "DHAN_CLIENT_ID present")
    fails += not _p(bool(s.dhan_access_token), "DHAN_ACCESS_TOKEN present")
    from .config import autostart_status, token_expiry
    exp = token_expiry(s.dhan_access_token or "")
    if exp:
        left = exp - dt.datetime.now(IST)
        _p(left.total_seconds() > 0 or None, "Dhan token expiry", f"valid till {exp:%a %d %b %H:%M} IST"
           + ("" if left.total_seconds() > 0 else " - EXPIRED, paste a new one in Settings"))
    au = autostart_status()
    _p(au["ok"], "Weekday autostart (Windows scheduled task)", au["text"])
    _p(bool(s.telegram_bot_token and s.telegram_chat_id) or None, "Telegram configured",
       "" if s.telegram_bot_token and s.telegram_chat_id else "optional now - set it in the UI Settings later")

    inst = I.load(today)
    ok = inst.source.startswith("dhan")
    fails += not _p(ok, "Dhan scrip master", f"NIFTY id {inst.nifty_id}, BANKNIFTY {inst.banknifty_id}, VIX {inst.vix_id}, "
                    f"Nifty lot {inst.nifty_lot or '?'}, F&O stocks {len(inst.stocks)}")
    for n in inst.notes:
        print("       note:", n)

    api = DhanApi(s.dhan_client_id, s.dhan_access_token)
    try:
        q = api.ohlc({"IDX_I": [inst.nifty_id, inst.banknifty_id, inst.vix_id]})
        idx = q.get("IDX_I") or {}
        got = {k: v.get("last_price") for k, v in idx.items()}
        fails += not _p(bool(got), "Index quotes (/marketfeed/ohlc)", str(got))
    except DhanAuthError as exc:
        _p(False, "Dhan login", f"{exc}. Generate a new access token in Dhan web and put it in .env (or the UI).")
        return 1
    except Exception as exc:
        fails += not _p(False, "Index quotes", str(exc))

    try:
        d = api.intraday(inst.nifty_id, "IDX_I", "INDEX", today - dt.timedelta(days=8), today + dt.timedelta(days=1), 15)
        cs = C.from_arrays(d, 15)
        last = cs[-1] if cs else None
        fails += not _p(bool(cs), "Nifty 15-min candles (/charts/intraday)",
                        f"{len(cs)} candles, last {last.start:%a %d %b %H:%M} close {last.close:,.2f}" if last else "none")
    except Exception as exc:
        fails += not _p(False, "Nifty 15-min candles", str(exc))

    try:
        ex = api.expiry_list(inst.nifty_id, "IDX_I")
        near, trade = OC.nearest_expiry(ex, today), OC.choose_expiry(ex, today)
        fails += not _p(bool(ex), "Option expiries", f"nearest {near}, would trade {trade}")
        raw = api.option_chain(inst.nifty_id, "IDX_I", near)
        ch = OC.parse_chain(raw, near)
        st = OC.stats(ch)
        has_prev = any(r.ce.prev_oi is not None for r in ch.rows)
        fails += not _p(bool(ch.rows), "Option chain (/optionchain)",
                        f"spot {ch.spot:,.2f}, {len(ch.rows)} strikes, call wall {st.call_wall}, put wall {st.put_wall}, "
                        f"PCR {(st.pcr or 0):.2f}, max pain {st.max_pain}, previous_oi field {'present' if has_prev else 'absent (OI change tracked from first snapshot)'}"
                        if ch.rows else "empty")
    except Exception as exc:
        fails += not _p(False, "Option chain", str(exc))

    if inst.stocks:
        st0 = inst.by_symbol().get("RELIANCE") or inst.stocks[0]
        try:
            d = api.daily(st0.security_id, "NSE_EQ", "EQUITY", today - dt.timedelta(days=45), today)
            n = len(d.get("close", []))
            fails += not _p(n > 5, f"Daily candles ({st0.symbol})", f"{n} days via {api.daily_mode}")
        except Exception as exc:
            fails += not _p(False, "Daily candles", str(exc))

    if send_telegram and s.telegram_bot_token and s.telegram_chat_id:
        ok = Notifier(s.telegram_bot_token, s.telegram_chat_id).send("✅ Signal Desk self-check: Telegram works.")
        fails += not _p(ok, "Telegram test message")
    print(f"\nAPI calls used: {api.calls}.  " + ("All good - start it with start_signal_desk.bat" if not fails else f"{fails} problem(s) above."))
    return 1 if fails else 0


def main(argv: list[str]) -> int:
    try:
        return run(send_telegram="--telegram" in argv)
    except Exception:
        traceback.print_exc()
        return 2
