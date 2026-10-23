from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .config import PKG_DIR

log = logging.getLogger("signal_desk.server")
INDEX = PKG_DIR / "web" / "index.html"


def make_handler(_engine: Any):
    class Handler(BaseHTTPRequestHandler):
        engine = _engine

        def log_message(self, fmt, *args):
            log.debug(fmt, *args)

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: Any, code: int = 200) -> None:
            self._send(code, json.dumps(obj, default=str).encode())

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return {}

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
            elif self.path.startswith("/api/state"):
                self._json(self.engine.snapshot())
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            d = self._body()
            try:
                if self.path == "/api/trade":
                    t = self.engine.add_trade(d)
                    self._json({"ok": True, "trade": t.to_json()})
                elif self.path == "/api/trade/close":
                    t = self.engine.close_trade(d["id"], float(d["exit"]))
                    self._json({"ok": bool(t)})
                elif self.path == "/api/trade/paper":
                    t = self.engine.set_trade_paper(d["id"], bool(d.get("paper")))
                    self._json({"ok": bool(t)})
                elif self.path == "/api/trade/delete":
                    self._json({"ok": self.engine.delete_trade(d["id"])})
                elif self.path == "/api/trade/update":
                    f = lambda k: float(d[k]) if d.get(k) not in (None, "") else None
                    t = self.engine.update_trade(d["id"], f("sl"), f("target"))
                    self._json({"ok": bool(t)})
                elif self.path == "/api/toggles":
                    self.engine.set_toggles(d)
                    self._json({"ok": True})
                elif self.path == "/api/token":
                    if not d.get("token"):
                        raise ValueError("token is empty")
                    self.engine.update_dhan_token(d["token"], d.get("client_id"))
                    self._json({"ok": True, "msg": "Token saved - checking it with Dhan now (watch the Dhan badge at the top)."})
                elif self.path == "/api/telegram":
                    self.engine.save_telegram(d.get("bot_token"), d.get("chat_id"))
                    self._json({"ok": True})
                elif self.path == "/api/telegram/find":
                    cid, msg = self.engine.notifier.find_chat_id()
                    if cid:
                        self.engine.save_telegram(None, cid)
                    self._json({"ok": bool(cid), "chat_id": cid, "msg": msg})
                elif self.path == "/api/risk":
                    self.engine.update_risk(d)
                    self._json({"ok": True, "msg": "Saved."})
                elif self.path == "/api/backtest":
                    self.engine.run_backtest(int(d.get("days") or 60))
                    self._json({"ok": True, "msg": "Backtest started - results appear in ~1 minute."})
                elif self.path == "/api/research":
                    self.engine.run_research(int(d.get("days") or 250))
                    self._json({"ok": True, "msg": "Rule testing started - takes 2-4 minutes (it fetches about a year of candles)."})
                elif self.path == "/api/stock_research":
                    msg = self.engine.run_stock_research(int(d.get("days") or 250), capital=float(d.get("capital") or 30000))
                    self._json({"ok": True, "msg": msg})
                elif self.path == "/api/shutdown":
                    self._json({"ok": True, "msg": "stopping"})
                    import threading
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                elif self.path == "/api/telegram/test":
                    ok = self.engine.notifier.send("✅ <b>Signal Desk</b> test alert - Telegram is connected.", kind="info")
                    self._json({"ok": ok, "msg": "Sent - check Telegram." if ok else (self.engine.notifier.last_error or "Telegram not configured")})
                else:
                    self._send(404, b"not found", "text/plain")
            except (KeyError, ValueError, TypeError) as exc:
                self._json({"ok": False, "msg": f"bad input: {exc}"}, 400)

    return Handler


def serve(engine: Any, port: int) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(engine))
    return httpd
