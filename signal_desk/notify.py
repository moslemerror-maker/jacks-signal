from __future__ import annotations

import datetime as dt
import logging
import re
import threading
from collections import deque

import requests

from .clock import IST

log = logging.getLogger("signal_desk.notify")


def _plain(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html)


class Notifier:
    def __init__(self, bot_token: str = "", chat_id: str = "", session: requests.Session | None = None) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._session = session or requests.Session()
        self.feed: deque[dict] = deque(maxlen=200)
        self.last_error: str | None = None
        self.sent = 0
        self._lock = threading.Lock()
        self.clock = None

    @property
    def configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send(self, html: str, kind: str = "info", push: bool = True) -> bool:
        now = self.clock.now() if self.clock else dt.datetime.now(IST)
        item = {"time": now.strftime("%H:%M:%S"), "kind": kind, "text": _plain(html), "telegram": False}
        with self._lock:
            self.feed.appendleft(item)
        if not push or not self.configured:
            return False
        ok = self._post(html)
        item["telegram"] = ok
        return ok

    def _post(self, html: str) -> bool:
        try:
            r = self._session.post(f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                                   json={"chat_id": self.chat_id, "text": html[:4000], "parse_mode": "HTML",
                                         "disable_web_page_preview": True}, timeout=15)
            if r.status_code == 200:
                self.sent += 1
                self.last_error = None
                return True
            r = self._session.post(f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                                   json={"chat_id": self.chat_id, "text": _plain(html)[:4000]}, timeout=15)
            if r.status_code == 200:
                self.sent += 1
                self.last_error = None
                return True
            self.last_error = f"Telegram HTTP {r.status_code}: {r.text[:150]}"
        except requests.RequestException as exc:
            self.last_error = f"Telegram error: {exc}"
        log.warning(self.last_error)
        return False

    def find_chat_id(self) -> tuple[str | None, str]:
        if not self.bot_token:
            return None, "Set TELEGRAM_BOT_TOKEN first."
        try:
            r = self._session.get(f"https://api.telegram.org/bot{self.bot_token}/getUpdates", timeout=15)
            data = r.json()
        except (requests.RequestException, ValueError) as exc:
            return None, f"Could not reach Telegram: {exc}"
        if not data.get("ok"):
            return None, f"Telegram said: {data.get('description', 'error')} (check the bot token)"
        for upd in reversed(data.get("result", [])):
            msg = upd.get("message") or upd.get("channel_post") or {}
            chat = msg.get("chat") or {}
            if chat.get("id"):
                return str(chat["id"]), f"Found chat with {chat.get('first_name') or chat.get('title') or 'you'}"
        return None, "No messages found. Open your bot in Telegram, press Start / send 'hi', then try again."
