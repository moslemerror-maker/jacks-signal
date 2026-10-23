from __future__ import annotations

import dataclasses
import os
import re
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"
STATE_DIR = PKG_DIR / "state"
CACHE_DIR = PKG_DIR / "cache"


def token_expiry(token: str):
    import base64
    import datetime as dt
    import json
    try:
        part = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
        exp = float(claims["exp"])
    except Exception:
        return None
    return dt.datetime.fromtimestamp(exp, dt.timezone(dt.timedelta(hours=5, minutes=30)))


def autostart_status() -> dict:
    import subprocess
    import sys
    if not sys.platform.startswith("win"):
        return {"ok": None, "text": "not Windows"}
    try:
        r = subprocess.run(["schtasks", "/query", "/tn", "Signal Desk", "/fo", "CSV", "/nh"], capture_output=True,
                           text=True, timeout=15, creationflags=0x08000000)
    except Exception as exc:
        return {"ok": None, "text": f"could not check ({exc})"}
    if r.returncode != 0:
        return {"ok": False, "text": "NOT installed - run signal_desk\\install_autostart.bat"}
    import csv
    rows = [x for x in csv.reader(r.stdout.splitlines()) if len(x) >= 3]
    if not rows:
        return {"ok": None, "text": "registered (details unavailable)"}
    nxt, status = rows[0][1], rows[0][2]
    return {"ok": status.strip().lower() != "disabled", "text": f"next run {nxt} ({status})"}


def read_env(path: Path = ENV_PATH) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.strip().strip('"').strip("'")
        out[k.strip()] = v
    return out


def write_env_value(key: str, value: str, path: Path = ENV_PATH) -> None:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines() if path.exists() else []
    pat = re.compile(rf"^\s*#?\s*{re.escape(key)}\s*=")
    done = False
    for i, line in enumerate(lines):
        if pat.match(line):
            lines[i] = f"{key}={value}"
            done = True
            break
    if not done:
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _get(env: dict[str, str], key: str, default: str = "") -> str:
    return os.environ.get(key) or env.get(key) or default


@dataclasses.dataclass
class Settings:
    dhan_client_id: str = ""
    dhan_access_token: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    daily_loss_warn: float = 1000.0
    daily_loss_limit: float = 1500.0
    max_trades: int = 2
    stock_risk: float = 500.0
    entry_cutoff: str = "14:30"
    nifty_trend: bool = True
    calls_mode: str = "paper"
    token_autorenew: bool = True
    h1_paper: bool = True
    expiry_day_contract: str = "next"
    holidays_extra: str = ""
    nifty_cutoff: str = "13:00"
    exit_reminder: str = "15:05"
    squareoff: str = "15:15"
    min_grade: str = "B"
    alerts_enabled: bool = True
    stock_alerts_enabled: bool = False
    nifty_id: int = 13
    banknifty_id: int = 25
    vix_id: int = 21
    nifty_lot: int = 0
    stock_universe_max: int = 0
    port: int = 8765

    @classmethod
    def load(cls) -> "Settings":
        e = read_env()
        s = cls(
            dhan_client_id=_get(e, "DHAN_CLIENT_ID"),
            dhan_access_token=_get(e, "DHAN_ACCESS_TOKEN"),
            telegram_bot_token=_get(e, "TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=_get(e, "TELEGRAM_CHAT_ID"),
        )
        num = {
            "SD_DAILY_LOSS_WARN": ("daily_loss_warn", float), "SD_DAILY_LOSS_LIMIT": ("daily_loss_limit", float),
            "SD_MAX_TRADES": ("max_trades", int), "SD_STOCK_RISK": ("stock_risk", float),
            "SD_NIFTY_ID": ("nifty_id", int), "SD_BANKNIFTY_ID": ("banknifty_id", int), "SD_VIX_ID": ("vix_id", int),
            "SD_NIFTY_LOT": ("nifty_lot", int), "SD_STOCK_UNIVERSE_MAX": ("stock_universe_max", int), "SD_PORT": ("port", int),
        }
        for key, (attr, typ) in num.items():
            v = _get(e, key)
            if v:
                try:
                    setattr(s, attr, typ(v))
                except ValueError:
                    pass
        for key, attr in (("SD_NIFTY_TREND", "nifty_trend"), ("SD_H1_PAPER", "h1_paper"), ("SD_STOCK_ALERTS", "stock_alerts_enabled"),
                          ("SD_TOKEN_AUTORENEW", "token_autorenew")):
            v = _get(e, key)
            if v:
                setattr(s, attr, v.strip().lower() in ("1", "true", "yes", "on"))
        if (_get(e, "SD_EXPIRY_DAY_CONTRACT") or "").strip().lower() in ("next", "today"):
            s.expiry_day_contract = _get(e, "SD_EXPIRY_DAY_CONTRACT").strip().lower()
        s.holidays_extra = _get(e, "SD_HOLIDAYS") or ""
        if (_get(e, "SD_CALLS_MODE") or "").strip().lower() in ("live", "paper"):
            s.calls_mode = _get(e, "SD_CALLS_MODE").strip().lower()
        for key, attr in (("SD_ENTRY_CUTOFF", "entry_cutoff"), ("SD_NIFTY_CUTOFF", "nifty_cutoff"), ("SD_EXIT_REMINDER", "exit_reminder"),
                          ("SD_SQUAREOFF", "squareoff"), ("SD_MIN_GRADE", "min_grade")):
            v = _get(e, key)
            if v:
                setattr(s, attr, v)
        return s

    def public(self) -> dict:
        d = dataclasses.asdict(self)
        for k in ("dhan_access_token", "telegram_bot_token"):
            d[k] = "set" if d[k] else ""
        d["dhan_client_id"] = "set" if self.dhan_client_id else ""
        d["telegram_chat_id"] = self.telegram_chat_id
        return d
