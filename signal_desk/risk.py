from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any

from .clock import at

OPTION_CHARGES = 60.0
STOCK_CHARGES_PCT = 0.0006


@dataclasses.dataclass
class Trade:
    id: str
    kind: str
    symbol: str
    side: str
    entry: float
    qty: int
    opened_at: dt.datetime
    sl: float | None = None
    target: float | None = None
    signal_id: str | None = None
    strike: float | None = None
    option: str | None = None
    expiry: str | None = None
    security_id: int | None = None
    ltp: float | None = None
    exit: float | None = None
    closed_at: dt.datetime | None = None
    be_alerted: bool = False
    sl_alerted: bool = False
    tgt_alerted: bool = False
    paper: bool = False

    @property
    def open(self) -> bool:
        return self.closed_at is None

    def pnl(self) -> float:
        px = self.exit if self.exit is not None else self.ltp
        if px is None:
            return 0.0
        gross = (px - self.entry) * self.qty if self.side == "BUY" else (self.entry - px) * self.qty
        return gross - self.charges()

    def charges(self) -> float:
        if self.kind == "OPTION":
            return OPTION_CHARGES
        px = self.exit if self.exit is not None else (self.ltp or self.entry)
        return STOCK_CHARGES_PCT * (self.entry + px) * self.qty

    def to_json(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        for k in ("opened_at", "closed_at"):
            v = getattr(self, k)
            d[k] = v.isoformat() if v else None
        d["pnl"] = round(self.pnl(), 2)
        d["is_open"] = self.open
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Trade":
        d = dict(d)
        for k in ("opened_at", "closed_at"):
            if d.get(k):
                d[k] = dt.datetime.fromisoformat(d[k])
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})


class RiskGuard:
    def __init__(self, warn: float, limit: float, max_trades: int, stock_risk: float, cutoff: str, squareoff: str) -> None:
        self.warn, self.limit, self.max_trades = warn, limit, max_trades
        self.stock_risk = stock_risk
        self.cutoff, self.squareoff = cutoff, squareoff
        self.trades: list[Trade] = []
        self.warn_sent = False
        self.limit_sent = False

    def today(self, day: dt.date) -> list[Trade]:
        return [t for t in self.trades if t.opened_at.date() == day]

    def real_today(self, day: dt.date) -> list[Trade]:
        return [t for t in self.today(day) if not t.paper]

    def day_pnl(self, day: dt.date) -> float:
        return sum(t.pnl() for t in self.real_today(day))

    def paper_pnl(self, day: dt.date) -> float:
        return sum(t.pnl() for t in self.today(day) if t.paper)

    def blocked_reason(self, now: dt.datetime) -> str | None:
        day = now.date()
        pnl = self.day_pnl(day)
        if pnl <= -self.limit:
            return f"daily loss limit ₹{self.limit:,.0f} reached (P&L ₹{pnl:,.0f}) - no more trades today"
        n = len(self.real_today(day))
        if n >= self.max_trades:
            return f"max {self.max_trades} trades for the day already taken"
        if now > at(day, self.cutoff):
            return f"no new entries after {self.cutoff}"
        return None

    def budget(self, now: dt.datetime, kind: str = "OPTION") -> float:
        loss = max(0.0, -self.day_pnl(now.date()))
        room = max(0.0, self.limit - loss)
        return room if kind == "OPTION" else min(self.stock_risk, room)

    def check_limits(self, now: dt.datetime) -> str | None:
        pnl = self.day_pnl(now.date())
        if pnl <= -self.limit and not self.limit_sent:
            self.limit_sent = self.warn_sent = True
            return f"🛑 Daily loss limit hit: P&L ₹{pnl:,.0f} (limit ₹{self.limit:,.0f}). Close open positions and stop trading for today."
        if pnl <= -self.warn and not self.warn_sent:
            self.warn_sent = True
            return f"⚠️ Day P&L ₹{pnl:,.0f} - past your ₹{self.warn:,.0f} warning level. Only A-grade setups from here, smaller size."
        return None

    def trade_alerts(self, t: Trade) -> list[str]:
        if not t.open or t.ltp is None:
            return []
        out = []
        buy = t.side == "BUY"
        if t.sl is not None and not t.sl_alerted and ((buy and t.ltp <= t.sl) or (not buy and t.ltp >= t.sl)):
            t.sl_alerted = True
            out.append(f"🔻 {t.symbol}: stop {t.sl:,.2f} hit (LTP {t.ltp:,.2f}). Exit now. P&L ≈ ₹{t.pnl():,.0f}")
        if t.target is not None and not t.tgt_alerted and ((buy and t.ltp >= t.target) or (not buy and t.ltp <= t.target)):
            t.tgt_alerted = True
            out.append(f"🎯 {t.symbol}: target {t.target:,.2f} reached (LTP {t.ltp:,.2f}). Book at least half. P&L ≈ ₹{t.pnl():,.0f}")
        if t.sl is not None and not t.be_alerted:
            r = abs(t.entry - t.sl)
            gain = (t.ltp - t.entry) if buy else (t.entry - t.ltp)
            if r > 0 and gain >= r:
                t.be_alerted = True
                out.append(f"🔒 {t.symbol}: up 1R (LTP {t.ltp:,.2f}). Move your stop to cost {t.entry:,.2f} - this trade can't become a loss.")
        return out
