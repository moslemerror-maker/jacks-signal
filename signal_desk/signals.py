from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any

PASS, WARN, FAIL, INFO = "pass", "warn", "fail", "info"
GRADE_ORDER = {"A": 0, "B": 1, "C": 2}


@dataclasses.dataclass
class Check:
    name: str
    status: str
    detail: str


@dataclasses.dataclass
class Signal:
    id: str
    kind: str
    symbol: str
    side: str
    time: dt.datetime
    level: float
    entry: float
    stop: float
    t1: float
    t2: float
    checks: list[Check] = dataclasses.field(default_factory=list)
    grade: str = "C"
    action: str = "SKIP"
    base_grade: str | None = None
    prev_range: float | None = None
    note: str = ""
    option: str | None = None
    strike: float | None = None
    expiry: str | None = None
    opt_ltp: float | None = None
    delta: float | None = None
    prem_stop: float | None = None
    prem_t1: float | None = None
    lot: int | None = None
    risk_per_lot: float | None = None
    lots: int | None = None
    security_id: int | None = None
    qty: int | None = None
    risk_rupees: float | None = None
    status: str = "open"
    outcome_time: dt.datetime | None = None
    t2_hit: bool = False
    mfe: float = 0.0
    mae: float = 0.0
    exit_ref: float | None = None
    notified: bool = False

    @property
    def risk_pts(self) -> float:
        return abs(self.entry - self.stop)

    def to_json(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        for k in ("time", "outcome_time"):
            v = getattr(self, k)
            d[k] = v.isoformat() if v else None
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Signal":
        d = dict(d)
        d["checks"] = [Check(**c) for c in d.get("checks", [])]
        for k in ("time", "outcome_time"):
            if d.get(k):
                d[k] = dt.datetime.fromisoformat(d[k])
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})


def grade(checks: list[Check]) -> tuple[str, str]:
    blocked = any(c.name == "Risk guard" and c.status == FAIL for c in checks)
    checks = [c for c in checks if c.name != "Risk guard"]
    g, a = _grade(checks)
    return g, ("SKIP" if blocked else a)


def _grade(checks: list[Check]) -> tuple[str, str]:
    fails = [c for c in checks if c.status == FAIL]
    warns = [c for c in checks if c.status == WARN]
    if fails:
        return "C", "SKIP"
    if not warns:
        return "A", "TAKE"
    if len(warns) == 1:
        return "B", "TAKE"
    return "C", "SKIP"


def grade_ok(g: str, minimum: str) -> bool:
    return GRADE_ORDER.get(g, 9) <= GRADE_ORDER.get(minimum, 1)


def _n(x: float | None, d: int = 0) -> str:
    if x is None:
        return "-"
    return f"{x:,.{d}f}"


def expiry_label(expiry: str | None, today: dt.date) -> str:
    if not expiry:
        return "-"
    try:
        e = dt.date.fromisoformat(str(expiry)[:10])
    except ValueError:
        return str(expiry)
    n = (e - today).days
    tail = "EXPIRES TODAY" if n == 0 else ("1 day left" if n == 1 else f"{n} days left")
    return f"{e:%a %d %b} ({tail})"


def format_signal(s: Signal) -> str:
    arrow = "🟢" if s.side == "LONG" else "🔴"
    blocked = any(c.name == "Risk guard" and c.status == "fail" for c in s.checks)
    head = "TAKE" if s.action == "TAKE" else ("BLOCKED by your risk limits" if blocked and s.grade in ("A", "B") else "SKIP (info)")
    t = s.time.strftime("%H:%M")
    lines = [f"{arrow} <b>{s.symbol} {s.side} - {head} - grade {s.grade}</b>  ({t})"]
    if s.kind == "NIFTY":
        lines.append(f"15-min close {_n(s.entry, 1)} {'above' if s.side == 'LONG' else 'below'} {_n(s.level, 1)}")
        if s.option and s.strike:
            lines.append(f"<b>Buy NIFTY {_n(s.strike)} {s.option}</b> @ ~{_n(s.opt_ltp, 1)} (delta {_n(s.delta, 2)})")
            lines.append(f"📅 <b>Expiry: {expiry_label(s.expiry, s.time.date())}</b>")
            lines.append(f"Premium SL ~{_n(s.prem_stop, 1)} | T1 ~{_n(s.prem_t1, 1)}")
            worst = (s.risk_per_lot or 0) + 60
            lines.append(f"<b>Worst case if SL hits: -₹{_n(worst)} per lot</b> (lot {s.lot}; {s.lots} lot(s) = -₹{_n(worst * (s.lots or 1))})")
        lines.append(f"Nifty SL {_n(s.stop, 1)} (15-min close) | T1 {_n(s.t1, 1)} | T2 {_n(s.t2, 1)}")
    else:
        lines.append(f"15-min close {_n(s.entry, 2)} {'above' if s.side == 'LONG' else 'below'} {_n(s.level, 2)}")
        lines.append(f"<b>{'Buy' if s.side == 'LONG' else 'Sell'} {s.symbol}</b> ~{_n(s.entry, 2)} | SL {_n(s.stop, 2)} | T1 {_n(s.t1, 2)} | T2 {_n(s.t2, 2)}")
        if s.qty:
            lines.append(f"Qty {s.qty} | <b>Worst case if SL hits: -₹{_n(s.risk_rupees)}</b> + charges")
    for c in s.checks:
        icon = {"pass": "✅", "warn": "⚠️", "fail": "❌", "info": "💰"}.get(c.status, "•")
        lines.append(f"{icon} {c.name}: {c.detail}")
    if s.note:
        lines.append(s.note)
    lines.append("<i>Alert only - you place the order. Square off by 3:15.</i>")
    return "\n".join(lines)
