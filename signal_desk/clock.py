from __future__ import annotations

import datetime as dt

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

MARKET_OPEN = dt.time(9, 15)
MARKET_CLOSE = dt.time(15, 30)


class Clock:

    def now(self) -> dt.datetime:
        return dt.datetime.now(IST)


def from_epoch(seconds: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(float(seconds), IST)


def at(day: dt.date, hhmm: str) -> dt.datetime:
    h, m = (int(x) for x in hhmm.split(":"))
    return dt.datetime.combine(day, dt.time(h, m), IST)


def is_weekday(d: dt.date) -> bool:
    return d.weekday() < 5


def market_open_now(now: dt.datetime) -> bool:
    return is_weekday(now.date()) and MARKET_OPEN <= now.time() < MARKET_CLOSE


def fmt_t(t: dt.datetime | None) -> str:
    return t.strftime("%H:%M") if t else "-"
