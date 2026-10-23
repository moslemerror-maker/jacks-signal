from __future__ import annotations

import datetime as dt

NSE_HOLIDAYS: dict[str, str] = {
    "2026-01-15": "Maharashtra municipal elections",
    "2026-01-26": "Republic Day",
    "2026-03-03": "Holi",
    "2026-03-26": "Shri Ram Navami",
    "2026-03-31": "Shri Mahavir Jayanti",
    "2026-04-03": "Good Friday",
    "2026-04-14": "Dr. Baba Saheb Ambedkar Jayanti",
    "2026-05-01": "Maharashtra Day",
    "2026-05-28": "Bakri Id",
    "2026-06-26": "Muharram",
    "2026-09-14": "Ganesh Chaturthi",
    "2026-10-02": "Mahatma Gandhi Jayanti",
    "2026-10-20": "Dussehra",
    "2026-11-10": "Diwali (Balipratipada)",
    "2026-11-24": "Guru Nanak Jayanti",
    "2026-12-25": "Christmas",
}


def holiday_name(day: dt.date, extra: str = "") -> str | None:
    key = day.isoformat()
    if key in NSE_HOLIDAYS:
        return NSE_HOLIDAYS[key]
    extras = {x.strip() for x in (extra or "").split(",") if x.strip()}
    return "market holiday" if key in extras else None


def next_trading_day(day: dt.date, extra: str = "") -> dt.date:
    d = day + dt.timedelta(days=1)
    while d.weekday() >= 5 or holiday_name(d, extra):
        d += dt.timedelta(days=1)
    return d
