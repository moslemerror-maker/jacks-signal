from __future__ import annotations

import bisect
import datetime
from typing import Iterable, Sequence

BLOCKER_THRESHOLD = 0.28
WATCH_THRESHOLD = 0.15
ISIN_MAX_LAG_SESSIONS = 2


def _first_bar_on_or_after(bar_dates: Sequence[datetime.date], day: datetime.date) -> int | None:
    i = bisect.bisect_left(bar_dates, day)
    return i if i < len(bar_dates) else None


def scan_close_discontinuities(
    bars: Sequence[tuple[datetime.date, float]],
    price_affecting_ex_dates: Iterable[datetime.date],
    *,
    blocker: float = BLOCKER_THRESHOLD,
    watch: float = WATCH_THRESHOLD,
) -> dict:
    ordered = sorted(bars)
    bar_dates = [d for d, _ in ordered]
    explained_bars = {}
    for ex in price_affecting_ex_dates:
        i = _first_bar_on_or_after(bar_dates, ex)
        if i is not None:
            explained_bars[bar_dates[i]] = ex

    jumps = []
    for (pd, pc), (d, c) in zip(ordered, ordered[1:]):
        if pc == 0:
            continue
        change = c / pc - 1.0
        if abs(change) < watch:
            continue
        jumps.append({
            "bar_date": d.isoformat(), "prev_date": pd.isoformat(), "prev_close": pc, "close": c,
            "change": round(change, 6),
            "tier": "BLOCKER" if abs(change) >= blocker else "WATCH",
            "explained_by_ex_date": explained_bars[d].isoformat() if d in explained_bars else None,
        })
    blockers = [j for j in jumps if j["tier"] == "BLOCKER" and j["explained_by_ex_date"] is None]
    return {
        "diagnostic_only": True,
        "thresholds": {"blocker": blocker, "watch": watch},
        "blocking_count": len(blockers),
        "unexplained_blockers": blockers,
        "explained_jumps": [j for j in jumps if j["explained_by_ex_date"] is not None],
        "watch_unexplained": [j for j in jumps if j["tier"] == "WATCH" and j["explained_by_ex_date"] is None],
        "statement": (
            "diagnostic heuristic only: it can block or annotate, never establish completeness; blind to "
            "actions whose price effect is below the thresholds"
        ),
    }


def isin_transition_corroboration(
    transition_dates: Iterable[datetime.date],
    bar_dates: Sequence[datetime.date],
    price_affecting_ex_dates: Iterable[datetime.date],
    *,
    max_lag_sessions: int = ISIN_MAX_LAG_SESSIONS,
) -> dict:
    dates = sorted(bar_dates)
    ex_bar_index = {}
    for ex in price_affecting_ex_dates:
        i = _first_bar_on_or_after(dates, ex)
        if i is not None:
            ex_bar_index[ex] = i

    rows = []
    for t in sorted(transition_dates):
        ti = _first_bar_on_or_after(dates, t)
        match = None
        if ti is not None:
            for ex, ei in sorted(ex_bar_index.items()):
                lag = ti - ei
                if 0 <= lag <= max_lag_sessions:
                    match = (ex, lag)
                    break
        rows.append({
            "transition_date": t.isoformat(), "corroborated": match is not None,
            "matched_ex_date": match[0].isoformat() if match else None,
            "lag_sessions": match[1] if match else None,
        })
    return {
        "diagnostic_only": True,
        "max_lag_sessions": max_lag_sessions,
        "transitions": rows,
        "uncorroborated_count": sum(1 for r in rows if not r["corroborated"]),
        "statement": (
            "corroboration only: the ISIN lag relative to ex-date varies by event; this never anchors an "
            "adjustment and never establishes completeness"
        ),
    }
