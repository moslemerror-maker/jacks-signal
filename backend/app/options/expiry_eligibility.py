from __future__ import annotations

import datetime as dt
from typing import Sequence

from .availability import session_close_utc


class NoEligibleExpiryError(ValueError):
    pass


def select_eligible_expiry(candidate_expiries: Sequence[dt.date], *, as_of: dt.datetime) -> dt.date:
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    eligible = sorted(e for e in candidate_expiries if session_close_utc(e) > as_of)
    if not eligible:
        raise NoEligibleExpiryError(f"no candidate expiry has a session close after {as_of.isoformat()} among {sorted(candidate_expiries)}")
    return eligible[0]
