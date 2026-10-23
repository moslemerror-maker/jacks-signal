from __future__ import annotations

import dataclasses
import datetime


@dataclasses.dataclass(frozen=True)
class Bar:
    trade_date: datetime.date
    close: float
    high: float
    low: float
