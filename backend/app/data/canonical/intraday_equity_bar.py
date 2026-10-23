from __future__ import annotations

import datetime as dt
from enum import Enum

from pydantic import BaseModel, model_validator

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


class IntradayProvider(str, Enum):
    DHAN = "DHAN"


class CanonicalIntradayEquityBar(BaseModel):

    provider: IntradayProvider
    exchange_segment: str
    security_id: int
    trading_symbol: str

    raw_timestamp_epoch: int
    bar_start_utc: dt.datetime
    bar_start_ist: dt.datetime
    trading_date: dt.date
    interval_minutes: int

    open: float
    high: float
    low: float
    close: float
    volume: float | None = None

    source_provider: str
    source_dataset: str
    source_key: str
    source_sha256: str
    normalizer_version: str

    @model_validator(mode="after")
    def _validate(self) -> "CanonicalIntradayEquityBar":
        if self.open <= 0 or self.high <= 0 or self.low <= 0 or self.close <= 0:
            raise ValueError(f"non-positive price: open={self.open} high={self.high} low={self.low} close={self.close}")
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) < low ({self.low})")
        if self.high < max(self.open, self.close):
            raise ValueError(f"high ({self.high}) < max(open, close) ({max(self.open, self.close)})")
        if self.low > min(self.open, self.close):
            raise ValueError(f"low ({self.low}) > min(open, close) ({min(self.open, self.close)})")
        if self.volume is not None and self.volume < 0:
            raise ValueError(f"negative volume: {self.volume}")
        if self.bar_start_utc.tzinfo is None or self.bar_start_ist.tzinfo is None:
            raise ValueError("bar_start_utc/bar_start_ist must be timezone-aware")
        if self.bar_start_utc.astimezone(dt.timezone.utc) != self.bar_start_ist.astimezone(dt.timezone.utc):
            raise ValueError("bar_start_utc and bar_start_ist do not represent the same instant")
        return self


class RejectedIntradayRow(BaseModel):

    source_key: str
    row_index: int
    raw_timestamp_epoch: int | None
    reason: str
