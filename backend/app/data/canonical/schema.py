from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, model_validator


class SourceFormat(str, Enum):

    NSE_LEGACY_CSV = "NSE_LEGACY_CSV"
    NSE_UDIFF = "NSE_UDIFF"
    NSE_INDEX_HISTORY = "NSE_INDEX_HISTORY"


INDEX_SERIES = "INDEX"


class CanonicalBar(BaseModel):

    symbol: str
    series: str
    isin: str | None = None
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    last_price: float | None = None
    prev_close: float | None = None
    total_traded_qty: float | None = None
    total_traded_val: float | None = None
    total_trades: int | None = None
    open_interest: float | None = None
    source_format: SourceFormat
    source_file: str

    @model_validator(mode="after")
    def _validate_ohlc(self) -> "CanonicalBar":
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) < low ({self.low})")
        if not (self.low <= self.open <= self.high):
            raise ValueError(
                f"open ({self.open}) outside [low, high] ({self.low}, {self.high})"
            )
        if not (self.low <= self.close <= self.high):
            raise ValueError(
                f"close ({self.close}) outside [low, high] ({self.low}, {self.high})"
            )
        return self


class RejectedRow(BaseModel):

    source_file: str
    row_index: int
    raw: dict
    reason: str
