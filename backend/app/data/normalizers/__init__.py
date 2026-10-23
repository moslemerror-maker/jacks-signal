from __future__ import annotations

from datetime import date
from pathlib import Path

from backend.app.data.canonical.schema import CanonicalBar, RejectedRow, SourceFormat
from backend.app.data.normalizers.legacy_csv import normalize_legacy_csv
from backend.app.data.normalizers.udiff import normalize_udiff

UDIFF_CUTOVER_DATE = date(2024, 7, 8)


def format_for_date(trade_date: date) -> SourceFormat:
    return (
        SourceFormat.NSE_UDIFF
        if trade_date >= UDIFF_CUTOVER_DATE
        else SourceFormat.NSE_LEGACY_CSV
    )


def normalize_bhavcopy(
    path: Path, trade_date: date
) -> tuple[list[CanonicalBar], list[RejectedRow]]:
    fmt = format_for_date(trade_date)
    if fmt is SourceFormat.NSE_UDIFF:
        return normalize_udiff(path)
    return normalize_legacy_csv(path)
