from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from backend.app.data.canonical.schema import (
    INDEX_SERIES,
    CanonicalBar,
    RejectedRow,
    SourceFormat,
)

_REQUIRED_FIELDS = (
    "EOD_INDEX_NAME",
    "EOD_TIMESTAMP",
    "EOD_OPEN_INDEX_VAL",
    "EOD_HIGH_INDEX_VAL",
    "EOD_LOW_INDEX_VAL",
    "EOD_CLOSE_INDEX_VAL",
)


def normalize_index_history(path: Path) -> tuple[list[CanonicalBar], list[RejectedRow]]:
    payload = json.loads(path.read_text())
    rows = payload.get("data", [])

    bars: list[CanonicalBar] = []
    rejected: list[RejectedRow] = []

    for idx, row in enumerate(rows):
        try:
            missing = [f for f in _REQUIRED_FIELDS if row.get(f) is None]
            if missing:
                raise ValueError(f"missing required fields: {missing}")

            trade_date = datetime.strptime(
                str(row["EOD_TIMESTAMP"]).strip().title(), "%d-%b-%Y"
            ).date()
            bar = CanonicalBar(
                symbol=str(row["EOD_INDEX_NAME"]).strip(),
                series=INDEX_SERIES,
                isin=None,
                trade_date=trade_date,
                open=float(row["EOD_OPEN_INDEX_VAL"]),
                high=float(row["EOD_HIGH_INDEX_VAL"]),
                low=float(row["EOD_LOW_INDEX_VAL"]),
                close=float(row["EOD_CLOSE_INDEX_VAL"]),
                last_price=None,
                prev_close=None,
                total_traded_qty=_clean_float(row.get("HIT_TRADED_QTY")),
                total_traded_val=_clean_float(row.get("HIT_TURN_OVER")),
                total_trades=None,
                open_interest=None,
                source_format=SourceFormat.NSE_INDEX_HISTORY,
                source_file=path.name,
            )
            bars.append(bar)
        except (ValidationError, ValueError, TypeError, KeyError) as exc:
            rejected.append(
                RejectedRow(
                    source_file=path.name,
                    row_index=idx,
                    raw=row,
                    reason=str(exc),
                )
            )

    bars.sort(key=lambda b: b.trade_date)
    return bars, rejected


def _clean_float(v: object) -> float | None:
    return None if v is None else float(v)
