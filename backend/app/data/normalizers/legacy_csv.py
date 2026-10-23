from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from backend.app.data.canonical.schema import CanonicalBar, RejectedRow, SourceFormat
from backend.app.data.normalizers._io import (
    clean_float,
    clean_int,
    clean_isin,
    jsonable,
    read_csv_from_zip,
)

_REQUIRED_COLUMNS = {
    "SYMBOL",
    "SERIES",
    "OPEN",
    "HIGH",
    "LOW",
    "CLOSE",
    "TIMESTAMP",
}


def normalize_legacy_csv(path: Path) -> tuple[list[CanonicalBar], list[RejectedRow]]:
    df = read_csv_from_zip(path)
    df.columns = [c.strip() for c in df.columns]

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing expected legacy columns {missing}")

    bars: list[CanonicalBar] = []
    rejected: list[RejectedRow] = []

    for idx, row in enumerate(df.to_dict(orient="records")):
        try:
            trade_date = datetime.strptime(str(row["TIMESTAMP"]).strip(), "%d-%b-%Y").date()
            bar = CanonicalBar(
                symbol=str(row["SYMBOL"]).strip(),
                series=str(row["SERIES"]).strip(),
                isin=clean_isin(row.get("ISIN")),
                trade_date=trade_date,
                open=float(row["OPEN"]),
                high=float(row["HIGH"]),
                low=float(row["LOW"]),
                close=float(row["CLOSE"]),
                last_price=clean_float(row.get("LAST")),
                prev_close=clean_float(row.get("PREVCLOSE")),
                total_traded_qty=clean_float(row.get("TOTTRDQTY")),
                total_traded_val=clean_float(row.get("TOTTRDVAL")),
                total_trades=clean_int(row.get("TOTALTRADES")),
                open_interest=None,
                source_format=SourceFormat.NSE_LEGACY_CSV,
                source_file=path.name,
            )
            bars.append(bar)
        except (ValidationError, ValueError, TypeError) as exc:
            rejected.append(
                RejectedRow(
                    source_file=path.name,
                    row_index=idx,
                    raw={k: jsonable(v) for k, v in row.items()},
                    reason=str(exc),
                )
            )

    return bars, rejected
