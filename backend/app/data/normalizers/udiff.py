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
    "TradDt",
    "TckrSymb",
    "SctySrs",
    "OpnPric",
    "HghPric",
    "LwPric",
    "ClsPric",
}


def normalize_udiff(path: Path) -> tuple[list[CanonicalBar], list[RejectedRow]]:
    df = read_csv_from_zip(path)
    df.columns = [c.strip() for c in df.columns]

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing expected UDiFF columns {missing}")

    bars: list[CanonicalBar] = []
    rejected: list[RejectedRow] = []

    for idx, row in enumerate(df.to_dict(orient="records")):
        try:
            trade_date = datetime.strptime(str(row["TradDt"]).strip(), "%Y-%m-%d").date()
            bar = CanonicalBar(
                symbol=str(row["TckrSymb"]).strip(),
                series=str(row["SctySrs"]).strip(),
                isin=clean_isin(row.get("ISIN")),
                trade_date=trade_date,
                open=float(row["OpnPric"]),
                high=float(row["HghPric"]),
                low=float(row["LwPric"]),
                close=float(row["ClsPric"]),
                last_price=clean_float(row.get("LastPric")),
                prev_close=clean_float(row.get("PrvsClsgPric")),
                total_traded_qty=clean_float(row.get("TtlTradgVol")),
                total_traded_val=clean_float(row.get("TtlTrfVal")),
                total_trades=clean_int(row.get("TtlNbOfTxsExctd")),
                open_interest=clean_float(row.get("OpnIntrst")),
                source_format=SourceFormat.NSE_UDIFF,
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
