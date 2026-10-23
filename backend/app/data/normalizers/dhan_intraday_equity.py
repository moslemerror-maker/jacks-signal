from __future__ import annotations

import dataclasses
import datetime as dt
import json
from collections import defaultdict
from typing import Sequence

from pydantic import ValidationError

from ..calendar.equity_trading_calendar import EXPECTED_BARS_PER_COMPLETE_SESSION, SessionShape, classify_session_shape, expected_minute_grid_ist
from ..canonical.intraday_equity_bar import IST, CanonicalIntradayEquityBar, IntradayProvider, RejectedIntradayRow
from ..raw.artifact import RawArtifactMeta
from ..raw.store import RawStore
from ..sources import dhan_equity_historical as deh

NORMALIZER_VERSION = "dhan_intraday_equity/1"
_EXPECTED_MINUTE_GRID = expected_minute_grid_ist()


def normalize_dhan_intraday_artifact(
    store: RawStore, *, source_key: str, trading_symbol: str, version: int | None = None,
) -> tuple[list[CanonicalIntradayEquityBar], list[RejectedIntradayRow]]:
    key = deh.parse_source_key(source_key)
    versions = store.versions("dhan", deh.DATASET, source_key)
    meta: RawArtifactMeta = versions[-1] if version is None else versions[version - 1]
    content = store.read_bytes("dhan", deh.DATASET, source_key, version=meta.version)
    deh.validate_intraday_json_shape(content)
    parsed = json.loads(content)

    n = len(parsed["timestamp"])
    rejected: list[RejectedIntradayRow] = []

    epoch_seen: dict[int, int] = {}
    bad_indices: set[int] = set()
    prev_epoch: int | None = None
    for i in range(n):
        e = parsed["timestamp"][i]
        if e in epoch_seen:
            bad_indices.add(i)
            bad_indices.add(epoch_seen[e])
        else:
            epoch_seen[e] = i
        if prev_epoch is not None and e < prev_epoch:
            bad_indices.add(i)
        prev_epoch = e
    for i in sorted(bad_indices):
        reason = "duplicate timestamp" if list(parsed["timestamp"]).count(parsed["timestamp"][i]) > 1 else "timestamp out of order"
        rejected.append(RejectedIntradayRow(source_key=source_key, row_index=i, raw_timestamp_epoch=parsed["timestamp"][i], reason=reason))

    bars: list[CanonicalIntradayEquityBar] = []
    for i in range(n):
        if i in bad_indices:
            continue
        epoch = parsed["timestamp"][i]
        try:
            bar_start_utc = dt.datetime.fromtimestamp(epoch, tz=dt.timezone.utc)
            bar_start_ist = bar_start_utc.astimezone(IST)
            bar = CanonicalIntradayEquityBar(
                provider=IntradayProvider.DHAN, exchange_segment="NSE_EQ", security_id=key.security_id, trading_symbol=trading_symbol,
                raw_timestamp_epoch=int(epoch), bar_start_utc=bar_start_utc, bar_start_ist=bar_start_ist, trading_date=bar_start_ist.date(),
                interval_minutes=key.interval, open=parsed["open"][i], high=parsed["high"][i], low=parsed["low"][i], close=parsed["close"][i],
                volume=parsed["volume"][i], source_provider="dhan", source_dataset=deh.DATASET, source_key=source_key, source_sha256=meta.sha256,
                normalizer_version=NORMALIZER_VERSION,
            )
            bars.append(bar)
        except (ValidationError, ValueError, TypeError) as exc:
            rejected.append(RejectedIntradayRow(source_key=source_key, row_index=i, raw_timestamp_epoch=int(epoch) if epoch is not None else None, reason=str(exc)))

    bars.sort(key=lambda b: b.bar_start_utc)
    rejected.sort(key=lambda r: r.row_index)
    return bars, rejected


@dataclasses.dataclass(frozen=True)
class SessionBoundaryReport:
    trading_date: dt.date
    bar_count: int
    first_bar_start_ist: dt.time
    last_bar_start_ist: dt.time
    is_complete_360: bool
    missing_minute_marks_ist: tuple[str, ...]
    session_shape: SessionShape


def session_boundary_reports(bars: Sequence[CanonicalIntradayEquityBar]) -> list[SessionBoundaryReport]:
    by_date: dict[dt.date, list[CanonicalIntradayEquityBar]] = defaultdict(list)
    for b in bars:
        by_date[b.trading_date].append(b)
    out = []
    for trading_date, day_bars in sorted(by_date.items()):
        day_bars = sorted(day_bars, key=lambda b: b.bar_start_ist)
        present = {b.bar_start_ist.time() for b in day_bars}
        missing = tuple(t.strftime("%H:%M") for t in _EXPECTED_MINUTE_GRID if t not in present)
        out.append(SessionBoundaryReport(
            trading_date=trading_date, bar_count=len(day_bars), first_bar_start_ist=day_bars[0].bar_start_ist.time(),
            last_bar_start_ist=day_bars[-1].bar_start_ist.time(), is_complete_360=(len(day_bars) == EXPECTED_BARS_PER_COMPLETE_SESSION and not missing),
            missing_minute_marks_ist=missing, session_shape=classify_session_shape(trading_date, present),
        ))
    return out


@dataclasses.dataclass(frozen=True)
class ReconciliationResult:
    source_key: str
    raw_row_count: int
    normalized_row_count: int
    rejected_row_count: int
    rows_accounted_for: bool
    earliest_timestamp_matches: bool
    latest_timestamp_matches: bool
    ohlc_matches: bool
    discrepancies: tuple[str, ...]


def reconcile(store: RawStore, *, source_key: str, bars: Sequence[CanonicalIntradayEquityBar], rejected: Sequence[RejectedIntradayRow]) -> ReconciliationResult:
    content = store.read_bytes("dhan", deh.DATASET, source_key)
    parsed = json.loads(content)
    n = len(parsed["timestamp"])
    discrepancies: list[str] = []

    rows_accounted_for = (len(bars) + len(rejected)) == n
    if not rows_accounted_for:
        discrepancies.append(f"raw rows {n} != normalized {len(bars)} + rejected {len(rejected)}")

    by_epoch = {b.raw_timestamp_epoch: b for b in bars}
    earliest_matches = latest_matches = True
    if n and bars:
        raw_first, raw_last = parsed["timestamp"][0], parsed["timestamp"][-1]
        if raw_first not in by_epoch and not any(r.raw_timestamp_epoch == raw_first for r in rejected):
            earliest_matches = False
            discrepancies.append(f"raw first timestamp {raw_first} not accounted for in normalized or rejected output")
        if raw_last not in by_epoch and not any(r.raw_timestamp_epoch == raw_last for r in rejected):
            latest_matches = False
            discrepancies.append(f"raw last timestamp {raw_last} not accounted for in normalized or rejected output")

    ohlc_matches = True
    for i in range(n):
        e = parsed["timestamp"][i]
        bar = by_epoch.get(e)
        if bar is None:
            continue
        if (bar.open, bar.high, bar.low, bar.close) != (parsed["open"][i], parsed["high"][i], parsed["low"][i], parsed["close"][i]):
            ohlc_matches = False
            discrepancies.append(f"row {i} (epoch {e}): OHLC mismatch between raw and normalized")
        if bar.volume != parsed["volume"][i]:
            ohlc_matches = False
            discrepancies.append(f"row {i} (epoch {e}): volume mismatch between raw and normalized")

    return ReconciliationResult(
        source_key=source_key, raw_row_count=n, normalized_row_count=len(bars), rejected_row_count=len(rejected),
        rows_accounted_for=rows_accounted_for, earliest_timestamp_matches=earliest_matches, latest_timestamp_matches=latest_matches,
        ohlc_matches=ohlc_matches, discrepancies=tuple(discrepancies),
    )
