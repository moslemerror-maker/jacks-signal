"""Phase 1 historical backfill orchestrator.

Ingests the approved Phase 1 universe (docs/16, docs/31; basket
defined in backend/app/data/instruments/universe.py):
- Equity bhavcopy (legacy CSV / UDiFF, dispatched by date) for the
  25-stock basket, filtered to equity series (EQ/BE/BZ).
- NIFTY 50 and NIFTY BANK index EOD history.

Does not pre-fetch a holiday calendar: NSE's holiday-master API only
appears to return the current year's holidays (verified 2026-09-14),
not historical years back to 2015, so this instead attempts every
weekday and treats a 404 from the bhavcopy archive as "not a trading
day" - simpler and self-correcting versus relying on a calendar with
unverified historical coverage.

Index history is fetched in monthly chunks. VERIFIED (not assumed):
the indicesHistory API silently caps responses at ~70 rows regardless
of the requested range and does not error - a year-long request for
2015 came back with only Jan-Apr 2015 and no indication of truncation.
Monthly chunks (a month has at most ~23 trading days) stay safely
under that cap; `_assert_chunk_not_truncated` is a defensive check in
case the real cap is ever lower still.

Resumable: raw files are never re-fetched once present (see
nse_bhavcopy.fetch_bhavcopy / nse_index_history.fetch_index_history),
and DB loads are idempotent upserts, so re-running this script after
an interruption picks up where it left off rather than redoing
completed work.

Malformed/corrupted content (a 200 response that isn't a usable
ZIP/JSON) is quarantined rather than committed to raw storage or
crashing the run - see nse_bhavcopy.py / nse_index_history.py /
quarantine.py. --on-invalid-content controls whether such a day is
skipped (default, matching how a persistent HTTP failure is already
handled) or is a hard stop; either way it's counted and logged
explicitly, not silently absorbed.

Every requested trading day (or index date-range chunk) is accounted
for in exactly one of four buckets: fetched, a confirmed non-trading
day (404), quarantined (invalid content), or failed (a persistent,
non-404 HTTP failure). A failed date/chunk is logged as an error,
listed by name in the final summary, and - unlike quarantine, which
has its own deliberate skip/fail policy - always causes the run to
exit unsuccessfully once all dates have been attempted: there is no
"skip forever" option for an unexplained fetch failure, because unlike
a 404 or a quarantined response, it isn't a recognized, deliberately-
handled outcome. A reconciliation assertion (requested candidates ==
sum of all four buckets) runs before the summary is logged, as a
defensive check against a future code change silently dropping a date
from every bucket - this should be structurally impossible given the
loop, and the assertion exists to keep it that way, not because it's
expected to fire.
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import requests

from backend.app.data.instruments.isin_links import expand_isin_chain
from backend.app.data.instruments.loader import upsert_bars
from backend.app.data.instruments.universe import (
    INDEX_SYMBOLS,
    is_equity_series,
    resolve_basket_isins,
)
from backend.app.data.normalizers import normalize_bhavcopy
from backend.app.data.normalizers.index_history import normalize_index_history
from backend.app.data.sources.errors import InvalidContentError
from backend.app.data.sources.nse_bhavcopy import fetch_bhavcopy
from backend.app.data.sources.nse_index_history import fetch_index_history
from backend.app.db.session import get_session

logger = logging.getLogger(__name__)

RAW_BHAVCOPY_DIR = Path("data/raw/bhavcopy")
RAW_INDEX_DIR = Path("data/raw/index_history")


def trading_day_candidates(start: date, end: date):
    current = start
    one_day = timedelta(days=1)
    while current <= end:
        if current.weekday() < 5:
            yield current
        current += one_day


def month_chunks(start: date, end: date):
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        chunk_start = max(start, date(year, month, 1))
        next_month_first = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
        chunk_end = min(end, next_month_first - timedelta(days=1))
        yield chunk_start, chunk_end
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


_MAX_CHUNK_COVERAGE_GAP_DAYS = 10


def _assert_chunk_not_truncated(
    bars: list, chunk_start: date, chunk_end: date, index_symbol: str
) -> None:
    if not bars:
        return
    latest = max(b.trade_date for b in bars)
    gap_days = (chunk_end - latest).days
    if gap_days > _MAX_CHUNK_COVERAGE_GAP_DAYS:
        raise RuntimeError(
            f"{index_symbol} {chunk_start}..{chunk_end}: latest bar returned is "
            f"{latest}, {gap_days} days short of the requested chunk end - looks "
            "like API truncation, not just a holiday gap. Refusing to silently "
            "load an incomplete range."
        )


def _reconcile_or_raise(
    expected_count: int, accounted_count: int, label: str, buckets: dict[str, int]
) -> None:
    if accounted_count != expected_count:
        raise RuntimeError(
            f"{label} reconciliation failed: {expected_count} candidates "
            f"expected, only {accounted_count} accounted for across buckets "
            f"{buckets} - {expected_count - accounted_count} candidate(s) were "
            "silently dropped somewhere in the loop, which should be "
            "impossible; this is a bug in the orchestrator, not a data problem"
        )


def _fetch_reference_bhavcopy(end: date, max_lookback_days: int = 10) -> tuple[Path, date]:
    candidate = end
    for _ in range(max_lookback_days):
        try:
            path = fetch_bhavcopy(candidate, RAW_BHAVCOPY_DIR)
            return path, candidate
        except InvalidContentError as exc:
            logger.warning("reference day %s quarantined, trying earlier: %s", candidate, exc)
            candidate -= timedelta(days=1)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status != 404:
                raise
            candidate -= timedelta(days=1)
    raise RuntimeError(
        f"could not find a trading day within {max_lookback_days} days before "
        f"{end} to resolve basket ISINs"
    )


def backfill_equities(
    start: date,
    end: date,
    reference_date: date | None = None,
    on_invalid_content: str = "skip",
) -> None:
    session = get_session()

    reference_path, resolved_reference_date = _fetch_reference_bhavcopy(
        reference_date or date.today()
    )
    reference_bars, _ = normalize_bhavcopy(reference_path, resolved_reference_date)
    current_isins = set(resolve_basket_isins(reference_bars).values())

    basket_isins: set[str] = set()
    for isin in current_isins:
        basket_isins |= expand_isin_chain(isin)

    logger.info(
        "resolved %d basket ISINs (%d current + predecessors) using reference day %s",
        len(basket_isins),
        len(current_isins),
        resolved_reference_date,
    )

    fetched, skipped_holidays, quarantined, loaded_bars = 0, 0, 0, 0
    failed_dates: list[date] = []

    for trade_date in trading_day_candidates(start, end):
        try:
            path = fetch_bhavcopy(trade_date, RAW_BHAVCOPY_DIR)
        except InvalidContentError as exc:
            quarantined += 1
            logger.error("%s: quarantined invalid content: %s", trade_date, exc)
            if on_invalid_content == "fail":
                session.close()
                raise
            continue
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 404:
                skipped_holidays += 1
                continue
            failed_dates.append(trade_date)
            logger.error(
                "%s: fetch failed, not a 404 (status=%s) - recorded as failed, "
                "not silently skipped: %s",
                trade_date,
                status,
                exc,
            )
            continue

        fetched += 1
        bars, rejected = normalize_bhavcopy(path, trade_date)
        if rejected:
            logger.warning("%s: %d rejected rows", trade_date, len(rejected))

        universe_bars = [
            b for b in bars if b.isin in basket_isins and is_equity_series(b.series)
        ]
        loaded_bars += upsert_bars(session, universe_bars)
        session.commit()

        if fetched % 50 == 0:
            logger.info(
                "progress: %s done, %d trading days, %d bars loaded",
                trade_date,
                fetched,
                loaded_bars,
            )

    session.close()

    _reconcile_or_raise(
        expected_count=sum(1 for _ in trading_day_candidates(start, end)),
        accounted_count=fetched + skipped_holidays + quarantined + len(failed_dates),
        label="equity backfill",
        buckets={
            "fetched": fetched,
            "holidays": skipped_holidays,
            "quarantined": quarantined,
            "failed": len(failed_dates),
        },
    )

    logger.info(
        "equity backfill complete: %d trading days, %d holidays skipped, "
        "%d quarantined, %d failed, %d bars loaded",
        fetched,
        skipped_holidays,
        quarantined,
        len(failed_dates),
        loaded_bars,
    )

    if failed_dates:
        logger.error(
            "%d date(s) failed and are neither loaded nor a confirmed "
            "non-trading day: %s",
            len(failed_dates),
            ", ".join(d.isoformat() for d in failed_dates),
        )
        raise RuntimeError(
            f"equity backfill did not complete successfully: {len(failed_dates)} "
            f"date(s) failed to fetch ({[d.isoformat() for d in failed_dates]}). "
            "Already-fetched days are cached and won't be re-fetched; re-run "
            "after investigating the failure(s) above."
        )


def backfill_indices(start: date, end: date, on_invalid_content: str = "skip") -> None:
    session = get_session()
    loaded_bars = 0
    processed_chunks = 0
    quarantined = 0
    failed_chunks: list[tuple[str, date, date]] = []

    for index_symbol in INDEX_SYMBOLS:
        for chunk_start, chunk_end in month_chunks(start, end):
            try:
                path = fetch_index_history(index_symbol, chunk_start, chunk_end, RAW_INDEX_DIR)
            except InvalidContentError as exc:
                quarantined += 1
                logger.error(
                    "%s %s-%s: quarantined invalid content: %s",
                    index_symbol,
                    chunk_start,
                    chunk_end,
                    exc,
                )
                if on_invalid_content == "fail":
                    session.close()
                    raise
                continue
            except requests.HTTPError as exc:
                failed_chunks.append((index_symbol, chunk_start, chunk_end))
                logger.error(
                    "%s %s-%s: fetch failed - recorded as failed, not silently "
                    "skipped: %s",
                    index_symbol,
                    chunk_start,
                    chunk_end,
                    exc,
                )
                continue

            processed_chunks += 1
            bars, rejected = normalize_index_history(path)
            if rejected:
                logger.warning(
                    "%s %s-%s: %d rejected rows",
                    index_symbol,
                    chunk_start,
                    chunk_end,
                    len(rejected),
                )
            _assert_chunk_not_truncated(bars, chunk_start, chunk_end, index_symbol)

            loaded_bars += upsert_bars(session, bars)
            session.commit()
            logger.info(
                "%s %s-%s: %d bars loaded", index_symbol, chunk_start, chunk_end, len(bars)
            )

    session.close()

    total_chunks_per_symbol = sum(1 for _ in month_chunks(start, end))
    expected_chunks = total_chunks_per_symbol * len(INDEX_SYMBOLS)
    _reconcile_or_raise(
        expected_count=expected_chunks,
        accounted_count=processed_chunks + quarantined + len(failed_chunks),
        label="index backfill",
        buckets={
            "processed": processed_chunks,
            "quarantined": quarantined,
            "failed": len(failed_chunks),
        },
    )

    logger.info(
        "index backfill complete: %d bars loaded, %d chunks processed, "
        "%d quarantined, %d failed",
        loaded_bars,
        processed_chunks,
        quarantined,
        len(failed_chunks),
    )

    if failed_chunks:
        failed_desc = [f"{sym} {s}..{e}" for sym, s, e in failed_chunks]
        logger.error(
            "%d chunk(s) failed and were not loaded: %s",
            len(failed_chunks),
            ", ".join(failed_desc),
        )
        raise RuntimeError(
            f"index backfill did not complete successfully: {len(failed_chunks)} "
            f"chunk(s) failed to fetch ({failed_desc}). Already-fetched chunks "
            "are cached and won't be re-fetched; re-run after investigating the "
            "failure(s) above."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2015, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument(
        "--reference-date",
        type=date.fromisoformat,
        default=None,
        help="Recent trading day used to resolve the equity basket's symbols to "
        "ISINs (default: today). Only override for testing against an older "
        "fixed dataset.",
    )
    parser.add_argument("--only", choices=["equities", "indices", "all"], default="all")
    parser.add_argument(
        "--on-invalid-content",
        choices=["skip", "fail"],
        default="skip",
        help="Policy when a fetched response is structurally invalid (not a "
        "usable ZIP/JSON) after being quarantined: 'skip' logs it and "
        "continues with the next date/chunk (default, matching how a "
        "persistent HTTP failure is already handled); 'fail' stops the run "
        "immediately.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.only in ("equities", "all"):
        backfill_equities(
            args.start,
            args.end,
            reference_date=args.reference_date,
            on_invalid_content=args.on_invalid_content,
        )
    if args.only in ("indices", "all"):
        backfill_indices(args.start, args.end, on_invalid_content=args.on_invalid_content)


if __name__ == "__main__":
    main()
