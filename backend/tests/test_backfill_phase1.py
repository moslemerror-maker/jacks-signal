from datetime import date

import pytest

from scripts.backfill_phase1 import (
    _assert_chunk_not_truncated,
    month_chunks,
    trading_day_candidates,
)


class TestTradingDayCandidates:
    def test_excludes_weekends(self):
        days = list(trading_day_candidates(date(2024, 7, 1), date(2024, 7, 7)))
        assert days == [
            date(2024, 7, 1),
            date(2024, 7, 2),
            date(2024, 7, 3),
            date(2024, 7, 4),
            date(2024, 7, 5),
        ]

    def test_single_day_range(self):
        assert list(trading_day_candidates(date(2024, 7, 1), date(2024, 7, 1))) == [
            date(2024, 7, 1)
        ]

    def test_single_weekend_day_yields_nothing(self):
        assert list(trading_day_candidates(date(2024, 7, 6), date(2024, 7, 6))) == []


class TestMonthChunks:
    def test_single_month_range_yields_one_chunk(self):
        chunks = list(month_chunks(date(2024, 3, 5), date(2024, 3, 20)))
        assert chunks == [(date(2024, 3, 5), date(2024, 3, 20))]

    def test_multi_month_range_splits_at_month_boundaries(self):
        chunks = list(month_chunks(date(2024, 11, 15), date(2025, 2, 10)))
        assert chunks == [
            (date(2024, 11, 15), date(2024, 11, 30)),
            (date(2024, 12, 1), date(2024, 12, 31)),
            (date(2025, 1, 1), date(2025, 1, 31)),
            (date(2025, 2, 1), date(2025, 2, 10)),
        ]

    def test_full_calendar_month_range(self):
        chunks = list(month_chunks(date(2020, 2, 1), date(2020, 2, 29)))
        assert chunks == [(date(2020, 2, 1), date(2020, 2, 29))]

    def test_december_to_january_year_boundary(self):
        chunks = list(month_chunks(date(2023, 12, 20), date(2024, 1, 10)))
        assert chunks == [
            (date(2023, 12, 20), date(2023, 12, 31)),
            (date(2024, 1, 1), date(2024, 1, 10)),
        ]

    def test_multi_year_range_stays_chunked_monthly_throughout(self):
        chunks = list(month_chunks(date(2015, 1, 1), date(2016, 1, 31)))
        assert len(chunks) == 13
        assert chunks[0] == (date(2015, 1, 1), date(2015, 1, 31))
        assert chunks[-1] == (date(2016, 1, 1), date(2016, 1, 31))


class TestAssertChunkNotTruncated:
    def _bar(self, trade_date):
        return type("Bar", (), {"trade_date": trade_date})()

    def test_passes_when_latest_bar_matches_chunk_end(self):
        bars = [self._bar(date(2024, 3, 29))]
        _assert_chunk_not_truncated(bars, date(2024, 3, 1), date(2024, 3, 31), "NIFTY 50")

    def test_passes_with_small_gap_from_month_end_holiday(self):
        bars = [self._bar(date(2024, 3, 25))]
        _assert_chunk_not_truncated(bars, date(2024, 3, 1), date(2024, 3, 31), "NIFTY 50")

    def test_raises_on_large_gap_suggesting_truncation(self):
        bars = [self._bar(date(2015, 4, 15))]
        with pytest.raises(RuntimeError, match="truncat"):
            _assert_chunk_not_truncated(
                bars, date(2015, 1, 1), date(2015, 12, 31), "NIFTY 50"
            )

    def test_empty_bars_does_not_raise(self):
        _assert_chunk_not_truncated([], date(2024, 3, 1), date(2024, 3, 31), "NIFTY 50")
