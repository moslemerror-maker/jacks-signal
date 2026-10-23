from __future__ import annotations

import datetime as dt

import pytest

from backend.app.data.live.staleness import CollectionState, classify_date_state

UTC = dt.timezone.utc
D = dt.date
TODAY = D(2026, 9, 29)


class TestSixStates:
    def test_error_overrides_everything_else(self):
        state = classify_date_state(
            trading_date=TODAY, today=TODAY, bar_count=0, expected_bars=360, last_bar_utc=None,
            now_utc=dt.datetime(2026, 9, 29, 6, 0, tzinfo=UTC), is_backfilling=True, had_error=True,
        )
        assert state is CollectionState.ERROR

    def test_backfilling_overrides_freshness_and_coverage(self):
        state = classify_date_state(
            trading_date=D(2026, 9, 28), today=TODAY, bar_count=200, expected_bars=360, last_bar_utc=None,
            is_backfilling=True,
        )
        assert state is CollectionState.BACKFILLING

    def test_today_with_fresh_last_bar_is_live(self):
        now = dt.datetime(2026, 9, 29, 6, 17, 24, tzinfo=UTC)
        last_bar = dt.datetime(2026, 9, 29, 6, 17, 0, tzinfo=UTC)
        state = classify_date_state(
            trading_date=TODAY, today=TODAY, bar_count=153, expected_bars=360, last_bar_utc=last_bar, now_utc=now,
        )
        assert state is CollectionState.LIVE

    def test_today_with_old_last_bar_is_stale(self):
        now = dt.datetime(2026, 9, 29, 7, 0, 0, tzinfo=UTC)
        last_bar = dt.datetime(2026, 9, 29, 6, 17, 0, tzinfo=UTC)
        state = classify_date_state(
            trading_date=TODAY, today=TODAY, bar_count=153, expected_bars=360, last_bar_utc=last_bar, now_utc=now,
        )
        assert state is CollectionState.STALE

    def test_today_with_no_bar_at_all_is_stale_not_live(self):
        now = dt.datetime(2026, 9, 29, 9, 20, 0, tzinfo=UTC)
        state = classify_date_state(
            trading_date=TODAY, today=TODAY, bar_count=0, expected_bars=360, last_bar_utc=None, now_utc=now,
        )
        assert state is CollectionState.STALE

    def test_today_requires_now_utc_or_raises(self):
        with pytest.raises(ValueError, match="now_utc"):
            classify_date_state(
                trading_date=TODAY, today=TODAY, bar_count=100, expected_bars=360,
                last_bar_utc=dt.datetime(2026, 9, 29, 6, 0, tzinfo=UTC), now_utc=None,
            )

    def test_past_date_with_full_coverage_is_complete(self):
        state = classify_date_state(
            trading_date=D(2026, 9, 25), today=TODAY, bar_count=360, expected_bars=360, last_bar_utc=None,
        )
        assert state is CollectionState.COMPLETE

    def test_past_date_with_the_known_375_bar_variant_still_counts_as_complete(self):
        state = classify_date_state(
            trading_date=D(2026, 7, 1), today=TODAY, bar_count=375, expected_bars=360, last_bar_utc=None,
        )
        assert state is CollectionState.COMPLETE

    def test_past_date_with_short_coverage_is_partial(self):
        state = classify_date_state(
            trading_date=D(2026, 9, 28), today=TODAY, bar_count=200, expected_bars=360, last_bar_utc=None,
        )
        assert state is CollectionState.PARTIAL

    def test_past_date_with_zero_bars_is_partial_not_a_seventh_state(self):
        state = classify_date_state(
            trading_date=D(2026, 9, 28), today=TODAY, bar_count=0, expected_bars=360, last_bar_utc=None,
        )
        assert state is CollectionState.PARTIAL

    def test_never_returns_a_value_outside_the_six_states(self):
        assert set(CollectionState) == {
            CollectionState.LIVE, CollectionState.STALE, CollectionState.PARTIAL,
            CollectionState.BACKFILLING, CollectionState.COMPLETE, CollectionState.ERROR,
        }
