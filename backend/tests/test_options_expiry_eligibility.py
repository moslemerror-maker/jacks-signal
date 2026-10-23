from __future__ import annotations

import datetime as dt

import pytest

from backend.app.options.availability import session_close_utc
from backend.app.options.expiry_eligibility import NoEligibleExpiryError, select_eligible_expiry

D = dt.date
UTC = dt.timezone.utc

TODAY = D(2026, 9, 22)
NEXT_WEEK = D(2026, 9, 29)
WEEK_AFTER = D(2026, 10, 6)


class TestSelectEligibleExpiry:
    def test_current_date_still_eligible_before_its_own_close(self):
        before_close = session_close_utc(TODAY) - dt.timedelta(minutes=5)
        assert select_eligible_expiry([TODAY, NEXT_WEEK], as_of=before_close) == TODAY

    def test_current_date_not_eligible_after_its_own_close(self):
        after_close = session_close_utc(TODAY) + dt.timedelta(minutes=10)
        assert select_eligible_expiry([TODAY, NEXT_WEEK], as_of=after_close) == NEXT_WEEK

    def test_expired_past_date_excluded(self):
        past = D(2026, 9, 1)
        as_of = session_close_utc(TODAY) - dt.timedelta(hours=1)
        assert select_eligible_expiry([past, TODAY, NEXT_WEEK], as_of=as_of) == TODAY

    def test_next_available_expiry_picked_when_nearest_is_ineligible(self):
        after_close = session_close_utc(TODAY) + dt.timedelta(minutes=1)
        assert select_eligible_expiry([TODAY, NEXT_WEEK, WEEK_AFTER], as_of=after_close) == NEXT_WEEK

    def test_multiple_future_expiries_nearest_wins(self):
        early = session_close_utc(TODAY) - dt.timedelta(hours=2)
        assert select_eligible_expiry([WEEK_AFTER, NEXT_WEEK, TODAY], as_of=early) == TODAY

    def test_no_eligible_expiry_raises(self):
        far_future_as_of = session_close_utc(WEEK_AFTER) + dt.timedelta(days=1)
        with pytest.raises(NoEligibleExpiryError, match="no candidate expiry"):
            select_eligible_expiry([TODAY, NEXT_WEEK, WEEK_AFTER], as_of=far_future_as_of)

    def test_empty_candidate_list_raises(self):
        with pytest.raises(NoEligibleExpiryError):
            select_eligible_expiry([], as_of=dt.datetime(2026, 9, 22, 10, 0, tzinfo=UTC))

    def test_naive_as_of_rejected(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            select_eligible_expiry([NEXT_WEEK], as_of=dt.datetime(2026, 9, 22, 10, 0))

    def test_deterministic(self):
        as_of = dt.datetime(2026, 9, 22, 10, 0, tzinfo=UTC)
        assert select_eligible_expiry([TODAY, NEXT_WEEK], as_of=as_of) == select_eligible_expiry([TODAY, NEXT_WEEK], as_of=as_of)
