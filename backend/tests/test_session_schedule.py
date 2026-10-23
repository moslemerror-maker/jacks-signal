from __future__ import annotations

import datetime as dt

from backend.app.options.session_schedule import IST, SESSION_START_IST, SESSION_STOP_IST, ist_today, session_start_utc, session_stop_utc

D = dt.date
UTC = dt.timezone.utc


class TestScheduleConstants:
    def test_session_start_is_0915(self):
        assert SESSION_START_IST == dt.time(9, 15)

    def test_session_stop_is_1515_not_1530(self):
        assert SESSION_STOP_IST == dt.time(15, 15)
        from backend.app.options.availability import SESSION_CLOSE_IST
        assert SESSION_CLOSE_IST == dt.time(15, 30)
        assert SESSION_STOP_IST != SESSION_CLOSE_IST


class TestSessionStartUtc:
    def test_converts_0915_ist_to_the_correct_utc_instant(self):
        start = session_start_utc(D(2026, 9, 24))
        assert start == dt.datetime(2026, 9, 24, 3, 45, tzinfo=UTC)

    def test_result_is_timezone_aware_utc(self):
        start = session_start_utc(D(2026, 9, 24))
        assert start.tzinfo is not None
        assert start.utcoffset() == dt.timedelta(0)


class TestSessionStopUtc:
    def test_converts_1515_ist_to_the_correct_utc_instant(self):
        stop = session_stop_utc(D(2026, 9, 24))
        assert stop == dt.datetime(2026, 9, 24, 9, 45, tzinfo=UTC)

    def test_stop_is_after_start_on_the_same_day(self):
        day = D(2026, 9, 24)
        assert session_stop_utc(day) > session_start_utc(day)

    def test_stop_minus_start_is_exactly_six_hours(self):
        day = D(2026, 9, 24)
        assert session_stop_utc(day) - session_start_utc(day) == dt.timedelta(hours=6)


class TestIstToday:
    def test_returns_the_ist_calendar_date(self):
        now = dt.datetime(2026, 9, 24, 3, 45, tzinfo=UTC)
        assert ist_today(now) == D(2026, 9, 24)

    def test_ist_date_can_differ_from_utc_date_around_midnight_ist(self):
        now = dt.datetime(2026, 9, 23, 19, 0, tzinfo=UTC)
        assert now.date() == D(2026, 9, 23)
        assert ist_today(now) == D(2026, 9, 24)

    def test_naive_datetime_is_rejected(self):
        import pytest
        with pytest.raises(ValueError, match="timezone-aware"):
            ist_today(dt.datetime(2026, 9, 24, 3, 45))


class TestIstIsAsiaKolkataByName:
    def test_zone_key_is_asia_kolkata(self):
        assert str(IST) == "Asia/Kolkata" or IST.key == "Asia/Kolkata"
