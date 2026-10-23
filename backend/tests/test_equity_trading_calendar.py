from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from backend.app.data.calendar import equity_trading_calendar as cal

FIXTURE = Path(__file__).parent / "fixtures" / "calendar" / "holiday_master.json"
D = dt.date


class TestIsTradingDay:
    def test_weekend_is_never_a_trading_day_even_with_empty_calendar(self):
        assert cal.is_trading_day(D(2026, 9, 26), frozenset()) is False
        assert cal.is_trading_day(D(2026, 9, 27), frozenset()) is False

    def test_ordinary_weekday_with_no_holiday_is_a_trading_day(self):
        assert cal.is_trading_day(D(2026, 9, 25), frozenset()) is True

    def test_weekday_holiday_is_not_a_trading_day(self):
        holidays = frozenset({cal.nse_date_string(D(2026, 9, 14))})
        assert cal.is_trading_day(D(2026, 9, 14), holidays) is False

    def test_never_infers_a_holiday_from_an_empty_set(self):
        for d in (D(2026, 9, 21), D(2026, 9, 22), D(2026, 9, 23), D(2026, 9, 24), D(2026, 9, 25)):
            assert cal.is_trading_day(d, frozenset()) is True


class TestNseDateString:
    def test_format_matches_nse_convention(self):
        assert cal.nse_date_string(D(2026, 9, 14)) == "14-Sep-2026"
        assert cal.nse_date_string(D(2026, 1, 5)) == "05-Jan-2026"


class TestLoadEquityHolidayDates:
    def test_loads_real_fixture(self):
        holidays = cal.load_equity_holiday_dates(FIXTURE)
        assert "14-Sep-2026" in holidays
        assert len(holidays) == 20

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(cal.HolidayCalendarUnavailable, match="not found"):
            cal.load_equity_holiday_dates(tmp_path / "nope.json")

    def test_malformed_json_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json")
        with pytest.raises(cal.HolidayCalendarUnavailable, match="not valid JSON"):
            cal.load_equity_holiday_dates(p)

    def test_missing_segment_key_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"FO": []}))
        with pytest.raises(cal.HolidayCalendarUnavailable, match="CM"):
            cal.load_equity_holiday_dates(p)


class TestExpectedTradingDays:
    def test_pure_computation_excludes_weekends_and_holidays(self):
        holidays = frozenset({cal.nse_date_string(D(2026, 9, 14))})
        days = cal.expected_trading_days(D(2026, 9, 11), D(2026, 9, 16), holidays)
        assert days == [D(2026, 9, 11), D(2026, 9, 15), D(2026, 9, 16)]


class TestExpectedMinuteGrid:
    def test_360_bars_09_15_through_15_14(self):
        grid = cal.expected_minute_grid_ist()
        assert len(grid) == 360
        assert grid[0] == dt.time(9, 15) and grid[-1] == dt.time(15, 14)

    def test_official_session_close_is_distinct_from_the_bar_grid_end(self):
        assert cal.SESSION_CLOSE_IST == dt.time(15, 30)
        assert dt.time(15, 30) not in cal.expected_minute_grid_ist()


class TestSessionShapeClassification:

    def test_current_grid_matches(self):
        shape = cal.classify_session_shape(D(2026, 8, 3), set(cal.expected_minute_grid_ist()))
        assert shape == cal.SessionShape.MATCHES_CURRENT_GRID

    def test_known_historical_variant_matches_on_or_before_boundary_date(self):
        shape = cal.classify_session_shape(D(2026, 6, 29), set(cal.historical_variant_minute_grid_ist()))
        assert shape == cal.SessionShape.MATCHES_KNOWN_HISTORICAL_VARIANT

    def test_boundary_date_itself_is_still_the_variant(self):
        shape = cal.classify_session_shape(cal.HISTORICAL_VARIANT_LAST_DATE, set(cal.historical_variant_minute_grid_ist()))
        assert shape == cal.SessionShape.MATCHES_KNOWN_HISTORICAL_VARIANT

    def test_variant_shaped_bars_after_the_boundary_date_are_unrecognized(self):
        shape = cal.classify_session_shape(cal.HISTORICAL_VARIANT_LAST_DATE + dt.timedelta(days=3), set(cal.historical_variant_minute_grid_ist()))
        assert shape == cal.SessionShape.UNRECOGNIZED_SHAPE

    def test_partial_or_gappy_session_is_unrecognized_not_rejected(self):
        partial = set(cal.expected_minute_grid_ist()[:100])
        shape = cal.classify_session_shape(D(2026, 8, 3), partial)
        assert shape == cal.SessionShape.UNRECOGNIZED_SHAPE

    def test_historical_variant_grid_is_375_bars_09_15_to_15_29(self):
        grid = cal.historical_variant_minute_grid_ist()
        assert len(grid) == 375
        assert grid[0] == dt.time(9, 15) and grid[-1] == dt.time(15, 29)


class TestPhaseBRealDataSessionShapeFindings:

    def test_real_375_bar_midcap_symbol_on_a_current_date_is_unrecognized_not_forced_into_the_historical_variant(self):
        grid = set(cal.historical_variant_minute_grid_ist())
        shape = cal.classify_session_shape(D(2026, 8, 26), grid)
        assert shape == cal.SessionShape.UNRECOGNIZED_SHAPE

    def test_the_boundary_date_rule_is_unaffected_by_this_finding(self):
        grid = set(cal.historical_variant_minute_grid_ist())
        assert cal.classify_session_shape(D(2026, 6, 29), grid) == cal.SessionShape.MATCHES_KNOWN_HISTORICAL_VARIANT

    def test_current_grid_symbols_are_unaffected_by_the_finding(self):
        grid = set(cal.expected_minute_grid_ist())
        assert cal.classify_session_shape(D(2026, 8, 26), grid) == cal.SessionShape.MATCHES_CURRENT_GRID


class TestRealDataCrossCheck:

    def test_expected_trading_days_matches_the_real_s1_2_pilot_dates_exactly(self):
        raw_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / "nse_calendar"
        holiday_master = raw_dir / "holiday_master.json"
        if not holiday_master.exists():
            pytest.skip("real cached holiday_master.json not present in this environment")
        holidays = cal.load_equity_holiday_dates(holiday_master)
        expected = cal.expected_trading_days(D(2026, 8, 26), D(2026, 9, 25), holidays)
        assert len(expected) == 22
        assert D(2026, 9, 14) not in expected
        assert D(2026, 8, 26) in expected and D(2026, 9, 25) in expected
