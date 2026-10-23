from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from backend.app.options.trading_calendar import (
    HolidayCalendarUnavailable, is_trading_day, load_fo_holiday_dates, nse_date_string,
)

D = dt.date
HOLIDAYS = frozenset({"14-Sep-2026", "02-Oct-2026", "25-Dec-2026"})


class TestNseDateString:
    def test_matches_nse_own_format(self):
        assert nse_date_string(D(2026, 9, 14)) == "14-Sep-2026"

    def test_zero_pads_single_digit_days(self):
        assert nse_date_string(D(2026, 1, 5)) == "05-Jan-2026"

    def test_never_depends_on_locale(self):
        assert nse_date_string(D(2026, 3, 1)) == "01-Mar-2026"


class TestIsTradingDay:
    def test_ordinary_weekday_not_a_holiday_is_a_trading_day(self):
        assert D(2026, 9, 15).weekday() == 1
        assert is_trading_day(D(2026, 9, 15), HOLIDAYS) is True

    def test_saturday_is_never_a_trading_day(self):
        saturday = D(2026, 9, 19)
        assert saturday.weekday() == 5
        assert is_trading_day(saturday, HOLIDAYS) is False

    def test_sunday_is_never_a_trading_day(self):
        sunday = D(2026, 9, 20)
        assert sunday.weekday() == 6
        assert is_trading_day(sunday, HOLIDAYS) is False

    def test_a_listed_holiday_on_a_weekday_is_not_a_trading_day(self):
        assert D(2026, 9, 14).weekday() == 0
        assert is_trading_day(D(2026, 9, 14), HOLIDAYS) is False

    def test_empty_holiday_set_still_respects_weekends(self):
        assert is_trading_day(D(2026, 9, 19), frozenset()) is False
        assert is_trading_day(D(2026, 9, 15), frozenset()) is True


class TestLoadFoHolidayDates:
    FIXTURES = Path(__file__).parent / "fixtures" / "calendar"
    HOLIDAY_MASTER_FILE = FIXTURES / "holiday_master.json"

    def test_loads_real_fixture_fo_dates(self):
        dates = load_fo_holiday_dates(self.HOLIDAY_MASTER_FILE)
        assert "14-Sep-2026" in dates
        assert len(dates) == 20

    def test_missing_file_raises_not_silently_assumes_no_holidays(self, tmp_path):
        with pytest.raises(HolidayCalendarUnavailable, match="not found"):
            load_fo_holiday_dates(tmp_path / "does_not_exist.json")

    def test_malformed_json_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        with pytest.raises(HolidayCalendarUnavailable, match="not valid JSON"):
            load_fo_holiday_dates(bad)

    def test_missing_fo_key_raises(self, tmp_path):
        no_fo = tmp_path / "no_fo.json"
        no_fo.write_text(json.dumps({"CM": [{"tradingDate": "01-Jan-2027"}]}))
        with pytest.raises(HolidayCalendarUnavailable, match="FO"):
            load_fo_holiday_dates(no_fo)

    def test_malformed_fo_rows_raise(self, tmp_path):
        bad_shape = tmp_path / "bad_shape.json"
        bad_shape.write_text(json.dumps({"FO": ["not_a_dict"]}))
        with pytest.raises(HolidayCalendarUnavailable):
            load_fo_holiday_dates(bad_shape)


class TestEndToEndWithRealFixture:

    FIXTURES = Path(__file__).parent / "fixtures" / "calendar"
    HOLIDAY_MASTER_FILE = FIXTURES / "holiday_master.json"

    def test_the_known_holiday_is_correctly_rejected(self):
        dates = load_fo_holiday_dates(self.HOLIDAY_MASTER_FILE)
        assert is_trading_day(D(2026, 9, 14), dates) is False

    def test_an_ordinary_nearby_weekday_is_accepted(self):
        dates = load_fo_holiday_dates(self.HOLIDAY_MASTER_FILE)
        assert is_trading_day(D(2026, 9, 15), dates) is True
