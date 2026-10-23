from pathlib import Path

from backend.app.data.sources.nse_calendar import equity_holiday_dates, fo_holiday_dates

FIXTURES = Path(__file__).parent / "fixtures" / "calendar"
HOLIDAY_MASTER_FILE = FIXTURES / "holiday_master.json"


def test_returns_cm_segment_holiday_dates():
    dates = equity_holiday_dates(HOLIDAY_MASTER_FILE)
    assert len(dates) == 20


def test_includes_known_current_holiday():
    dates = equity_holiday_dates(HOLIDAY_MASTER_FILE)
    assert "14-Sep-2026" in dates


def test_fo_segment_returns_its_own_holiday_dates():
    dates = fo_holiday_dates(HOLIDAY_MASTER_FILE)
    assert len(dates) == 20
    assert "14-Sep-2026" in dates


def test_fo_and_cm_are_read_from_genuinely_different_keys(tmp_path):
    import json
    synthetic = tmp_path / "holiday_master.json"
    synthetic.write_text(json.dumps({
        "CM": [{"tradingDate": "01-Jan-2027"}],
        "FO": [{"tradingDate": "02-Jan-2027"}, {"tradingDate": "03-Jan-2027"}],
    }))
    assert equity_holiday_dates(synthetic) == {"01-Jan-2027"}
    assert fo_holiday_dates(synthetic) == {"02-Jan-2027", "03-Jan-2027"}
