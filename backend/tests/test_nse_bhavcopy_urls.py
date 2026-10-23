from datetime import date

from backend.app.data.sources.nse_bhavcopy import bhavcopy_url


def test_legacy_url_matches_verified_poc_pattern():
    assert bhavcopy_url(date(2024, 7, 5)) == (
        "https://nsearchives.nseindia.com/content/historical/EQUITIES/"
        "2024/JUL/cm05JUL2024bhav.csv.zip"
    )


def test_udiff_url_matches_verified_poc_pattern():
    assert bhavcopy_url(date(2025, 9, 15)) == (
        "https://nsearchives.nseindia.com/content/cm/"
        "BhavCopy_NSE_CM_0_0_0_20250915_F_0000.csv.zip"
    )


def test_url_selects_udiff_on_cutover_date():
    assert "BhavCopy_NSE_CM" in bhavcopy_url(date(2024, 7, 8))


def test_url_selects_legacy_the_day_before_cutover():
    assert "bhav.csv.zip" in bhavcopy_url(date(2024, 7, 7))
