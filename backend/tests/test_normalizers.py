from datetime import date
from pathlib import Path

import pytest

from backend.app.data.canonical.schema import SourceFormat
from backend.app.data.normalizers import (
    UDIFF_CUTOVER_DATE,
    format_for_date,
    normalize_bhavcopy,
)
from backend.app.data.normalizers.legacy_csv import normalize_legacy_csv
from backend.app.data.normalizers.udiff import normalize_udiff

FIXTURES = Path(__file__).parent / "fixtures" / "bhavcopy"
LEGACY_FILE = FIXTURES / "cm05JUL2024bhav.csv.zip"
UDIFF_FILE = FIXTURES / "BhavCopy_NSE_CM_0_0_0_20250915_F_0000.csv.zip"


class TestLegacyNormalizer:
    def test_parses_all_rows_with_no_rejections(self):
        bars, rejected = normalize_legacy_csv(LEGACY_FILE)
        assert len(bars) == 2775
        assert rejected == []

    def test_reliance_row_matches_known_values(self):
        bars, _ = normalize_legacy_csv(LEGACY_FILE)
        reliance = next(b for b in bars if b.symbol == "RELIANCE" and b.series == "EQ")
        assert reliance.isin == "INE002A01018"
        assert reliance.trade_date == date(2024, 7, 5)
        assert reliance.open == 3107.65
        assert reliance.high == 3197.0
        assert reliance.low == 3096.0
        assert reliance.close == 3177.25
        assert reliance.open_interest is None
        assert reliance.source_format is SourceFormat.NSE_LEGACY_CSV

    def test_deterministic_across_runs(self):
        bars1, _ = normalize_legacy_csv(LEGACY_FILE)
        bars2, _ = normalize_legacy_csv(LEGACY_FILE)
        assert [b.model_dump() for b in bars1] == [b.model_dump() for b in bars2]


class TestUdiffNormalizer:
    def test_parses_all_rows_with_no_rejections(self):
        bars, rejected = normalize_udiff(UDIFF_FILE)
        assert len(bars) == 3183
        assert rejected == []

    def test_reliance_row_matches_known_values(self):
        bars, _ = normalize_udiff(UDIFF_FILE)
        reliance = next(b for b in bars if b.symbol == "RELIANCE" and b.series == "EQ")
        assert reliance.isin == "INE002A01018"
        assert reliance.trade_date == date(2025, 9, 15)
        assert reliance.open == 1393.0
        assert reliance.high == 1400.8
        assert reliance.low == 1388.6
        assert reliance.close == 1399.3
        assert reliance.source_format is SourceFormat.NSE_UDIFF

    def test_bundles_multiple_series_not_just_equity(self):
        bars, _ = normalize_udiff(UDIFF_FILE)
        series_seen = {b.series for b in bars}
        assert "EQ" in series_seen
        assert "GB" in series_seen

    def test_deterministic_across_runs(self):
        bars1, _ = normalize_udiff(UDIFF_FILE)
        bars2, _ = normalize_udiff(UDIFF_FILE)
        assert [b.model_dump() for b in bars1] == [b.model_dump() for b in bars2]


class TestFormatDispatch:
    @pytest.mark.parametrize(
        "trade_date,expected",
        [
            (date(2024, 7, 5), SourceFormat.NSE_LEGACY_CSV),
            (date(2024, 7, 7), SourceFormat.NSE_LEGACY_CSV),
            (UDIFF_CUTOVER_DATE, SourceFormat.NSE_UDIFF),
            (date(2024, 7, 9), SourceFormat.NSE_UDIFF),
            (date(2015, 1, 2), SourceFormat.NSE_LEGACY_CSV),
        ],
    )
    def test_format_for_date_boundary(self, trade_date, expected):
        assert format_for_date(trade_date) is expected

    def test_dispatch_picks_legacy_normalizer_by_date(self):
        bars, _ = normalize_bhavcopy(LEGACY_FILE, date(2024, 7, 5))
        assert bars[0].source_format is SourceFormat.NSE_LEGACY_CSV

    def test_dispatch_picks_udiff_normalizer_by_date(self):
        bars, _ = normalize_bhavcopy(UDIFF_FILE, date(2025, 9, 15))
        assert bars[0].source_format is SourceFormat.NSE_UDIFF
