from datetime import date

import pytest
from pydantic import ValidationError

from backend.app.data.canonical.schema import CanonicalBar, SourceFormat


def _make_bar(**overrides) -> CanonicalBar:
    fields = dict(
        symbol="RELIANCE",
        series="EQ",
        isin="INE002A01018",
        trade_date=date(2024, 7, 5),
        open=100.0,
        high=110.0,
        low=95.0,
        close=105.0,
        source_format=SourceFormat.NSE_LEGACY_CSV,
        source_file="cm05JUL2024bhav.csv.zip",
    )
    fields.update(overrides)
    return CanonicalBar(**fields)


def test_valid_bar_constructs():
    bar = _make_bar()
    assert bar.symbol == "RELIANCE"
    assert bar.high == 110.0


def test_rejects_high_below_low():
    with pytest.raises(ValidationError, match="high"):
        _make_bar(high=90.0, low=95.0)


def test_rejects_open_outside_high_low_range():
    with pytest.raises(ValidationError, match="open"):
        _make_bar(open=200.0)


def test_rejects_close_outside_high_low_range():
    with pytest.raises(ValidationError, match="close"):
        _make_bar(close=1.0)
