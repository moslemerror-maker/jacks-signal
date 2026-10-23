from pathlib import Path

import pytest

from backend.app.data.instruments.universe import (
    EQUITY_BASKET,
    INDEX_SYMBOLS,
    is_equity_series,
    resolve_basket_isins,
)
from backend.app.data.normalizers.udiff import normalize_udiff

FIXTURES = Path(__file__).parent / "fixtures" / "bhavcopy"
UDIFF_FILE = FIXTURES / "BhavCopy_NSE_CM_0_0_0_20250915_F_0000.csv.zip"


def test_basket_has_25_stocks():
    assert len(EQUITY_BASKET) == 25
    assert len(set(EQUITY_BASKET)) == 25


def test_index_symbols_are_nifty_and_banknifty():
    assert INDEX_SYMBOLS == ("NIFTY 50", "NIFTY BANK")


def test_equity_series_accepts_mainboard_and_trade_for_trade():
    assert is_equity_series("EQ")
    assert is_equity_series("BE")
    assert is_equity_series("BZ")


def test_equity_series_rejects_non_equity_series():
    assert not is_equity_series("BL")
    assert not is_equity_series("N3")
    assert not is_equity_series("GB")


class TestResolveBasketIsins:
    def test_resolves_all_25_symbols_from_real_fixture(self):
        bars, _ = normalize_udiff(UDIFF_FILE)
        isins = resolve_basket_isins(bars)

        assert len(isins) == 25
        assert set(isins.keys()) == set(EQUITY_BASKET)

    def test_eternal_resolves_to_zomatos_isin(self):
        bars, _ = normalize_udiff(UDIFF_FILE)
        isins = resolve_basket_isins(bars)
        assert isins["ETERNAL"] == "INE758T01015"

    def test_raises_on_incomplete_reference_day(self):
        bars, _ = normalize_udiff(UDIFF_FILE)
        incomplete = [b for b in bars if b.symbol != "RELIANCE"]
        with pytest.raises(ValueError, match="RELIANCE"):
            resolve_basket_isins(incomplete)
