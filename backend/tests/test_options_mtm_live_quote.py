from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from backend.app.options.contracts import ContractIdentity
from backend.app.options.live_quote import LiveQuote
from backend.app.options.mtm import valuation_from_live_quote

D = dt.date
UTC = dt.timezone.utc
FIXTURE = "DEVELOPMENT_FIXTURE"
EXPIRY = D(2026, 10, 6)
NOW = dt.datetime(2026, 9, 25, 5, 0, tzinfo=UTC)


def contract(strike=24500, cp="CE"):
    return ContractIdentity("NIFTY", "IDO", EXPIRY, strike * 100, cp)


def quote(*, last=None, bid=None, ask=None, observed_at=NOW, strike=24500, cp="CE"):
    return LiveQuote.from_raw(
        contract=contract(strike, cp), observed_at=observed_at,
        last_price=Decimal(last) if last is not None else None, bid=Decimal(bid) if bid is not None else None,
        ask=Decimal(ask) if ask is not None else None, volume=500, open_interest=1000,
        underlying_last_price=Decimal("24500"), provider_reported_iv=None, provider_reported_greeks=None,
        source="dhan", source_row_hash="mtm-test", provenance_class=FIXTURE,
    )


class TestValuationFromLiveQuoteBasic:
    def test_last_price_is_the_default_price_kind(self):
        q = quote(last="140", bid="138", ask="142")
        v = valuation_from_live_quote(q)
        assert v is not None
        assert v.price == Decimal("140")
        assert v.source == "OBSERVED_MARKET_PRICE"
        assert v.contract == q.contract
        assert v.valuation_at == NOW

    def test_explicit_last_price_kind(self):
        q = quote(last="140", bid="138", ask="142")
        v = valuation_from_live_quote(q, price_kind="LAST_PRICE")
        assert v.price == Decimal("140")


class TestExecutableBidExitValuation:
    def test_long_option_close_uses_bid(self):
        q = quote(last="140", bid="135", ask="145")
        v = valuation_from_live_quote(q, price_kind="BID")
        assert v.price == Decimal("135")
        assert v.source == "OBSERVED_MARKET_PRICE"

    def test_ask_price_kind_for_completeness(self):
        q = quote(last="140", bid="135", ask="145")
        v = valuation_from_live_quote(q, price_kind="ASK")
        assert v.price == Decimal("145")

    def test_invalid_price_kind_rejected(self):
        q = quote(last="140", bid="135", ask="145")
        with pytest.raises(ValueError, match="price_kind must be one of"):
            valuation_from_live_quote(q, price_kind="MID")  # type: ignore[arg-type]


class TestMissingOrInvalidQuoteHandling:
    def test_no_quote_status_returns_none_never_fabricated(self):
        q = quote()
        assert valuation_from_live_quote(q) is None
        assert valuation_from_live_quote(q, price_kind="BID") is None

    def test_quoted_but_no_bid_returns_none_for_bid_price_kind(self):
        q = quote(last="140")
        assert valuation_from_live_quote(q, price_kind="LAST_PRICE") is not None
        assert valuation_from_live_quote(q, price_kind="BID") is None
        assert valuation_from_live_quote(q, price_kind="ASK") is None

    def test_anomalous_quote_returns_none(self):
        q = quote(last="140", bid="145", ask="140")
        assert valuation_from_live_quote(q, price_kind="LAST_PRICE") is None
        assert valuation_from_live_quote(q, price_kind="BID") is None

    def test_no_look_ahead(self):
        q = quote(last="140", observed_at=NOW)
        with pytest.raises(ValueError, match="no look-ahead"):
            valuation_from_live_quote(q, valuation_at=NOW - dt.timedelta(minutes=1))

    def test_valuation_at_must_be_timezone_aware(self):
        q = quote(last="140", observed_at=NOW)
        with pytest.raises(ValueError, match="timezone-aware"):
            valuation_from_live_quote(q, valuation_at=dt.datetime(2026, 9, 25, 5, 1))

    def test_defaults_valuation_at_to_observed_at(self):
        q = quote(last="140", observed_at=NOW)
        v = valuation_from_live_quote(q)
        assert v.valuation_at == NOW
