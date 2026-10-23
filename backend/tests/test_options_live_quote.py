from __future__ import annotations

import dataclasses
import datetime as dt
from decimal import Decimal

import pytest

from backend.app.options.contracts import ContractIdentity
from backend.app.options.live_quote import LiveQuote, classify_live_quote, is_stale, quote_age

D = dt.date
UTC = dt.timezone.utc
FIXTURE = "DEVELOPMENT_FIXTURE"
CONTRACT = ContractIdentity("NIFTY", "IDO", D(2024, 7, 25), 2200000, "CE")
OBSERVED_AT = dt.datetime(2024, 7, 5, 10, 30, tzinfo=UTC)


def mkquote(**overrides) -> LiveQuote:
    fields = dict(
        contract=CONTRACT, observed_at=OBSERVED_AT, last_price=Decimal("11"), bid=Decimal("10.8"), ask=Decimal("11.2"),
        volume=100, open_interest=5000, underlying_last_price=Decimal("22050"), provider_reported_iv=Decimal("14.5"),
        provider_reported_greeks={"delta": Decimal("0.52")}, source="dhan", source_row_hash=None, provenance_class=FIXTURE,
    )
    fields.update(overrides)
    return LiveQuote.from_raw(**fields)


class TestClassifyLiveQuote:
    def test_quoted(self):
        assert classify_live_quote(Decimal(11), Decimal("10.8"), Decimal("11.2")) == "QUOTED"

    def test_no_quote(self):
        assert classify_live_quote(None, None, None) == "NO_QUOTE"

    def test_quoted_with_only_last_price(self):
        assert classify_live_quote(Decimal(11), None, None) == "QUOTED"

    def test_nonpositive_last_price(self):
        assert classify_live_quote(Decimal(0), None, None) == "ANOMALY_NONPOSITIVE_PRICE"

    def test_negative_bid(self):
        assert classify_live_quote(None, Decimal(-1), Decimal(5)) == "ANOMALY_NONPOSITIVE_PRICE"

    def test_crossed_book(self):
        assert classify_live_quote(Decimal(11), Decimal(12), Decimal(10)) == "ANOMALY_BID_ABOVE_ASK"

    def test_nonpositive_checked_before_crossed_book(self):
        assert classify_live_quote(None, Decimal(0), Decimal(5)) == "ANOMALY_NONPOSITIVE_PRICE"


class TestLiveQuoteConstruction:
    def test_from_raw_derives_status(self):
        q = mkquote()
        assert q.status == "QUOTED"

    def test_from_raw_no_quote(self):
        q = mkquote(last_price=None, bid=None, ask=None)
        assert q.status == "NO_QUOTE" and q.is_usable is False

    def test_is_usable_true_only_for_quoted(self):
        assert mkquote().is_usable is True
        assert mkquote(last_price=None, bid=None, ask=None).is_usable is False

    def test_direct_construction_rejects_inconsistent_status(self):
        with pytest.raises(ValueError, match="inconsistent"):
            LiveQuote(
                contract=CONTRACT, observed_at=OBSERVED_AT, last_price=Decimal(11), bid=None, ask=None, volume=None,
                open_interest=None, underlying_last_price=None, provider_reported_iv=None, provider_reported_greeks=None,
                status="NO_QUOTE", source="dhan", source_row_hash=None, provenance_class=FIXTURE,
            )

    def test_naive_observed_at_rejected(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            mkquote(observed_at=dt.datetime(2024, 7, 5, 10, 30))

    def test_nonpositive_underlying_last_price_rejected(self):
        with pytest.raises(ValueError, match="underlying_last_price"):
            mkquote(underlying_last_price=Decimal(0))

    def test_nonpositive_provider_reported_iv_rejected(self):
        with pytest.raises(ValueError, match="provider_reported_iv"):
            mkquote(provider_reported_iv=Decimal(-1))

    def test_empty_source_rejected(self):
        with pytest.raises(ValueError, match="source"):
            mkquote(source="")

    def test_unknown_provenance_class_rejected(self):
        with pytest.raises(ValueError, match="provenance_class"):
            mkquote(provenance_class="MADE_UP")  # type: ignore[arg-type]

    def test_provider_reported_greeks_never_becomes_our_own_greeks_type(self):
        from backend.app.options.valuation import Greeks
        q = mkquote()
        assert isinstance(q.provider_reported_greeks, dict)
        assert not isinstance(q.provider_reported_greeks, Greeks)

    def test_frozen(self):
        q = mkquote()
        with pytest.raises(dataclasses.FrozenInstanceError):
            q.last_price = Decimal(0)  # type: ignore[misc]


class TestQuoteAgeAndStaleness:
    def test_age_computed_correctly(self):
        q = mkquote()
        later = OBSERVED_AT + dt.timedelta(minutes=5)
        assert quote_age(q, later) == dt.timedelta(minutes=5)

    def test_as_of_before_observed_at_rejected(self):
        q = mkquote()
        with pytest.raises(ValueError, match="no look-ahead"):
            quote_age(q, OBSERVED_AT - dt.timedelta(seconds=1))

    def test_naive_as_of_rejected(self):
        q = mkquote()
        with pytest.raises(ValueError, match="timezone-aware"):
            quote_age(q, dt.datetime(2024, 7, 5, 10, 35))

    def test_is_stale_true_when_older_than_max_age(self):
        q = mkquote()
        later = OBSERVED_AT + dt.timedelta(minutes=10)
        assert is_stale(q, later, dt.timedelta(minutes=5)) is True

    def test_is_stale_false_when_within_max_age(self):
        q = mkquote()
        later = OBSERVED_AT + dt.timedelta(minutes=2)
        assert is_stale(q, later, dt.timedelta(minutes=5)) is False

    def test_negative_max_age_rejected(self):
        q = mkquote()
        with pytest.raises(ValueError, match="max_age"):
            is_stale(q, OBSERVED_AT, dt.timedelta(seconds=-1))

    def test_exactly_at_max_age_is_not_stale(self):
        q = mkquote()
        later = OBSERVED_AT + dt.timedelta(minutes=5)
        assert is_stale(q, later, dt.timedelta(minutes=5)) is False
