from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from backend.app.data.sources.dhan_live_quote import parse_dhan_optionchain

D = dt.date
UTC = dt.timezone.utc
FIXTURE = "DEVELOPMENT_FIXTURE"
OBSERVED_AT = dt.datetime(2026, 9, 22, 10, 0, tzinfo=UTC)
EXPIRY = D(2026, 9, 23)


_DEFAULT_GREEKS = {"delta": 0.52, "theta": -5.2, "gamma": 0.002, "vega": 10.1}
_UNSET = object()


def leg(last_price=150.5, top_bid_price=150.0, top_ask_price=151.0, volume=12345, oi=6789, implied_volatility=14.5,
        greeks=_UNSET) -> dict:
    body = {
        "last_price": last_price, "top_bid_price": top_bid_price, "top_ask_price": top_ask_price, "volume": volume,
        "oi": oi, "implied_volatility": implied_volatility,
    }
    body["greeks"] = dict(_DEFAULT_GREEKS) if greeks is _UNSET else greeks
    return body


def double_nested_payload(**strike_legs) -> dict:
    return {"status": "success", "data": {"data": {"last_price": 24500.15, "oc": strike_legs}}}


def single_nested_payload(**strike_legs) -> dict:
    return {"status": "success", "data": {"last_price": 24500.15, "oc": strike_legs}}


class TestParseDhanOptionChain:
    def test_double_nested_shape_parsed(self):
        raw = double_nested_payload(**{"24500.000000": {"ce": leg(), "pe": leg(last_price=140.0)}})
        quotes = parse_dhan_optionchain(raw, underlying="NIFTY", expiry=EXPIRY, observed_at=OBSERVED_AT, source_row_hash=None, provenance_class=FIXTURE)
        assert len(quotes) == 2
        ce = next(q for q in quotes if q.contract.option_type == "CE")
        assert ce.contract.strike_paise == 2450000
        assert ce.contract.underlying == "NIFTY" and ce.contract.expiry == EXPIRY
        assert ce.last_price == Decimal("150.5")
        assert ce.bid == Decimal("150") and ce.ask == Decimal("151")
        assert ce.volume == 12345 and ce.open_interest == 6789
        assert ce.underlying_last_price == Decimal("24500.15")
        assert ce.provider_reported_iv == Decimal("14.5")
        assert ce.provider_reported_greeks == {"delta": Decimal("0.52"), "theta": Decimal("-5.2"), "gamma": Decimal("0.002"), "vega": Decimal("10.1")}
        assert ce.source == "dhan" and ce.provenance_class == FIXTURE
        assert ce.status == "QUOTED"

    def test_single_nested_shape_also_tolerated(self):
        raw = single_nested_payload(**{"24500.000000": {"ce": leg(), "pe": leg()}})
        quotes = parse_dhan_optionchain(raw, underlying="NIFTY", expiry=EXPIRY, observed_at=OBSERVED_AT, source_row_hash=None, provenance_class=FIXTURE)
        assert len(quotes) == 2

    def test_missing_iv_produces_none_not_zero(self):
        raw = double_nested_payload(**{"24500.000000": {"ce": leg(implied_volatility=None)}})
        (q,) = parse_dhan_optionchain(raw, underlying="NIFTY", expiry=EXPIRY, observed_at=OBSERVED_AT, source_row_hash=None, provenance_class=FIXTURE)
        assert q.provider_reported_iv is None

    def test_zero_last_price_treated_as_not_reported(self):
        raw = double_nested_payload(**{"24500.000000": {"ce": leg(last_price=0, top_bid_price=0, top_ask_price=0)}})
        (q,) = parse_dhan_optionchain(raw, underlying="NIFTY", expiry=EXPIRY, observed_at=OBSERVED_AT, source_row_hash=None, provenance_class=FIXTURE)
        assert q.last_price is None and q.bid is None and q.ask is None
        assert q.status == "NO_QUOTE"

    def test_illiquid_leg_with_nothing_reported_is_no_quote(self):
        raw = double_nested_payload(**{"24500.000000": {"ce": leg(last_price=None, top_bid_price=None, top_ask_price=None, volume=0, oi=0, implied_volatility=None, greeks={})}})
        (q,) = parse_dhan_optionchain(raw, underlying="NIFTY", expiry=EXPIRY, observed_at=OBSERVED_AT, source_row_hash=None, provenance_class=FIXTURE)
        assert q.status == "NO_QUOTE"

    def test_only_ce_present_for_a_strike_emits_one_quote(self):
        raw = double_nested_payload(**{"24500.000000": {"ce": leg()}})
        quotes = parse_dhan_optionchain(raw, underlying="NIFTY", expiry=EXPIRY, observed_at=OBSERVED_AT, source_row_hash=None, provenance_class=FIXTURE)
        assert len(quotes) == 1 and quotes[0].contract.option_type == "CE"

    def test_unparsable_strike_key_skipped_without_crashing(self):
        raw = double_nested_payload(**{"not-a-number": {"ce": leg()}, "24500.000000": {"ce": leg()}})
        quotes = parse_dhan_optionchain(raw, underlying="NIFTY", expiry=EXPIRY, observed_at=OBSERVED_AT, source_row_hash=None, provenance_class=FIXTURE)
        assert len(quotes) == 1

    def test_empty_option_chain_returns_empty_list(self):
        raw = double_nested_payload()
        assert parse_dhan_optionchain(raw, underlying="NIFTY", expiry=EXPIRY, observed_at=OBSERVED_AT, source_row_hash=None, provenance_class=FIXTURE) == []

    def test_missing_greeks_returns_none(self):
        raw = double_nested_payload(**{"24500.000000": {"ce": leg(greeks=None)}})
        (q,) = parse_dhan_optionchain(raw, underlying="NIFTY", expiry=EXPIRY, observed_at=OBSERVED_AT, source_row_hash=None, provenance_class=FIXTURE)
        assert q.provider_reported_greeks is None

    def test_naive_observed_at_rejected(self):
        raw = double_nested_payload(**{"24500.000000": {"ce": leg()}})
        with pytest.raises(ValueError, match="timezone-aware"):
            parse_dhan_optionchain(raw, underlying="NIFTY", expiry=EXPIRY, observed_at=dt.datetime(2026, 9, 22, 10, 0), source_row_hash=None, provenance_class=FIXTURE)

    def test_source_row_hash_passthrough(self):
        raw = double_nested_payload(**{"24500.000000": {"ce": leg()}})
        (q,) = parse_dhan_optionchain(raw, underlying="NIFTY", expiry=EXPIRY, observed_at=OBSERVED_AT, source_row_hash="abc123", provenance_class=FIXTURE)
        assert q.source_row_hash == "abc123"

    def test_production_provenance_passthrough(self):
        raw = double_nested_payload(**{"24500.000000": {"ce": leg()}})
        (q,) = parse_dhan_optionchain(raw, underlying="NIFTY", expiry=EXPIRY, observed_at=OBSERVED_AT, source_row_hash=None, provenance_class="PRODUCTION_RESEARCH_DATASET")
        assert q.provenance_class == "PRODUCTION_RESEARCH_DATASET"
