from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from backend.app.options.contracts import ContractIdentity
from backend.app.options.liquidity import assess_leg_liquidity
from backend.app.options.live_quote import LiveQuote

D = dt.date
UTC = dt.timezone.utc
FIXTURE = "DEVELOPMENT_FIXTURE"
CONTRACT = ContractIdentity("NIFTY", "IDO", D(2026, 9, 29), 2450000, "CE")
OBSERVED_AT = dt.datetime(2026, 9, 22, 10, 0, tzinfo=UTC)


def mkquote(**overrides) -> LiveQuote:
    fields = dict(
        contract=CONTRACT, observed_at=OBSERVED_AT, last_price=Decimal("150"), bid=Decimal("149"), ask=Decimal("151"),
        volume=1000, open_interest=5000, underlying_last_price=Decimal("24500"), provider_reported_iv=None,
        provider_reported_greeks=None, source="dhan", source_row_hash=None, provenance_class=FIXTURE,
    )
    fields.update(overrides)
    return LiveQuote.from_raw(**fields)


DEFAULT_KW = dict(as_of=OBSERVED_AT + dt.timedelta(minutes=1), max_quote_age=dt.timedelta(minutes=5), max_spread_fraction=Decimal("0.05"), min_volume=None)


class TestAssessLegLiquidity:
    def test_good_quote_is_acceptable(self):
        r = assess_leg_liquidity(mkquote(), **DEFAULT_KW)
        assert r.is_acceptable is True and r.reasons == ()

    def test_no_quote_rejected_with_single_reason(self):
        r = assess_leg_liquidity(mkquote(last_price=None, bid=None, ask=None), **DEFAULT_KW)
        assert r.is_acceptable is False
        assert len(r.reasons) == 1 and "NO_QUOTE" in r.reasons[0]

    def test_stale_quote_rejected(self):
        r = assess_leg_liquidity(mkquote(), as_of=OBSERVED_AT + dt.timedelta(minutes=10), max_quote_age=dt.timedelta(minutes=5), max_spread_fraction=Decimal("0.05"), min_volume=None)
        assert r.is_acceptable is False and any("stale" in x for x in r.reasons)

    def test_wide_spread_rejected(self):
        r = assess_leg_liquidity(mkquote(bid=Decimal("100"), ask=Decimal("200")), **DEFAULT_KW)
        assert r.is_acceptable is False and any("spread" in x for x in r.reasons)

    def test_missing_bid_ask_rejected(self):
        r = assess_leg_liquidity(mkquote(bid=None, ask=None), **DEFAULT_KW)
        assert r.is_acceptable is False and any("bid/ask" in x for x in r.reasons)

    def test_volume_below_minimum_rejected(self):
        r = assess_leg_liquidity(mkquote(volume=10), as_of=OBSERVED_AT, max_quote_age=dt.timedelta(minutes=5), max_spread_fraction=Decimal("0.05"), min_volume=100)
        assert r.is_acceptable is False and any("volume" in x for x in r.reasons)

    def test_volume_at_or_above_minimum_accepted(self):
        r = assess_leg_liquidity(mkquote(volume=100), as_of=OBSERVED_AT, max_quote_age=dt.timedelta(minutes=5), max_spread_fraction=Decimal("0.05"), min_volume=100)
        assert r.is_acceptable is True

    def test_missing_volume_when_required_is_rejected_not_skipped(self):
        r = assess_leg_liquidity(mkquote(volume=None), as_of=OBSERVED_AT, max_quote_age=dt.timedelta(minutes=5), max_spread_fraction=Decimal("0.05"), min_volume=100)
        assert r.is_acceptable is False and any("volume" in x for x in r.reasons)

    def test_min_volume_none_means_no_volume_check(self):
        r = assess_leg_liquidity(mkquote(volume=None), **DEFAULT_KW)
        assert r.is_acceptable is True

    def test_multiple_failures_all_reported(self):
        r = assess_leg_liquidity(mkquote(bid=None, ask=None, volume=1), as_of=OBSERVED_AT, max_quote_age=dt.timedelta(minutes=5), max_spread_fraction=Decimal("0.05"), min_volume=100)
        assert len(r.reasons) >= 2

    def test_result_invariant_enforced(self):
        from backend.app.options.liquidity import LiquidityAssessment
        with pytest.raises(ValueError, match="is_acceptable"):
            LiquidityAssessment(is_acceptable=True, reasons=("x",))

    def test_deterministic(self):
        q = mkquote()
        assert assess_leg_liquidity(q, **DEFAULT_KW) == assess_leg_liquidity(q, **DEFAULT_KW)
