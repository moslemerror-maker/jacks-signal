from __future__ import annotations

import datetime as dt
from decimal import Decimal

from backend.app.options.candidate_generator import CandidateGenerationConfig, generate_candidates
from backend.app.options.contracts import ContractIdentity
from backend.app.options.forward import resolve_forward_spot_proxy
from backend.app.options.live_quote import LiveQuote

D = dt.date
UTC = dt.timezone.utc
FIXTURE = "DEVELOPMENT_FIXTURE"
EXPIRY = D(2026, 9, 29)
NOW = dt.datetime(2026, 9, 22, 10, 0, tzinfo=UTC)
FORWARD = resolve_forward_spot_proxy(Decimal("24500"))

CONFIG = CandidateGenerationConfig(
    lots=1, lot_size=75, lot_size_source="CALLER_SUPPLIED", risk_free_rate=Decimal("0.065"),
    max_quote_age=dt.timedelta(minutes=5), max_spread_fraction=Decimal("0.10"), min_volume=None,
)


def opt(strike_rupees: int, cp: str) -> ContractIdentity:
    return ContractIdentity("NIFTY", "IDO", EXPIRY, strike_rupees * 100, cp)


def mkq(strike: int, cp: str, *, last_price=None, bid=None, ask=None, observed_at=NOW) -> LiveQuote:
    return LiveQuote.from_raw(
        contract=opt(strike, cp), observed_at=observed_at, last_price=last_price, bid=bid, ask=ask, volume=500,
        open_interest=1000, underlying_last_price=Decimal("24500"), provider_reported_iv=None, provider_reported_greeks=None,
        source="dhan", source_row_hash=None, provenance_class=FIXTURE,
    )


def full_chain() -> list[LiveQuote]:
    return [
        mkq(24500, "CE", last_price=Decimal("140"), bid=Decimal("138"), ask=Decimal("142")),
        mkq(24500, "PE", last_price=Decimal("135"), bid=Decimal("133"), ask=Decimal("137")),
        mkq(24400, "CE", last_price=Decimal("190"), bid=Decimal("188"), ask=Decimal("192")),
        mkq(24400, "PE", last_price=Decimal("95"), bid=Decimal("93"), ask=Decimal("97")),
        mkq(24600, "CE", last_price=Decimal("100"), bid=Decimal("98"), ask=Decimal("102")),
        mkq(24600, "PE", last_price=Decimal("185"), bid=Decimal("183"), ask=Decimal("187")),
    ]


def gen(quotes, **overrides):
    kwargs = dict(quotes=quotes, expiry=EXPIRY, forward_resolution=FORWARD, config=CONFIG, timestamp=NOW, as_of=NOW)
    kwargs.update(overrides)
    return generate_candidates(**kwargs)


class TestAllThreeShapesFromAFullChain:
    def test_all_three_succeed(self):
        outcomes = gen(full_chain())
        assert {o.shape for o in outcomes} == {"LONG_CALL_ATM", "LONG_PUT_ATM", "LONG_STRADDLE_ATM"}
        assert all(o.candidate is not None for o in outcomes)

    def test_long_call_uses_the_atm_strike(self):
        outcomes = gen(full_chain())
        call = next(o for o in outcomes if o.shape == "LONG_CALL_ATM")
        assert call.candidate.structure.legs[0].contract.strike_paise == 2450000
        assert call.candidate.structure.legs[0].side == "BUY"

    def test_straddle_has_two_legs_same_strike(self):
        outcomes = gen(full_chain())
        straddle = next(o for o in outcomes if o.shape == "LONG_STRADDLE_ATM")
        legs = straddle.candidate.structure.legs
        assert len(legs) == 2
        assert {leg.contract.option_type for leg in legs} == {"CE", "PE"}
        assert legs[0].contract.strike_paise == legs[1].contract.strike_paise == 2450000

    def test_forward_resolution_attached_to_every_candidate(self):
        outcomes = gen(full_chain())
        for o in outcomes:
            assert o.candidate.forward_resolution == FORWARD

    def test_iv_and_greeks_context_populated_when_recoverable(self):
        outcomes = gen(full_chain())
        call = next(o for o in outcomes if o.shape == "LONG_CALL_ATM")
        assert len(call.candidate.iv_context) == 1
        assert call.candidate.iv_context[0].status == "OK"
        assert len(call.candidate.greeks_context) == 1
        assert call.candidate.data_quality == "EOD_RESOLVED"

    def test_entry_price_source_is_live_quote(self):
        outcomes = gen(full_chain())
        call = next(o for o in outcomes if o.shape == "LONG_CALL_ATM")
        assert call.candidate.structure.legs[0].entry_price_source == "LIVE_QUOTE"


class TestPerShapeRejection:
    def test_missing_pe_at_all_strikes_rejects_put_and_straddle_but_not_call(self):
        quotes = [q for q in full_chain() if q.contract.option_type != "PE"]
        outcomes = gen(quotes)
        by_shape = {o.shape: o for o in outcomes}
        assert by_shape["LONG_CALL_ATM"].candidate is not None
        assert by_shape["LONG_PUT_ATM"].candidate is None
        assert by_shape["LONG_STRADDLE_ATM"].candidate is None

    def test_illiquid_nearest_strike_falls_back_to_next_nearest(self):
        quotes = full_chain()
        quotes = [q if not (q.contract.strike_paise == 2450000 and q.contract.option_type == "CE") else mkq(24500, "CE") for q in quotes]
        outcomes = gen(quotes)
        call = next(o for o in outcomes if o.shape == "LONG_CALL_ATM")
        assert call.candidate is not None
        assert call.candidate.structure.legs[0].contract.strike_paise != 2450000

    def test_no_usable_strike_anywhere_rejects_all_three(self):
        quotes = [mkq(24500, "CE"), mkq(24500, "PE")]
        outcomes = gen(quotes)
        assert all(o.candidate is None for o in outcomes)
        assert all("no strike" in o.rejected_reason for o in outcomes)

    def test_wide_spread_leg_is_rejected_with_liquidity_reason(self):
        quotes = [mkq(24500, "CE", last_price=Decimal("140"), bid=Decimal("50"), ask=Decimal("230")),
                  mkq(24500, "PE", last_price=Decimal("135"), bid=Decimal("133"), ask=Decimal("137"))]
        outcomes = gen(quotes)
        call = next(o for o in outcomes if o.shape == "LONG_CALL_ATM")
        assert call.candidate is None and "liquidity" in call.rejected_reason

    def test_empty_quote_list_rejects_all_three(self):
        outcomes = gen([])
        assert all(o.candidate is None for o in outcomes)


class TestLegQuoteAndLiquidityPreservation:

    def test_single_leg_shape_preserves_the_exact_quote_used(self):
        outcomes = gen(full_chain())
        call = next(o for o in outcomes if o.shape == "LONG_CALL_ATM")
        assert len(call.leg_quotes) == 1
        used = call.leg_quotes[0]
        assert used.contract.strike_paise == 2450000 and used.contract.option_type == "CE"
        assert used.last_price == Decimal("140") and used.bid == Decimal("138") and used.ask == Decimal("142")
        assert used.volume == 500 and used.open_interest == 1000

    def test_straddle_preserves_both_legs_in_structure_order(self):
        outcomes = gen(full_chain())
        straddle = next(o for o in outcomes if o.shape == "LONG_STRADDLE_ATM")
        assert len(straddle.leg_quotes) == 2
        assert [q.contract.option_type for q in straddle.leg_quotes] == [leg.contract.option_type for leg in straddle.candidate.structure.legs]

    def test_liquidity_assessment_is_acceptable_for_every_used_leg(self):
        outcomes = gen(full_chain())
        call = next(o for o in outcomes if o.shape == "LONG_CALL_ATM")
        assert len(call.leg_liquidity) == 1
        assert call.leg_liquidity[0].is_acceptable is True
        assert call.leg_liquidity[0].reasons == ()

    def test_fallback_strike_preserves_the_quote_actually_selected_not_the_rejected_one(self):
        quotes = full_chain()
        quotes = [q if not (q.contract.strike_paise == 2450000 and q.contract.option_type == "CE") else mkq(24500, "CE") for q in quotes]
        outcomes = gen(quotes)
        call = next(o for o in outcomes if o.shape == "LONG_CALL_ATM")
        assert call.leg_quotes[0].contract.strike_paise == call.candidate.structure.legs[0].contract.strike_paise
        assert call.leg_quotes[0].contract.strike_paise != 2450000

    def test_rejected_shape_carries_no_leg_quotes_or_liquidity(self):
        quotes = [mkq(24500, "CE"), mkq(24500, "PE")]
        outcomes = gen(quotes)
        assert all(o.leg_quotes == () and o.leg_liquidity == () for o in outcomes)

    def test_provider_iv_and_greeks_survive_on_the_preserved_quote_when_present(self):
        quotes = full_chain()
        quotes = [q if not (q.contract.strike_paise == 2450000 and q.contract.option_type == "CE") else
                 LiveQuote.from_raw(contract=opt(24500, "CE"), observed_at=NOW, last_price=Decimal("140"), bid=Decimal("138"), ask=Decimal("142"),
                                    volume=500, open_interest=1000, underlying_last_price=Decimal("24500"),
                                    provider_reported_iv=Decimal("0.15"), provider_reported_greeks={"delta": Decimal("0.5")},
                                    source="dhan", source_row_hash=None, provenance_class=FIXTURE)
                 for q in quotes]
        outcomes = gen(quotes)
        call = next(o for o in outcomes if o.shape == "LONG_CALL_ATM")
        assert call.leg_quotes[0].provider_reported_iv == Decimal("0.15")
        assert call.leg_quotes[0].provider_reported_greeks == {"delta": Decimal("0.5")}
