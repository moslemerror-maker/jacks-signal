from __future__ import annotations

import dataclasses
import datetime as dt
from decimal import Decimal

import pytest

from backend.app.options import availability as avail
from backend.app.options.bars import DerivativeBar
from backend.app.options.contracts import ContractIdentity, ContractSpec
from backend.app.options.structures import (
    OptionLeg, OptionStructure, PnLExtremum, expiry_pnl_curve, leg_expiry_payoff, leg_expiry_pnl, leg_from_live_quote,
    leg_from_traded_bar, lot_size_from_spec, option_expiry_intrinsic_value_per_unit, structure_break_even,
    structure_entry_cash_flow, structure_expiry_payoff, structure_expiry_pnl, structure_max_loss, structure_max_profit,
)
from scripts.run_o1_0b_sample_gate import _sidecar, load, sample_path, udiff_bars

D = dt.date
UTC = dt.timezone.utc
FIXTURE = "DEVELOPMENT_FIXTURE"
EXPIRY = D(2024, 7, 25)


def opt(strike_rupees: int, cp: str, underlying: str = "NIFTY", expiry: dt.date = EXPIRY, cls: str = "IDO") -> ContractIdentity:
    return ContractIdentity(underlying, cls, expiry, strike_rupees * 100, cp)


def mkleg(strike: int, cp: str, side: str, entry_price, *, lots: int = 1, lot_size: int = 50) -> OptionLeg:
    return OptionLeg(
        contract=opt(strike, cp), side=side, lots=lots, lot_size=lot_size, lot_size_source="CALLER_SUPPLIED",
        entry_price=Decimal(entry_price), entry_price_source="CALLER_SUPPLIED", provenance_class=FIXTURE,
    )


class TestIntrinsicValuePerUnit:
    def test_call_intrinsic_below_strike_is_zero(self):
        assert option_expiry_intrinsic_value_per_unit(opt(100, "CE"), Decimal(90)) == Decimal(0)

    def test_call_intrinsic_above_strike_is_the_difference(self):
        assert option_expiry_intrinsic_value_per_unit(opt(100, "CE"), Decimal(130)) == Decimal(30)

    def test_put_intrinsic_above_strike_is_zero(self):
        assert option_expiry_intrinsic_value_per_unit(opt(100, "PE"), Decimal(110)) == Decimal(0)

    def test_put_intrinsic_below_strike_is_the_difference(self):
        assert option_expiry_intrinsic_value_per_unit(opt(100, "PE"), Decimal(70)) == Decimal(30)

    def test_at_the_money_is_zero_for_both(self):
        assert option_expiry_intrinsic_value_per_unit(opt(100, "CE"), Decimal(100)) == Decimal(0)
        assert option_expiry_intrinsic_value_per_unit(opt(100, "PE"), Decimal(100)) == Decimal(0)

    def test_negative_settlement_price_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            option_expiry_intrinsic_value_per_unit(opt(100, "CE"), Decimal(-1))

    def test_futures_contract_rejected_26(self):
        fut = ContractIdentity("NIFTY", "IDF", EXPIRY, None, None)
        with pytest.raises(ValueError, match="option contract"):
            option_expiry_intrinsic_value_per_unit(fut, Decimal(100))

    def test_american_call_put_use_same_terminal_formula(self):
        c = ContractIdentity("SBIN", "STO", D(2010, 7, 29), 30000, "CA")
        p = ContractIdentity("SBIN", "STO", D(2010, 7, 29), 30000, "PA")
        assert option_expiry_intrinsic_value_per_unit(c, Decimal(350)) == Decimal(50)
        assert option_expiry_intrinsic_value_per_unit(p, Decimal(250)) == Decimal(50)


class TestLongCall:
    def test_payoff_and_pnl(self):
        leg = mkleg(100, "CE", "BUY", "5")
        assert leg.entry_cash_flow == Decimal(-250)
        assert leg_expiry_payoff(leg, Decimal(120)) == Decimal(1000)
        assert leg_expiry_pnl(leg, Decimal(120)) == Decimal(750)
        assert leg_expiry_payoff(leg, Decimal(90)) == Decimal(0)
        assert leg_expiry_pnl(leg, Decimal(90)) == Decimal(-250)

    def test_max_profit_unbounded_max_loss_bounded_break_even(self):
        s = OptionStructure((mkleg(100, "CE", "BUY", "5"),))
        assert structure_max_profit(s).boundedness == "UNBOUNDED"
        loss = structure_max_loss(s)
        assert loss.boundedness == "BOUNDED" and loss.value == Decimal(-250)
        be = structure_break_even(s)
        assert be.points == (Decimal(105),) and be.ranges == ()


class TestLongPut:
    def test_payoff_and_pnl(self):
        leg = mkleg(100, "PE", "BUY", "4")
        assert leg.entry_cash_flow == Decimal(-200)
        assert leg_expiry_payoff(leg, Decimal(80)) == Decimal(1000)
        assert leg_expiry_pnl(leg, Decimal(80)) == Decimal(800)

    def test_bounded_profit_and_loss_break_even(self):
        s = OptionStructure((mkleg(100, "PE", "BUY", "4"),))
        profit = structure_max_profit(s)
        assert profit.boundedness == "BOUNDED" and profit.value == Decimal(4800) and profit.at_settlement_price == Decimal(0)
        loss = structure_max_loss(s)
        assert loss.boundedness == "BOUNDED" and loss.value == Decimal(-200)
        assert structure_break_even(s).points == (Decimal(96),)


class TestShortCall:
    def test_bounded_profit_unbounded_loss(self):
        s = OptionStructure((mkleg(100, "CE", "SELL", "5"),))
        profit = structure_max_profit(s)
        assert profit.boundedness == "BOUNDED" and profit.value == Decimal(250)
        assert structure_max_loss(s).boundedness == "UNBOUNDED"
        assert structure_break_even(s).points == (Decimal(105),)


class TestShortPut:
    def test_bounded_profit_bounded_but_large_loss(self):
        s = OptionStructure((mkleg(100, "PE", "SELL", "4"),))
        profit = structure_max_profit(s)
        assert profit.boundedness == "BOUNDED" and profit.value == Decimal(200)
        loss = structure_max_loss(s)
        assert loss.boundedness == "BOUNDED" and loss.value == Decimal(-4800)
        assert structure_break_even(s).points == (Decimal(96),)


class TestBullCallSpread:
    def test_max_profit_loss_break_even(self):
        s = OptionStructure((mkleg(100, "CE", "BUY", "6"), mkleg(110, "CE", "SELL", "2")))
        assert structure_entry_cash_flow(s) == Decimal(-200)
        profit = structure_max_profit(s)
        assert profit.boundedness == "BOUNDED" and profit.value == Decimal(300)
        loss = structure_max_loss(s)
        assert loss.boundedness == "BOUNDED" and loss.value == Decimal(-200)
        assert structure_break_even(s).points == (Decimal(104),)


class TestBearPutSpread:
    def test_max_profit_loss_break_even(self):
        s = OptionStructure((mkleg(100, "PE", "BUY", "6"), mkleg(90, "PE", "SELL", "2")))
        assert structure_entry_cash_flow(s) == Decimal(-200)
        profit = structure_max_profit(s)
        assert profit.boundedness == "BOUNDED" and profit.value == Decimal(300)
        loss = structure_max_loss(s)
        assert loss.boundedness == "BOUNDED" and loss.value == Decimal(-200)
        assert structure_break_even(s).points == (Decimal(96),)


class TestBearCallSpread:
    def test_max_profit_loss_break_even(self):
        s = OptionStructure((mkleg(100, "CE", "SELL", "6"), mkleg(110, "CE", "BUY", "2")))
        assert structure_entry_cash_flow(s) == Decimal(200)
        profit = structure_max_profit(s)
        assert profit.boundedness == "BOUNDED" and profit.value == Decimal(200)
        loss = structure_max_loss(s)
        assert loss.boundedness == "BOUNDED" and loss.value == Decimal(-300)
        assert structure_break_even(s).points == (Decimal(104),)


class TestBullPutSpread:
    def test_max_profit_loss_break_even(self):
        s = OptionStructure((mkleg(100, "PE", "SELL", "6"), mkleg(90, "PE", "BUY", "2")))
        assert structure_entry_cash_flow(s) == Decimal(200)
        profit = structure_max_profit(s)
        assert profit.boundedness == "BOUNDED" and profit.value == Decimal(200)
        loss = structure_max_loss(s)
        assert loss.boundedness == "BOUNDED" and loss.value == Decimal(-300)
        assert structure_break_even(s).points == (Decimal(96),)


class TestLongStraddle:
    def test_unbounded_profit_bounded_loss_two_break_evens(self):
        s = OptionStructure((mkleg(100, "CE", "BUY", "5"), mkleg(100, "PE", "BUY", "4")))
        assert structure_max_profit(s).boundedness == "UNBOUNDED"
        loss = structure_max_loss(s)
        assert loss.boundedness == "BOUNDED" and loss.value == Decimal(-450) and loss.at_settlement_price == Decimal(100)
        assert structure_break_even(s).points == (Decimal(91), Decimal(109))


class TestShortStraddle:
    def test_bounded_profit_unbounded_loss_two_break_evens(self):
        s = OptionStructure((mkleg(100, "CE", "SELL", "5"), mkleg(100, "PE", "SELL", "4")))
        profit = structure_max_profit(s)
        assert profit.boundedness == "BOUNDED" and profit.value == Decimal(450) and profit.at_settlement_price == Decimal(100)
        assert structure_max_loss(s).boundedness == "UNBOUNDED"
        assert structure_break_even(s).points == (Decimal(91), Decimal(109))


class TestLongStrangle:
    def test_unbounded_profit_bounded_flat_loss_two_break_evens(self):
        s = OptionStructure((mkleg(110, "CE", "BUY", "3"), mkleg(90, "PE", "BUY", "3")))
        assert structure_max_profit(s).boundedness == "UNBOUNDED"
        loss = structure_max_loss(s)
        assert loss.boundedness == "BOUNDED" and loss.value == Decimal(-300)
        assert loss.at_settlement_price in (Decimal(90), Decimal(110))
        assert structure_break_even(s).points == (Decimal(84), Decimal(116))


class TestShortStrangle:
    def test_bounded_flat_profit_unbounded_loss_two_break_evens(self):
        s = OptionStructure((mkleg(110, "CE", "SELL", "3"), mkleg(90, "PE", "SELL", "3")))
        profit = structure_max_profit(s)
        assert profit.boundedness == "BOUNDED" and profit.value == Decimal(300)
        assert structure_max_loss(s).boundedness == "UNBOUNDED"
        assert structure_break_even(s).points == (Decimal(84), Decimal(116))


class TestIronCondor:
    def test_bounded_profit_bounded_loss_two_break_evens(self):
        s = OptionStructure((
            mkleg(90, "PE", "SELL", "2"), mkleg(80, "PE", "BUY", "1"),
            mkleg(110, "CE", "SELL", "2"), mkleg(120, "CE", "BUY", "1"),
        ))
        assert structure_entry_cash_flow(s) == Decimal(100)
        profit = structure_max_profit(s)
        assert profit.boundedness == "BOUNDED" and profit.value == Decimal(100)
        loss = structure_max_loss(s)
        assert loss.boundedness == "BOUNDED" and loss.value == Decimal(-400)
        assert structure_break_even(s).points == (Decimal(88), Decimal(112))

    def test_structure_payoff_equals_sum_of_leg_payoffs(self):
        s = OptionStructure((
            mkleg(90, "PE", "SELL", "2"), mkleg(80, "PE", "BUY", "1"),
            mkleg(110, "CE", "SELL", "2"), mkleg(120, "CE", "BUY", "1"),
        ))
        for S in (Decimal(0), Decimal(85), Decimal(100), Decimal(115), Decimal(500)):
            assert structure_expiry_payoff(s, S) == sum((leg_expiry_payoff(leg, S) for leg in s.legs), Decimal(0))


class TestSignQuantityLotSizeInvariants:
    def test_reversing_buy_sell_reverses_signed_payoff_before_premium(self):
        buy = mkleg(100, "CE", "BUY", "5")
        sell = mkleg(100, "CE", "SELL", "5")
        for S in (Decimal(50), Decimal(100), Decimal(150)):
            assert leg_expiry_payoff(buy, S) == -leg_expiry_payoff(sell, S)

    def test_doubling_lots_doubles_pnl(self):
        one = mkleg(100, "CE", "BUY", "5", lots=1)
        two = mkleg(100, "CE", "BUY", "5", lots=2)
        assert leg_expiry_pnl(two, Decimal(130)) == leg_expiry_pnl(one, Decimal(130)) * 2

    def test_doubling_lot_size_doubles_pnl(self):
        one = mkleg(100, "CE", "BUY", "5", lot_size=50)
        two = mkleg(100, "CE", "BUY", "5", lot_size=100)
        assert leg_expiry_pnl(two, Decimal(130)) == leg_expiry_pnl(one, Decimal(130)) * 2

    def test_buy_pays_premium_sell_receives_premium(self):
        assert mkleg(100, "CE", "BUY", "5").entry_cash_flow < 0
        assert mkleg(100, "CE", "SELL", "5").entry_cash_flow > 0

    def test_expiry_pnl_is_deterministic(self):
        leg = mkleg(100, "CE", "BUY", "5")
        assert leg_expiry_pnl(leg, Decimal(120)) == leg_expiry_pnl(leg, Decimal(120))

    def test_contract_identity_preserved_through_structure(self):
        c = opt(100, "CE")
        leg = OptionLeg(contract=c, side="BUY", lots=1, lot_size=50, lot_size_source="CALLER_SUPPLIED",
                        entry_price=Decimal(5), entry_price_source="CALLER_SUPPLIED", provenance_class=FIXTURE)
        s = OptionStructure((leg,))
        assert s.legs[0].contract == c


class TestNoBreakEven:
    def test_always_profitable_structure_has_no_break_even(self):
        s = OptionStructure((mkleg(100, "CE", "SELL", "50"), mkleg(110, "CE", "BUY", "1")))
        for S in (Decimal(0), Decimal(100), Decimal(110), Decimal(1000)):
            assert structure_expiry_pnl(s, S) > 0
        be = structure_break_even(s)
        assert be.points == () and be.ranges == ()

    def test_always_losing_structure_has_no_break_even(self):
        s = OptionStructure((mkleg(100, "CE", "BUY", "50"), mkleg(110, "CE", "SELL", "1")))
        for S in (Decimal(0), Decimal(100), Decimal(110), Decimal(1000)):
            assert structure_expiry_pnl(s, S) < 0
        be = structure_break_even(s)
        assert be.points == () and be.ranges == ()


class TestQuantityValidation:
    def test_zero_lots_rejected(self):
        with pytest.raises(ValueError, match="lots"):
            mkleg(100, "CE", "BUY", "5", lots=0)

    def test_negative_lots_rejected(self):
        with pytest.raises(ValueError, match="lots"):
            mkleg(100, "CE", "BUY", "5", lots=-1)

    def test_non_integer_lots_rejected(self):
        with pytest.raises(ValueError, match="lots"):
            OptionLeg(contract=opt(100, "CE"), side="BUY", lots=1.5, lot_size=50, lot_size_source="CALLER_SUPPLIED",  # type: ignore[arg-type]
                     entry_price=Decimal(5), entry_price_source="CALLER_SUPPLIED", provenance_class=FIXTURE)

    def test_zero_lot_size_rejected(self):
        with pytest.raises(ValueError, match="lot_size"):
            mkleg(100, "CE", "BUY", "5", lot_size=0)


class TestInvalidContractRejection:
    def test_futures_contract_rejected_as_a_leg(self):
        fut = ContractIdentity("NIFTY", "IDF", EXPIRY, None, None)
        with pytest.raises(ValueError, match="option contract"):
            OptionLeg(contract=fut, side="BUY", lots=1, lot_size=50, lot_size_source="CALLER_SUPPLIED",
                     entry_price=Decimal(100), entry_price_source="CALLER_SUPPLIED", provenance_class=FIXTURE)

    def test_negative_entry_price_rejected(self):
        with pytest.raises(ValueError, match="entry_price"):
            mkleg(100, "CE", "BUY", "-1")

    def test_invalid_side_rejected(self):
        with pytest.raises(ValueError, match="side"):
            mkleg(100, "CE", "HOLD", "5")  # type: ignore[arg-type]

    def test_structure_rejects_mixed_underlyings(self):
        other_underlying_leg = OptionLeg(contract=opt(100, "CE", underlying="BANKNIFTY"), side="BUY", lots=1, lot_size=50,
                                         lot_size_source="CALLER_SUPPLIED", entry_price=Decimal(5), entry_price_source="CALLER_SUPPLIED", provenance_class=FIXTURE)
        with pytest.raises(ValueError, match="underlying"):
            OptionStructure((mkleg(100, "CE", "BUY", "5"), other_underlying_leg))

    def test_structure_rejects_mixed_expiries(self):
        other_expiry_leg = OptionLeg(contract=opt(100, "CE", expiry=D(2024, 8, 29)), side="BUY", lots=1, lot_size=50,
                                     lot_size_source="CALLER_SUPPLIED", entry_price=Decimal(5), entry_price_source="CALLER_SUPPLIED", provenance_class=FIXTURE)
        with pytest.raises(ValueError, match="expiry"):
            OptionStructure((mkleg(100, "CE", "BUY", "5"), other_expiry_leg))

    def test_structure_rejects_mixed_provenance(self):
        prod_leg = OptionLeg(contract=opt(110, "CE"), side="BUY", lots=1, lot_size=50, lot_size_source="CALLER_SUPPLIED",
                             entry_price=Decimal(5), entry_price_source="CALLER_SUPPLIED", provenance_class="PRODUCTION_RESEARCH_DATASET")
        with pytest.raises(ValueError, match="provenance_class"):
            OptionStructure((mkleg(100, "CE", "BUY", "5"), prod_leg))

    def test_structure_rejects_empty_legs(self):
        with pytest.raises(ValueError, match="at least one leg"):
            OptionStructure(())


class TestLotSizeAuthoritativeness:
    def test_caller_supplied_lot_size_flagged_non_authoritative(self):
        leg = mkleg(100, "CE", "BUY", "5")
        assert leg.is_non_authoritative is True
        s = OptionStructure((leg,))
        assert s.uses_non_authoritative_lot_size is True

    def test_contract_spec_lot_size_flagged_authoritative(self):
        c = opt(100, "CE")
        spec = ContractSpec(contract=c, lot_size=50, effective_from=D(2024, 1, 1), effective_to=None, provenance_class=FIXTURE)
        lot_size, source = lot_size_from_spec([spec], c, D(2024, 7, 1))
        assert lot_size == 50 and source == "CONTRACT_SPEC"
        leg = OptionLeg(contract=c, side="BUY", lots=1, lot_size=lot_size, lot_size_source=source,
                        entry_price=Decimal(5), entry_price_source="CALLER_SUPPLIED", provenance_class=FIXTURE)
        assert leg.is_non_authoritative is False

    def test_lot_size_from_spec_raises_when_no_spec_covers_the_date(self):
        c = opt(100, "CE")
        spec = ContractSpec(contract=c, lot_size=50, effective_from=D(2024, 1, 1), effective_to=D(2024, 3, 31), provenance_class=FIXTURE)
        with pytest.raises(ValueError, match="no ContractSpec covers"):
            lot_size_from_spec([spec], c, D(2024, 7, 1))


def make_bar(*, trade_state_inputs, contract=None, session_date=D(2024, 7, 5)) -> DerivativeBar:
    bar_end = avail.session_close_utc(session_date)
    available_at, basis = avail.next_session_open_utc(session_date)
    o, h, l, c, v = trade_state_inputs
    return DerivativeBar.from_raw(
        contract=contract or opt(100, "CE"), session_date=session_date, bar_end_at=bar_end, interval="1D",
        open_=Decimal(o), high=Decimal(h), low=Decimal(l), close=Decimal(c), volume=v, open_interest=None, oi_change=None,
        turnover=None, settle_raw=None, underlying_price=None, observed_lot_size=None, available_at=available_at,
        availability_basis=basis, source="nse_archives", source_row_hash=None, provenance_class=FIXTURE,
    )


class TestLegFromTradedBar:
    def test_traded_bar_produces_a_leg_with_pit_timestamp(self):
        b = make_bar(trade_state_inputs=("10", "12", "9", "11", 100))
        leg = leg_from_traded_bar(b, side="BUY", lots=1, lot_size=50, lot_size_source="CALLER_SUPPLIED")
        assert leg.entry_price == Decimal("11") and leg.entry_price_source == "TRADED_BAR"
        assert leg.entry_observed_at == b.available_at

    def test_carried_close_bar_is_refused_not_used_as_entry_price(self):
        b = make_bar(trade_state_inputs=("0", "0", "0", "26.65", 0))
        with pytest.raises(ValueError, match="not TRADED"):
            leg_from_traded_bar(b, side="BUY", lots=1, lot_size=50, lot_size_source="CALLER_SUPPLIED")

    def test_anomalous_bar_is_refused(self):
        b = make_bar(trade_state_inputs=("0", "0", "0", "1361.7", 2))
        with pytest.raises(ValueError, match="not TRADED"):
            leg_from_traded_bar(b, side="BUY", lots=1, lot_size=50, lot_size_source="CALLER_SUPPLIED")

    def test_entry_price_source_traded_bar_requires_entry_observed_at(self):
        with pytest.raises(ValueError, match="entry_observed_at"):
            OptionLeg(contract=opt(100, "CE"), side="BUY", lots=1, lot_size=50, lot_size_source="CALLER_SUPPLIED",
                     entry_price=Decimal(5), entry_price_source="TRADED_BAR", provenance_class=FIXTURE)

    def test_naive_entry_observed_at_rejected(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            OptionLeg(contract=opt(100, "CE"), side="BUY", lots=1, lot_size=50, lot_size_source="CALLER_SUPPLIED",
                     entry_price=Decimal(5), entry_price_source="TRADED_BAR", provenance_class=FIXTURE,
                     entry_observed_at=dt.datetime(2024, 7, 5))


def make_live_quote(*, last_price, bid=None, ask=None, contract=None, observed_at=None, provenance_class=FIXTURE):
    from backend.app.options.live_quote import LiveQuote
    return LiveQuote.from_raw(
        contract=contract or opt(100, "CE"), observed_at=observed_at or dt.datetime(2026, 9, 22, 10, 0, tzinfo=UTC),
        last_price=last_price, bid=bid, ask=ask, volume=100, open_interest=5000, underlying_last_price=Decimal("23329"),
        provider_reported_iv=None, provider_reported_greeks=None, source="dhan", source_row_hash=None, provenance_class=provenance_class,
    )


class TestLegFromLiveQuote:
    def test_usable_quote_produces_a_leg_with_pit_timestamp(self):
        q = make_live_quote(last_price=Decimal("11"))
        leg = leg_from_live_quote(q, side="BUY", lots=1, lot_size=50, lot_size_source="CALLER_SUPPLIED")
        assert leg.entry_price == Decimal("11") and leg.entry_price_source == "LIVE_QUOTE"
        assert leg.entry_observed_at == q.observed_at
        assert leg.provenance_class == FIXTURE

    def test_no_quote_is_refused_not_used_as_entry_price(self):
        q = make_live_quote(last_price=None, bid=None, ask=None)
        with pytest.raises(ValueError, match="not usable"):
            leg_from_live_quote(q, side="BUY", lots=1, lot_size=50, lot_size_source="CALLER_SUPPLIED")

    def test_anomalous_quote_is_refused(self):
        q = make_live_quote(last_price=Decimal(0))
        with pytest.raises(ValueError, match="not usable"):
            leg_from_live_quote(q, side="BUY", lots=1, lot_size=50, lot_size_source="CALLER_SUPPLIED")

    def test_live_quote_source_accepted_on_direct_construction(self):
        leg = OptionLeg(contract=opt(100, "CE"), side="BUY", lots=1, lot_size=50, lot_size_source="CALLER_SUPPLIED",
                        entry_price=Decimal(5), entry_price_source="LIVE_QUOTE", provenance_class=FIXTURE,
                        entry_observed_at=dt.datetime(2026, 9, 22, 10, 0, tzinfo=UTC))
        assert leg.entry_price_source == "LIVE_QUOTE"

    def test_production_provenance_passthrough(self):
        q = make_live_quote(last_price=Decimal("11"), provenance_class="PRODUCTION_RESEARCH_DATASET")
        leg = leg_from_live_quote(q, side="SELL", lots=1, lot_size=50, lot_size_source="CALLER_SUPPLIED")
        assert leg.provenance_class == "PRODUCTION_RESEARCH_DATASET"


class TestPayoffCurve:
    def test_curve_is_deterministic_and_matches_structure_functions(self):
        s = OptionStructure((mkleg(100, "CE", "BUY", "6"), mkleg(110, "CE", "SELL", "2")))
        prices = [Decimal(x) for x in (0, 90, 100, 104, 110, 200)]
        c1 = expiry_pnl_curve(s, prices)
        c2 = expiry_pnl_curve(s, prices)
        assert c1 == c2
        for point in c1:
            assert point.payoff == structure_expiry_payoff(s, point.settlement_price)
            assert point.pnl == structure_expiry_pnl(s, point.settlement_price)
        assert c1[3].pnl == Decimal(0)

    def test_empty_settlement_prices_rejected(self):
        s = OptionStructure((mkleg(100, "CE", "BUY", "6"),))
        with pytest.raises(ValueError, match="must not be empty"):
            expiry_pnl_curve(s, [])


class TestResultTypeInvariants:
    def test_pnl_extremum_invariant(self):
        with pytest.raises(ValueError, match="iff boundedness"):
            PnLExtremum(boundedness="BOUNDED", value=None, at_settlement_price=None)
        with pytest.raises(ValueError, match="iff boundedness"):
            PnLExtremum(boundedness="UNBOUNDED", value=Decimal(1), at_settlement_price=Decimal(1))

    def test_leg_and_structure_are_frozen(self):
        leg = mkleg(100, "CE", "BUY", "5")
        with pytest.raises(dataclasses.FrozenInstanceError):
            leg.entry_price = Decimal(10)  # type: ignore[misc]


class TestRealFixtureReplay:
    def real_traded_option_bars(self, day, underlying):
        got = load("udiff", day)
        if got is None:
            pytest.skip(f"O1.0B NSE sample not present on disk for udiff:{day}")
        import json
        side = json.loads(_sidecar(sample_path("udiff", day)).read_text())
        lm = avail.parse_last_modified((side.get("response_headers") or {}).get("Last-Modified"))
        av = avail.availability_for_eod_file(day, None, lm)
        out = []
        for b in udiff_bars(*got):
            if b.key.underlying != underlying or b.key.dtype != "IDO" or b.volume <= 0:
                continue
            contract = ContractIdentity(b.key.underlying, b.key.dtype, b.key.expiry, b.key.strike_paise, b.key.cp)
            bar = DerivativeBar.from_raw(
                contract=contract, session_date=day, bar_end_at=avail.session_close_utc(day), interval="1D",
                open_=b.open, high=b.high, low=b.low, close=b.close, volume=b.volume, open_interest=b.oi, oi_change=b.oi_change,
                turnover=b.turnover, settle_raw=b.settle, underlying_price=b.underlying_price, observed_lot_size=b.lot,
                available_at=av["available_at"], availability_basis=av["availability_basis"], source="nse_archives",
                source_row_hash=b.row_hash, provenance_class=FIXTURE, is_expiry_day=(b.key.expiry == day),
            )
            out.append(bar)
        return out

    def test_bull_call_spread_from_two_real_traded_nifty_bars(self):
        day = D(2026, 9, 18)
        bars_ = self.real_traded_option_bars(day, "NIFTY")
        nearest_expiry = min(b.contract.expiry for b in bars_)
        calls = sorted((b for b in bars_ if b.contract.option_type == "CE" and b.contract.expiry == nearest_expiry), key=lambda b: b.contract.strike_paise)
        assert len(calls) >= 2, "expected at least two traded NIFTY calls in this fixture's nearest expiry"
        lower, higher = calls[0], calls[len(calls) // 2]
        assert lower.contract.strike_paise < higher.contract.strike_paise
        long_leg = leg_from_traded_bar(lower, side="BUY", lots=1, lot_size=50, lot_size_source="CALLER_SUPPLIED")
        short_leg = leg_from_traded_bar(higher, side="SELL", lots=1, lot_size=50, lot_size_source="CALLER_SUPPLIED")
        s = OptionStructure((long_leg, short_leg))
        assert s.provenance_class == FIXTURE
        expected_cash_flow = -(lower.close * 50) + (higher.close * 50)
        assert structure_entry_cash_flow(s) == expected_cash_flow
        settlement = Decimal(higher.contract.strike_paise) / 100 + 1000
        width = (Decimal(higher.contract.strike_paise) - Decimal(lower.contract.strike_paise)) / 100 * 50
        assert structure_expiry_payoff(s, settlement) == width
