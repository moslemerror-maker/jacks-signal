from __future__ import annotations

import dataclasses
import datetime as dt
import json
from decimal import Decimal

import pytest

from backend.app.options import availability as avail
from backend.app.options.bars import DerivativeBar
from backend.app.options.contracts import ContractIdentity
from backend.app.options.mtm import (
    CostItem, LegCosts, LegValuation, SlippageAssumption, mark_to_market_leg, mark_to_market_structure, net_pnl,
    slippage_amount, slippage_cost_per_unit, structure_costs, valuation_from_model, valuation_from_traded_bar,
)
from backend.app.options.structures import OptionLeg, OptionStructure, leg_expiry_pnl, leg_from_traded_bar
from backend.app.options.valuation import ValuationInputs, black76_implied_volatility
from scripts.run_o1_0b_sample_gate import _sidecar, load, sample_path, udiff_bars

D = dt.date
UTC = dt.timezone.utc
FIXTURE = "DEVELOPMENT_FIXTURE"
EXPIRY = D(2024, 7, 25)


def opt(strike_rupees: int, cp: str, underlying: str = "NIFTY", expiry: dt.date = EXPIRY) -> ContractIdentity:
    return ContractIdentity(underlying, "IDO", expiry, strike_rupees * 100, cp)


def mkleg(strike: int, cp: str, side: str, entry_price, *, lots: int = 1, lot_size: int = 50, entry_observed_at=None) -> OptionLeg:
    return OptionLeg(
        contract=opt(strike, cp), side=side, lots=lots, lot_size=lot_size, lot_size_source="CALLER_SUPPLIED",
        entry_price=Decimal(entry_price), entry_price_source="TRADED_BAR" if entry_observed_at else "CALLER_SUPPLIED",
        provenance_class=FIXTURE, entry_observed_at=entry_observed_at,
    )


def val(contract: ContractIdentity, price, *, at=dt.datetime(2024, 7, 5, 3, 45, tzinfo=UTC), source: str = "OBSERVED_MARKET_PRICE") -> LegValuation:
    return LegValuation(contract=contract, price=Decimal(price), source=source, valuation_at=at)


class TestSingleLegMTM:
    def test_long_call(self):
        leg = mkleg(100, "CE", "BUY", "5")
        r = mark_to_market_leg(leg, val(leg.contract, "8"))
        assert r.mtm_pnl == Decimal(150)

    def test_long_put(self):
        leg = mkleg(100, "PE", "BUY", "4")
        r = mark_to_market_leg(leg, val(leg.contract, "6"))
        assert r.mtm_pnl == Decimal(100)

    def test_short_call(self):
        leg = mkleg(100, "CE", "SELL", "5")
        r = mark_to_market_leg(leg, val(leg.contract, "8"))
        assert r.mtm_pnl == Decimal(-150)

    def test_short_put(self):
        leg = mkleg(100, "PE", "SELL", "4")
        r = mark_to_market_leg(leg, val(leg.contract, "2"))
        assert r.mtm_pnl == Decimal(100)


class TestMultiLegMTM:
    def test_bull_call_spread(self):
        long_leg = mkleg(100, "CE", "BUY", "6")
        short_leg = mkleg(110, "CE", "SELL", "2")
        s = OptionStructure((long_leg, short_leg))
        valuations = {long_leg.contract: val(long_leg.contract, "9"), short_leg.contract: val(short_leg.contract, "3")}
        r = mark_to_market_structure(s, valuations)
        assert r.mtm_pnl == Decimal(100)

    def test_missing_valuation_for_a_leg_is_refused(self):
        long_leg = mkleg(100, "CE", "BUY", "6")
        short_leg = mkleg(110, "CE", "SELL", "2")
        s = OptionStructure((long_leg, short_leg))
        with pytest.raises(ValueError, match="no valuation supplied"):
            mark_to_market_structure(s, {long_leg.contract: val(long_leg.contract, "9")})


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


class TestValuationSource:
    def test_observed_market_price_from_a_traded_bar(self):
        b = make_bar(trade_state_inputs=("10", "12", "9", "11", 100))
        v = valuation_from_traded_bar(b)
        assert v.source == "OBSERVED_MARKET_PRICE" and v.price == Decimal("11")

    def test_model_valuation_via_black76(self):
        contract = opt(50, "CE")
        expiry_at = avail.session_close_utc(EXPIRY)
        inputs = ValuationInputs(option_type="CE", forward=Decimal(49), strike=Decimal(50), valuation_at=dt.datetime(2024, 1, 1, tzinfo=UTC),
                                 expiry_at=dt.datetime(2024, 1, 1, tzinfo=UTC) + dt.timedelta(days=140), risk_free_rate=Decimal("0.05"))
        v = valuation_from_model(contract, inputs, Decimal("0.2"))
        assert v.source == "MODEL_VALUATION" and v.price > 0

    def test_model_valuation_rejects_mismatched_strike(self):
        contract = opt(999, "CE")
        inputs = ValuationInputs(option_type="CE", forward=Decimal(49), strike=Decimal(50), valuation_at=dt.datetime(2024, 1, 1, tzinfo=UTC),
                                 expiry_at=dt.datetime(2024, 1, 1, tzinfo=UTC) + dt.timedelta(days=140), risk_free_rate=Decimal("0.05"))
        with pytest.raises(ValueError, match="strike"):
            valuation_from_model(contract, inputs, Decimal("0.2"))

    def test_model_valuation_rejects_mismatched_option_type(self):
        contract = opt(50, "PE")
        inputs = ValuationInputs(option_type="CE", forward=Decimal(49), strike=Decimal(50), valuation_at=dt.datetime(2024, 1, 1, tzinfo=UTC),
                                 expiry_at=dt.datetime(2024, 1, 1, tzinfo=UTC) + dt.timedelta(days=140), risk_free_rate=Decimal("0.05"))
        with pytest.raises(ValueError, match="option_type"):
            valuation_from_model(contract, inputs, Decimal("0.2"))

    def test_carried_close_bar_is_refused(self):
        b = make_bar(trade_state_inputs=("0", "0", "0", "26.65", 0))
        with pytest.raises(ValueError, match="not TRADED"):
            valuation_from_traded_bar(b)

    def test_non_traded_no_price_bar_is_refused(self):
        b = make_bar(trade_state_inputs=("0", "0", "0", "0", 0))
        with pytest.raises(ValueError, match="not TRADED"):
            valuation_from_traded_bar(b)

    def test_anomalous_bar_is_refused(self):
        b = make_bar(trade_state_inputs=("0", "0", "0", "1361.7", 2))
        with pytest.raises(ValueError, match="not TRADED"):
            valuation_from_traded_bar(b)

    def test_valuation_before_bar_available_at_is_refused(self):
        b = make_bar(trade_state_inputs=("10", "12", "9", "11", 100))
        too_early = b.available_at - dt.timedelta(minutes=1)
        with pytest.raises(ValueError, match="no look-ahead"):
            valuation_from_traded_bar(b, valuation_at=too_early)

    def test_valuation_before_entry_is_refused(self):
        b = make_bar(trade_state_inputs=("10", "12", "9", "11", 100), session_date=D(2024, 7, 5))
        leg = leg_from_traded_bar(b, side="BUY", lots=1, lot_size=50, lot_size_source="CALLER_SUPPLIED")
        too_early_valuation = val(leg.contract, "20", at=leg.entry_observed_at - dt.timedelta(days=1))
        with pytest.raises(ValueError, match="impossible chronology"):
            mark_to_market_leg(leg, too_early_valuation)

    def test_same_availability_instant_is_flagged_not_invented(self):
        b = make_bar(trade_state_inputs=("10", "12", "9", "11", 100), session_date=D(2024, 7, 5))
        leg = leg_from_traded_bar(b, side="BUY", lots=1, lot_size=50, lot_size_source="CALLER_SUPPLIED")
        same_instant_valuation = val(leg.contract, "13", at=leg.entry_observed_at)
        r = mark_to_market_leg(leg, same_instant_valuation)
        assert r.entry_and_valuation_share_same_availability_instant is True

    def test_different_instant_not_flagged(self):
        b = make_bar(trade_state_inputs=("10", "12", "9", "11", 100), session_date=D(2024, 7, 5))
        leg = leg_from_traded_bar(b, side="BUY", lots=1, lot_size=50, lot_size_source="CALLER_SUPPLIED")
        later = val(leg.contract, "13", at=leg.entry_observed_at + dt.timedelta(days=1))
        r = mark_to_market_leg(leg, later)
        assert r.entry_and_valuation_share_same_availability_instant is False


class TestEntryPriceHandling:
    def test_mtm_uses_legs_own_entry_price_exactly(self):
        leg = mkleg(100, "CE", "BUY", "5.25")
        r = mark_to_market_leg(leg, val(leg.contract, "9"))
        assert r.mtm_pnl == (Decimal("9") - Decimal("5.25")) * 50

    def test_chronology_check_skipped_when_entry_observed_at_is_none(self):
        leg = mkleg(100, "CE", "BUY", "5")
        assert leg.entry_observed_at is None
        r = mark_to_market_leg(leg, val(leg.contract, "9", at=dt.datetime(2000, 1, 1, tzinfo=UTC)))
        assert r.mtm_pnl == Decimal(200)


class TestExpiryVsPreExpiryDistinction:
    def test_mtm_and_expiry_pnl_are_mathematically_different_paths(self):
        leg = mkleg(100, "CE", "BUY", "5")
        settlement = Decimal(103)
        expiry_pnl = leg_expiry_pnl(leg, settlement)
        mtm_pnl = mark_to_market_leg(leg, val(leg.contract, "9")).mtm_pnl
        assert expiry_pnl == Decimal(-100)
        assert mtm_pnl == Decimal(200)
        assert expiry_pnl != mtm_pnl


class TestCostsAndSlippage:
    def test_gross_pnl_is_the_mtm_pnl_before_costs(self):
        leg = mkleg(100, "CE", "BUY", "5")
        gross = mark_to_market_leg(leg, val(leg.contract, "8")).mtm_pnl
        assert gross == Decimal(150)

    def test_entry_costs(self):
        lc = LegCosts(contract=opt(100, "CE"), entry_costs=(CostItem("BROKERAGE", Decimal(20)), CostItem("STT", Decimal(5))))
        assert lc.entry_total == Decimal(25) and lc.exit_total == Decimal(0)

    def test_exit_costs_only_present_when_explicitly_supplied(self):
        lc_open = LegCosts(contract=opt(100, "CE"), entry_costs=(CostItem("BROKERAGE", Decimal(20)),))
        assert lc_open.exit_total == Decimal(0)
        lc_closed = dataclasses.replace(lc_open, exit_costs=(CostItem("BROKERAGE", Decimal(20)), CostItem("STT", Decimal(5))))
        assert lc_closed.exit_total == Decimal(25)

    def test_fixed_price_slippage(self):
        a = SlippageAssumption("FIXED_PRICE", Decimal("0.5"))
        assert slippage_cost_per_unit(a, Decimal(100)) == Decimal("0.5")

    def test_basis_points_slippage(self):
        a = SlippageAssumption("BASIS_POINTS", Decimal(10))
        assert slippage_cost_per_unit(a, Decimal(200)) == Decimal("0.2")

    def test_slippage_amount_scales_by_total_units(self):
        leg = mkleg(100, "CE", "BUY", "5", lots=2, lot_size=50)
        a = SlippageAssumption("FIXED_PRICE", Decimal("0.5"))
        assert slippage_amount(leg, a, Decimal(100)) == Decimal("0.5") * 100

    def test_gross_to_net_reconciliation(self):
        gross = Decimal(150)
        lc = LegCosts(contract=opt(100, "CE"), entry_costs=(CostItem("BROKERAGE", Decimal(20)),), entry_slippage=Decimal(5),
                     exit_costs=(CostItem("STT", Decimal(3)),), exit_slippage=Decimal(2))
        assert lc.total == Decimal(30)
        assert net_pnl(gross, lc.total) == Decimal(120)

    def test_multi_leg_cost_aggregation_no_double_counting(self):
        long_leg = mkleg(100, "CE", "BUY", "6")
        short_leg = mkleg(110, "CE", "SELL", "2")
        s = OptionStructure((long_leg, short_leg))
        costs = {
            long_leg.contract: LegCosts(contract=long_leg.contract, entry_costs=(CostItem("BROKERAGE", Decimal(20)),)),
            short_leg.contract: LegCosts(contract=short_leg.contract, entry_costs=(CostItem("BROKERAGE", Decimal(20)),)),
        }
        sc = structure_costs(s, costs)
        assert sc.total == Decimal(40)
        assert sc.entry_total == Decimal(40) and sc.exit_total == Decimal(0)

    def test_structure_costs_requires_every_leg(self):
        long_leg = mkleg(100, "CE", "BUY", "6")
        short_leg = mkleg(110, "CE", "SELL", "2")
        s = OptionStructure((long_leg, short_leg))
        with pytest.raises(ValueError, match="no costs supplied"):
            structure_costs(s, {long_leg.contract: LegCosts(contract=long_leg.contract)})

    def test_net_pnl_applies_costs_exactly_once(self):
        gross = Decimal(1000)
        total_costs = Decimal(40)
        n1 = net_pnl(gross, total_costs)
        n2 = net_pnl(n1 + total_costs, total_costs)
        assert n1 == n2 == Decimal(960)


class TestScaling:
    def test_doubling_lots_doubles_mtm_pnl(self):
        one = mkleg(100, "CE", "BUY", "5", lots=1)
        two = mkleg(100, "CE", "BUY", "5", lots=2)
        v_one, v_two = val(one.contract, "8"), val(two.contract, "8")
        assert mark_to_market_leg(two, v_two).mtm_pnl == mark_to_market_leg(one, v_one).mtm_pnl * 2

    def test_doubling_lot_size_doubles_mtm_pnl(self):
        one = mkleg(100, "CE", "BUY", "5", lot_size=50)
        two = mkleg(100, "CE", "BUY", "5", lot_size=100)
        v_one, v_two = val(one.contract, "8"), val(two.contract, "8")
        assert mark_to_market_leg(two, v_two).mtm_pnl == mark_to_market_leg(one, v_one).mtm_pnl * 2


class TestValidation:
    def test_negative_cost_amount_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            CostItem("BROKERAGE", Decimal(-1))

    def test_unknown_cost_kind_rejected(self):
        with pytest.raises(ValueError, match="kind"):
            CostItem("MADE_UP_KIND", Decimal(1))  # type: ignore[arg-type]

    def test_negative_slippage_value_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            SlippageAssumption("FIXED_PRICE", Decimal(-1))

    def test_unknown_slippage_model_rejected(self):
        with pytest.raises(ValueError, match="model"):
            SlippageAssumption("VWAP_DRIFT", Decimal(1))  # type: ignore[arg-type]

    def test_non_positive_valuation_price_rejected(self):
        with pytest.raises(ValueError, match="price"):
            LegValuation(contract=opt(100, "CE"), price=Decimal(0), source="OBSERVED_MARKET_PRICE", valuation_at=dt.datetime(2024, 1, 1, tzinfo=UTC))

    def test_unknown_valuation_source_rejected(self):
        with pytest.raises(ValueError, match="source"):
            LegValuation(contract=opt(100, "CE"), price=Decimal(5), source="RUMOUR", valuation_at=dt.datetime(2024, 1, 1, tzinfo=UTC))  # type: ignore[arg-type]

    def test_naive_valuation_at_rejected(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            LegValuation(contract=opt(100, "CE"), price=Decimal(5), source="OBSERVED_MARKET_PRICE", valuation_at=dt.datetime(2024, 1, 1))

    def test_zero_cost_and_slippage_are_valid_but_must_not_be_read_as_free(self):
        lc = LegCosts(contract=opt(100, "CE"))
        assert lc.total == Decimal(0)

    def test_negative_reference_price_rejected_in_slippage_per_unit(self):
        with pytest.raises(ValueError, match="reference_price"):
            slippage_cost_per_unit(SlippageAssumption("BASIS_POINTS", Decimal(10)), Decimal(0))


class TestDeterminism:
    def test_mark_to_market_leg_is_deterministic(self):
        leg = mkleg(100, "CE", "BUY", "5")
        v = val(leg.contract, "8")
        assert mark_to_market_leg(leg, v).mtm_pnl == mark_to_market_leg(leg, v).mtm_pnl


class TestRealFixtureReplay:

    def real_traded_option_bars(self, day, underlying):
        got = load("udiff", day)
        if got is None:
            pytest.skip(f"O1.0B NSE sample not present on disk for udiff:{day}")
        side = json.loads(_sidecar(sample_path("udiff", day)).read_text())
        lm = avail.parse_last_modified((side.get("response_headers") or {}).get("Last-Modified"))
        av = avail.availability_for_eod_file(day, None, lm)
        out = {}
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
            out[contract] = bar
        return out

    def overlapping_contract(self, underlying):
        day1, day2 = D(2024, 7, 4), D(2024, 7, 5)
        bars1 = self.real_traded_option_bars(day1, underlying)
        bars2 = self.real_traded_option_bars(day2, underlying)
        common = sorted(set(bars1) & set(bars2), key=lambda c: c.strike_paise)
        if not common:
            pytest.skip(f"no {underlying} contract genuinely TRADED on both {day1} and {day2} in this fixture")
        c = common[len(common) // 2]
        return bars1[c], bars2[c]

    def test_nifty_real_entry_to_exit_mtm(self):
        entry_bar, exit_bar = self.overlapping_contract("NIFTY")
        leg = leg_from_traded_bar(entry_bar, side="BUY", lots=1, lot_size=50, lot_size_source="CALLER_SUPPLIED")
        exit_valuation = valuation_from_traded_bar(exit_bar)
        r = mark_to_market_leg(leg, exit_valuation)
        assert r.mtm_pnl == (exit_bar.close - entry_bar.close) * 50
        assert r.entry_and_valuation_share_same_availability_instant is False

    def test_banknifty_real_entry_to_exit_mtm(self):
        entry_bar, exit_bar = self.overlapping_contract("BANKNIFTY")
        leg = leg_from_traded_bar(entry_bar, side="SELL", lots=1, lot_size=15, lot_size_source="CALLER_SUPPLIED")
        exit_valuation = valuation_from_traded_bar(exit_bar)
        r = mark_to_market_leg(leg, exit_valuation)
        assert r.mtm_pnl == (entry_bar.close - exit_bar.close) * 15

    def test_model_valuation_using_o2c_against_a_real_traded_entry(self):
        entry_bar, exit_bar = self.overlapping_contract("NIFTY")
        forward = Decimal(entry_bar.contract.strike_paise) / 100
        expiry_at = avail.session_close_utc(entry_bar.contract.expiry)
        entry_inputs = ValuationInputs(option_type=entry_bar.contract.option_type, forward=forward,
                                       strike=Decimal(entry_bar.contract.strike_paise) / 100, valuation_at=entry_bar.available_at,
                                       expiry_at=expiry_at, risk_free_rate=Decimal("0.065"))
        iv_result = black76_implied_volatility(entry_inputs, entry_bar.close)
        if iv_result.status != "OK":
            pytest.skip("this real fixture row did not produce a solvable IV under the illustrative forward assumption")
        exit_inputs = ValuationInputs(option_type=entry_bar.contract.option_type, forward=forward,
                                      strike=Decimal(entry_bar.contract.strike_paise) / 100, valuation_at=exit_bar.available_at,
                                      expiry_at=expiry_at, risk_free_rate=Decimal("0.065"))
        model_valuation = valuation_from_model(entry_bar.contract, exit_inputs, iv_result.iv)
        leg = leg_from_traded_bar(entry_bar, side="BUY", lots=1, lot_size=50, lot_size_source="CALLER_SUPPLIED")
        r = mark_to_market_leg(leg, model_valuation)
        assert model_valuation.source == "MODEL_VALUATION"
        assert r.mtm_pnl == (model_valuation.price - entry_bar.close) * 50
