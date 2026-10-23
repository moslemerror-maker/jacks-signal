from __future__ import annotations

import dataclasses
import datetime as dt
import json
from decimal import Decimal

import pytest

from backend.app.options import availability as avail
from backend.app.options.bars import DerivativeBar
from backend.app.options.contracts import ContractIdentity
from backend.app.options.probability import (
    Black76RiskNeutralDistribution, SettlementDistribution, probability_metrics_from_distribution,
    probability_positive_net_pnl_by_expiry, structure_expected_payoff, structure_expected_value,
)
from backend.app.options.recommendation import CandidateStrategy, RecommendationResult
from backend.app.options.structures import OptionLeg, OptionStructure, structure_break_even, structure_max_loss
from backend.app.options.valuation import ValuationInputs, black76_implied_volatility, black76_price
from scripts.run_o1_0b_sample_gate import _sidecar, load, sample_path, udiff_bars

D = dt.date
UTC = dt.timezone.utc
FIXTURE = "DEVELOPMENT_FIXTURE"
EXPIRY = D(2024, 7, 25)

REF_F, REF_K, REF_R, REF_SIGMA, REF_T = Decimal("49"), Decimal("50"), Decimal("0.05"), Decimal("0.2"), Decimal("0.3846")
REF_CALL_PRICE = Decimal("1.942417578252269")
REF_PUT_PRICE = Decimal("2.9233712951919637")
TOL = Decimal("0.005")


def opt(strike_rupees: int, cp: str, underlying: str = "NIFTY", expiry: dt.date = EXPIRY) -> ContractIdentity:
    return ContractIdentity(underlying, "IDO", expiry, strike_rupees * 100, cp)


def mkleg(strike: int, cp: str, side: str, entry_price, *, lots: int = 1, lot_size: int = 50) -> OptionLeg:
    return OptionLeg(
        contract=opt(strike, cp), side=side, lots=lots, lot_size=lot_size, lot_size_source="CALLER_SUPPLIED",
        entry_price=Decimal(entry_price), entry_price_source="CALLER_SUPPLIED", provenance_class=FIXTURE,
    )


def close(a: Decimal, b: Decimal, tol: Decimal = TOL) -> bool:
    return abs(a - b) < tol


def dist(forward=REF_F, sigma=REF_SIGMA, t=REF_T) -> Black76RiskNeutralDistribution:
    return Black76RiskNeutralDistribution(forward=forward, volatility=sigma, time_to_expiry_years=t)


class TestDistributionValidation:
    def test_nonpositive_forward_rejected(self):
        with pytest.raises(ValueError, match="forward"):
            Black76RiskNeutralDistribution(forward=Decimal(0), volatility=REF_SIGMA, time_to_expiry_years=REF_T)

    def test_nonpositive_volatility_rejected(self):
        with pytest.raises(ValueError, match="volatility"):
            Black76RiskNeutralDistribution(forward=REF_F, volatility=Decimal(0), time_to_expiry_years=REF_T)

    def test_nonpositive_time_rejected(self):
        with pytest.raises(ValueError, match="time_to_expiry_years"):
            Black76RiskNeutralDistribution(forward=REF_F, volatility=REF_SIGMA, time_to_expiry_years=Decimal(0))

    def test_conforms_to_settlement_distribution_protocol(self):
        assert isinstance(dist(), SettlementDistribution)

    def test_frozen(self):
        d = dist()
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.forward = Decimal(100)  # type: ignore[misc]


class TestProbabilityBasics:
    def test_above_and_below_sum_to_one(self):
        d = dist()
        assert close(d.probability_above(REF_F) + d.probability_below(REF_F), Decimal(1), Decimal("0.0001"))

    def test_above_zero_or_negative_level_is_certain(self):
        d = dist()
        assert d.probability_above(Decimal(0)) == Decimal(1)
        assert d.probability_above(Decimal(-5)) == Decimal(1)

    def test_far_below_forward_is_near_certain(self):
        d = dist()
        assert d.probability_above(Decimal("0.01")) > Decimal("0.999")

    def test_far_above_forward_is_near_impossible(self):
        d = dist()
        assert d.probability_above(Decimal(100000)) < Decimal("0.001")

    def test_monotonically_decreasing(self):
        d = dist()
        levels = [Decimal(10), Decimal(30), Decimal(49), Decimal(70), Decimal(100)]
        probs = [d.probability_above(x) for x in levels]
        assert probs == sorted(probs, reverse=True)

    def test_probability_between_matches_difference_of_above(self):
        d = dist()
        assert d.probability_between(Decimal(40), Decimal(60)) == d.probability_above(Decimal(40)) - d.probability_above(Decimal(60))

    def test_probability_between_rejects_hi_less_than_lo(self):
        d = dist()
        with pytest.raises(ValueError, match="hi"):
            d.probability_between(Decimal(60), Decimal(40))

    def test_deterministic(self):
        d = dist()
        assert d.probability_above(REF_F) == d.probability_above(REF_F)


class TestCrossCheckAgainstO2C:
    def test_discounted_expected_call_payoff_matches_black76_price(self):
        d = dist()
        expected_payoff = structure_expected_payoff_of_single_call(d)
        import math
        discount = Decimal(repr(math.exp(-float(REF_R) * float(REF_T))))
        assert close(discount * expected_payoff, REF_CALL_PRICE, Decimal("0.001"))

    def test_discounted_expected_put_payoff_matches_black76_price(self):
        d = dist()
        expected_payoff = d.expected_value(lambda s: max(REF_K - s, Decimal(0)))
        import math
        discount = Decimal(repr(math.exp(-float(REF_R) * float(REF_T))))
        assert close(discount * expected_payoff, REF_PUT_PRICE, Decimal("0.001"))

    def test_second_reference_example(self):
        d = Black76RiskNeutralDistribution(forward=Decimal(100), volatility=Decimal("0.2"), time_to_expiry_years=Decimal("0.5"))
        expected_payoff = d.expected_value(lambda s: max(s - Decimal(100), Decimal(0)))
        import math
        discount = Decimal(repr(math.exp(-0.02 * 0.5)))
        assert close(discount * expected_payoff, Decimal("5.5811067246048118"), Decimal("0.002"))


def structure_expected_payoff_of_single_call(d: Black76RiskNeutralDistribution) -> Decimal:
    return d.expected_value(lambda s: max(s - REF_K, Decimal(0)))


class TestProbabilityPositiveNetPnl:
    def test_bull_call_spread_matches_probability_above_its_own_break_even(self):
        s = OptionStructure((mkleg(100, "CE", "BUY", "6"), mkleg(110, "CE", "SELL", "2")))
        be = structure_break_even(s).points[0]
        d = Black76RiskNeutralDistribution(forward=Decimal(105), volatility=Decimal("0.25"), time_to_expiry_years=Decimal("0.1"))
        p = probability_positive_net_pnl_by_expiry(s, d)
        assert p == d.probability_above(be)

    def test_deep_itm_long_call_cheap_premium_near_certain_profit(self):
        s = OptionStructure((mkleg(50, "CE", "BUY", "1"),))
        d = Black76RiskNeutralDistribution(forward=Decimal(200), volatility=Decimal("0.2"), time_to_expiry_years=Decimal("0.1"))
        assert probability_positive_net_pnl_by_expiry(s, d) > Decimal("0.99")

    def test_deep_otm_long_call_near_certain_loss(self):
        s = OptionStructure((mkleg(500, "CE", "BUY", "5"),))
        d = Black76RiskNeutralDistribution(forward=Decimal(100), volatility=Decimal("0.2"), time_to_expiry_years=Decimal("0.1"))
        assert probability_positive_net_pnl_by_expiry(s, d) < Decimal("0.01")

    def test_straddle_at_the_money_is_an_intermediate_probability(self):
        s = OptionStructure((mkleg(100, "CE", "BUY", "5"), mkleg(100, "PE", "BUY", "5")))
        d = Black76RiskNeutralDistribution(forward=Decimal(100), volatility=Decimal("0.3"), time_to_expiry_years=Decimal("0.2"))
        p = probability_positive_net_pnl_by_expiry(s, d)
        assert Decimal("0.05") < p < Decimal("0.95")

    def test_naked_short_call_probability_is_one_minus_bull_call_breakeven_case(self):
        s = OptionStructure((mkleg(100, "CE", "SELL", "5"),))
        be = structure_break_even(s).points[0]
        d = Black76RiskNeutralDistribution(forward=Decimal(95), volatility=Decimal("0.25"), time_to_expiry_years=Decimal("0.1"))
        assert probability_positive_net_pnl_by_expiry(s, d) == d.probability_below(be)


class TestStructureExpectedValue:
    def test_entering_at_model_price_gives_near_zero_ev(self):
        d = dist()
        model_price = black76_price(ValuationInputs(
            option_type="CE", forward=REF_F, strike=REF_K, valuation_at=dt.datetime(2024, 1, 1, tzinfo=UTC),
            expiry_at=dt.datetime(2024, 1, 1, tzinfo=UTC) + dt.timedelta(seconds=float(REF_T) * 365 * 24 * 3600),
            risk_free_rate=REF_R,
        ), REF_SIGMA)
        s = OptionStructure((mkleg(50, "CE", "BUY", model_price),))
        ev = structure_expected_value(s, d, REF_R, REF_T)
        assert abs(ev) < Decimal("1")

    def test_rejects_negative_risk_free_rate(self):
        with pytest.raises(ValueError, match="risk_free_rate"):
            structure_expected_value(OptionStructure((mkleg(100, "CE", "BUY", "5"),)), dist(), Decimal("-0.01"), REF_T)

    def test_rejects_nonpositive_time_to_expiry(self):
        with pytest.raises(ValueError, match="time_to_expiry_years"):
            structure_expected_value(OptionStructure((mkleg(100, "CE", "BUY", "5"),)), dist(), REF_R, Decimal(0))

    def test_deterministic(self):
        s = OptionStructure((mkleg(100, "CE", "BUY", "5"),))
        d = dist(forward=Decimal(100))
        assert structure_expected_value(s, d, REF_R, REF_T) == structure_expected_value(s, d, REF_R, REF_T)


class TestO2FIntegration:
    def test_probability_metrics_has_exactly_one_populated_key(self):
        s = OptionStructure((mkleg(100, "CE", "BUY", "5"),))
        d = dist(forward=Decimal(100))
        metrics = probability_metrics_from_distribution(s, d)
        assert set(metrics.keys()) == {"POSITIVE_NET_PNL_BY_EXPIRY"}
        assert metrics["POSITIVE_NET_PNL_BY_EXPIRY"] == probability_positive_net_pnl_by_expiry(s, d)

    def test_probability_metrics_feeds_a_valid_recommendation_result(self):
        s = OptionStructure((mkleg(100, "CE", "BUY", "5"),))
        d = dist(forward=Decimal(100))
        metrics = probability_metrics_from_distribution(s, d)
        ts = dt.datetime(2024, 7, 4, 10, 0, tzinfo=UTC)
        candidate = CandidateStrategy(
            timestamp=ts, structure=s, rationale="illustrative", market_context=None, iv_context=(), greeks_context=(),
            max_profit=structure_max_loss(s), max_loss=structure_max_loss(s), break_even=structure_break_even(s),
            target=None, stop=None, time_exit_at=None, expected_holding_period=None,
            required_probability_propositions=("POSITIVE_NET_PNL_BY_EXPIRY",), expected_value_inputs=None,
        )
        result = RecommendationResult(
            status="TRADE", timestamp=ts, underlying="NIFTY", candidate=candidate,
            probability_metrics=metrics, expected_value=structure_expected_value(s, d, REF_R, REF_T),
        )
        assert result.probability_metrics["POSITIVE_NET_PNL_BY_EXPIRY"] == metrics["POSITIVE_NET_PNL_BY_EXPIRY"]


def sidecar_last_modified(family: str, day: dt.date):
    side = json.loads(_sidecar(sample_path(family, day)).read_text())
    return avail.parse_last_modified((side.get("response_headers") or {}).get("Last-Modified"))


def near_future_close(day: dt.date, underlying: str) -> Decimal:
    got = load("udiff", day)
    futs = sorted((b for b in udiff_bars(*got) if b.key.underlying == underlying and b.key.dtype == "IDF"), key=lambda b: b.key.expiry)
    assert futs and futs[0].volume > 0
    return futs[0].close


class TestRealFixtureReplay:

    RISK_FREE_RATE = Decimal("0.065")

    def test_real_nifty_snapshot_probability_and_ev(self):
        day = D(2026, 9, 18)
        got = load("udiff", day)
        if got is None:
            pytest.skip("O1.0B NSE sample not present on disk")
        forward = near_future_close(day, "NIFTY")
        lm = sidecar_last_modified("udiff", day)
        av = avail.availability_for_eod_file(day, None, lm)
        real_call = None
        for b in udiff_bars(*got):
            if b.key.underlying != "NIFTY" or b.key.dtype != "IDO" or b.key.cp != "CE" or b.volume <= 0:
                continue
            real_call = b
            if abs(Decimal(b.key.strike_paise) / 100 - forward) < 200:
                break
        if real_call is None:
            pytest.skip("no traded NIFTY call found in this fixture day")
        contract = ContractIdentity("NIFTY", "IDO", real_call.key.expiry, real_call.key.strike_paise, "CE")
        bar = DerivativeBar.from_raw(
            contract=contract, session_date=day, bar_end_at=avail.session_close_utc(day), interval="1D",
            open_=real_call.open, high=real_call.high, low=real_call.low, close=real_call.close, volume=real_call.volume,
            open_interest=real_call.oi, oi_change=real_call.oi_change, turnover=real_call.turnover, settle_raw=real_call.settle,
            underlying_price=real_call.underlying_price, observed_lot_size=real_call.lot, available_at=av["available_at"],
            availability_basis=av["availability_basis"], source="nse_archives", source_row_hash=real_call.row_hash,
            provenance_class=FIXTURE, is_expiry_day=(real_call.key.expiry == day),
        )
        expiry_at = avail.session_close_utc(contract.expiry)
        inputs = ValuationInputs(option_type="CE", forward=forward, strike=Decimal(contract.strike_paise) / 100,
                                 valuation_at=bar.available_at, expiry_at=expiry_at, risk_free_rate=self.RISK_FREE_RATE)
        iv_result = black76_implied_volatility(inputs, bar.close)
        if iv_result.status != "OK":
            pytest.skip("this real fixture row did not produce a solvable IV under the illustrative forward assumption")
        t = inputs.t
        distribution = Black76RiskNeutralDistribution(forward=forward, volatility=iv_result.iv, time_to_expiry_years=t)
        leg = OptionLeg(contract=contract, side="BUY", lots=1, lot_size=50, lot_size_source="CALLER_SUPPLIED",
                        entry_price=bar.close, entry_price_source="TRADED_BAR", provenance_class=FIXTURE, entry_observed_at=bar.available_at)
        s = OptionStructure((leg,))
        p = probability_positive_net_pnl_by_expiry(s, distribution)
        ev = structure_expected_value(s, distribution, self.RISK_FREE_RATE, t)
        assert Decimal(0) <= p <= Decimal(1)
        assert abs(ev) < Decimal(1000)
