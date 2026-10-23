from __future__ import annotations

import dataclasses
import datetime as dt
import json
from decimal import Decimal

import pytest

from backend.app.options import availability as avail
from backend.app.options.bars import DerivativeBar
from backend.app.options.contracts import ContractIdentity
from backend.app.options.valuation import (
    EuropeanOptionType, Greeks, ImpliedVolatilityResult, ValuationInputs, black76_greeks, black76_implied_volatility,
    black76_price, implied_volatility_from_bar, time_to_expiry_years,
)
from scripts.run_o1_0b_sample_gate import _sidecar, load, sample_path, udiff_bars

D = dt.date
UTC = dt.timezone.utc
FIXTURE = "DEVELOPMENT_FIXTURE"

REF_F, REF_K, REF_R, REF_SIGMA, REF_T = Decimal("49"), Decimal("50"), Decimal("0.05"), Decimal("0.2"), Decimal("0.3846")
REF_CALL_PRICE = Decimal("1.942417578252269")
REF_PUT_PRICE = Decimal("2.9233712951919637")
REF_DELTA = Decimal("0.45107017482201828")
REF_GAMMA = Decimal("0.0640646705882")
REF_THETA_CALL = Decimal("-0.00816236877462")
REF_THETA_PUT = Decimal("-0.00802799155312")
REF_VEGA = Decimal("0.118317785624")
REF_RHO_CALL = Decimal("-0.0074705380059582258")
REF_RHO_PUT = Decimal("-0.011243286001308292")
TOL = Decimal("0.00001")


def inputs(option_type: EuropeanOptionType = "CE", forward=REF_F, strike=REF_K, r=REF_R, t_years=REF_T, valuation_at=None, expiry_at=None) -> ValuationInputs:
    if valuation_at is None:
        valuation_at = dt.datetime(2024, 1, 1, tzinfo=UTC)
    if expiry_at is None:
        expiry_at = valuation_at + dt.timedelta(seconds=float(t_years) * 365 * 24 * 3600)
    return ValuationInputs(option_type=option_type, forward=forward, strike=strike, valuation_at=valuation_at, expiry_at=expiry_at, risk_free_rate=r)


def close(a: Decimal, b: Decimal, tol: Decimal = TOL) -> bool:
    return abs(a - b) < tol


class TestBlack76Pricing:
    def test_call_price_matches_reference(self):
        assert close(black76_price(inputs("CE"), REF_SIGMA), REF_CALL_PRICE)

    def test_put_price_matches_reference(self):
        assert close(black76_price(inputs("PE"), REF_SIGMA), REF_PUT_PRICE)

    def test_result_is_a_plain_decimal_not_a_numpy_scalar_wrapper(self):
        price = black76_price(inputs("CE"), REF_SIGMA)
        assert type(price) is Decimal

    def test_second_reference_example_atm(self):
        i = inputs("CE", forward=Decimal(100), strike=Decimal(100), r=Decimal("0.02"), t_years=Decimal("0.5"))
        assert close(black76_price(i, Decimal("0.2")), Decimal("5.5811067246048118"))


class TestPutCallParity:
    def test_black76_parity_c_minus_p_equals_discounted_forward_minus_strike(self):
        import math
        c = black76_price(inputs("CE"), REF_SIGMA)
        p = black76_price(inputs("PE"), REF_SIGMA)
        expected = Decimal(repr(math.exp(-float(REF_R) * float(REF_T)) * float(REF_F - REF_K)))
        assert close(c - p, expected, Decimal("0.0001"))


class TestPricingDomainValidation:
    def test_rejects_non_positive_forward(self):
        with pytest.raises(ValueError, match="forward"):
            black76_price(inputs(forward=Decimal(0)), REF_SIGMA)

    def test_rejects_non_positive_strike(self):
        with pytest.raises(ValueError, match="strike"):
            black76_price(inputs(strike=Decimal(-1)), REF_SIGMA)

    def test_rejects_non_positive_volatility(self):
        with pytest.raises(ValueError, match="volatility"):
            black76_price(inputs(), Decimal(0))

    def test_rejects_zero_time_to_expiry(self):
        v = dt.datetime(2024, 1, 1, tzinfo=UTC)
        with pytest.raises(ValueError, match="time_to_expiry_years is 0"):
            black76_price(inputs(valuation_at=v, expiry_at=v), REF_SIGMA)

    def test_construction_rejects_expiry_before_valuation_is_just_zero_time_not_negative(self):
        v = dt.datetime(2024, 1, 2, tzinfo=UTC)
        e = dt.datetime(2024, 1, 1, tzinfo=UTC)
        i = inputs(valuation_at=v, expiry_at=e)
        assert i.t == Decimal(0)
        with pytest.raises(ValueError, match="time_to_expiry_years is 0"):
            black76_price(i, REF_SIGMA)

    def test_option_type_restricted_to_european(self):
        with pytest.raises(ValueError, match="European"):
            inputs(option_type="CA")  # type: ignore[arg-type]

    def test_valuation_inputs_requires_timezone_aware_timestamps(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            ValuationInputs(option_type="CE", forward=REF_F, strike=REF_K, valuation_at=dt.datetime(2024, 1, 1),
                            expiry_at=dt.datetime(2024, 6, 1, tzinfo=UTC), risk_free_rate=REF_R)

    def test_unknown_model_rejected(self):
        with pytest.raises(ValueError, match="model"):
            dataclasses.replace(inputs(), model="BJERKSUND_STENSLAND")  # type: ignore[arg-type]


class TestImpliedVolatility:
    def test_recovers_known_volatility_from_a_model_price(self):
        i = inputs("CE")
        price = black76_price(i, REF_SIGMA)
        r = black76_implied_volatility(i, price)
        assert r.status == "OK" and close(r.iv, REF_SIGMA, Decimal("0.0001"))

    def test_second_reference_iv_round_trip(self):
        i = inputs("CE", forward=Decimal(100), strike=Decimal(100), r=Decimal("0.02"), t_years=Decimal("0.5"))
        r = black76_implied_volatility(i, Decimal("5.5811067246048118"))
        assert r.status == "OK" and close(r.iv, Decimal("0.2"), Decimal("0.00001"))

    def test_price_below_intrinsic_is_refused_not_a_number(self):
        i = inputs("CE", forward=Decimal(100), strike=Decimal(80))
        r = black76_implied_volatility(i, Decimal("0.5"))
        assert r.status == "BELOW_INTRINSIC" and r.iv is None and r.detail

    def test_absurdly_high_price_is_refused_not_a_number(self):
        i = inputs("CE", forward=Decimal(100), strike=Decimal(100))
        r = black76_implied_volatility(i, Decimal(500))
        assert r.status == "ABOVE_MAXIMUM" and r.iv is None

    def test_non_positive_price_is_refused(self):
        i = inputs("CE")
        assert black76_implied_volatility(i, Decimal(0)).status == "INVALID_INPUT"
        assert black76_implied_volatility(i, Decimal(-1)).status == "INVALID_INPUT"

    def test_zero_time_to_expiry_is_refused_not_infinity(self):
        v = dt.datetime(2024, 1, 1, tzinfo=UTC)
        i = inputs(valuation_at=v, expiry_at=v)
        r = black76_implied_volatility(i, Decimal(5))
        assert r.status == "EXPIRED_OR_ZERO_TIME" and r.iv is None

    def test_near_expiry_small_positive_time_still_solves(self):
        v = dt.datetime(2024, 1, 1, tzinfo=UTC)
        i = inputs(forward=Decimal(100), strike=Decimal(100), valuation_at=v, expiry_at=v + dt.timedelta(hours=1))
        price = black76_price(i, REF_SIGMA)
        r = black76_implied_volatility(i, price)
        assert r.status == "OK" and close(r.iv, REF_SIGMA, Decimal("0.01"))

    def test_iv_result_records_every_input_for_explainability(self):
        i = inputs("CE")
        r = black76_implied_volatility(i, black76_price(i, REF_SIGMA))
        assert r.forward == REF_F and r.strike == REF_K and r.risk_free_rate == REF_R and r.model == "BLACK_76"

    def test_iv_result_invariant_iv_set_iff_status_ok(self):
        with pytest.raises(ValueError, match="iff status is OK"):
            ImpliedVolatilityResult(status="OK", iv=None, market_price=None, forward=None, strike=None, time_to_expiry_years=None, risk_free_rate=None, model="BLACK_76")
        with pytest.raises(ValueError, match="iff status is OK"):
            ImpliedVolatilityResult(status="INVALID_INPUT", iv=Decimal(1), market_price=None, forward=None, strike=None, time_to_expiry_years=None, risk_free_rate=None, model="BLACK_76")


class TestGreeks:
    def test_all_five_match_reference_values_call(self):
        g = black76_greeks(inputs("CE"), REF_SIGMA)
        assert close(g.delta, REF_DELTA) and close(g.gamma, REF_GAMMA) and close(g.theta_per_day, REF_THETA_CALL)
        assert close(g.vega_per_vol_point, REF_VEGA) and close(g.rho_per_rate_point, REF_RHO_CALL)

    def test_put_theta_and_rho_differ_from_call_gamma_and_vega_do_not(self):
        gc = black76_greeks(inputs("CE"), REF_SIGMA)
        gp = black76_greeks(inputs("PE"), REF_SIGMA)
        assert close(gp.theta_per_day, REF_THETA_PUT) and close(gp.rho_per_rate_point, REF_RHO_PUT)
        assert close(gc.gamma, gp.gamma) and close(gc.vega_per_vol_point, gp.vega_per_vol_point)

    def test_theta_is_verified_per_calendar_day_via_finite_difference_of_our_own_pricing_function(self):
        v = dt.datetime(2024, 1, 1, tzinfo=UTC)
        e = v + dt.timedelta(days=round(float(REF_T) * 365))
        i_now = inputs("CE", valuation_at=v, expiry_at=e)
        i_one_day_later = inputs("CE", valuation_at=v + dt.timedelta(days=1), expiry_at=e)
        price_now = black76_price(i_now, REF_SIGMA)
        price_one_day_less = black76_price(i_one_day_later, REF_SIGMA)
        empirical_one_day_decay = price_one_day_less - price_now
        g = black76_greeks(i_now, REF_SIGMA)
        ratio = g.theta_per_day / empirical_one_day_decay
        assert Decimal("0.9") < ratio < Decimal("1.1")

    def test_vega_matches_empirical_price_change_for_a_001_vol_move_not_a_100_move(self):
        price_now = black76_price(inputs("CE"), REF_SIGMA)
        price_plus_1pt = black76_price(inputs("CE"), REF_SIGMA + Decimal("0.01"))
        empirical_vega = price_plus_1pt - price_now
        g = black76_greeks(inputs("CE"), REF_SIGMA)
        assert close(g.vega_per_vol_point, empirical_vega, Decimal("0.001"))
        price_plus_100pt = black76_price(inputs("CE"), REF_SIGMA + Decimal("1.00"))
        assert abs(g.vega_per_vol_point - (price_plus_100pt - price_now)) > Decimal("1")

    def test_greeks_reject_zero_time_to_expiry(self):
        v = dt.datetime(2024, 1, 1, tzinfo=UTC)
        with pytest.raises(ValueError, match="time_to_expiry_years is 0"):
            black76_greeks(inputs(valuation_at=v, expiry_at=v), REF_SIGMA)

    def test_greeks_frozen_and_typed(self):
        g = black76_greeks(inputs("CE"), REF_SIGMA)
        with pytest.raises(dataclasses.FrozenInstanceError):
            g.delta = Decimal(0)  # type: ignore[misc]


class TestTimeToExpiry:
    def test_basic_one_year(self):
        v = dt.datetime(2024, 1, 1, tzinfo=UTC)
        e = dt.datetime(2025, 1, 1, tzinfo=UTC)
        assert close(time_to_expiry_years(v, e), Decimal("1.0"), Decimal("0.01"))

    def test_expiry_before_valuation_is_exactly_zero_not_negative(self):
        v = dt.datetime(2024, 6, 1, tzinfo=UTC)
        e = dt.datetime(2024, 1, 1, tzinfo=UTC)
        assert time_to_expiry_years(v, e) == Decimal(0)

    def test_equal_timestamps_is_zero(self):
        v = dt.datetime(2024, 1, 1, tzinfo=UTC)
        assert time_to_expiry_years(v, v) == Decimal(0)

    def test_requires_timezone_aware(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            time_to_expiry_years(dt.datetime(2024, 1, 1), dt.datetime(2024, 6, 1, tzinfo=UTC))


def make_bar(*, trade_state_inputs, session_date=D(2024, 7, 5), contract=None, available_at=None) -> DerivativeBar:
    bar_end = avail.session_close_utc(session_date)
    if available_at is None:
        available_at, basis = avail.next_session_open_utc(session_date)
    else:
        basis = "OBSERVED_HTTP_LAST_MODIFIED"
    o, h, l, c, v = trade_state_inputs
    return DerivativeBar.from_raw(
        contract=contract or ContractIdentity("NIFTY", "IDO", D(2024, 7, 25), 2200000, "CE"), session_date=session_date, bar_end_at=bar_end, interval="1D",
        open_=Decimal(o), high=Decimal(h), low=Decimal(l), close=Decimal(c), volume=v, open_interest=None, oi_change=None, turnover=None,
        settle_raw=None, underlying_price=None, observed_lot_size=None, available_at=available_at, availability_basis=basis,
        source="nse_archives", source_row_hash=None, provenance_class=FIXTURE,
    )


class TestBarIntegration:
    def test_traded_bar_produces_a_real_iv(self):
        b = make_bar(trade_state_inputs=("10", "12", "9", "11", 100))
        r = implied_volatility_from_bar(b, forward=Decimal("22005"), risk_free_rate=Decimal("0.065"))
        assert r.status == "OK" and r.iv is not None

    def test_carried_close_bar_is_refused_not_used_as_a_market_price(self):
        b = make_bar(trade_state_inputs=("0", "0", "0", "26.65", 0))
        r = implied_volatility_from_bar(b, forward=Decimal("22005"), risk_free_rate=Decimal("0.065"))
        assert r.status == "UNAVAILABLE_NOT_TRADED" and r.iv is None and r.market_price is None

    def test_no_price_bar_is_refused(self):
        b = make_bar(trade_state_inputs=("0", "0", "0", "0", 0))
        assert implied_volatility_from_bar(b, forward=Decimal("22005"), risk_free_rate=Decimal("0.065")).status == "UNAVAILABLE_NOT_TRADED"

    def test_anomalous_bar_is_refused(self):
        b = make_bar(trade_state_inputs=("0", "0", "0", "1361.7", 2))
        assert implied_volatility_from_bar(b, forward=Decimal("22005"), risk_free_rate=Decimal("0.065")).status == "UNAVAILABLE_NOT_TRADED"

    def test_valuation_before_available_at_is_refused_no_look_ahead(self):
        b = make_bar(trade_state_inputs=("10", "12", "9", "11", 100))
        too_early = b.available_at - dt.timedelta(minutes=1)
        r = implied_volatility_from_bar(b, forward=Decimal("22005"), risk_free_rate=Decimal("0.065"), valuation_at=too_early)
        assert r.status == "UNAVAILABLE_BEFORE_KNOWN"

    def test_valuation_at_exactly_available_at_is_accepted(self):
        b = make_bar(trade_state_inputs=("10", "12", "9", "11", 100))
        r = implied_volatility_from_bar(b, forward=Decimal("22005"), risk_free_rate=Decimal("0.065"), valuation_at=b.available_at)
        assert r.status == "OK"

    def test_default_valuation_at_is_the_bars_own_available_at(self):
        b = make_bar(trade_state_inputs=("10", "12", "9", "11", 100))
        r1 = implied_volatility_from_bar(b, forward=Decimal("22005"), risk_free_rate=Decimal("0.065"))
        r2 = implied_volatility_from_bar(b, forward=Decimal("22005"), risk_free_rate=Decimal("0.065"), valuation_at=b.available_at)
        assert r1.iv == r2.iv

    def test_american_option_type_is_refused_unsupported_exercise_style(self):
        c = ContractIdentity("SBIN", "STO", D(2010, 7, 29), 300000, "CA")
        b = make_bar(trade_state_inputs=("10", "12", "9", "11", 100), contract=c)
        r = implied_volatility_from_bar(b, forward=Decimal(300), risk_free_rate=Decimal("0.065"))
        assert r.status == "UNSUPPORTED_EXERCISE_STYLE"

    def test_naive_valuation_at_rejected(self):
        b = make_bar(trade_state_inputs=("10", "12", "9", "11", 100))
        r = implied_volatility_from_bar(b, forward=Decimal("22005"), risk_free_rate=Decimal("0.065"), valuation_at=dt.datetime(2024, 7, 8))
        assert r.status == "INVALID_INPUT"


def sidecar_last_modified(family: str, day: dt.date):
    side = json.loads(_sidecar(sample_path(family, day)).read_text())
    return avail.parse_last_modified((side.get("response_headers") or {}).get("Last-Modified"))


def near_future_close(day: dt.date, underlying: str) -> Decimal:
    got = load("udiff", day)
    futs = sorted((b for b in udiff_bars(*got) if b.key.underlying == underlying and b.key.dtype == "IDF"), key=lambda b: b.key.expiry)
    assert futs and futs[0].volume > 0, "expected a genuinely TRADED near-month future in the fixture (verified present on the sampled dates)"
    return futs[0].close


class TestRealFixtureReplay:

    RISK_FREE_RATE = Decimal("0.065")

    def real_bars(self, day, underlying):
        got = load("udiff", day)
        if got is None:
            pytest.skip(f"O1.0B NSE sample not present on disk for udiff:{day}")
        lm = sidecar_last_modified("udiff", day)
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

    def test_nifty_ce_and_pe_iv_recoverable_for_traded_near_atm_strikes(self):
        day = D(2026, 9, 18)
        forward = near_future_close(day, "NIFTY")
        bars_ = self.real_bars(day, "NIFTY")
        near_atm = sorted(bars_, key=lambda b: abs(Decimal(b.contract.strike_paise) / 100 - forward))[:6]
        assert {b.contract.option_type for b in near_atm} == {"CE", "PE"}
        results = [implied_volatility_from_bar(b, forward=forward, risk_free_rate=self.RISK_FREE_RATE) for b in near_atm]
        ok = [r for r in results if r.status == "OK"]
        assert ok, "expected at least one near-ATM NIFTY contract to produce a valid IV"
        assert all(r.iv > 0 for r in ok)

    def test_banknifty_ce_and_pe_iv_recoverable_for_traded_near_atm_strikes(self):
        day = D(2025, 9, 30)
        forward = near_future_close(day, "BANKNIFTY")
        bars_ = self.real_bars(day, "BANKNIFTY")
        near_atm = sorted(bars_, key=lambda b: abs(Decimal(b.contract.strike_paise) / 100 - forward))[:6]
        assert {b.contract.option_type for b in near_atm} <= {"CE", "PE"}
        results = [implied_volatility_from_bar(b, forward=forward, risk_free_rate=self.RISK_FREE_RATE) for b in near_atm]
        ok = [r for r in results if r.status == "OK"]
        assert ok, "expected at least one near-ATM BANKNIFTY contract to produce a valid IV"

    def test_real_greeks_are_within_theoretical_bounds_for_a_traded_nifty_call(self):
        day = D(2026, 9, 18)
        forward = near_future_close(day, "NIFTY")
        bars_ = self.real_bars(day, "NIFTY")
        calls = [b for b in bars_ if b.contract.option_type == "CE"]
        near_atm_call = min(calls, key=lambda b: abs(Decimal(b.contract.strike_paise) / 100 - forward))
        r = implied_volatility_from_bar(near_atm_call, forward=forward, risk_free_rate=self.RISK_FREE_RATE)
        assert r.status == "OK"
        expiry_at = avail.session_close_utc(near_atm_call.contract.expiry)
        i = ValuationInputs(option_type="CE", forward=forward, strike=Decimal(near_atm_call.contract.strike_paise) / 100,
                            valuation_at=near_atm_call.available_at, expiry_at=expiry_at, risk_free_rate=self.RISK_FREE_RATE)
        g = black76_greeks(i, r.iv)
        assert Decimal(0) < g.delta < Decimal(1)
        assert g.gamma > 0
        assert g.vega_per_vol_point > 0

    def test_far_otm_illiquid_strikes_may_legitimately_fail_iv_without_crashing(self):
        day = D(2026, 9, 18)
        forward = near_future_close(day, "NIFTY")
        bars_ = self.real_bars(day, "NIFTY")
        far = sorted(bars_, key=lambda b: abs(Decimal(b.contract.strike_paise) / 100 - forward))[-10:]
        results = [implied_volatility_from_bar(b, forward=forward, risk_free_rate=self.RISK_FREE_RATE) for b in far]
        assert all(r.status in ("OK", "BELOW_INTRINSIC", "ABOVE_MAXIMUM") for r in results)
