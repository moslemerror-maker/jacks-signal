from __future__ import annotations

import ast
import datetime as dt
import inspect
import json
from decimal import Decimal

import pytest

from backend.app.options import availability as avail
from backend.app.options import bars as barsmod
from backend.app.options.bars import DerivativeBar, classify_expiry_settle, classify_trade_state, settle_semantic_for
from backend.app.options.contracts import ContractIdentity
from scripts.run_o1_0b_sample_gate import _sidecar, load, sample_path, udiff_bars

D = dt.date
UTC = dt.timezone.utc
FIXTURE = "DEVELOPMENT_FIXTURE"


def nifty_call(strike=2200000, expiry=D(2024, 7, 25)) -> ContractIdentity:
    return ContractIdentity("NIFTY", "IDO", expiry, strike, "CE")


def dec(s):
    return Decimal(s)


def make_bar(*, is_expiry_day=False, session_date=D(2024, 7, 5), volume=100, o="10", h="12", l="9", c="11", settle="11", underlying="24000",
            contract=None) -> DerivativeBar:
    bar_end = avail.session_close_utc(session_date)
    available_at, basis = avail.next_session_open_utc(session_date)
    return DerivativeBar.from_raw(
        contract=contract or nifty_call(), session_date=session_date, bar_end_at=bar_end, interval="1D",
        open_=dec(o), high=dec(h), low=dec(l), close=dec(c), volume=volume, open_interest=500, oi_change=10,
        turnover=dec("1000"), settle_raw=dec(settle) if settle is not None else None, underlying_price=dec(underlying) if underlying else None,
        observed_lot_size=50, available_at=available_at, availability_basis=basis, source="nse_archives", source_row_hash="abc123",
        provenance_class=FIXTURE, is_expiry_day=is_expiry_day,
    )


class TestClassifyTradeState:
    def test_traded(self):
        assert classify_trade_state(10, dec(5), dec(6), dec(4), dec("5.5")) == (True, "TRADED")

    def test_carried_close_is_not_an_observation(self):
        assert classify_trade_state(0, dec(0), dec(0), dec(0), dec("26.65")) == (False, "NOT_TRADED_PRICE_CARRIED")

    def test_no_price(self):
        assert classify_trade_state(0, dec(0), dec(0), dec(0), dec(0)) == (False, "NOT_TRADED_NO_PRICE")

    def test_zero_volume_with_ohlc_is_an_anomaly(self):
        assert classify_trade_state(0, dec(1), dec(2), dec(1), dec(1)) == (False, "ANOMALY_NOT_TRADED_WITH_OHLC")

    def test_traded_with_zero_open_is_an_anomaly(self):
        assert classify_trade_state(2, dec(0), dec(0), dec(0), dec("1361.7")) == (False, "ANOMALY_TRADED_NONPOSITIVE_PRICE")

    def test_close_outside_range_is_an_anomaly(self):
        assert classify_trade_state(33, dec("463.05"), dec("465.45"), dec(458), dec(457)) == (False, "ANOMALY_TRADED_OHLC_ORDER")

    def test_negative_volume(self):
        assert classify_trade_state(-1, dec(1), dec(1), dec(1), dec(1)) == (False, "ANOMALY_NEGATIVE_VOLUME")


class TestClassifyExpirySettle:
    def c(self, settle, close="10", spot="22326.9", strike="22300", cp="CE"):
        return classify_expiry_settle(dec(settle), dec(close), dec(spot), dec(strike), cp)

    def test_zero_wins_over_zero_intrinsic(self):
        assert self.c("0", strike="22400") == "ZERO"

    def test_underlying_settlement_price(self):
        assert self.c("22326.9") == "UNDERLYING_SETTLEMENT_PRICE"

    def test_intrinsic(self):
        assert self.c("26.9") == "INTRINSIC_VALUE"

    def test_equals_close_and_other(self):
        assert self.c("10", close="10") == "EQUALS_LAST_CLOSE"
        assert self.c("11", close="10") == "OTHER"


class TestSettleSemanticFor:
    def test_none_when_no_settle_value(self):
        assert settle_semantic_for(None, dec(10), dec(24000), dec(22000), "CE", is_expiry_day=True) is None

    def test_live_daily_settlement_on_a_non_expiry_day(self):
        assert settle_semantic_for(dec("11.5"), dec(10), dec(24000), dec(22000), "CE", is_expiry_day=False) == "LIVE_DAILY_SETTLEMENT"

    def test_unknown_on_expiry_day_without_spot(self):
        assert settle_semantic_for(dec("0"), dec(0), None, dec(22000), "CE", is_expiry_day=True) == "UNKNOWN"

    def test_classifies_via_classify_expiry_settle_on_a_real_expiry_day(self):
        assert settle_semantic_for(dec("24000"), dec(10), dec(24000), dec(22000), "CE", is_expiry_day=True) == "UNDERLYING_SETTLEMENT_PRICE"


class TestDerivativeBarFromRaw:
    def test_traded_bar(self):
        b = make_bar(volume=100, o="10", h="12", l="9", c="11")
        assert b.is_traded and b.trade_state == "TRADED"
        assert (b.open, b.high, b.low, b.close) == (dec(10), dec(12), dec(9), dec(11))
        assert b.carried_close is None
        assert (b.raw_open, b.raw_high, b.raw_low, b.raw_close) == (dec(10), dec(12), dec(9), dec(11))

    def test_carried_close_bar_has_no_canonical_ohlc(self):
        b = make_bar(volume=0, o="0", h="0", l="0", c="26.65", settle=None)
        assert not b.is_traded and b.trade_state == "NOT_TRADED_PRICE_CARRIED"
        assert (b.open, b.high, b.low, b.close) == (None, None, None, None)
        assert b.carried_close == dec("26.65")
        assert b.raw_close == dec("26.65")

    def test_no_price_bar_has_neither_canonical_ohlc_nor_carried_close(self):
        b = make_bar(volume=0, o="0", h="0", l="0", c="0", settle="0")
        assert b.trade_state == "NOT_TRADED_NO_PRICE"
        assert (b.open, b.high, b.low, b.close, b.carried_close) == (None, None, None, None, None)

    def test_anomalous_row_gets_no_price_either(self):
        b = make_bar(volume=2, o="0", h="0", l="0", c="1361.7", settle=None)
        assert b.trade_state == "ANOMALY_TRADED_NONPOSITIVE_PRICE" and not b.is_traded
        assert (b.open, b.high, b.low, b.close, b.carried_close) == (None, None, None, None, None)
        assert b.raw_close == dec("1361.7")

    def test_observed_lot_size_is_carried_but_never_authoritative_by_itself(self):
        b = make_bar()
        assert b.observed_lot_size == 50

    def test_settle_semantic_on_a_real_expiry_day(self):
        b = make_bar(is_expiry_day=True, session_date=D(2024, 7, 25), settle="24000", underlying="24000",
                     contract=nifty_call(expiry=D(2024, 7, 25)))
        assert b.settle_semantic == "UNDERLYING_SETTLEMENT_PRICE"

    def test_settle_semantic_on_a_non_expiry_day(self):
        b = make_bar(is_expiry_day=False, settle="11.5")
        assert b.settle_semantic == "LIVE_DAILY_SETTLEMENT"

    def test_provenance_class_is_mandatory(self):
        with pytest.raises(TypeError):
            DerivativeBar.from_raw(  # type: ignore[call-arg]
                contract=nifty_call(), session_date=D(2024, 7, 5), bar_end_at=avail.session_close_utc(D(2024, 7, 5)), interval="1D",
                open_=dec(1), high=dec(1), low=dec(1), close=dec(1), volume=1, open_interest=None, oi_change=None, turnover=None,
                settle_raw=None, underlying_price=None, observed_lot_size=None,
                available_at=avail.session_close_utc(D(2024, 7, 5)) + dt.timedelta(hours=1), availability_basis="x", source="nse_archives",
                source_row_hash=None,
            )


class TestDerivativeBarDirectConstructionValidation:

    def base_kwargs(self, **over):
        bar_end = avail.session_close_utc(D(2024, 7, 5))
        d = dict(contract=nifty_call(), session_date=D(2024, 7, 5), bar_end_at=bar_end, interval="1D",
                raw_open=dec(10), raw_high=dec(12), raw_low=dec(9), raw_close=dec(11), volume=100, open_interest=None, oi_change=None,
                turnover=None, settle_raw=None, underlying_price=None, observed_lot_size=None,
                is_traded=True, trade_state="TRADED", open=dec(10), high=dec(12), low=dec(9), close=dec(11), carried_close=None, settle_semantic=None,
                available_at=bar_end, availability_basis="RULE_NEXT_SESSION_OPEN", source="nse_archives", source_row_hash=None, provenance_class=FIXTURE)
        d.update(over)
        return d

    def test_valid_direct_construction(self):
        DerivativeBar(**self.base_kwargs())

    def test_is_traded_must_agree_with_trade_state(self):
        with pytest.raises(ValueError, match="inconsistent with trade_state"):
            DerivativeBar(**self.base_kwargs(is_traded=False))

    def test_traded_bar_must_have_full_ohlc(self):
        with pytest.raises(ValueError, match="must have open/high/low/close"):
            DerivativeBar(**self.base_kwargs(close=None))

    def test_traded_bar_must_not_have_carried_close(self):
        with pytest.raises(ValueError, match="must not carry a carried_close"):
            DerivativeBar(**self.base_kwargs(carried_close=dec(11)))

    def test_traded_ohlc_order_enforced(self):
        with pytest.raises(ValueError, match="out of order"):
            DerivativeBar(**self.base_kwargs(high=dec(1)))

    def test_non_traded_bar_must_not_have_canonical_ohlc(self):
        kwargs = self.base_kwargs(is_traded=False, trade_state="NOT_TRADED_NO_PRICE", open=None, high=None, low=None, close=dec(5))
        with pytest.raises(ValueError, match="must not have canonical open/high/low/close"):
            DerivativeBar(**kwargs)

    def test_not_traded_price_carried_requires_carried_close(self):
        kwargs = self.base_kwargs(is_traded=False, trade_state="NOT_TRADED_PRICE_CARRIED", open=None, high=None, low=None, close=None, carried_close=None)
        with pytest.raises(ValueError, match="requires carried_close"):
            DerivativeBar(**kwargs)

    def test_other_states_must_not_have_a_carried_close(self):
        kwargs = self.base_kwargs(is_traded=False, trade_state="NOT_TRADED_NO_PRICE", open=None, high=None, low=None, close=None, carried_close=dec(5))
        with pytest.raises(ValueError, match="must not have a carried_close"):
            DerivativeBar(**kwargs)

    def test_no_look_ahead_available_at_before_bar_end_at_rejected(self):
        bar_end = avail.session_close_utc(D(2024, 7, 5))
        with pytest.raises(ValueError, match="no look-ahead"):
            DerivativeBar(**self.base_kwargs(available_at=bar_end - dt.timedelta(minutes=1)))

    def test_naive_bar_end_at_rejected(self):
        with pytest.raises(ValueError, match="bar_end_at must be timezone-aware"):
            DerivativeBar(**self.base_kwargs(bar_end_at=dt.datetime(2024, 7, 5, 15, 30)))

    def test_naive_available_at_rejected(self):
        bar_end = avail.session_close_utc(D(2024, 7, 5))
        with pytest.raises(ValueError, match="available_at must be timezone-aware"):
            DerivativeBar(**self.base_kwargs(available_at=dt.datetime(2024, 7, 5, 15, 30)))

    def test_invalid_trade_state_rejected(self):
        with pytest.raises(ValueError, match="trade_state"):
            DerivativeBar(**self.base_kwargs(trade_state="MADE_UP"))

    def test_invalid_settle_semantic_rejected(self):
        with pytest.raises(ValueError, match="settle_semantic"):
            DerivativeBar(**self.base_kwargs(settle_semantic="MADE_UP"))

    def test_provenance_class_rejects_invalid_value(self):
        with pytest.raises(ValueError, match="provenance_class"):
            DerivativeBar(**self.base_kwargs(provenance_class="MADE_UP"))

    def test_frozen(self):
        b = DerivativeBar(**self.base_kwargs())
        import dataclasses
        with pytest.raises(dataclasses.FrozenInstanceError):
            b.volume = 999  # type: ignore[misc]


class TestAvailability:
    def lm(self, y, m, d, hh, mm):
        return dt.datetime(y, m, d, hh, mm, tzinfo=avail.IST).astimezone(UTC)

    def test_observed_last_modified_between_close_and_next_open(self):
        r = avail.availability_for_eod_file(D(2024, 7, 5), None, self.lm(2024, 7, 5, 19, 8))
        assert r["availability_basis"] == "OBSERVED_HTTP_LAST_MODIFIED"

    def test_reupload_after_next_open_falls_back_to_the_rule(self):
        r = avail.availability_for_eod_file(D(2015, 1, 5), None, self.lm(2019, 1, 17, 17, 38))
        assert r["availability_basis"] == "RULE_NEXT_SESSION_OPEN"
        assert r["available_at"] == self.lm(2015, 1, 6, 9, 15)

    def test_same_day_close_decision_can_never_see_post_close_data(self):
        close = avail.session_close_utc(D(2024, 7, 5))
        for lm in (None, self.lm(2024, 7, 5, 15, 30), self.lm(2024, 7, 5, 19, 8), self.lm(2019, 1, 17, 17, 38)):
            assert avail.availability_for_eod_file(D(2024, 7, 5), None, lm)["available_at"] >= close

    def test_last_modified_header_parses_as_utc(self):
        t = avail.parse_last_modified("Fri, 05 Jul 2024 13:38:00 GMT")
        assert t == dt.datetime(2024, 7, 5, 13, 38, tzinfo=UTC)

    def test_bad_or_missing_last_modified(self):
        assert avail.parse_last_modified(None) is None
        assert avail.parse_last_modified("not a date") is None


class TestContractLinkage:
    def test_bar_carries_the_o2a_contract_identity_object(self):
        c = nifty_call()
        b = make_bar(contract=c)
        assert b.contract is c and b.contract.underlying == "NIFTY"

    def test_two_bars_for_the_same_real_contract_share_an_equal_identity(self):
        b1 = make_bar(contract=nifty_call())
        b2 = make_bar(session_date=D(2024, 7, 8), contract=nifty_call())
        assert b1.contract == b2.contract


def sidecar_last_modified(family: str, day: dt.date) -> dt.datetime | None:
    side = json.loads(_sidecar(sample_path(family, day)).read_text())
    return avail.parse_last_modified((side.get("response_headers") or {}).get("Last-Modified"))


def load_real_bars(family: str, day: dt.date, underlying: str, dclass: str) -> list[DerivativeBar]:
    got = load(family, day)
    if got is None:
        pytest.skip(f"O1.0B NSE sample not present on disk for {family}:{day}")
    src_bars = udiff_bars(*got)
    lm = sidecar_last_modified(family, day)
    av = avail.availability_for_eod_file(day, None, lm)
    out = []
    for b in src_bars:
        if b.key.underlying != underlying or b.key.dtype != dclass:
            continue
        strike = None if b.key.cp is None else b.key.strike_paise
        contract = ContractIdentity(b.key.underlying, b.key.dtype, b.key.expiry, strike, b.key.cp)
        out.append(DerivativeBar.from_raw(
            contract=contract, session_date=day, bar_end_at=avail.session_close_utc(day), interval="1D",
            open_=b.open, high=b.high, low=b.low, close=b.close, volume=b.volume, open_interest=b.oi, oi_change=b.oi_change,
            turnover=b.turnover, settle_raw=b.settle, underlying_price=b.underlying_price, observed_lot_size=b.lot,
            available_at=av["available_at"], availability_basis=av["availability_basis"], source="nse_archives",
            source_row_hash=b.row_hash, provenance_class=FIXTURE, is_expiry_day=(b.key.expiry == day),
        ))
    return out


class TestRealFixtureReplay:
    def test_nifty_and_banknifty_2025_09_30_simultaneous_expiry(self):
        nifty = load_real_bars("udiff", D(2025, 9, 30), "NIFTY", "IDO")
        bn = load_real_bars("udiff", D(2025, 9, 30), "BANKNIFTY", "IDO")
        assert len(nifty) == 1572 and len(bn) == 921
        assert all(b.provenance_class == FIXTURE for b in nifty + bn)
        bn_expiring = [b for b in bn if b.session_date == b.contract.expiry]
        assert bn_expiring and all(b.settle_semantic == "UNDERLYING_SETTLEMENT_PRICE" for b in bn_expiring)

    def test_2024_07_05_zero_volume_share_matches_the_documented_finding(self):
        bn = load_real_bars("udiff", D(2024, 7, 5), "BANKNIFTY", "IDO")
        traded = sum(1 for b in bn if b.is_traded)
        assert len(bn) == 1636 and traded == 755

    def test_no_bar_ever_exposes_a_canonical_price_for_a_non_traded_row(self):
        bn = load_real_bars("udiff", D(2024, 7, 5), "BANKNIFTY", "IDO")
        for b in bn:
            if not b.is_traded:
                assert b.open is None and b.high is None and b.low is None and b.close is None

    def test_availability_is_after_the_session_close_for_every_real_bar(self):
        nifty = load_real_bars("udiff", D(2026, 9, 18), "NIFTY", "IDO")
        close = avail.session_close_utc(D(2026, 9, 18))
        assert nifty and all(b.available_at >= close for b in nifty)

    def test_no_anomalies_among_real_nifty_banknifty_option_rows(self):
        for underlying in ("NIFTY", "BANKNIFTY"):
            bars_ = load_real_bars("udiff", D(2026, 9, 18), underlying, "IDO")
            assert bars_ and not any(b.trade_state.startswith("ANOMALY") for b in bars_)


def _imported_module_names(module) -> set[str]:
    tree = ast.parse(inspect.getsource(module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


class TestNoEquityCorporateActionCoupling:
    @pytest.mark.parametrize("module", [barsmod, avail])
    def test_no_forbidden_import(self, module):
        imported = _imported_module_names(module)
        forbidden = {n for n in imported if "adjustment_factors" in n or "corporate_action" in n or n.endswith("point_in_time")}
        assert forbidden == set()

    def test_bars_module_has_no_forbidden_mention_outside_its_own_explanatory_docstring(self):
        tree = ast.parse(inspect.getsource(barsmod))
        body_without_docstring = tree.body[1:] if (tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant)) else tree.body
        code_text = "\n".join(ast.unparse(n) for n in body_without_docstring)
        assert "adjustment_factors" not in code_text and "corporate_action" not in code_text.lower()
