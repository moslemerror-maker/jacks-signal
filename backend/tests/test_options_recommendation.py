from __future__ import annotations

import dataclasses
import datetime as dt
from decimal import Decimal

import pytest

from backend.app.options.contracts import ContractIdentity
from backend.app.options.mtm import LegValuation, StructureCosts
from backend.app.options.recommendation import (
    PROBABILITY_PROPOSITIONS, CandidateStrategy, RecommendationResult, StrategyOutcome, available_capital_bases,
    probability_proposition,
)
from backend.app.options.structures import (
    OptionLeg, OptionStructure, leg_expiry_pnl, structure_break_even, structure_expiry_pnl, structure_max_loss,
)

D = dt.date
UTC = dt.timezone.utc
FIXTURE = "DEVELOPMENT_FIXTURE"
EXPIRY = D(2024, 7, 25)


def opt(strike_rupees: int, cp: str, underlying: str = "NIFTY", expiry: dt.date = EXPIRY) -> ContractIdentity:
    return ContractIdentity(underlying, "IDO", expiry, strike_rupees * 100, cp)


def mkleg(strike: int, cp: str, side: str, entry_price, *, lots: int = 1, lot_size: int = 50) -> OptionLeg:
    return OptionLeg(
        contract=opt(strike, cp), side=side, lots=lots, lot_size=lot_size, lot_size_source="CALLER_SUPPLIED",
        entry_price=Decimal(entry_price), entry_price_source="CALLER_SUPPLIED", provenance_class=FIXTURE,
    )


def val(contract, price, at) -> LegValuation:
    return LegValuation(contract=contract, price=Decimal(price), source="OBSERVED_MARKET_PRICE", valuation_at=at)


ENTRY_AT = dt.datetime(2024, 7, 4, 10, 0, tzinfo=UTC)
EXIT_AT = dt.datetime(2024, 7, 25, 10, 0, tzinfo=UTC)


def base_outcome(**overrides) -> StrategyOutcome:
    leg = mkleg(100, "CE", "BUY", "5")
    s = OptionStructure((leg,))
    fields = dict(
        structure=s, entry_at=ENTRY_AT, exit_mode="EXPIRY_EXIT", exit_at=EXIT_AT,
        exit_prices=(val(leg.contract, "8", EXIT_AT),), gross_pnl=Decimal(150), costs=None, net_pnl=Decimal(150),
    )
    fields.update(overrides)
    return StrategyOutcome(**fields)


class TestExitModeValidation:
    def test_all_documented_exit_modes_accepted(self):
        for mode in ("EXPIRY_EXIT", "TARGET_EXIT", "STOP_EXIT", "TIME_EXIT"):
            base_outcome(exit_mode=mode)

    def test_unknown_exit_mode_rejected(self):
        with pytest.raises(ValueError, match="exit_mode"):
            base_outcome(exit_mode="MADE_UP_EXIT")  # type: ignore[arg-type]

    def test_ambiguous_unavailable_requires_no_exit_at_and_no_gross_pnl(self):
        outcome = base_outcome(exit_mode="AMBIGUOUS_UNAVAILABLE", exit_at=None, gross_pnl=None, net_pnl=None)
        assert outcome.exit_at is None and outcome.gross_pnl is None

    def test_ambiguous_unavailable_with_exit_at_rejected(self):
        with pytest.raises(ValueError, match="AMBIGUOUS_UNAVAILABLE"):
            base_outcome(exit_mode="AMBIGUOUS_UNAVAILABLE", gross_pnl=None, net_pnl=None)

    def test_ambiguous_unavailable_with_invented_gross_pnl_rejected(self):
        with pytest.raises(ValueError, match="invented"):
            base_outcome(exit_mode="AMBIGUOUS_UNAVAILABLE", exit_at=None)

    def test_open_not_yet_exited_requires_no_exit_at(self):
        outcome = base_outcome(exit_mode="OPEN_NOT_YET_EXITED", exit_at=None)
        assert outcome.exit_at is None
        assert outcome.gross_pnl == Decimal(150)

    def test_open_not_yet_exited_with_exit_at_rejected(self):
        with pytest.raises(ValueError, match="OPEN_NOT_YET_EXITED"):
            base_outcome(exit_mode="OPEN_NOT_YET_EXITED")

    def test_resolved_exit_mode_requires_exit_at(self):
        with pytest.raises(ValueError, match="requires an exit_at"):
            base_outcome(exit_mode="TARGET_EXIT", exit_at=None)


class TestChronology:
    def test_exit_before_entry_rejected(self):
        with pytest.raises(ValueError, match="impossible chronology"):
            base_outcome(exit_at=ENTRY_AT - dt.timedelta(days=1))

    def test_exit_equal_to_entry_accepted(self):
        base_outcome(exit_at=ENTRY_AT)

    def test_naive_entry_at_rejected(self):
        with pytest.raises(ValueError, match="entry_at must be timezone-aware"):
            base_outcome(entry_at=dt.datetime(2024, 7, 4, 10, 0))

    def test_naive_exit_at_rejected(self):
        with pytest.raises(ValueError, match="exit_at must be timezone-aware"):
            base_outcome(exit_at=dt.datetime(2024, 7, 25, 10, 0))


class TestStrategyOutcomeInvariants:
    def test_gross_and_net_pnl_must_both_be_set_or_both_none(self):
        with pytest.raises(ValueError, match="net_pnl must be set iff gross_pnl"):
            base_outcome(net_pnl=None)

    def test_exit_price_for_a_foreign_contract_rejected(self):
        foreign = opt(999, "PE")
        with pytest.raises(ValueError, match="not a leg of this structure"):
            base_outcome(exit_prices=(val(foreign, "1", EXIT_AT),))

    def test_positive_mae_rejected(self):
        with pytest.raises(ValueError, match="mae"):
            base_outcome(mae=Decimal(1))

    def test_negative_mfe_rejected(self):
        with pytest.raises(ValueError, match="mfe"):
            base_outcome(mfe=Decimal(-1))

    def test_zero_mae_and_mfe_accepted(self):
        base_outcome(mae=Decimal(0), mfe=Decimal(0))

    def test_unknown_data_quality_rejected(self):
        with pytest.raises(ValueError, match="data_quality"):
            base_outcome(data_quality="SUSPICIOUSLY_GOOD")  # type: ignore[arg-type]

    def test_deterministic_construction(self):
        assert base_outcome() == base_outcome()

    def test_frozen(self):
        outcome = base_outcome()
        with pytest.raises(dataclasses.FrozenInstanceError):
            outcome.gross_pnl = Decimal(0)  # type: ignore[misc]


class TestProbabilityPropositions:
    def test_exactly_four_documented_propositions(self):
        assert {p.kind for p in PROBABILITY_PROPOSITIONS} == {
            "POSITIVE_NET_PNL_BY_EXPIRY", "TARGET_BEFORE_STOP", "RETURN_ABOVE_THRESHOLD", "MAX_LOSS_OR_INVALIDATION",
        }

    def test_expiry_pnl_proposition_is_eod_sufficient(self):
        assert probability_proposition("POSITIVE_NET_PNL_BY_EXPIRY").eod_support == "EOD_SUFFICIENT"

    def test_target_before_stop_requires_intraday(self):
        assert probability_proposition("TARGET_BEFORE_STOP").eod_support == "REQUIRES_INTRADAY"

    def test_return_above_threshold_and_max_loss_partially_supported(self):
        assert probability_proposition("RETURN_ABOVE_THRESHOLD").eod_support == "PARTIALLY_SUPPORTED_EOD"
        assert probability_proposition("MAX_LOSS_OR_INVALIDATION").eod_support == "PARTIALLY_SUPPORTED_EOD"

    def test_unknown_proposition_kind_rejected(self):
        with pytest.raises(ValueError, match="kind"):
            probability_proposition("MADE_UP_PROPOSITION")  # type: ignore[arg-type]

    def test_every_proposition_documents_a_limitation(self):
        assert all(p.limitation for p in PROBABILITY_PROPOSITIONS)


class TestCapitalBases:
    def test_net_debit_bounded_structure_has_all_three_definable_bases(self):
        long_leg = mkleg(100, "CE", "BUY", "6")
        short_leg = mkleg(110, "CE", "SELL", "2")
        s = OptionStructure((long_leg, short_leg))
        bases = available_capital_bases(s, structure_max_loss(s))
        assert set(bases) == {"STRATEGY_NOTIONAL", "NET_ENTRY_DEBIT", "MAX_DEFINED_LOSS"}

    def test_naked_short_call_only_has_notional(self):
        s = OptionStructure((mkleg(100, "CE", "SELL", "5"),))
        bases = available_capital_bases(s, structure_max_loss(s))
        assert bases == ("STRATEGY_NOTIONAL",)

    def test_short_put_credit_bounded_loss_excludes_net_entry_debit(self):
        s = OptionStructure((mkleg(100, "PE", "SELL", "4"),))
        bases = available_capital_bases(s, structure_max_loss(s))
        assert set(bases) == {"STRATEGY_NOTIONAL", "MAX_DEFINED_LOSS"}
        assert "NET_ENTRY_DEBIT" not in bases

    def test_margin_requirement_never_returned(self):
        s = OptionStructure((mkleg(100, "CE", "SELL", "5"),))
        assert "MARGIN_REQUIREMENT" not in available_capital_bases(s, structure_max_loss(s))


class TestCandidateStrategy:
    def base_kwargs(self, **overrides):
        leg = mkleg(100, "CE", "BUY", "5")
        s = OptionStructure((leg,))
        fields = dict(
            timestamp=ENTRY_AT, structure=s, rationale="illustrative test candidate", market_context=None,
            iv_context=(), greeks_context=(), max_profit=structure_max_loss(s), max_loss=structure_max_loss(s),
            break_even=structure_break_even(s),
            target=None, stop=None, time_exit_at=None, expected_holding_period=None,
            required_probability_propositions=("POSITIVE_NET_PNL_BY_EXPIRY",), expected_value_inputs=None,
        )
        fields.update(overrides)
        return fields

    def test_valid_candidate_constructs(self):
        CandidateStrategy(**self.base_kwargs())

    def test_naive_timestamp_rejected(self):
        with pytest.raises(ValueError, match="timestamp must be timezone-aware"):
            CandidateStrategy(**self.base_kwargs(timestamp=dt.datetime(2024, 7, 4, 10, 0)))

    def test_naive_time_exit_at_rejected(self):
        with pytest.raises(ValueError, match="time_exit_at must be timezone-aware"):
            CandidateStrategy(**self.base_kwargs(time_exit_at=dt.datetime(2024, 7, 10)))

    def test_negative_expected_holding_period_rejected(self):
        with pytest.raises(ValueError, match="expected_holding_period"):
            CandidateStrategy(**self.base_kwargs(expected_holding_period=dt.timedelta(days=-1)))

    def test_mismatched_iv_context_length_rejected(self):
        with pytest.raises(ValueError, match="iv_context"):
            CandidateStrategy(**self.base_kwargs(iv_context=(1, 2)))  # type: ignore[arg-type]  # structure has 1 leg

    def test_mismatched_greeks_context_length_rejected(self):
        with pytest.raises(ValueError, match="greeks_context"):
            CandidateStrategy(**self.base_kwargs(greeks_context=(1, 2)))  # type: ignore[arg-type]

    def test_empty_context_lists_accepted(self):
        CandidateStrategy(**self.base_kwargs(iv_context=(), greeks_context=()))

    def test_unknown_required_probability_proposition_rejected(self):
        with pytest.raises(ValueError, match="kind"):
            CandidateStrategy(**self.base_kwargs(required_probability_propositions=("NOT_A_REAL_PROPOSITION",)))  # type: ignore[arg-type]


class TestRecommendationResult:
    def candidate(self):
        leg = mkleg(100, "CE", "BUY", "5")
        s = OptionStructure((leg,))
        return CandidateStrategy(
            timestamp=ENTRY_AT, structure=s, rationale="x", market_context=None, iv_context=(), greeks_context=(),
            max_profit=structure_max_loss(s), max_loss=structure_max_loss(s), break_even=structure_break_even(s),
            target=None, stop=None, time_exit_at=None, expected_holding_period=None,
            required_probability_propositions=(), expected_value_inputs=None,
        )

    def test_trade_requires_a_candidate(self):
        with pytest.raises(ValueError, match="TRADE requires a candidate"):
            RecommendationResult(status="TRADE", timestamp=ENTRY_AT, underlying="NIFTY", candidate=None)

    def test_trade_with_candidate_accepted(self):
        RecommendationResult(status="TRADE", timestamp=ENTRY_AT, underlying="NIFTY", candidate=self.candidate())

    def test_wait_must_not_carry_a_candidate(self):
        with pytest.raises(ValueError, match="must not carry a concrete candidate"):
            RecommendationResult(status="WAIT", timestamp=ENTRY_AT, underlying="NIFTY", candidate=self.candidate())

    def test_no_trade_must_not_carry_a_candidate(self):
        with pytest.raises(ValueError, match="must not carry a concrete candidate"):
            RecommendationResult(status="NO_TRADE", timestamp=ENTRY_AT, underlying="NIFTY", candidate=self.candidate())

    def test_wait_and_no_trade_without_candidate_accepted(self):
        RecommendationResult(status="WAIT", timestamp=ENTRY_AT, underlying="NIFTY", candidate=None)
        RecommendationResult(status="NO_TRADE", timestamp=ENTRY_AT, underlying="NIFTY", candidate=None)

    def test_unknown_status_rejected(self):
        with pytest.raises(ValueError, match="status"):
            RecommendationResult(status="MAYBE", timestamp=ENTRY_AT, underlying="NIFTY", candidate=None)  # type: ignore[arg-type]

    def test_naive_timestamp_rejected(self):
        with pytest.raises(ValueError, match="timestamp must be timezone-aware"):
            RecommendationResult(status="WAIT", timestamp=dt.datetime(2024, 7, 4), underlying="NIFTY", candidate=None)

    def test_unknown_probability_metric_key_rejected(self):
        with pytest.raises(ValueError, match="kind"):
            RecommendationResult(
                status="WAIT", timestamp=ENTRY_AT, underlying="NIFTY", candidate=None,
                probability_metrics={"NOT_A_REAL_PROPOSITION": Decimal("0.5")},  # type: ignore[dict-item]
            )

    def test_known_probability_metric_key_accepted(self):
        RecommendationResult(
            status="WAIT", timestamp=ENTRY_AT, underlying="NIFTY", candidate=None,
            probability_metrics={"POSITIVE_NET_PNL_BY_EXPIRY": Decimal("0.5")},
        )


class TestOutcomeLevelSeparation:

    def test_correct_direction_can_still_be_an_unprofitable_option_trade(self):
        leg = mkleg(200, "CE", "BUY", "15")
        underlying_went_up = True
        settlement = Decimal(205)
        option_pnl = leg_expiry_pnl(leg, settlement)
        assert underlying_went_up is True
        assert option_pnl == Decimal(-500)

    def test_profitable_leg_can_still_be_part_of_an_unprofitable_structure(self):
        long_leg = mkleg(100, "CE", "BUY", "5")
        long_only_leg_costly = mkleg(100, "PE", "BUY", "50")
        s = OptionStructure((long_leg, long_only_leg_costly))
        settlement = Decimal(130)
        call_leg_pnl = leg_expiry_pnl(long_leg, settlement)
        structure_pnl = structure_expiry_pnl(s, settlement)
        assert call_leg_pnl > 0
        assert structure_pnl < 0
        assert call_leg_pnl != structure_pnl
