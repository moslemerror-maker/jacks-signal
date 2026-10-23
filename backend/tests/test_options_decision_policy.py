from __future__ import annotations

import datetime as dt
from decimal import Decimal

from backend.app.options import decision_policy as dp
from backend.app.options.candidate_generator import CandidateGenerationOutcome
from backend.app.options.contracts import ContractIdentity
from backend.app.options.forward import resolve_forward_spot_proxy
from backend.app.options.probability import Black76RiskNeutralDistribution
from backend.app.options.recommendation import CandidateStrategy
from backend.app.options.structures import OptionLeg, OptionStructure, structure_break_even, structure_max_loss

D = dt.date
UTC = dt.timezone.utc
FIXTURE = "DEVELOPMENT_FIXTURE"
EXPIRY = D(2026, 9, 29)
NOW = dt.datetime(2026, 9, 22, 10, 0, tzinfo=UTC)
FORWARD_VERIFIED = resolve_forward_spot_proxy(Decimal("24500"))

THRESHOLDS = dp.DecisionPolicyThresholds(min_probability_positive_net_pnl=Decimal("0.5"), require_verified_forward=False)


def mkcandidate(forward_resolution=FORWARD_VERIFIED) -> CandidateStrategy:
    leg = OptionLeg(contract=ContractIdentity("NIFTY", "IDO", EXPIRY, 2450000, "CE"), side="BUY", lots=1, lot_size=75,
                    lot_size_source="CALLER_SUPPLIED", entry_price=Decimal(140), entry_price_source="LIVE_QUOTE",
                    provenance_class=FIXTURE, entry_observed_at=NOW)
    s = OptionStructure((leg,))
    return CandidateStrategy(
        timestamp=NOW, structure=s, rationale="test", market_context=None, iv_context=(), greeks_context=(),
        max_profit=structure_max_loss(s), max_loss=structure_max_loss(s), break_even=structure_break_even(s),
        target=None, stop=None, time_exit_at=None, expected_holding_period=None,
        required_probability_propositions=("POSITIVE_NET_PNL_BY_EXPIRY",), expected_value_inputs=None,
        forward_resolution=forward_resolution,
    )


def favorable_distribution(_candidate) -> Black76RiskNeutralDistribution:
    return Black76RiskNeutralDistribution(forward=Decimal(50000), volatility=Decimal("0.15"), time_to_expiry_years=Decimal("0.05"))


def unfavorable_distribution(_candidate) -> Black76RiskNeutralDistribution:
    return Black76RiskNeutralDistribution(forward=Decimal(100), volatility=Decimal("0.15"), time_to_expiry_years=Decimal("0.05"))


class TestNoTradeWhenCandidateMissing:
    def test_rejected_outcome_becomes_no_trade(self):
        outcome = CandidateGenerationOutcome(shape="LONG_CALL_ATM", candidate=None, rejected_reason="no usable quote")
        (result,) = dp.decide_for_underlying([outcome], distribution_factory=lambda c: None, timestamp=NOW, underlying="NIFTY",
                                             thresholds=THRESHOLDS, model_version="v0", strategy_version="v0")
        assert result.status == "NO_TRADE" and result.candidate is None
        assert "no usable quote" in result.reasons[0]


class TestWaitWhenDistributionUnavailable:
    def test_insufficient_iv_data_becomes_wait(self):
        outcome = CandidateGenerationOutcome(shape="LONG_CALL_ATM", candidate=mkcandidate(), rejected_reason=None)
        (result,) = dp.decide_for_underlying([outcome], distribution_factory=lambda c: None, timestamp=NOW, underlying="NIFTY",
                                             thresholds=THRESHOLDS, model_version="v0", strategy_version="v0")
        assert result.status == "WAIT" and result.candidate is None
        assert any("insufficient IV/Greeks" in r for r in result.reasons)


class TestNeverFabricatesTradeWhileNotEstablished:
    def test_high_probability_never_produces_trade_while_not_established(self):
        assert dp.PREDICTIVE_PROBABILITY_STATUS == "NOT_ESTABLISHED"
        outcome = CandidateGenerationOutcome(shape="LONG_CALL_ATM", candidate=mkcandidate(), rejected_reason=None)
        (result,) = dp.decide_for_underlying([outcome], distribution_factory=favorable_distribution, timestamp=NOW, underlying="NIFTY",
                                             thresholds=THRESHOLDS, model_version="v0", strategy_version="v0")
        assert result.status == "WAIT"
        assert result.candidate is None
        assert any("NOT_ESTABLISHED" in r for r in result.reasons)
        assert any("RISK-NEUTRAL MODEL BASELINE" in r for r in result.reasons)

    def test_low_probability_also_wait(self):
        outcome = CandidateGenerationOutcome(shape="LONG_CALL_ATM", candidate=mkcandidate(), rejected_reason=None)
        (result,) = dp.decide_for_underlying([outcome], distribution_factory=unfavorable_distribution, timestamp=NOW, underlying="NIFTY",
                                             thresholds=THRESHOLDS, model_version="v0", strategy_version="v0")
        assert result.status == "WAIT"


class TestTradeReachableOnlyWhenEstablished:

    def test_trade_reachable_when_established_and_threshold_met(self, monkeypatch):
        monkeypatch.setattr(dp, "PREDICTIVE_PROBABILITY_STATUS", "ESTABLISHED")
        outcome = CandidateGenerationOutcome(shape="LONG_CALL_ATM", candidate=mkcandidate(), rejected_reason=None)
        (result,) = dp.decide_for_underlying([outcome], distribution_factory=favorable_distribution, timestamp=NOW, underlying="NIFTY",
                                             thresholds=THRESHOLDS, model_version="v0", strategy_version="v0")
        assert result.status == "TRADE" and result.candidate is not None

    def test_still_wait_when_established_but_threshold_not_met(self, monkeypatch):
        monkeypatch.setattr(dp, "PREDICTIVE_PROBABILITY_STATUS", "ESTABLISHED")
        outcome = CandidateGenerationOutcome(shape="LONG_CALL_ATM", candidate=mkcandidate(), rejected_reason=None)
        (result,) = dp.decide_for_underlying([outcome], distribution_factory=unfavorable_distribution, timestamp=NOW, underlying="NIFTY",
                                             thresholds=THRESHOLDS, model_version="v0", strategy_version="v0")
        assert result.status == "WAIT"

    def test_unverified_forward_blocks_trade_when_required(self, monkeypatch):
        monkeypatch.setattr(dp, "PREDICTIVE_PROBABILITY_STATUS", "ESTABLISHED")
        strict = dp.DecisionPolicyThresholds(min_probability_positive_net_pnl=Decimal("0.5"), require_verified_forward=True)
        outcome = CandidateGenerationOutcome(shape="LONG_CALL_ATM", candidate=mkcandidate(forward_resolution=resolve_forward_spot_proxy(Decimal(100))), rejected_reason=None)
        (result,) = dp.decide_for_underlying([outcome], distribution_factory=favorable_distribution, timestamp=NOW, underlying="NIFTY",
                                             thresholds=strict, model_version="v0", strategy_version="v0")
        assert result.status == "WAIT"
        assert any("SPOT_PROXY_UNVERIFIED" in r for r in result.reasons)

    def test_unverified_forward_allowed_when_not_required(self, monkeypatch):
        monkeypatch.setattr(dp, "PREDICTIVE_PROBABILITY_STATUS", "ESTABLISHED")
        outcome = CandidateGenerationOutcome(shape="LONG_CALL_ATM", candidate=mkcandidate(forward_resolution=resolve_forward_spot_proxy(Decimal(100))), rejected_reason=None)
        (result,) = dp.decide_for_underlying([outcome], distribution_factory=favorable_distribution, timestamp=NOW, underlying="NIFTY",
                                             thresholds=THRESHOLDS, model_version="v0", strategy_version="v0")
        assert result.status == "TRADE"


class TestResultCountMatchesOutcomeCount:
    def test_multiple_outcomes_produce_matching_results(self):
        outcomes = [
            CandidateGenerationOutcome(shape="LONG_CALL_ATM", candidate=mkcandidate(), rejected_reason=None),
            CandidateGenerationOutcome(shape="LONG_PUT_ATM", candidate=None, rejected_reason="no PE quote"),
        ]
        results = dp.decide_for_underlying(outcomes, distribution_factory=lambda c: None, timestamp=NOW, underlying="NIFTY",
                                           thresholds=THRESHOLDS, model_version="v0", strategy_version="v0")
        assert len(results) == 2
        assert results[0].status == "WAIT" and results[1].status == "NO_TRADE"
