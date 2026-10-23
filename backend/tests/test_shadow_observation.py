from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest
from sqlalchemy import text

from backend.app.options.candidate_generator import CandidateGenerationConfig
from backend.app.options.contracts import ContractIdentity
from backend.app.options.decision_policy import DecisionPolicyThresholds
from backend.app.options.live_quote import LiveQuote
from backend.app.options.live_session import ChainSnapshot, OrchestratorConfig, SyntheticLiveChainSource, run_one_cycle
from backend.app.options.shadow_observation import (
    NotAForwardObservationError,
    UnknownRecommendationError,
    attach_forward_observation,
    create_shadow_observation,
    list_forward_observations,
    shadow_observation_status,
)

D = dt.date
UTC = dt.timezone.utc
FIXTURE = "DEVELOPMENT_FIXTURE"
EXPIRY = D(2026, 10, 6)
T0 = dt.datetime(2026, 9, 24, 3, 45, tzinfo=UTC)


def opt(strike: int, cp: str) -> ContractIdentity:
    return ContractIdentity("NIFTY", "IDO", EXPIRY, strike * 100, cp)


def q(strike, cp, last, bid, ask, *, observed_at) -> LiveQuote:
    return LiveQuote.from_raw(
        contract=opt(strike, cp), observed_at=observed_at, last_price=Decimal(last), bid=Decimal(bid), ask=Decimal(ask),
        volume=500, open_interest=1000, underlying_last_price=Decimal("24500"), provider_reported_iv=None,
        provider_reported_greeks=None, source="dhan", source_row_hash=None, provenance_class=FIXTURE,
    )


def good_snapshot(as_of: dt.datetime) -> ChainSnapshot:
    quotes = (q(24500, "CE", "140", "138", "142", observed_at=as_of), q(24500, "PE", "135", "133", "137", observed_at=as_of))
    return ChainSnapshot(source="dhan", underlying="NIFTY", observed_at=as_of, expiry_list_dates=(EXPIRY,), quotes=quotes, raw_response_sha256=None)


def empty_snapshot(as_of: dt.datetime) -> ChainSnapshot:
    return ChainSnapshot(source="dhan", underlying="NIFTY", observed_at=as_of, expiry_list_dates=(EXPIRY,), quotes=(), raw_response_sha256=None)


def mkconfig(source) -> OrchestratorConfig:
    return OrchestratorConfig(
        mode="LIVE_PAPER", underlying="NIFTY", source=source,
        candidate_config=CandidateGenerationConfig(lots=1, lot_size=75, lot_size_source="CALLER_SUPPLIED",
                                                   risk_free_rate=Decimal("0.065"), max_quote_age=dt.timedelta(minutes=15),
                                                   max_spread_fraction=Decimal("0.10"), min_volume=None),
        decision_thresholds=DecisionPolicyThresholds(min_probability_positive_net_pnl=Decimal("0.55"), require_verified_forward=False),
        model_version="phase1-test-v0", strategy_version="phase1-test-v0",
    )


def persist_cycle(db_session, as_of: dt.datetime, source=None):
    config = mkconfig(source or SyntheticLiveChainSource(factory=good_snapshot))
    result = run_one_cycle(db=db_session, config=config, now=as_of)
    assert result.outcome == "PERSISTED"
    return result.recommendation_id


class TestCreateShadowObservation:
    def test_creates_a_row_linked_to_an_existing_recommendation(self, db_session):
        rec_id = persist_cycle(db_session, T0)
        obs_id = create_shadow_observation(
            db_session, recommendation_id=rec_id, research_phase_id="PHASE1-TEST",
            data_version="v1", feature_version="v1", cost_model_version="UNVERIFIED",
        )
        row = db_session.execute(text("SELECT recommendation_id, research_phase_id, hypothesis_id FROM option_shadow_observations WHERE id=:i"), {"i": str(obs_id)}).one()
        assert str(row.recommendation_id) == str(rec_id)
        assert row.research_phase_id == "PHASE1-TEST"
        assert row.hypothesis_id is None

    def test_raises_on_unknown_recommendation_id(self, db_session):
        import uuid
        with pytest.raises(UnknownRecommendationError):
            create_shadow_observation(db_session, recommendation_id=uuid.uuid4(), research_phase_id="X",
                                      data_version="v1", feature_version="v1", cost_model_version="UNVERIFIED")

    def test_does_not_create_a_paper_trade_event(self, db_session):
        rec_id = persist_cycle(db_session, T0)
        create_shadow_observation(db_session, recommendation_id=rec_id, research_phase_id="PHASE1-TEST",
                                  data_version="v1", feature_version="v1", cost_model_version="UNVERIFIED")
        n = db_session.execute(text("SELECT count(*) FROM option_paper_trade_events")).scalar()
        assert n == 0

    def test_is_immutable(self, db_session):
        rec_id = persist_cycle(db_session, T0)
        obs_id = create_shadow_observation(db_session, recommendation_id=rec_id, research_phase_id="PHASE1-TEST",
                                           data_version="v1", feature_version="v1", cost_model_version="UNVERIFIED")
        with pytest.raises(Exception, match="immutable"):
            db_session.execute(text("UPDATE option_shadow_observations SET research_phase_id='CHANGED' WHERE id=:i"), {"i": str(obs_id)})
        db_session.rollback()

    def test_cannot_be_deleted(self, db_session):
        rec_id = persist_cycle(db_session, T0)
        obs_id = create_shadow_observation(db_session, recommendation_id=rec_id, research_phase_id="PHASE1-TEST",
                                           data_version="v1", feature_version="v1", cost_model_version="UNVERIFIED")
        with pytest.raises(Exception, match="cannot be deleted"):
            db_session.execute(text("DELETE FROM option_shadow_observations WHERE id=:i"), {"i": str(obs_id)})
        db_session.rollback()


class TestShadowObservationStatus:
    def test_awaiting_outcome_before_any_forward_observation(self, db_session):
        rec_id = persist_cycle(db_session, T0)
        obs_id = create_shadow_observation(db_session, recommendation_id=rec_id, research_phase_id="PHASE1-TEST",
                                           data_version="v1", feature_version="v1", cost_model_version="UNVERIFIED")
        assert shadow_observation_status(db_session, obs_id) == "AWAITING_OUTCOME"

    def test_outcome_recorded_after_one_forward_observation(self, db_session):
        rec_id = persist_cycle(db_session, T0)
        obs_id = create_shadow_observation(db_session, recommendation_id=rec_id, research_phase_id="PHASE1-TEST",
                                           data_version="v1", feature_version="v1", cost_model_version="UNVERIFIED")
        later_rec_id = persist_cycle(db_session, T0 + dt.timedelta(minutes=5))
        attach_forward_observation(db_session, shadow_observation_id=obs_id, forward_recommendation_id=later_rec_id)
        assert shadow_observation_status(db_session, obs_id) == "OUTCOME_RECORDED"


class TestAttachForwardObservation:
    def test_records_elapsed_seconds_and_quote_available(self, db_session):
        rec_id = persist_cycle(db_session, T0)
        obs_id = create_shadow_observation(db_session, recommendation_id=rec_id, research_phase_id="PHASE1-TEST",
                                           data_version="v1", feature_version="v1", cost_model_version="UNVERIFIED")
        later_rec_id = persist_cycle(db_session, T0 + dt.timedelta(minutes=15))
        fwd_id = attach_forward_observation(db_session, shadow_observation_id=obs_id, forward_recommendation_id=later_rec_id)
        row = db_session.execute(text("SELECT elapsed_seconds, quote_status FROM option_shadow_forward_observations WHERE id=:i"), {"i": str(fwd_id)}).one()
        assert row.elapsed_seconds == 900.0
        assert row.quote_status == "QUOTE_AVAILABLE"

    def test_missing_quote_is_recorded_not_silently_dropped(self, db_session):
        rec_id = persist_cycle(db_session, T0)
        obs_id = create_shadow_observation(db_session, recommendation_id=rec_id, research_phase_id="PHASE1-TEST",
                                           data_version="v1", feature_version="v1", cost_model_version="UNVERIFIED")
        later_rec_id = persist_cycle(db_session, T0 + dt.timedelta(minutes=10), source=SyntheticLiveChainSource(factory=empty_snapshot))
        fwd_id = attach_forward_observation(db_session, shadow_observation_id=obs_id, forward_recommendation_id=later_rec_id)
        row = db_session.execute(text("SELECT quote_status FROM option_shadow_forward_observations WHERE id=:i"), {"i": str(fwd_id)}).one()
        assert row.quote_status == "QUOTE_MISSING"

    def test_rejects_a_backward_or_same_instant_attachment_no_future_leakage_at_entry(self, db_session):
        rec_id = persist_cycle(db_session, T0 + dt.timedelta(minutes=10))
        obs_id = create_shadow_observation(db_session, recommendation_id=rec_id, research_phase_id="PHASE1-TEST",
                                           data_version="v1", feature_version="v1", cost_model_version="UNVERIFIED")
        earlier_rec_id = persist_cycle(db_session, T0)
        with pytest.raises(NotAForwardObservationError):
            attach_forward_observation(db_session, shadow_observation_id=obs_id, forward_recommendation_id=earlier_rec_id)

    def test_rejects_the_same_recommendation_as_its_own_forward_observation(self, db_session):
        rec_id = persist_cycle(db_session, T0)
        obs_id = create_shadow_observation(db_session, recommendation_id=rec_id, research_phase_id="PHASE1-TEST",
                                           data_version="v1", feature_version="v1", cost_model_version="UNVERIFIED")
        with pytest.raises(NotAForwardObservationError):
            attach_forward_observation(db_session, shadow_observation_id=obs_id, forward_recommendation_id=rec_id)

    def test_multiple_forward_observations_can_be_attached_over_time(self, db_session):
        rec_id = persist_cycle(db_session, T0)
        obs_id = create_shadow_observation(db_session, recommendation_id=rec_id, research_phase_id="PHASE1-TEST",
                                           data_version="v1", feature_version="v1", cost_model_version="UNVERIFIED")
        r1 = persist_cycle(db_session, T0 + dt.timedelta(minutes=5))
        r2 = persist_cycle(db_session, T0 + dt.timedelta(minutes=10))
        attach_forward_observation(db_session, shadow_observation_id=obs_id, forward_recommendation_id=r1)
        attach_forward_observation(db_session, shadow_observation_id=obs_id, forward_recommendation_id=r2)
        records = list_forward_observations(db_session, obs_id)
        assert len(records) == 2
        assert records[0].elapsed_seconds < records[1].elapsed_seconds

    def test_is_immutable_and_cannot_be_deleted(self, db_session):
        rec_id = persist_cycle(db_session, T0)
        obs_id = create_shadow_observation(db_session, recommendation_id=rec_id, research_phase_id="PHASE1-TEST",
                                           data_version="v1", feature_version="v1", cost_model_version="UNVERIFIED")
        later_rec_id = persist_cycle(db_session, T0 + dt.timedelta(minutes=5))
        fwd_id = attach_forward_observation(db_session, shadow_observation_id=obs_id, forward_recommendation_id=later_rec_id)
        with pytest.raises(Exception, match="immutable"):
            db_session.execute(text("UPDATE option_shadow_forward_observations SET quote_status='QUOTE_MISSING' WHERE id=:i"), {"i": str(fwd_id)})
        db_session.rollback()
