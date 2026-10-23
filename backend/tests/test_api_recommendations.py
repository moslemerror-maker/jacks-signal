from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from backend.app.api.app import create_app
from backend.app.api.dependencies import get_db
from backend.app.options.candidate_generator import CandidateGenerationConfig
from backend.app.options.contracts import ContractIdentity
from backend.app.options.decision_policy import DecisionPolicyThresholds
from backend.app.options.live_quote import LiveQuote
from backend.app.options.live_session import ChainSnapshot, OrchestratorConfig, SyntheticLiveChainSource, run_one_cycle

D = dt.date
UTC = dt.timezone.utc
FIXTURE = "DEVELOPMENT_FIXTURE"
EXPIRY = D(2026, 10, 6)
NOW = dt.datetime(2026, 9, 23, 5, 0, tzinfo=UTC)
LATER = dt.datetime(2026, 9, 23, 5, 5, tzinfo=UTC)


def opt(strike_rupees: int, cp: str) -> ContractIdentity:
    return ContractIdentity("NIFTY", "IDO", EXPIRY, strike_rupees * 100, cp)


def q(strike, cp, last, bid, ask, *, observed_at=NOW, volume=500, oi=1000) -> LiveQuote:
    return LiveQuote.from_raw(
        contract=opt(strike, cp), observed_at=observed_at, last_price=Decimal(last), bid=Decimal(bid), ask=Decimal(ask),
        volume=volume, open_interest=oi, underlying_last_price=Decimal("24500"), provider_reported_iv=Decimal("0.15"),
        provider_reported_greeks={"delta": Decimal("0.5")}, source="dhan", source_row_hash="api-test", provenance_class=FIXTURE,
    )


def chain(observed_at=NOW) -> tuple[LiveQuote, ...]:
    return (
        q(24500, "CE", "140", "138", "142", observed_at=observed_at),
        q(24500, "PE", "135", "133", "137", observed_at=observed_at),
    )


def snapshot_source(observed_at=NOW) -> SyntheticLiveChainSource:
    return SyntheticLiveChainSource(factory=lambda as_of: ChainSnapshot(
        source="dhan", underlying="NIFTY", observed_at=observed_at, expiry_list_dates=(EXPIRY,),
        quotes=chain(observed_at), raw_response_sha256="deadbeef" * 4,
    ))


def mkconfig(source) -> OrchestratorConfig:
    return OrchestratorConfig(
        mode="SYNTHETIC", underlying="NIFTY", source=source,
        candidate_config=CandidateGenerationConfig(lots=1, lot_size=75, lot_size_source="CALLER_SUPPLIED",
                                                   risk_free_rate=Decimal("0.065"), max_quote_age=dt.timedelta(minutes=15),
                                                   max_spread_fraction=Decimal("0.10"), min_volume=None),
        decision_thresholds=DecisionPolicyThresholds(min_probability_positive_net_pnl=Decimal("0.5"), require_verified_forward=False),
        model_version="api-test-model-v0", strategy_version="api-test-strategy-v0",
    )


def make_client(db_session) -> TestClient:
    app = create_app()

    def _override():
        yield db_session

    app.dependency_overrides[get_db] = _override
    return TestClient(app)


def all_columns(db_session, rec_id) -> dict:
    return dict(db_session.execute(text("SELECT * FROM option_recommendations WHERE id=:i"), {"i": str(rec_id)}).mappings().one())


class TestLatestRecommendation:
    def test_returns_404_when_no_recommendation_exists(self, db_session):
        client = make_client(db_session)
        r = client.get("/api/recommendations/latest")
        assert r.status_code == 404

    def test_returns_the_latest_persisted_recommendation(self, db_session):
        result = run_one_cycle(db=db_session, config=mkconfig(snapshot_source()), now=NOW)
        client = make_client(db_session)
        r = client.get("/api/recommendations/latest")
        assert r.status_code == 200
        body = r.json()
        assert body["metadata"]["recommendation_id"] == str(result.recommendation_id)
        assert body["metadata"]["underlying"] == "NIFTY"
        assert body["metadata"]["source"] == "dhan"

    def test_defaults_to_nifty(self, db_session):
        run_one_cycle(db=db_session, config=mkconfig(snapshot_source()), now=NOW)
        client = make_client(db_session)
        default_r = client.get("/api/recommendations/latest")
        explicit_r = client.get("/api/recommendations/latest", params={"underlying": "NIFTY"})
        assert default_r.json() == explicit_r.json()

    def test_unsupported_underlying_is_rejected_not_silently_empty(self, db_session):
        client = make_client(db_session)
        r = client.get("/api/recommendations/latest", params={"underlying": "BANKNIFTY"})
        assert r.status_code == 400

    def test_ordering_returns_the_most_recent_decision_timestamp(self, db_session):
        older = run_one_cycle(db=db_session, config=mkconfig(snapshot_source(NOW)), now=NOW)
        newer = run_one_cycle(db=db_session, config=mkconfig(snapshot_source(LATER)), now=LATER)
        assert older.recommendation_id != newer.recommendation_id
        client = make_client(db_session)
        r = client.get("/api/recommendations/latest")
        assert r.json()["metadata"]["recommendation_id"] == str(newer.recommendation_id)

    def test_no_recommendation_mutation_occurs(self, db_session):
        result = run_one_cycle(db=db_session, config=mkconfig(snapshot_source()), now=NOW)
        before = all_columns(db_session, result.recommendation_id)
        client = make_client(db_session)
        client.get("/api/recommendations/latest")
        client.get("/api/recommendations/latest")
        after = all_columns(db_session, result.recommendation_id)
        assert before == after

    def test_only_one_row_exists_after_two_api_calls(self, db_session):
        run_one_cycle(db=db_session, config=mkconfig(snapshot_source()), now=NOW)
        client = make_client(db_session)
        client.get("/api/recommendations/latest")
        client.get("/api/recommendations/latest")
        count = db_session.execute(text("SELECT COUNT(*) FROM option_recommendations")).scalar()
        assert count == 1

    def test_typed_response_shape_has_every_required_section(self, db_session):
        run_one_cycle(db=db_session, config=mkconfig(snapshot_source()), now=NOW)
        client = make_client(db_session)
        body = client.get("/api/recommendations/latest").json()
        assert set(body) == {"metadata", "market_snapshot", "selected_recommendation", "candidates", "probability", "cost_risk_status", "data_quality", "provenance"}

    def test_all_three_candidate_shapes_are_preserved_not_just_selected(self, db_session):
        run_one_cycle(db=db_session, config=mkconfig(snapshot_source()), now=NOW)
        client = make_client(db_session)
        body = client.get("/api/recommendations/latest").json()
        assert {c["shape"] for c in body["candidates"]} == {"LONG_CALL_ATM", "LONG_PUT_ATM", "LONG_STRADDLE_ATM"}

    def test_decision_snapshot_leg_fields_are_present_and_correct(self, db_session):
        run_one_cycle(db=db_session, config=mkconfig(snapshot_source()), now=NOW)
        client = make_client(db_session)
        body = client.get("/api/recommendations/latest").json()
        call = next(c for c in body["candidates"] if c["shape"] == "LONG_CALL_ATM")
        assert call["formed"] is True
        leg = call["legs"][0]
        assert leg["contract"]["strike_paise"] == 2450000
        assert leg["entry_price"] == "140"
        assert leg["last_price"] == "140" and leg["bid"] == "138" and leg["ask"] == "142"
        assert leg["volume"] == 500 and leg["open_interest"] == 1000
        assert leg["provider_reported_iv"] == "0.15"
        assert leg["provider_reported_greeks"] == {"delta": "0.5"}
        assert leg["liquidity_assessment"]["is_acceptable"] is True

    def test_risk_neutral_probability_is_distinct_from_predictive_probability(self, db_session):
        run_one_cycle(db=db_session, config=mkconfig(snapshot_source()), now=NOW)
        client = make_client(db_session)
        body = client.get("/api/recommendations/latest").json()
        prob = body["probability"]
        assert "risk_neutral_probability_metrics" in prob
        assert "win_probability" not in str(body)
        assert prob["predictive_probability_status"] == "NOT_ESTABLISHED"

    def test_not_modeled_fields_remain_not_modeled(self, db_session):
        run_one_cycle(db=db_session, config=mkconfig(snapshot_source()), now=NOW)
        client = make_client(db_session)
        body = client.get("/api/recommendations/latest").json()
        crs = body["cost_risk_status"]
        assert crs["costs_slippage_assumptions"]["status"] == "NOT_MODELED"
        assert crs["risk"]["status"] == "NOT_MODELED"
        assert crs["expected_value"]["status"] == "NOT_MODELED"

    def test_missing_optional_quote_fields_stay_null_never_zero(self, db_session):
        no_volume_chain = (
            LiveQuote.from_raw(contract=opt(24500, "CE"), observed_at=NOW, last_price=Decimal("140"), bid=Decimal("138"),
                               ask=Decimal("142"), volume=None, open_interest=None, underlying_last_price=Decimal("24500"),
                               provider_reported_iv=None, provider_reported_greeks=None, source="dhan",
                               source_row_hash=None, provenance_class=FIXTURE),
            LiveQuote.from_raw(contract=opt(24500, "PE"), observed_at=NOW, last_price=Decimal("135"), bid=Decimal("133"),
                               ask=Decimal("137"), volume=None, open_interest=None, underlying_last_price=Decimal("24500"),
                               provider_reported_iv=None, provider_reported_greeks=None, source="dhan",
                               source_row_hash=None, provenance_class=FIXTURE),
        )
        source = SyntheticLiveChainSource(factory=lambda as_of: ChainSnapshot(
            source="dhan", underlying="NIFTY", observed_at=NOW, expiry_list_dates=(EXPIRY,), quotes=no_volume_chain, raw_response_sha256=None,
        ))
        run_one_cycle(db=db_session, config=mkconfig(source), now=NOW)
        client = make_client(db_session)
        body = client.get("/api/recommendations/latest").json()
        call = next(c for c in body["candidates"] if c["shape"] == "LONG_CALL_ATM")
        leg = call["legs"][0]
        assert leg["volume"] is None and leg["open_interest"] is None
        assert leg["provider_reported_iv"] is None and leg["provider_reported_greeks"] is None

    def test_raw_response_sha256_is_labeled_project_level_not_wire_level(self, db_session):
        run_one_cycle(db=db_session, config=mkconfig(snapshot_source()), now=NOW)
        client = make_client(db_session)
        body = client.get("/api/recommendations/latest").json()
        prov = body["provenance"]
        assert prov["raw_response_sha256"] == "deadbeef" * 4
        assert "wire" not in prov["description"].lower() or "not" in prov["description"].lower()
        assert "project-level" in prov["description"].lower() or "not a cryptographic digest of the provider" in prov["description"].lower()
