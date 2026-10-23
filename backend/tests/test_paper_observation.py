from __future__ import annotations

import datetime as dt
import json
import uuid
from decimal import Decimal

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from backend.app.options import decision_policy as dp
from backend.app.options import paper_observation as po
from backend.app.options.candidate_generator import CandidateGenerationConfig
from backend.app.options.contracts import ContractIdentity
from backend.app.options.decision_policy import DecisionPolicyThresholds
from backend.app.options.live_quote import LiveQuote
from backend.app.options.live_session import ChainSnapshot, OrchestratorConfig, SyntheticLiveChainSource, run_one_cycle
from backend.app.options.mtm import valuation_from_live_quote

D = dt.date
UTC = dt.timezone.utc
FIXTURE = "DEVELOPMENT_FIXTURE"
EXPIRY = D(2026, 10, 6)
NOW = dt.datetime(2026, 9, 25, 5, 0, tzinfo=UTC)


def opt(strike, cp):
    return ContractIdentity("NIFTY", "IDO", EXPIRY, strike * 100, cp)


def mkq(strike, cp, *, last=None, bid=None, ask=None, u=Decimal("1000"), observed_at=NOW):
    return LiveQuote.from_raw(
        contract=opt(strike, cp), observed_at=observed_at,
        last_price=Decimal(last) if last is not None else None, bid=Decimal(bid) if bid is not None else None,
        ask=Decimal(ask) if ask is not None else None, volume=500, open_interest=1000, underlying_last_price=u,
        provider_reported_iv=None, provider_reported_greeks=None, source="dhan", source_row_hash="po-test",
        provenance_class=FIXTURE,
    )


def snapshot(quotes, observed_at=NOW) -> ChainSnapshot:
    return ChainSnapshot(source="dhan", underlying="NIFTY", observed_at=observed_at, expiry_list_dates=(EXPIRY,),
                         quotes=tuple(quotes), raw_response_sha256="po" * 16)


def leg_dict(strike, cp, side, entry_price, *, lots=1, lot_size=75, entry_observed_at=NOW):
    return {
        "contract": {"underlying": "NIFTY", "derivative_class": "IDO", "expiry": EXPIRY.isoformat(), "strike_paise": strike * 100, "option_type": cp},
        "side": side, "lots": lots, "lot_size": lot_size, "lot_size_source": "CALLER_SUPPLIED",
        "entry_price": str(entry_price), "entry_price_source": "LIVE_QUOTE",
        "entry_observed_at": entry_observed_at.isoformat(), "provenance_class": FIXTURE,
        "entry_ltp": str(entry_price), "entry_bid": None, "entry_ask": None, "entry_volume": 500,
        "entry_open_interest": 1000, "entry_quote_status": "QUOTED", "provider_reported_iv": None,
        "provider_reported_greeks": None, "liquidity_assessment": {"is_acceptable": True, "reasons": []},
    }


def insert_traded_position(db_session, *, legs, decision_timestamp=NOW, underlying="NIFTY") -> uuid.UUID:
    rec_id = uuid.uuid4()
    db_session.execute(text(
        "INSERT INTO option_recommendations (id, decision_timestamp, underlying, status, shape, data_quality, source, payload) "
        "VALUES (:id, :dt, :u, 'TRADE', :shape, 'EOD_RESOLVED', 'dhan', CAST('{}' AS JSONB))"
    ), {"id": rec_id, "dt": decision_timestamp, "u": underlying, "shape": "LONG_STRADDLE_ATM" if len(legs) > 1 else "LONG_CALL_ATM"})
    payload = {
        "shape": "LONG_STRADDLE_ATM" if len(legs) > 1 else "LONG_CALL_ATM",
        "recommendation_timestamp": decision_timestamp.isoformat(), "model_version": "po-test-v0",
        "strategy_version": "po-test-v0", "capital_basis": "NOT_MODELED", "forward_resolution": None,
        "legs": legs, "candidate": None, "reasons": [],
    }
    db_session.execute(text(
        "INSERT INTO option_paper_trade_events (id, recommendation_id, event_type, event_timestamp, payload) "
        "VALUES (gen_random_uuid(), :rid, 'PAPER_ENTRY', :ts, CAST(:payload AS JSONB))"
    ), {"rid": str(rec_id), "ts": decision_timestamp, "payload": json.dumps(payload)})
    db_session.commit()
    return rec_id


def get_entry_payload(db_session, recommendation_id) -> dict:
    return db_session.execute(text(
        "SELECT payload FROM option_paper_trade_events WHERE recommendation_id=:r AND event_type='PAPER_ENTRY'"
    ), {"r": str(recommendation_id)}).scalar()


def observations(db_session, recommendation_id) -> list[dict]:
    rows = db_session.execute(text(
        "SELECT payload FROM option_paper_trade_events WHERE recommendation_id=:r AND event_type='OBSERVATION' ORDER BY event_timestamp"
    ), {"r": str(recommendation_id)}).scalars().all()
    return list(rows)


class TestOpenPaperPositions:
    def test_finds_a_real_open_position(self, db_session):
        rec_id = insert_traded_position(db_session, legs=[leg_dict(100, "CE", "BUY", "10")])
        positions = po.open_paper_positions(db_session, underlying="NIFTY")
        assert len(positions) == 1
        assert positions[0].recommendation_id == rec_id
        assert positions[0].entry_payload["legs"]

    def test_none_when_nothing_ever_traded(self, db_session):
        assert po.open_paper_positions(db_session, underlying="NIFTY") == ()


class TestObservationCreation:
    def test_observation_created_for_an_open_position(self, db_session):
        rec_id = insert_traded_position(db_session, legs=[leg_dict(100, "CE", "BUY", "10"), leg_dict(100, "PE", "BUY", "5")])
        later = NOW + dt.timedelta(minutes=5)
        snap = snapshot((mkq(100, "CE", last="12", bid="11", ask="13", observed_at=later), mkq(100, "PE", last="4", bid="3", ask="5", observed_at=later)), later)
        ids = po.run_observation_cycle(db_session, underlying="NIFTY", snapshot=snap, now=later)
        db_session.commit()
        assert len(ids) == 1
        obs = observations(db_session, rec_id)
        assert len(obs) == 1
        assert len(obs[0]["legs"]) == 2

    def test_exact_contract_matching_uses_current_quotes(self, db_session):
        rec_id = insert_traded_position(db_session, legs=[leg_dict(100, "CE", "BUY", "10")])
        later = NOW + dt.timedelta(minutes=5)
        snap = snapshot((mkq(100, "CE", last="950", bid="945", ask="955", observed_at=later),), later)
        po.run_observation_cycle(db_session, underlying="NIFTY", snapshot=snap, now=later)
        db_session.commit()
        obs = observations(db_session, rec_id)[0]
        assert obs["legs"][0]["ltp"] == "950"

    def test_mtm_pnl_computed_from_entry_to_current_last_price(self, db_session):
        rec_id = insert_traded_position(db_session, legs=[leg_dict(100, "CE", "BUY", "10")])
        later = NOW + dt.timedelta(minutes=5)
        snap = snapshot((mkq(100, "CE", last="15", bid="14", ask="16", observed_at=later),), later)
        po.run_observation_cycle(db_session, underlying="NIFTY", snapshot=snap, now=later)
        db_session.commit()
        obs = observations(db_session, rec_id)[0]
        assert obs["legs"][0]["mtm_pnl"] == "375"
        assert obs["structure_mtm_pnl"] == "375"
        assert obs["legs"][0]["valuation_source"] == "OBSERVED_MARKET_PRICE"

    def test_missing_quote_for_a_leg_is_degraded_never_fabricated(self, db_session):
        rec_id = insert_traded_position(db_session, legs=[leg_dict(100, "CE", "BUY", "10"), leg_dict(100, "PE", "BUY", "5")])
        later = NOW + dt.timedelta(minutes=5)
        only_ce = snapshot((mkq(100, "CE", last="950", bid="945", ask="955", observed_at=later),), later)
        po.run_observation_cycle(db_session, underlying="NIFTY", snapshot=only_ce, now=later)
        db_session.commit()
        obs = observations(db_session, rec_id)[0]
        pe_leg = next(l for l in obs["legs"] if l["contract"]["option_type"] == "PE")
        assert pe_leg["mtm_pnl"] is None
        assert pe_leg["data_quality"] == "UNAVAILABLE_INSUFFICIENT_DATA"
        assert pe_leg["unavailable_reason"] is not None
        assert obs["structure_mtm_pnl"] is None
        assert obs["data_quality"] == "UNAVAILABLE_INSUFFICIENT_DATA"


class TestObservationIdempotency:
    def test_repeated_call_with_same_snapshot_does_not_duplicate(self, db_session):
        rec_id = insert_traded_position(db_session, legs=[leg_dict(100, "CE", "BUY", "10")])
        later = NOW + dt.timedelta(minutes=5)
        snap = snapshot((mkq(100, "CE", last="12", bid="11", ask="13", observed_at=later),), later)
        ids1 = po.run_observation_cycle(db_session, underlying="NIFTY", snapshot=snap, now=later)
        db_session.commit()
        ids2 = po.run_observation_cycle(db_session, underlying="NIFTY", snapshot=snap, now=later)
        db_session.commit()
        assert ids1 == ids2
        assert len(observations(db_session, rec_id)) == 1

    def test_different_snapshot_timestamps_create_separate_observations(self, db_session):
        rec_id = insert_traded_position(db_session, legs=[leg_dict(100, "CE", "BUY", "10")])
        t1, t2 = NOW + dt.timedelta(minutes=5), NOW + dt.timedelta(minutes=10)
        po.run_observation_cycle(db_session, underlying="NIFTY", snapshot=snapshot((mkq(100, "CE", last="11", bid="10", ask="12", observed_at=t1),), t1), now=t1)
        db_session.commit()
        po.run_observation_cycle(db_session, underlying="NIFTY", snapshot=snapshot((mkq(100, "CE", last="12", bid="11", ask="13", observed_at=t2),), t2), now=t2)
        db_session.commit()
        assert len(observations(db_session, rec_id)) == 2


class TestPostExitSuppression:
    def test_no_observation_created_after_exit(self, db_session):
        rec_id = insert_traded_position(db_session, legs=[leg_dict(100, "CE", "BUY", "10")])
        entry_payload = get_entry_payload(db_session, rec_id)
        exit_quote = mkq(100, "CE", last="1", bid="1", ask="2", observed_at=NOW + dt.timedelta(minutes=1))
        exit_valuations = {opt(100, "CE"): valuation_from_live_quote(exit_quote, price_kind="BID")}
        po.create_manual_paper_exit(db_session, recommendation_id=rec_id, entry_payload=entry_payload, now=NOW + dt.timedelta(minutes=1), exit_valuations=exit_valuations)
        db_session.commit()

        assert po.open_paper_positions(db_session, underlying="NIFTY") == ()
        later = NOW + dt.timedelta(minutes=5)
        ids = po.run_observation_cycle(db_session, underlying="NIFTY", snapshot=snapshot((mkq(100, "CE", last="2", bid="1", ask="3", observed_at=later),), later), now=later)
        db_session.commit()
        assert ids == ()
        assert observations(db_session, rec_id) == []


class TestManualPaperExit:
    def test_gross_pnl_computed_and_net_is_costs_not_modeled(self, db_session):
        rec_id = insert_traded_position(db_session, legs=[leg_dict(100, "CE", "BUY", "10")])
        entry_payload = get_entry_payload(db_session, rec_id)
        exit_quote = mkq(100, "CE", last="20", bid="18", ask="22", observed_at=NOW + dt.timedelta(minutes=5))
        exit_valuations = {opt(100, "CE"): valuation_from_live_quote(exit_quote, price_kind="BID")}
        event_id = po.create_manual_paper_exit(db_session, recommendation_id=rec_id, entry_payload=entry_payload,
                                               now=NOW + dt.timedelta(minutes=5), exit_valuations=exit_valuations)
        db_session.commit()
        row = db_session.execute(text("SELECT payload FROM option_paper_trade_events WHERE id=:i"), {"i": str(event_id)}).scalar()
        assert row["exit_mode"] == "MANUAL_PAPER_EXIT"
        assert row["gross_pnl"] == "600"
        assert row["net_pnl"] == "COSTS_NOT_MODELED"

    def test_idempotent_at_most_one_exit(self, db_session):
        rec_id = insert_traded_position(db_session, legs=[leg_dict(100, "CE", "BUY", "10")])
        entry_payload = get_entry_payload(db_session, rec_id)
        q = mkq(100, "CE", last="20", bid="18", ask="22", observed_at=NOW + dt.timedelta(minutes=5))
        vals = {opt(100, "CE"): valuation_from_live_quote(q, price_kind="BID")}
        id1 = po.create_manual_paper_exit(db_session, recommendation_id=rec_id, entry_payload=entry_payload, now=NOW + dt.timedelta(minutes=5), exit_valuations=vals)
        db_session.commit()
        id2 = po.create_manual_paper_exit(db_session, recommendation_id=rec_id, entry_payload=entry_payload, now=NOW + dt.timedelta(minutes=10), exit_valuations=vals)
        db_session.commit()
        assert id1 == id2
        n = db_session.execute(text("SELECT count(*) FROM option_paper_trade_events WHERE recommendation_id=:r AND event_type='EXIT'"), {"r": str(rec_id)}).scalar()
        assert n == 1

    def test_missing_leg_valuation_raises_never_silently_partial(self, db_session):
        rec_id = insert_traded_position(db_session, legs=[leg_dict(100, "CE", "BUY", "10"), leg_dict(100, "PE", "BUY", "5")])
        entry_payload = get_entry_payload(db_session, rec_id)
        q = mkq(100, "CE", last="20", bid="18", ask="22", observed_at=NOW)
        only_one_leg_valued = {opt(100, "CE"): valuation_from_live_quote(q, price_kind="BID")}
        with pytest.raises(ValueError, match="every leg must be valued"):
            po.create_manual_paper_exit(db_session, recommendation_id=rec_id, entry_payload=entry_payload, now=NOW, exit_valuations=only_one_leg_valued)


class TestAmbiguousUnavailableExit:
    def test_never_carries_a_gross_pnl(self, db_session):
        rec_id = insert_traded_position(db_session, legs=[leg_dict(100, "CE", "BUY", "10")])
        event_id = po.create_ambiguous_unavailable_exit(db_session, recommendation_id=rec_id, now=NOW + dt.timedelta(minutes=5), reason="contract stopped being quoted")
        db_session.commit()
        row = db_session.execute(text("SELECT payload FROM option_paper_trade_events WHERE id=:i"), {"i": str(event_id)}).scalar()
        assert row["exit_mode"] == "AMBIGUOUS_UNAVAILABLE"
        assert row["gross_pnl"] is None


class TestNoFakeExpirySettlement:
    def test_requires_an_explicit_settlement_source_detail(self, db_session):
        rec_id = insert_traded_position(db_session, legs=[leg_dict(100, "CE", "BUY", "10")])
        entry_payload = get_entry_payload(db_session, rec_id)
        with pytest.raises(ValueError, match="settlement_source_detail"):
            po.create_expiry_settlement_exit(db_session, recommendation_id=rec_id, entry_payload=entry_payload, now=NOW, settlement_price=Decimal("100"), settlement_source_detail="")

    def test_gross_pnl_computed_via_existing_expiry_payoff_engine(self, db_session):
        rec_id = insert_traded_position(db_session, legs=[leg_dict(100, "CE", "BUY", "10")])
        entry_payload = get_entry_payload(db_session, rec_id)
        event_id = po.create_expiry_settlement_exit(
            db_session, recommendation_id=rec_id, entry_payload=entry_payload, now=NOW,
            settlement_price=Decimal("120"), settlement_source_detail="NSE official settlement (test-supplied, verified)",
        )
        db_session.commit()
        row = db_session.execute(text("SELECT payload FROM option_paper_trade_events WHERE id=:i"), {"i": str(event_id)}).scalar()
        assert row["gross_pnl"] == "750"
        assert row["net_pnl"] == "COSTS_NOT_MODELED"
        assert row["exit_mode"] == "EXPIRY_EXIT"

    def test_never_invoked_automatically_by_the_observation_cycle(self, db_session):
        rec_id = insert_traded_position(db_session, legs=[leg_dict(100, "CE", "BUY", "10")])
        for minutes in (5, 10, 15, 20):
            t = NOW + dt.timedelta(minutes=minutes)
            po.run_observation_cycle(db_session, underlying="NIFTY", snapshot=snapshot((mkq(100, "CE", last="12", bid="11", ask="13", observed_at=t),), t), now=t)
            db_session.commit()
        n_exit = db_session.execute(text("SELECT count(*) FROM option_paper_trade_events WHERE recommendation_id=:r AND event_type='EXIT'"), {"r": str(rec_id)}).scalar()
        assert n_exit == 0


class TestStraddleMultiLeg:
    def test_per_leg_and_aggregate_mtm_both_present(self, db_session):
        rec_id = insert_traded_position(db_session, legs=[leg_dict(100, "CE", "BUY", "10"), leg_dict(100, "PE", "BUY", "5")])
        later = NOW + dt.timedelta(minutes=5)
        moved = (mkq(100, "CE", last="15", bid="14", ask="16", observed_at=later), mkq(100, "PE", last="3", bid="2", ask="4", observed_at=later))
        po.run_observation_cycle(db_session, underlying="NIFTY", snapshot=snapshot(moved, later), now=later)
        db_session.commit()
        obs = observations(db_session, rec_id)[0]
        assert len(obs["legs"]) == 2
        ce_pnl = Decimal(next(l["mtm_pnl"] for l in obs["legs"] if l["contract"]["option_type"] == "CE"))
        pe_pnl = Decimal(next(l["mtm_pnl"] for l in obs["legs"] if l["contract"]["option_type"] == "PE"))
        assert Decimal(obs["structure_mtm_pnl"]) == ce_pnl + pe_pnl
        assert ce_pnl == Decimal("375")
        assert pe_pnl == Decimal("-150")

    def test_reuses_existing_structure_abstractions_not_a_special_case(self, db_session):
        rec_id = insert_traded_position(db_session, legs=[leg_dict(100, "CE", "BUY", "10")])
        later = NOW + dt.timedelta(minutes=5)
        po.run_observation_cycle(db_session, underlying="NIFTY", snapshot=snapshot((mkq(100, "CE", last="12", bid="11", ask="13", observed_at=later),), later), now=later)
        db_session.commit()
        obs = observations(db_session, rec_id)[0]
        assert len(obs["legs"]) == 1
        assert obs["structure_mtm_pnl"] == obs["legs"][0]["mtm_pnl"]


class TestMfeMae:
    def test_reproducible_from_raw_observations(self, db_session):
        rec_id = insert_traded_position(db_session, legs=[leg_dict(100, "CE", "BUY", "10")])
        for minutes, last in ((5, "15"), (10, "5"), (15, "12")):
            t = NOW + dt.timedelta(minutes=minutes)
            po.run_observation_cycle(db_session, underlying="NIFTY", snapshot=snapshot((mkq(100, "CE", last=last, bid=last, ask=last, observed_at=t),), t), now=t)
            db_session.commit()
        obs_payloads = observations(db_session, rec_id)
        assert len(obs_payloads) == 3
        mfe, mae = po.compute_mfe_mae(obs_payloads)
        assert mfe == Decimal("375")
        assert mae == Decimal("-375")

    def test_none_when_no_observation_has_a_valid_mtm(self):
        assert po.compute_mfe_mae([{"structure_mtm_pnl": None}, {"structure_mtm_pnl": None}]) == (None, None)

    def test_empty_sequence(self):
        assert po.compute_mfe_mae([]) == (None, None)


class TestAppendOnlyIntegrity:
    def test_observation_events_cannot_be_updated(self, db_session):
        rec_id = insert_traded_position(db_session, legs=[leg_dict(100, "CE", "BUY", "10")])
        later = NOW + dt.timedelta(minutes=5)
        po.run_observation_cycle(db_session, underlying="NIFTY", snapshot=snapshot((mkq(100, "CE", last="12", bid="11", ask="13", observed_at=later),), later), now=later)
        db_session.commit()
        event_id = db_session.execute(text("SELECT id FROM option_paper_trade_events WHERE recommendation_id=:r AND event_type='OBSERVATION'"), {"r": str(rec_id)}).scalar()
        with pytest.raises(ProgrammingError, match="immutable"):
            db_session.execute(text("UPDATE option_paper_trade_events SET payload='{}' WHERE id=:i"), {"i": str(event_id)})
        db_session.rollback()

    def test_observation_events_cannot_be_deleted(self, db_session):
        rec_id = insert_traded_position(db_session, legs=[leg_dict(100, "CE", "BUY", "10")])
        later = NOW + dt.timedelta(minutes=5)
        po.run_observation_cycle(db_session, underlying="NIFTY", snapshot=snapshot((mkq(100, "CE", last="12", bid="11", ask="13", observed_at=later),), later), now=later)
        db_session.commit()
        event_id = db_session.execute(text("SELECT id FROM option_paper_trade_events WHERE recommendation_id=:r AND event_type='OBSERVATION'"), {"r": str(rec_id)}).scalar()
        with pytest.raises(ProgrammingError, match="cannot be deleted"):
            db_session.execute(text("DELETE FROM option_paper_trade_events WHERE id=:i"), {"i": str(event_id)})
        db_session.rollback()


class TestEndToEndWiringViaRunOneCycle:

    def favorable_source(self, now=NOW):
        quotes = (mkq(100, "CE", last="910", bid="905", ask="915", observed_at=now), mkq(100, "PE", last="5", bid="4", ask="6", observed_at=now))
        return SyntheticLiveChainSource(factory=lambda as_of: snapshot(quotes, as_of))

    def mkconfig(self, source, *, min_probability=Decimal("0.01")):
        return OrchestratorConfig(
            mode="SYNTHETIC", underlying="NIFTY", source=source,
            candidate_config=CandidateGenerationConfig(lots=1, lot_size=75, lot_size_source="CALLER_SUPPLIED",
                                                       risk_free_rate=Decimal("0.065"), max_quote_age=dt.timedelta(minutes=30),
                                                       max_spread_fraction=Decimal("0.20"), min_volume=None),
            decision_thresholds=DecisionPolicyThresholds(min_probability_positive_net_pnl=min_probability, require_verified_forward=False),
            model_version="po-e2e-v0", strategy_version="po-e2e-v0",
        )

    def test_real_paper_entry_is_observable(self, db_session, monkeypatch):
        monkeypatch.setattr(dp, "PREDICTIVE_PROBABILITY_STATUS", "ESTABLISHED")
        result = run_one_cycle(db=db_session, config=self.mkconfig(self.favorable_source()), now=NOW)
        assert result.status == "TRADE"
        assert len(observations(db_session, result.recommendation_id)) == 1

        later = NOW + dt.timedelta(minutes=5)
        ids = po.run_observation_cycle(db_session, underlying="NIFTY", snapshot=snapshot((mkq(100, "CE", last="920", bid="915", ask="925", observed_at=later), mkq(100, "PE", last="4", bid="3", ask="5", observed_at=later)), later), now=later)
        db_session.commit()
        assert len(ids) == 1
        assert len(observations(db_session, result.recommendation_id)) == 2

    def test_reuses_the_same_cycle_snapshot_no_second_fetch(self, db_session, monkeypatch):
        monkeypatch.setattr(dp, "PREDICTIVE_PROBABILITY_STATUS", "ESTABLISHED")
        calls = {"n": 0}
        def factory(as_of):
            calls["n"] += 1
            if calls["n"] > 1:
                raise AssertionError("fetch_snapshot called a second time - Dhan would be called twice per cycle")
            return snapshot((mkq(100, "CE", last="910", bid="905", ask="915", observed_at=as_of), mkq(100, "PE", last="5", bid="4", ask="6", observed_at=as_of)), as_of)
        source = SyntheticLiveChainSource(factory=factory)
        result = run_one_cycle(db=db_session, config=self.mkconfig(source), now=NOW)
        assert result.status == "TRADE"
        assert calls["n"] == 1
