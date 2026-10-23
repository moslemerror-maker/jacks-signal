from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError

from backend.app.options import decision_policy as dp
from backend.app.options.availability import session_close_utc
from backend.app.options.candidate_generator import CandidateGenerationConfig
from backend.app.options.contracts import ContractIdentity
from backend.app.options.decision_policy import DecisionPolicyThresholds
from backend.app.options.live_quote import LiveQuote
from backend.app.options.live_session import (
    ChainSnapshot, OrchestratorConfig, RunnerConfig, SyntheticLiveChainSource, run_one_cycle, run_session,
)

D = dt.date
UTC = dt.timezone.utc
FIXTURE = "DEVELOPMENT_FIXTURE"

EXPIRY = D(2026, 10, 6)
TODAY_EXPIRED = D(2026, 9, 22)
NOW = dt.datetime(2026, 9, 22, 17, 0, tzinfo=UTC)


def opt(strike_rupees: int, cp: str, expiry: dt.date = EXPIRY) -> ContractIdentity:
    return ContractIdentity("NIFTY", "IDO", expiry, strike_rupees * 100, cp)


def fut(expiry: dt.date = EXPIRY) -> ContractIdentity:
    return ContractIdentity("NIFTY", "IDF", expiry, None, None)


def q(strike, cp, last, bid, ask, *, underlying_last_price=Decimal("24500"), observed_at=NOW, expiry=EXPIRY,
     volume=500, iv=None, greeks=None) -> LiveQuote:
    return LiveQuote.from_raw(
        contract=opt(strike, cp, expiry), observed_at=observed_at, last_price=None if last is None else Decimal(last),
        bid=None if bid is None else Decimal(bid), ask=None if ask is None else Decimal(ask), volume=volume,
        open_interest=1000, underlying_last_price=underlying_last_price, provider_reported_iv=iv,
        provider_reported_greeks=greeks, source="dhan", source_row_hash="e2e-test", provenance_class=FIXTURE,
    )


def realistic_nifty_chain(observed_at=NOW, expiry=EXPIRY, underlying_last_price=Decimal("24500")) -> tuple[LiveQuote, ...]:
    rows = [
        (24300, "165", "163", "167", "60", "58", "62"),
        (24400, "115", "113", "117", "80", "78", "82"),
        (24500, "78", "76", "80", "108", "106", "110"),
        (24600, "50", "48", "52", "145", "143", "147"),
        (24700, "30", "28", "32", "190", "188", "192"),
    ]
    quotes: list[LiveQuote] = []
    for strike, ce_last, ce_bid, ce_ask, pe_last, pe_bid, pe_ask in rows:
        quotes.append(q(strike, "CE", ce_last, ce_bid, ce_ask, underlying_last_price=underlying_last_price, observed_at=observed_at, expiry=expiry))
        quotes.append(q(strike, "PE", pe_last, pe_bid, pe_ask, underlying_last_price=underlying_last_price, observed_at=observed_at, expiry=expiry))
    return tuple(quotes)


def mkconfig(source, *, min_probability=Decimal("0.5"), max_quote_age=dt.timedelta(minutes=15),
            max_spread_fraction=Decimal("0.10"), require_verified_forward=False) -> OrchestratorConfig:
    return OrchestratorConfig(
        mode="SYNTHETIC", underlying="NIFTY", source=source,
        candidate_config=CandidateGenerationConfig(lots=1, lot_size=75, lot_size_source="CALLER_SUPPLIED",
                                                   risk_free_rate=Decimal("0.065"), max_quote_age=max_quote_age,
                                                   max_spread_fraction=max_spread_fraction, min_volume=None),
        decision_thresholds=DecisionPolicyThresholds(min_probability_positive_net_pnl=min_probability, require_verified_forward=require_verified_forward),
        model_version="e2e-test-model-v0", strategy_version="e2e-test-strategy-v0",
    )


def snapshot_source(quotes, *, expiry_list_dates=(EXPIRY,), source="dhan") -> SyntheticLiveChainSource:
    return SyntheticLiveChainSource(factory=lambda as_of: ChainSnapshot(
        source=source, underlying="NIFTY", observed_at=as_of, expiry_list_dates=tuple(expiry_list_dates),
        quotes=tuple(quotes), raw_response_sha256="e2e" * 16))


def count(db_session, table, where="1=1") -> int:
    return db_session.execute(text(f"SELECT count(*) FROM {table} WHERE {where}")).scalar()


class TestCleanNiftySnapshotEndToEnd:
    def test_full_pipeline_completes_and_persists_without_fabricating_data(self, db_session):
        source = snapshot_source(realistic_nifty_chain())
        result = run_one_cycle(db=db_session, config=mkconfig(source), now=NOW)
        assert result.outcome == "PERSISTED"
        assert result.status in ("WAIT", "NO_TRADE")
        assert count(db_session, "option_recommendations") == 1
        payload = db_session.execute(text("SELECT payload FROM option_recommendations WHERE id=:i"), {"i": str(result.recommendation_id)}).scalar()
        assert payload["validation_notes"]["expiry_used"] == EXPIRY.isoformat()
        assert len(payload["per_shape_results"]) == 3


class TestWaitOrNoTradeIsExpectedBehavior:
    def test_recommendation_persisted_with_reasons_and_no_paper_entry(self, db_session):
        assert dp.PREDICTIVE_PROBABILITY_STATUS == "NOT_ESTABLISHED"
        source = snapshot_source(realistic_nifty_chain())
        result = run_one_cycle(db=db_session, config=mkconfig(source), now=NOW)
        assert result.status != "TRADE"
        row = db_session.execute(text("SELECT payload FROM option_recommendations WHERE id=:i"), {"i": str(result.recommendation_id)}).scalar()
        assert len(row["selected_recommendation"]["reasons"]) > 0
        assert any("NOT_ESTABLISHED" in r for r in row["selected_recommendation"]["reasons"])
        assert all(entry["candidate"] is not None or entry["rejected_reason"] is not None for entry in row["per_shape_results"])
        assert count(db_session, "option_paper_trade_events") == 0


class TestControlledPaperEntryScenario:
    def favorable_source(self):
        quotes = (
            q(100, "CE", "910", "905", "915", underlying_last_price=Decimal("1000")),
            q(100, "PE", "5", "4", "6", underlying_last_price=Decimal("1000")),
        )
        return snapshot_source(quotes)

    def test_paper_entry_created_only_via_existing_monkeypatch_seam(self, db_session, monkeypatch):
        monkeypatch.setattr(dp, "PREDICTIVE_PROBABILITY_STATUS", "ESTABLISHED")
        config = mkconfig(self.favorable_source(), min_probability=Decimal("0.01"))
        result = run_one_cycle(db=db_session, config=config, now=NOW)
        assert result.status == "TRADE"
        assert result.paper_entry_event_id is not None
        assert count(db_session, "option_paper_trade_events") == 2
        event_types = {r[0] for r in db_session.execute(text(
            "SELECT event_type FROM option_paper_trade_events WHERE recommendation_id=:r"
        ), {"r": str(result.recommendation_id)})}
        assert event_types == {"PAPER_ENTRY", "OBSERVATION"}

    def test_paper_entry_references_recommendation_and_is_reconstructable(self, db_session, monkeypatch):
        monkeypatch.setattr(dp, "PREDICTIVE_PROBABILITY_STATUS", "ESTABLISHED")
        config = mkconfig(self.favorable_source(), min_probability=Decimal("0.01"))
        result = run_one_cycle(db=db_session, config=config, now=NOW)
        row = db_session.execute(text("SELECT recommendation_id, event_type, payload FROM option_paper_trade_events WHERE id=:i"),
                                 {"i": str(result.paper_entry_event_id)}).one()
        assert str(row.recommendation_id) == str(result.recommendation_id)
        assert row.event_type == "PAPER_ENTRY"
        assert row.payload["candidate"]["structure"]["legs"][0]["contract"]["strike_paise"] == 10000
        assert "reasons" in row.payload

    def test_without_the_seam_trade_stays_blocked_no_execution_path_exists(self, db_session):
        assert dp.PREDICTIVE_PROBABILITY_STATUS == "NOT_ESTABLISHED"
        config = mkconfig(self.favorable_source(), min_probability=Decimal("0.01"))
        result = run_one_cycle(db=db_session, config=config, now=NOW)
        assert result.status != "TRADE" and result.paper_entry_event_id is None


class TestInvalidOrDegradedMarketData:
    def test_empty_option_chain_no_fabricated_recommendation(self, db_session):
        source = snapshot_source(())
        result = run_one_cycle(db=db_session, config=mkconfig(source), now=NOW)
        assert result.status == "NO_TRADE"
        payload = db_session.execute(text("SELECT payload FROM option_recommendations WHERE id=:i"), {"i": str(result.recommendation_id)}).scalar()
        assert payload["validation_notes"]["validation_failure"] == "empty_snapshot"

    def test_missing_underlying_price_no_fabricated_forward(self, db_session):
        quotes = tuple(q(s, cp, "100", "98", "102", underlying_last_price=None) for s, cp in ((24500, "CE"), (24500, "PE")))
        source = snapshot_source(quotes)
        result = run_one_cycle(db=db_session, config=mkconfig(source), now=NOW)
        assert result.status == "NO_TRADE"
        assert count(db_session, "option_session_events", "category='MISSING_UNDERLYING_PRICE'") == 1

    def test_all_quotes_stale_rejects_every_shape_via_liquidity_not_a_crash(self, db_session):
        stale_at = NOW - dt.timedelta(hours=2)
        quotes = realistic_nifty_chain(observed_at=stale_at)
        source = snapshot_source(quotes)
        config = mkconfig(source, max_quote_age=dt.timedelta(minutes=15))
        result = run_one_cycle(db=db_session, config=config, now=NOW)
        assert result.status == "NO_TRADE"
        payload = db_session.execute(text("SELECT payload FROM option_recommendations WHERE id=:i"), {"i": str(result.recommendation_id)}).scalar()
        assert all(e["candidate"] is None and "stale" in (e["rejected_reason"] or "") for e in payload["per_shape_results"])

    def test_unusable_crossed_bid_ask_across_chain_no_fabricated_price(self, db_session):
        quotes = tuple(q(s, cp, "100", "150", "50") for s, cp in ((24500, "CE"), (24500, "PE")))
        source = snapshot_source(quotes)
        result = run_one_cycle(db=db_session, config=mkconfig(source), now=NOW)
        assert result.status == "NO_TRADE"

    def test_missing_put_side_rejects_put_and_straddle_but_call_may_still_work(self, db_session):
        ce_only = tuple(q(s, "CE", "100", "98", "102") for s in (24400, 24500, 24600))
        source = snapshot_source(ce_only)
        result = run_one_cycle(db=db_session, config=mkconfig(source), now=NOW)
        payload = db_session.execute(text("SELECT payload FROM option_recommendations WHERE id=:i"), {"i": str(result.recommendation_id)}).scalar()
        by_shape = {e["shape"]: e for e in payload["per_shape_results"]}
        assert by_shape["LONG_PUT_ATM"]["candidate"] is None
        assert by_shape["LONG_STRADDLE_ATM"]["candidate"] is None
        assert by_shape["LONG_CALL_ATM"]["candidate"] is not None

    def test_no_eligible_future_expiry_no_fabricated_decision(self, db_session):
        far_past = D(2020, 1, 1)
        source = snapshot_source(realistic_nifty_chain(expiry=far_past), expiry_list_dates=(far_past,))
        result = run_one_cycle(db=db_session, config=mkconfig(source), now=NOW)
        assert result.status == "NO_TRADE"
        assert count(db_session, "option_session_events", "category='NO_ELIGIBLE_EXPIRY'") == 1


class TestAfterHoursExpiryRegressionMandatory:
    def test_future_eligible_expiry_selected_not_todays_expired_one(self, db_session):
        mixed = realistic_nifty_chain(expiry=TODAY_EXPIRED) + realistic_nifty_chain(expiry=EXPIRY)
        source = snapshot_source(mixed, expiry_list_dates=(TODAY_EXPIRED, EXPIRY))
        result = run_one_cycle(db=db_session, config=mkconfig(source), now=NOW)
        payload = db_session.execute(text("SELECT payload FROM option_recommendations WHERE id=:i"), {"i": str(result.recommendation_id)}).scalar()
        assert payload["validation_notes"]["expiry_used"] == EXPIRY.isoformat()
        for entry in payload["per_shape_results"]:
            if entry["candidate"] is not None:
                for leg in entry["candidate"]["structure"]["legs"]:
                    assert leg["contract"]["expiry"] == EXPIRY.isoformat()


class TestForwardResolutionSemantics:
    def test_spot_proxy_used_and_explicitly_tagged_when_no_futures_quote_present(self, db_session):
        source = snapshot_source(realistic_nifty_chain())
        result = run_one_cycle(db=db_session, config=mkconfig(source), now=NOW)
        payload = db_session.execute(text("SELECT payload FROM option_recommendations WHERE id=:i"), {"i": str(result.recommendation_id)}).scalar()
        assert payload["validation_notes"]["forward_resolution_source"] == "SPOT_PROXY_UNVERIFIED"
        for entry in payload["per_shape_results"]:
            if entry["candidate"] is not None:
                fr = entry["candidate"]["forward_resolution"]
                assert fr["source"] == "SPOT_PROXY_UNVERIFIED"
                assert fr["forward"] == fr["underlying_last_price"]

    def test_verified_futures_forward_preferred_when_a_usable_futures_quote_exists(self, db_session):
        future_quote = LiveQuote.from_raw(
            contract=fut(), observed_at=NOW, last_price=Decimal("24550"), bid=Decimal("24548"), ask=Decimal("24552"),
            volume=1000, open_interest=5000, underlying_last_price=Decimal("24500"), provider_reported_iv=None,
            provider_reported_greeks=None, source="dhan", source_row_hash="e2e-test", provenance_class=FIXTURE,
        )
        quotes = realistic_nifty_chain() + (future_quote,)
        source = snapshot_source(quotes)
        result = run_one_cycle(db=db_session, config=mkconfig(source), now=NOW)
        payload = db_session.execute(text("SELECT payload FROM option_recommendations WHERE id=:i"), {"i": str(result.recommendation_id)}).scalar()
        assert payload["validation_notes"]["forward_resolution_source"] == "VERIFIED_LIVE_FUTURES"
        for entry in payload["per_shape_results"]:
            if entry["candidate"] is not None:
                fr = entry["candidate"]["forward_resolution"]
                assert fr["source"] == "VERIFIED_LIVE_FUTURES"
                assert fr["forward"] == "24550"


class TestPayloadReconstructionExactValues:
    def test_payload_contains_actual_values_not_just_field_presence(self, db_session):
        source = snapshot_source(realistic_nifty_chain())
        result = run_one_cycle(db=db_session, config=mkconfig(source), now=NOW)
        payload = db_session.execute(text("SELECT payload FROM option_recommendations WHERE id=:i"), {"i": str(result.recommendation_id)}).scalar()
        row = db_session.execute(text("SELECT decision_timestamp, source, underlying, status, model_version, strategy_version FROM option_recommendations WHERE id=:i"),
                                 {"i": str(result.recommendation_id)}).one()

        assert row.underlying == "NIFTY"
        assert row.source == "dhan"
        assert row.decision_timestamp.replace(tzinfo=UTC) == NOW
        assert row.model_version == "e2e-test-model-v0" and row.strategy_version == "e2e-test-strategy-v0"

        assert payload["cycle"]["total_quotes_in_snapshot"] == 10
        assert payload["cycle"]["observed_at"] == NOW.isoformat()
        assert payload["candidate_config"]["risk_free_rate"] == "0.065"
        assert payload["candidate_config"]["lot_size"] == 75
        assert payload["decision_thresholds"]["min_probability_positive_net_pnl"] == "0.5"
        assert payload["costs_slippage_assumptions"]["status"] == "NOT_MODELED"
        assert payload["risk_and_expected_value_status"]["risk"]["status"] == "NOT_MODELED"
        assert payload["risk_and_expected_value_status"]["expected_value"]["status"] == "NOT_MODELED"
        assert payload["predictive_probability_status"] == "NOT_ESTABLISHED"
        assert payload["selected_recommendation"]["status"] == row.status
        assert payload["selected_recommendation"]["underlying"] == "NIFTY"

        call_entry = next(e for e in payload["per_shape_results"] if e["shape"] == "LONG_CALL_ATM")
        assert call_entry["candidate"] is not None
        leg = call_entry["candidate"]["structure"]["legs"][0]
        assert leg["contract"]["strike_paise"] == 2450000
        assert leg["entry_price"] == "78"
        assert leg["entry_price_source"] == "LIVE_QUOTE"
        assert leg["side"] == "BUY"

        snap = call_entry["decision_snapshot_legs"][0]
        assert snap["contract"]["strike_paise"] == 2450000
        assert snap["quote_status"] == "QUOTED"
        assert snap["last_price"] == "78"
        assert snap["bid"] is not None and snap["ask"] is not None
        assert snap["volume"] == 500
        assert snap["open_interest"] == 1000
        assert snap["liquidity_assessment"]["is_acceptable"] is True
        assert snap["liquidity_assessment"]["reasons"] == []

        for entry in payload["per_shape_results"]:
            if entry["candidate"] is None:
                assert entry["decision_snapshot_legs"] is None


class TestIdempotencyAcrossRepeatedCycles:
    def test_same_logical_cycle_run_twice_does_not_duplicate(self, db_session):
        source = snapshot_source(realistic_nifty_chain())
        config = mkconfig(source)
        r1 = run_one_cycle(db=db_session, config=config, now=NOW)
        r2 = run_one_cycle(db=db_session, config=config, now=NOW)
        assert r1.recommendation_id == r2.recommendation_id
        assert r2.was_duplicate is True
        assert count(db_session, "option_recommendations") == 1
        original_payload = db_session.execute(text("SELECT payload FROM option_recommendations WHERE id=:i"), {"i": str(r1.recommendation_id)}).scalar()
        assert original_payload["cycle"]["observed_at"] == NOW.isoformat()

    def test_new_decision_timestamp_creates_a_genuinely_new_recommendation(self, db_session):
        source = snapshot_source(realistic_nifty_chain())
        config = mkconfig(source)
        r1 = run_one_cycle(db=db_session, config=config, now=NOW)
        r2 = run_one_cycle(db=db_session, config=config, now=NOW + dt.timedelta(minutes=15))
        assert r1.recommendation_id != r2.recommendation_id
        assert count(db_session, "option_recommendations") == 2


class TestDatabaseIntegrityWithinE2E:
    def test_fk_check_and_immutability_all_hold_for_a_real_e2e_cycle(self, db_session, monkeypatch):
        monkeypatch.setattr(dp, "PREDICTIVE_PROBABILITY_STATUS", "ESTABLISHED")
        favorable = (q(100, "CE", "910", "905", "915", underlying_last_price=Decimal("1000")),
                    q(100, "PE", "5", "4", "6", underlying_last_price=Decimal("1000")))
        config = mkconfig(snapshot_source(favorable), min_probability=Decimal("0.01"))
        result = run_one_cycle(db=db_session, config=config, now=NOW)
        assert result.status == "TRADE" and result.paper_entry_event_id is not None

        fk_ok = db_session.execute(text(
            "SELECT count(*) FROM option_paper_trade_events e JOIN option_recommendations r ON e.recommendation_id = r.id WHERE e.id=:i"
        ), {"i": str(result.paper_entry_event_id)}).scalar()
        assert fk_ok == 1

        with pytest.raises((IntegrityError, DBAPIError)):
            with db_session.begin_nested():
                db_session.execute(text("INSERT INTO option_recommendations (id, decision_timestamp, underlying, status, data_quality, source, payload) "
                                        "VALUES (gen_random_uuid(), :ts, 'NIFTY', 'MAYBE', 'EOD_RESOLVED', 'dhan', '{}'::jsonb)"), {"ts": NOW})

        with pytest.raises(DBAPIError, match="immutable"):
            with db_session.begin_nested():
                db_session.execute(text("UPDATE option_recommendations SET status='WAIT' WHERE id=:i"), {"i": str(result.recommendation_id)})
        with pytest.raises(DBAPIError, match="cannot be deleted"):
            with db_session.begin_nested():
                db_session.execute(text("DELETE FROM option_paper_trade_events WHERE id=:i"), {"i": str(result.paper_entry_event_id)})
