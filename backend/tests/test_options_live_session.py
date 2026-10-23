from __future__ import annotations

import dataclasses
import datetime as dt
from decimal import Decimal

import pytest
from sqlalchemy import text

from backend.app.options import decision_policy as dp
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
EXPIRY = D(2026, 9, 29)
NOW = dt.datetime(2026, 9, 22, 8, 0, tzinfo=UTC)


def opt(strike_rupees: int, cp: str, expiry: dt.date = EXPIRY) -> ContractIdentity:
    return ContractIdentity("NIFTY", "IDO", expiry, strike_rupees * 100, cp)


def mkq(strike: int, cp: str, *, last_price=None, bid=None, ask=None, underlying_last_price=Decimal("24500"),
        expiry: dt.date = EXPIRY, observed_at=NOW) -> LiveQuote:
    return LiveQuote.from_raw(
        contract=opt(strike, cp, expiry), observed_at=observed_at, last_price=last_price, bid=bid, ask=ask, volume=500,
        open_interest=1000, underlying_last_price=underlying_last_price, provider_reported_iv=None,
        provider_reported_greeks=None, source="dhan", source_row_hash=None, provenance_class=FIXTURE,
    )


def good_quotes(expiry: dt.date = EXPIRY, observed_at=NOW) -> tuple[LiveQuote, ...]:
    return (
        mkq(24500, "CE", last_price=Decimal("140"), bid=Decimal("138"), ask=Decimal("142"), expiry=expiry, observed_at=observed_at),
        mkq(24500, "PE", last_price=Decimal("135"), bid=Decimal("133"), ask=Decimal("137"), expiry=expiry, observed_at=observed_at),
        mkq(24400, "CE", last_price=Decimal("190"), bid=Decimal("188"), ask=Decimal("192"), expiry=expiry, observed_at=observed_at),
        mkq(24400, "PE", last_price=Decimal("95"), bid=Decimal("93"), ask=Decimal("97"), expiry=expiry, observed_at=observed_at),
        mkq(24600, "CE", last_price=Decimal("100"), bid=Decimal("98"), ask=Decimal("102"), expiry=expiry, observed_at=observed_at),
        mkq(24600, "PE", last_price=Decimal("185"), bid=Decimal("183"), ask=Decimal("187"), expiry=expiry, observed_at=observed_at),
    )


def good_snapshot(as_of: dt.datetime, expiry_list=(EXPIRY,)) -> ChainSnapshot:
    return ChainSnapshot(source="dhan", underlying="NIFTY", observed_at=as_of, expiry_list_dates=tuple(expiry_list),
                         quotes=good_quotes(observed_at=as_of), raw_response_sha256="deadbeef" * 8)


def mkconfig(source, *, min_probability=Decimal("0.5"), mode="SYNTHETIC") -> OrchestratorConfig:
    return OrchestratorConfig(
        mode=mode, underlying="NIFTY", source=source,
        candidate_config=CandidateGenerationConfig(lots=1, lot_size=75, lot_size_source="CALLER_SUPPLIED",
                                                   risk_free_rate=Decimal("0.065"), max_quote_age=dt.timedelta(minutes=30),
                                                   max_spread_fraction=Decimal("0.20"), min_volume=None),
        decision_thresholds=DecisionPolicyThresholds(min_probability_positive_net_pnl=min_probability, require_verified_forward=False),
        model_version="test-model-v0", strategy_version="test-strategy-v0",
    )


def count_recommendations(db_session) -> int:
    return db_session.execute(text("SELECT count(*) FROM option_recommendations")).scalar()


def raising_source(exc: Exception) -> SyntheticLiveChainSource:
    def factory(as_of):
        raise exc
    return SyntheticLiveChainSource(factory=factory)


class TestSuccessfulCycle:
    def test_full_cycle_persists_one_recommendation(self, db_session):
        source = SyntheticLiveChainSource(factory=good_snapshot)
        config = mkconfig(source)
        result = run_one_cycle(db=db_session, config=config, now=NOW)
        assert result.outcome == "PERSISTED"
        assert result.status in ("WAIT", "NO_TRADE")
        assert result.recommendation_id is not None
        assert count_recommendations(db_session) == 1

    def test_persisted_row_has_expected_flat_columns(self, db_session):
        source = SyntheticLiveChainSource(factory=good_snapshot)
        config = mkconfig(source)
        result = run_one_cycle(db=db_session, config=config, now=NOW)
        row = db_session.execute(text("SELECT underlying, source, model_version, strategy_version FROM option_recommendations WHERE id=:i"),
                                 {"i": str(result.recommendation_id)}).one()
        assert row.underlying == "NIFTY" and row.source == "dhan" and row.model_version == "test-model-v0"


class TestFetchFailure:
    def test_no_recommendation_and_session_event_recorded(self, db_session):
        source = raising_source(RuntimeError("simulated Dhan auth failure"))
        config = mkconfig(source)
        result = run_one_cycle(db=db_session, config=config, now=NOW)
        assert result.outcome == "FETCH_FAILURE" and result.recommendation_id is None
        assert count_recommendations(db_session) == 0
        from backend.app.options.pipeline_reliability import FETCH_FAILURE_CATEGORIES
        category = db_session.execute(text("SELECT category FROM option_session_events")).scalar()
        assert category in FETCH_FAILURE_CATEGORIES


class TestMalformedData:
    def test_empty_snapshot_produces_no_trade_not_a_fabricated_decision(self, db_session):
        source = SyntheticLiveChainSource(factory=lambda as_of: ChainSnapshot(
            source="dhan", underlying="NIFTY", observed_at=as_of, expiry_list_dates=(EXPIRY,), quotes=(), raw_response_sha256=None))
        config = mkconfig(source)
        result = run_one_cycle(db=db_session, config=config, now=NOW)
        assert result.outcome == "PERSISTED" and result.status == "NO_TRADE"
        payload = db_session.execute(text("SELECT payload FROM option_recommendations WHERE id=:i"), {"i": str(result.recommendation_id)}).scalar()
        assert payload["validation_notes"]["validation_failure"] == "empty_snapshot"

    def test_missing_underlying_price_produces_no_trade_and_session_event(self, db_session):
        quotes = tuple(dataclasses.replace(q, underlying_last_price=None) for q in good_quotes())
        source = SyntheticLiveChainSource(factory=lambda as_of: ChainSnapshot(
            source="dhan", underlying="NIFTY", observed_at=as_of, expiry_list_dates=(EXPIRY,), quotes=quotes, raw_response_sha256=None))
        config = mkconfig(source)
        result = run_one_cycle(db=db_session, config=config, now=NOW)
        assert result.status == "NO_TRADE"
        n = db_session.execute(text("SELECT count(*) FROM option_session_events WHERE category='MISSING_UNDERLYING_PRICE'")).scalar()
        assert n == 1


class TestExpirySelection:
    def test_only_eligible_expiry_quotes_are_used(self, db_session):
        closed_expiry = D(2026, 9, 20)
        mixed_quotes = good_quotes(expiry=closed_expiry, observed_at=NOW) + good_quotes(expiry=EXPIRY, observed_at=NOW)
        source = SyntheticLiveChainSource(factory=lambda as_of: ChainSnapshot(
            source="dhan", underlying="NIFTY", observed_at=as_of, expiry_list_dates=(closed_expiry, EXPIRY),
            quotes=mixed_quotes, raw_response_sha256=None))
        config = mkconfig(source)
        result = run_one_cycle(db=db_session, config=config, now=NOW)
        payload = db_session.execute(text("SELECT payload FROM option_recommendations WHERE id=:i"), {"i": str(result.recommendation_id)}).scalar()
        assert payload["validation_notes"]["expiry_used"] == EXPIRY.isoformat()

    def test_no_eligible_expiry_produces_no_trade_and_session_event(self, db_session):
        past = D(2020, 1, 1)
        source = SyntheticLiveChainSource(factory=lambda as_of: ChainSnapshot(
            source="dhan", underlying="NIFTY", observed_at=as_of, expiry_list_dates=(past,), quotes=good_quotes(expiry=past), raw_response_sha256=None))
        config = mkconfig(source)
        result = run_one_cycle(db=db_session, config=config, now=NOW)
        assert result.status == "NO_TRADE"
        n = db_session.execute(text("SELECT count(*) FROM option_session_events WHERE category='NO_ELIGIBLE_EXPIRY'")).scalar()
        assert n == 1


class TestForwardProvenance:
    def test_spot_proxy_tag_present_in_payload(self, db_session):
        source = SyntheticLiveChainSource(factory=good_snapshot)
        config = mkconfig(source)
        result = run_one_cycle(db=db_session, config=config, now=NOW)
        payload = db_session.execute(text("SELECT payload FROM option_recommendations WHERE id=:i"), {"i": str(result.recommendation_id)}).scalar()
        assert payload["validation_notes"]["forward_resolution_source"] == "SPOT_PROXY_UNVERIFIED"
        for entry in payload["per_shape_results"]:
            if entry["candidate"] is not None:
                assert entry["candidate"]["forward_resolution"]["source"] == "SPOT_PROXY_UNVERIFIED"


class TestWaitAndNoTradePersistence:
    def test_wait_is_persisted_with_no_candidate_column_value_implying_trade(self, db_session):
        source = SyntheticLiveChainSource(factory=good_snapshot)
        result = run_one_cycle(db=db_session, config=mkconfig(source), now=NOW)
        row = db_session.execute(text("SELECT status FROM option_recommendations WHERE id=:i"), {"i": str(result.recommendation_id)}).one()
        assert row.status in ("WAIT", "NO_TRADE")


class TestPaperEntryLifecycle:
    def favorable_source(self):
        quotes = (
            mkq(100, "CE", last_price=Decimal("910"), bid=Decimal("905"), ask=Decimal("915"), underlying_last_price=Decimal("1000")),
            mkq(100, "PE", last_price=Decimal("5"), bid=Decimal("4"), ask=Decimal("6"), underlying_last_price=Decimal("1000")),
        )
        return SyntheticLiveChainSource(factory=lambda as_of: ChainSnapshot(
            source="dhan", underlying="NIFTY", observed_at=as_of, expiry_list_dates=(EXPIRY,), quotes=quotes, raw_response_sha256=None))

    def test_trade_blocked_by_default_no_paper_entry(self, db_session):
        assert dp.PREDICTIVE_PROBABILITY_STATUS == "NOT_ESTABLISHED"
        config = mkconfig(self.favorable_source(), min_probability=Decimal("0.01"))
        result = run_one_cycle(db=db_session, config=config, now=NOW)
        assert result.status != "TRADE"
        assert result.paper_entry_event_id is None
        n = db_session.execute(text("SELECT count(*) FROM option_paper_trade_events")).scalar()
        assert n == 0

    def test_paper_entry_created_when_policy_permits_trade(self, db_session, monkeypatch):
        monkeypatch.setattr(dp, "PREDICTIVE_PROBABILITY_STATUS", "ESTABLISHED")
        config = mkconfig(self.favorable_source(), min_probability=Decimal("0.01"))
        result = run_one_cycle(db=db_session, config=config, now=NOW)
        assert result.status == "TRADE"
        assert result.paper_entry_event_id is not None
        row = db_session.execute(text("SELECT event_type, recommendation_id FROM option_paper_trade_events WHERE id=:i"),
                                 {"i": str(result.paper_entry_event_id)}).one()
        assert row.event_type == "PAPER_ENTRY" and str(row.recommendation_id) == str(result.recommendation_id)


class TestPaperEntryEnrichment:

    def favorable_source(self):
        quotes = (
            mkq(100, "CE", last_price=Decimal("910"), bid=Decimal("905"), ask=Decimal("915"), underlying_last_price=Decimal("1000")),
            mkq(100, "PE", last_price=Decimal("5"), bid=Decimal("4"), ask=Decimal("6"), underlying_last_price=Decimal("1000")),
        )
        return SyntheticLiveChainSource(factory=lambda as_of: ChainSnapshot(
            source="dhan", underlying="NIFTY", observed_at=as_of, expiry_list_dates=(EXPIRY,), quotes=quotes, raw_response_sha256=None))

    def _paper_entry_payload(self, db_session, event_id) -> dict:
        return db_session.execute(text("SELECT payload FROM option_paper_trade_events WHERE id=:i"), {"i": str(event_id)}).scalar()

    def test_entry_snapshot_contains_per_leg_bid_ask_volume_oi(self, db_session, monkeypatch):
        monkeypatch.setattr(dp, "PREDICTIVE_PROBABILITY_STATUS", "ESTABLISHED")
        config = mkconfig(self.favorable_source(), min_probability=Decimal("0.01"))
        result = run_one_cycle(db=db_session, config=config, now=NOW)
        assert result.status == "TRADE"
        payload = self._paper_entry_payload(db_session, result.paper_entry_event_id)
        assert payload["shape"] in ("LONG_CALL_ATM", "LONG_PUT_ATM")
        assert len(payload["legs"]) >= 1
        leg = payload["legs"][0]
        assert leg["entry_bid"] is not None and leg["entry_ask"] is not None
        assert leg["entry_volume"] == 500 and leg["entry_open_interest"] == 1000
        assert leg["side"] == "BUY"
        assert leg["lots"] == 1 and leg["lot_size"] == 75
        assert leg["contract"]["underlying"] == "NIFTY"

    def test_entry_snapshot_includes_capital_basis_not_modeled(self, db_session, monkeypatch):
        monkeypatch.setattr(dp, "PREDICTIVE_PROBABILITY_STATUS", "ESTABLISHED")
        config = mkconfig(self.favorable_source(), min_probability=Decimal("0.01"))
        result = run_one_cycle(db=db_session, config=config, now=NOW)
        payload = self._paper_entry_payload(db_session, result.paper_entry_event_id)
        assert payload["capital_basis"] == "NOT_MODELED"

    def test_entry_snapshot_includes_shape_timestamp_and_versions(self, db_session, monkeypatch):
        monkeypatch.setattr(dp, "PREDICTIVE_PROBABILITY_STATUS", "ESTABLISHED")
        config = mkconfig(self.favorable_source(), min_probability=Decimal("0.01"))
        result = run_one_cycle(db=db_session, config=config, now=NOW)
        payload = self._paper_entry_payload(db_session, result.paper_entry_event_id)
        assert payload["model_version"] == "test-model-v0"
        assert payload["strategy_version"] == "test-strategy-v0"
        assert payload["recommendation_timestamp"] is not None
        assert payload["forward_resolution"]["source"] in ("SPOT_PROXY_UNVERIFIED", "VERIFIED_LIVE_FUTURES")

    def test_entry_snapshot_immutable_after_a_later_cycle(self, db_session, monkeypatch):
        monkeypatch.setattr(dp, "PREDICTIVE_PROBABILITY_STATUS", "ESTABLISHED")
        config = mkconfig(self.favorable_source(), min_probability=Decimal("0.01"))
        result = run_one_cycle(db=db_session, config=config, now=NOW)
        before = self._paper_entry_payload(db_session, result.paper_entry_event_id)

        later_source = SyntheticLiveChainSource(factory=good_snapshot)
        run_one_cycle(db=db_session, config=mkconfig(later_source), now=NOW + dt.timedelta(minutes=5))

        after = self._paper_entry_payload(db_session, result.paper_entry_event_id)
        assert before == after

    def test_missing_outcome_degrades_gracefully_never_crashes(self, db_session, monkeypatch):
        from backend.app.options import live_session as ls

        monkeypatch.setattr(dp, "PREDICTIVE_PROBABILITY_STATUS", "ESTABLISHED")
        config = mkconfig(self.favorable_source(), min_probability=Decimal("0.01"))

        real_persist = ls._persist_and_finish

        def patched(db, **kwargs):
            kwargs["per_shape"] = []
            return real_persist(db, **kwargs)

        monkeypatch.setattr(ls, "_persist_and_finish", patched)
        result = run_one_cycle(db=db_session, config=config, now=NOW)
        assert result.status == "TRADE"
        payload = self._paper_entry_payload(db_session, result.paper_entry_event_id)
        assert payload["shape"] is None
        assert payload["legs"][0]["entry_bid"] is None


class TestIdempotency:
    def test_identical_snapshot_run_twice_does_not_duplicate(self, db_session):
        source = SyntheticLiveChainSource(factory=good_snapshot)
        config = mkconfig(source)
        r1 = run_one_cycle(db=db_session, config=config, now=NOW)
        r2 = run_one_cycle(db=db_session, config=config, now=NOW)
        assert r1.recommendation_id == r2.recommendation_id
        assert r1.was_duplicate is False and r2.was_duplicate is True
        assert count_recommendations(db_session) == 1

    def test_different_observed_at_creates_a_separate_recommendation(self, db_session):
        source = SyntheticLiveChainSource(factory=good_snapshot)
        config = mkconfig(source)
        r1 = run_one_cycle(db=db_session, config=config, now=NOW)
        r2 = run_one_cycle(db=db_session, config=config, now=NOW + dt.timedelta(minutes=5))
        assert r1.recommendation_id != r2.recommendation_id
        assert count_recommendations(db_session) == 2


class TestBoundedRunner:
    def test_bounded_session_runs_exactly_max_cycles(self, db_session):
        source = SyntheticLiveChainSource(factory=good_snapshot)
        config = mkconfig(source)
        times = iter([NOW, NOW + dt.timedelta(minutes=5), NOW + dt.timedelta(minutes=10)])
        results = run_session(session_factory=lambda: db_session, config=config, runner_config=RunnerConfig(
            polling_interval=dt.timedelta(seconds=1), max_cycles=3), now_fn=lambda: next(times), sleep_fn=lambda s: None)
        assert len(results) == 3
        assert all(r.outcome == "PERSISTED" for r in results)
        assert count_recommendations(db_session) == 3

    def test_stop_flag_halts_before_max_cycles(self, db_session):
        source = SyntheticLiveChainSource(factory=good_snapshot)
        config = mkconfig(source)
        counter = {"n": 0}
        def stop():
            return counter["n"] >= 1
        def now_fn():
            counter["n"] += 1
            return NOW + dt.timedelta(minutes=counter["n"])
        results = run_session(session_factory=lambda: db_session, config=config, runner_config=RunnerConfig(
            polling_interval=dt.timedelta(seconds=1), max_cycles=5), now_fn=now_fn, sleep_fn=lambda s: None, stop_flag=stop)
        assert len(results) == 1


def _monotonic_clock(start: dt.datetime, step: dt.timedelta):
    state = {"n": 0}
    def clock() -> dt.datetime:
        t = start + state["n"] * step
        state["n"] += 1
        return t
    return clock


class TestStopAt:
    def test_stop_at_prevents_a_new_cycle_from_starting(self, db_session):
        source = SyntheticLiveChainSource(factory=good_snapshot)
        config = mkconfig(source)
        clock = _monotonic_clock(NOW, dt.timedelta(minutes=1))
        stop_at = NOW + dt.timedelta(minutes=30)
        results = run_session(session_factory=lambda: db_session, config=config, runner_config=RunnerConfig(
            polling_interval=dt.timedelta(seconds=1), max_cycles=None, stop_at=stop_at), now_fn=clock, sleep_fn=lambda s: None)
        assert 1 <= len(results) <= 30
        assert all(r.outcome == "PERSISTED" for r in results)
        assert count_recommendations(db_session) == len(results)

    def test_stop_at_already_passed_before_the_first_cycle_runs_zero_cycles(self, db_session):
        source = SyntheticLiveChainSource(factory=good_snapshot)
        config = mkconfig(source)
        stop_at = NOW - dt.timedelta(minutes=1)
        results = run_session(session_factory=lambda: db_session, config=config, runner_config=RunnerConfig(
            polling_interval=dt.timedelta(seconds=1), max_cycles=None, stop_at=stop_at), now_fn=lambda: NOW, sleep_fn=lambda s: None)
        assert results == []
        assert count_recommendations(db_session) == 0

    def test_max_cycles_and_stop_at_together_whichever_is_reached_first(self, db_session):
        source = SyntheticLiveChainSource(factory=good_snapshot)
        config = mkconfig(source)
        clock = _monotonic_clock(NOW, dt.timedelta(minutes=1))
        stop_at = NOW + dt.timedelta(days=1)
        results = run_session(session_factory=lambda: db_session, config=config, runner_config=RunnerConfig(
            polling_interval=dt.timedelta(seconds=1), max_cycles=2, stop_at=stop_at), now_fn=clock, sleep_fn=lambda s: None)
        assert len(results) == 2

    def test_stop_at_none_preserves_unbounded_behavior(self, db_session):
        source = SyntheticLiveChainSource(factory=good_snapshot)
        config = mkconfig(source)
        clock = _monotonic_clock(NOW, dt.timedelta(minutes=1))
        results = run_session(session_factory=lambda: db_session, config=config, runner_config=RunnerConfig(
            polling_interval=dt.timedelta(seconds=1), max_cycles=3), now_fn=clock, sleep_fn=lambda s: None)
        assert len(results) == 3

    def test_naive_stop_at_is_rejected(self):
        with pytest.raises(ValueError, match="stop_at must be timezone-aware"):
            RunnerConfig(stop_at=dt.datetime(2026, 9, 24, 9, 15))


class TestMaxConsecutiveFetchFailures:

    def test_stops_after_exactly_the_configured_number_of_consecutive_failures(self, db_session):
        source = raising_source(RuntimeError("simulated Dhan auth failure"))
        config = mkconfig(source)
        clock = _monotonic_clock(NOW, dt.timedelta(minutes=5))
        results = run_session(session_factory=lambda: db_session, config=config, runner_config=RunnerConfig(
            polling_interval=dt.timedelta(seconds=1), max_cycles=None, max_consecutive_fetch_failures=6),
            now_fn=clock, sleep_fn=lambda s: None)
        assert len(results) == 6
        assert all(r.outcome == "FETCH_FAILURE" for r in results)

    def test_an_interleaved_success_resets_the_counter_so_the_session_keeps_going(self, db_session):
        calls = {"n": 0}

        def flaky(as_of):
            calls["n"] += 1
            if calls["n"] == 3:
                return good_snapshot(as_of)
            raise RuntimeError("simulated transient failure")

        source = SyntheticLiveChainSource(factory=flaky)
        config = mkconfig(source)
        clock = _monotonic_clock(NOW, dt.timedelta(minutes=5))
        results = run_session(session_factory=lambda: db_session, config=config, runner_config=RunnerConfig(
            polling_interval=dt.timedelta(seconds=1), max_cycles=5, max_consecutive_fetch_failures=3),
            now_fn=clock, sleep_fn=lambda s: None)
        assert len(results) == 5
        assert [r.outcome for r in results] == ["FETCH_FAILURE", "FETCH_FAILURE", "PERSISTED", "FETCH_FAILURE", "FETCH_FAILURE"]

    def test_none_preserves_unbounded_retry_behavior(self, db_session):
        source = raising_source(RuntimeError("simulated failure"))
        config = mkconfig(source)
        clock = _monotonic_clock(NOW, dt.timedelta(minutes=5))
        results = run_session(session_factory=lambda: db_session, config=config, runner_config=RunnerConfig(
            polling_interval=dt.timedelta(seconds=1), max_cycles=10, max_consecutive_fetch_failures=None),
            now_fn=clock, sleep_fn=lambda s: None)
        assert len(results) == 10

    def test_rejects_a_non_positive_threshold(self):
        with pytest.raises(ValueError, match="max_consecutive_fetch_failures must be >= 1"):
            RunnerConfig(max_consecutive_fetch_failures=0)


class RaisingDB:
    def execute(self, *a, **kw):
        raise RuntimeError("simulated DB failure during persistence")
    def commit(self):
        pass
    def rollback(self):
        pass
    def close(self):
        pass


class TestDbFailurePropagates:
    def test_run_one_cycle_raises_not_returns_fake_success(self):
        source = SyntheticLiveChainSource(factory=good_snapshot)
        config = mkconfig(source)
        with pytest.raises(RuntimeError, match="simulated DB failure"):
            run_one_cycle(db=RaisingDB(), config=config, now=NOW)

    def test_run_session_catches_it_logs_and_continues(self):
        source = SyntheticLiveChainSource(factory=good_snapshot)
        config = mkconfig(source)
        state = {"n": 0}
        real = {}

        def factory():
            state["n"] += 1
            return RaisingDB()

        results = run_session(session_factory=factory, config=config, runner_config=RunnerConfig(
            polling_interval=dt.timedelta(seconds=1), max_cycles=2), now_fn=lambda: NOW, sleep_fn=lambda s: None)
        assert len(results) == 2
        assert all(r.outcome == "PERSISTENCE_FAILURE" for r in results)
        assert all(r.recommendation_id is None for r in results)


class TestPayloadReconstruction:
    def test_payload_contains_required_decision_context(self, db_session):
        source = SyntheticLiveChainSource(factory=good_snapshot)
        config = mkconfig(source)
        result = run_one_cycle(db=db_session, config=config, now=NOW)
        payload = db_session.execute(text("SELECT payload FROM option_recommendations WHERE id=:i"), {"i": str(result.recommendation_id)}).scalar()
        assert payload["cycle"]["source"] == "dhan"
        assert payload["cycle"]["underlying"] == "NIFTY"
        assert payload["cycle"]["observed_at"] == NOW.isoformat()
        assert payload["cycle"]["total_quotes_in_snapshot"] == 6
        assert payload["candidate_config"]["risk_free_rate"] == "0.065"
        assert payload["decision_thresholds"] is not None
        assert len(payload["per_shape_results"]) == 3
        assert {e["shape"] for e in payload["per_shape_results"]} == {"LONG_CALL_ATM", "LONG_PUT_ATM", "LONG_STRADDLE_ATM"}
        assert payload["selected_recommendation"]["status"] == result.status
        for entry in payload["per_shape_results"]:
            if entry["candidate"] is not None:
                assert entry["candidate"]["structure"]["legs"][0]["entry_price_source"] == "LIVE_QUOTE"


class TestImmutabilityStillIntact:
    def test_persisted_recommendation_cannot_be_updated(self, db_session):
        from sqlalchemy.exc import DBAPIError
        source = SyntheticLiveChainSource(factory=good_snapshot)
        result = run_one_cycle(db=db_session, config=mkconfig(source), now=NOW)
        with pytest.raises(DBAPIError, match="immutable"):
            with db_session.begin_nested():
                db_session.execute(text("UPDATE option_recommendations SET status='TRADE' WHERE id=:i"), {"i": str(result.recommendation_id)})
