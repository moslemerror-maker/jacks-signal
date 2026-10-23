from __future__ import annotations

import datetime as dt

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from backend.app.options.session_scheduler import (
    NOT_TRADING_DAY_CATEGORY, check_db_reachable, decide_session_start, determine_session_status,
    latest_session_state, log_not_trading_day, log_session_state, session_status_snapshot,
    todays_session_state_events, windows_safe_stop_flag,
)

D = dt.date
UTC = dt.timezone.utc
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
TODAY = D(2026, 9, 24)
NOON_IST = dt.datetime.combine(TODAY, dt.time(12, 0), tzinfo=IST).astimezone(UTC)


def _events(db, session_date=TODAY, now=NOON_IST):
    return todays_session_state_events(db, session_date=session_date, now=now)


class _BrokenSession:
    def execute(self, *a, **kw):
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))


class TestLogSessionState:
    def test_writes_a_correctly_categorized_row(self, db_session):
        log_session_state(db_session, state="LIVE", detail="polling started", event_timestamp=NOON_IST)
        db_session.flush()
        row = db_session.execute(text("SELECT category, severity, detail FROM option_session_events ORDER BY recorded_at DESC LIMIT 1")).mappings().one()
        assert row["category"] == "SESSION_LIVE"
        assert row["severity"] == "INFO"
        assert row["detail"] == "polling started"

    def test_rejects_an_unknown_state(self, db_session):
        import pytest
        with pytest.raises(ValueError, match="state must be one of"):
            log_session_state(db_session, state="BOGUS", detail="x", event_timestamp=NOON_IST)  # type: ignore[arg-type]


class TestLogNotTradingDay:
    def test_writes_the_dedicated_category_never_a_session_state(self, db_session):
        log_not_trading_day(db_session, detail="Saturday", event_timestamp=NOON_IST)
        db_session.flush()
        row = db_session.execute(text("SELECT category FROM option_session_events ORDER BY recorded_at DESC LIMIT 1")).mappings().one()
        assert row["category"] == NOT_TRADING_DAY_CATEGORY

    def test_is_excluded_from_todays_session_state_events(self, db_session):
        log_not_trading_day(db_session, detail="Sunday", event_timestamp=NOON_IST)
        db_session.flush()
        assert _events(db_session) == []
        assert determine_session_status(_events(db_session)) == "NOT_STARTED"


class TestCheckDbReachable:
    def test_true_for_a_real_session(self, db_session):
        assert check_db_reachable(db_session) is True

    def test_false_when_the_query_raises(self):
        assert check_db_reachable(_BrokenSession()) is False


class TestTodaysSessionStateEvents:
    def test_empty_when_nothing_recorded(self, db_session):
        assert _events(db_session) == []

    def test_only_session_category_rows_are_returned_not_ordinary_cycle_events(self, db_session):
        from backend.app.options.live_session import log_session_event
        log_session_event(db_session, event_timestamp=NOON_IST, severity="ERROR", category="PROVIDER_FETCH_FAILURE", detail="x")
        log_session_state(db_session, state="LIVE", detail="y", event_timestamp=NOON_IST)
        db_session.flush()
        events = _events(db_session)
        assert len(events) == 1
        assert events[0]["category"] == "SESSION_LIVE"

    def test_excludes_events_outside_todays_ist_window(self, db_session):
        yesterday_utc = NOON_IST - dt.timedelta(days=1)
        log_session_state(db_session, state="COMPLETED", detail="yesterday", event_timestamp=yesterday_utc)
        db_session.flush()
        assert _events(db_session) == []

    def test_returns_events_in_chronological_order(self, db_session):
        t1, t2 = NOON_IST, NOON_IST + dt.timedelta(minutes=5)
        log_session_state(db_session, state="STARTING", detail="1", event_timestamp=t1)
        log_session_state(db_session, state="LIVE", detail="2", event_timestamp=t2)
        db_session.flush()
        events = _events(db_session, now=t2 + dt.timedelta(minutes=1))
        assert [e["category"] for e in events] == ["SESSION_STARTING", "SESSION_LIVE"]


class TestDetermineSessionStatus:
    def test_no_events_is_not_started(self):
        assert determine_session_status([]) == "NOT_STARTED"

    def test_starting_only_is_in_progress(self):
        assert determine_session_status([{"category": "SESSION_STARTING"}]) == "IN_PROGRESS"

    def test_live_is_in_progress(self):
        assert determine_session_status([{"category": "SESSION_STARTING"}, {"category": "SESSION_LIVE"}]) == "IN_PROGRESS"

    def test_degraded_is_in_progress(self):
        assert determine_session_status([{"category": "SESSION_LIVE"}, {"category": "SESSION_DEGRADED"}]) == "IN_PROGRESS"

    def test_paused_is_in_progress(self):
        assert determine_session_status([{"category": "SESSION_LIVE"}, {"category": "SESSION_PAUSED"}]) == "IN_PROGRESS"

    def test_completed_is_terminal(self):
        assert determine_session_status([{"category": "SESSION_LIVE"}, {"category": "SESSION_COMPLETED"}]) == "COMPLETED"

    def test_failed_is_terminal(self):
        assert determine_session_status([{"category": "SESSION_STARTING"}, {"category": "SESSION_FAILED"}]) == "FAILED"

    def test_only_the_last_event_decides_status(self):
        events = [{"category": "SESSION_LIVE"}, {"category": "SESSION_DEGRADED"}, {"category": "SESSION_LIVE"}]
        assert determine_session_status(events) == "IN_PROGRESS"


class TestDecideSessionStart:
    def test_no_prior_session_starts_fresh(self, db_session):
        decision = decide_session_start(db_session, session_date=TODAY, now=NOON_IST)
        assert decision.action == "START"
        assert decision.status_before == "NOT_STARTED"

    def test_in_progress_session_resumes(self, db_session):
        log_session_state(db_session, state="LIVE", detail="already running", event_timestamp=NOON_IST)
        db_session.flush()
        decision = decide_session_start(db_session, session_date=TODAY, now=NOON_IST + dt.timedelta(minutes=1))
        assert decision.action == "RESUME"
        assert decision.status_before == "IN_PROGRESS"

    def test_completed_session_refuses_a_second_start(self, db_session):
        log_session_state(db_session, state="LIVE", detail="ran", event_timestamp=NOON_IST)
        log_session_state(db_session, state="COMPLETED", detail="done", event_timestamp=NOON_IST + dt.timedelta(hours=1))
        db_session.flush()
        decision = decide_session_start(db_session, session_date=TODAY, now=NOON_IST + dt.timedelta(hours=2))
        assert decision.action == "REFUSE_ALREADY_COMPLETED"

    def test_failed_session_refuses_a_second_start_without_force(self, db_session):
        log_session_state(db_session, state="FAILED", detail="preflight failed", event_timestamp=NOON_IST)
        db_session.flush()
        decision = decide_session_start(db_session, session_date=TODAY, now=NOON_IST + dt.timedelta(minutes=1))
        assert decision.action == "REFUSE_ALREADY_FAILED"

    def test_force_always_starts_even_after_completed(self, db_session):
        log_session_state(db_session, state="COMPLETED", detail="done", event_timestamp=NOON_IST)
        db_session.flush()
        decision = decide_session_start(db_session, session_date=TODAY, now=NOON_IST + dt.timedelta(minutes=1), force=True)
        assert decision.action == "START"

    def test_force_always_starts_even_after_failed(self, db_session):
        log_session_state(db_session, state="FAILED", detail="broken", event_timestamp=NOON_IST)
        db_session.flush()
        decision = decide_session_start(db_session, session_date=TODAY, now=NOON_IST + dt.timedelta(minutes=1), force=True)
        assert decision.action == "START"


class TestWindowsSafeStopFlag:

    def test_false_when_neither_signal_nor_file_is_set(self, tmp_path):
        stop_flag = windows_safe_stop_flag(signal_state={"requested": False}, stop_file=tmp_path / "stop")
        assert stop_flag() is False

    def test_true_when_the_signal_flag_is_set(self, tmp_path):
        stop_flag = windows_safe_stop_flag(signal_state={"requested": True}, stop_file=tmp_path / "stop")
        assert stop_flag() is True

    def test_true_when_the_stop_file_exists_even_with_no_signal_at_all(self, tmp_path):
        stop_file = tmp_path / "stop"
        stop_file.write_text("")
        stop_flag = windows_safe_stop_flag(signal_state={"requested": False}, stop_file=stop_file)
        assert stop_flag() is True

    def test_stop_file_created_after_construction_is_still_detected(self, tmp_path):
        stop_file = tmp_path / "stop"
        stop_flag = windows_safe_stop_flag(signal_state={"requested": False}, stop_file=stop_file)
        assert stop_flag() is False
        stop_file.write_text("")
        assert stop_flag() is True

    def test_end_to_end_with_a_real_run_session_loop(self, db_session, tmp_path):
        import datetime as _dt
        from decimal import Decimal
        from backend.app.options.candidate_generator import CandidateGenerationConfig
        from backend.app.options.contracts import ContractIdentity
        from backend.app.options.decision_policy import DecisionPolicyThresholds
        from backend.app.options.live_quote import LiveQuote
        from backend.app.options.live_session import ChainSnapshot, OrchestratorConfig, RunnerConfig, SyntheticLiveChainSource, run_session

        UTC = _dt.timezone.utc
        FIXTURE = "DEVELOPMENT_FIXTURE"
        EXPIRY = _dt.date(2026, 10, 6)
        NOW = _dt.datetime(2026, 9, 24, 5, 0, tzinfo=UTC)

        def opt(strike, cp):
            return ContractIdentity("NIFTY", "IDO", EXPIRY, strike * 100, cp)

        def q(strike, cp, last, bid, ask):
            return LiveQuote.from_raw(contract=opt(strike, cp), observed_at=NOW, last_price=Decimal(last), bid=Decimal(bid),
                                      ask=Decimal(ask), volume=500, open_interest=1000, underlying_last_price=Decimal("24500"),
                                      provider_reported_iv=None, provider_reported_greeks=None, source="dhan",
                                      source_row_hash=None, provenance_class=FIXTURE)

        source = SyntheticLiveChainSource(factory=lambda as_of: ChainSnapshot(
            source="dhan", underlying="NIFTY", observed_at=as_of, expiry_list_dates=(EXPIRY,),
            quotes=(q(24500, "CE", "140", "138", "142"), q(24500, "PE", "135", "133", "137")), raw_response_sha256=None))
        config = OrchestratorConfig(
            mode="SYNTHETIC", underlying="NIFTY", source=source,
            candidate_config=CandidateGenerationConfig(lots=1, lot_size=75, lot_size_source="CALLER_SUPPLIED", risk_free_rate=Decimal("0.065"),
                                                       max_quote_age=_dt.timedelta(minutes=15), max_spread_fraction=Decimal("0.10"), min_volume=None),
            decision_thresholds=DecisionPolicyThresholds(min_probability_positive_net_pnl=Decimal("0.5"), require_verified_forward=False),
            model_version="test-v0", strategy_version="test-v0",
        )

        stop_file = tmp_path / "stop"
        signal_state = {"requested": False}
        stop_flag = windows_safe_stop_flag(signal_state=signal_state, stop_file=stop_file)

        calls = {"n": 0}
        def sleep_and_create_stop_file(_seconds):
            calls["n"] += 1
            stop_file.write_text("")

        results = run_session(session_factory=lambda: db_session, config=config, runner_config=RunnerConfig(
            polling_interval=_dt.timedelta(seconds=1), max_cycles=None), now_fn=lambda: NOW, sleep_fn=sleep_and_create_stop_file, stop_flag=stop_flag)

        assert len(results) == 1
        assert calls["n"] == 1


def _insert_recommendation(db, *, decision_timestamp, status="WAIT", underlying="NIFTY", source="dhan", data_quality="EOD_RESOLVED"):
    import uuid
    rec_id = uuid.uuid4()
    db.execute(text(
        "INSERT INTO option_recommendations (id, decision_timestamp, underlying, status, data_quality, source, payload) "
        "VALUES (:id, :dt, :u, :status, :dq, :src, CAST('{}' AS JSONB))"
    ), {"id": rec_id, "dt": decision_timestamp, "u": underlying, "status": status, "dq": data_quality, "src": source})
    return rec_id


class TestLatestSessionState:
    def test_no_events_is_not_started(self):
        assert latest_session_state([]) == "NOT_STARTED"

    def test_starting_stays_starting_not_coarsened(self):
        assert latest_session_state([{"category": "SESSION_STARTING"}]) == "STARTING"

    def test_live_stays_live(self):
        assert latest_session_state([{"category": "SESSION_STARTING"}, {"category": "SESSION_LIVE"}]) == "LIVE"

    def test_degraded_stays_degraded(self):
        assert latest_session_state([{"category": "SESSION_LIVE"}, {"category": "SESSION_DEGRADED"}]) == "DEGRADED"

    def test_paused_stays_paused(self):
        assert latest_session_state([{"category": "SESSION_LIVE"}, {"category": "SESSION_PAUSED"}]) == "PAUSED"

    def test_completed_is_completed(self):
        assert latest_session_state([{"category": "SESSION_LIVE"}, {"category": "SESSION_COMPLETED"}]) == "COMPLETED"

    def test_failed_is_failed(self):
        assert latest_session_state([{"category": "SESSION_STARTING"}, {"category": "SESSION_FAILED"}]) == "FAILED"

    def test_unparseable_category_is_unknown_defensively(self):
        assert latest_session_state([{"category": "SESSION_NOT_A_REAL_STATE"}]) == "UNKNOWN"


class TestSessionStatusSnapshot:

    def test_no_session_today_and_no_recommendation_ever_is_not_started_with_nulls(self, db_session):
        snap = session_status_snapshot(db_session, underlying="NIFTY", session_date=TODAY, now=NOON_IST)
        assert snap.session_state == "NOT_STARTED"
        assert snap.cycle_count == 0
        assert snap.session_start_timestamp is None
        assert snap.latest_recommendation_timestamp is None
        assert snap.latest_recommendation_status is None
        assert snap.latest_recommendation_id is None
        assert snap.data_quality is None
        assert snap.source is None

    def test_starting_event_is_reported_as_starting(self, db_session):
        log_session_state(db_session, state="STARTING", detail="automatic daily session starting", event_timestamp=NOON_IST)
        db_session.flush()
        snap = session_status_snapshot(db_session, underlying="NIFTY", session_date=TODAY, now=NOON_IST + dt.timedelta(seconds=1))
        assert snap.session_state == "STARTING"
        assert snap.session_start_timestamp == NOON_IST

    def test_live_event_is_reported_as_live(self, db_session):
        log_session_state(db_session, state="STARTING", detail="starting", event_timestamp=NOON_IST)
        log_session_state(db_session, state="LIVE", detail="polling started", event_timestamp=NOON_IST + dt.timedelta(seconds=3))
        db_session.flush()
        snap = session_status_snapshot(db_session, underlying="NIFTY", session_date=TODAY, now=NOON_IST + dt.timedelta(minutes=1))
        assert snap.session_state == "LIVE"
        assert snap.session_start_timestamp == NOON_IST
        assert snap.completion_timestamp is None and snap.failure_timestamp is None

    def test_completed_event_is_reported_as_completed_with_timestamp(self, db_session):
        t_start, t_done = NOON_IST, NOON_IST + dt.timedelta(hours=6)
        log_session_state(db_session, state="STARTING", detail="starting", event_timestamp=t_start)
        log_session_state(db_session, state="COMPLETED", detail="cycles=15 status_counts={'WAIT': 15}", event_timestamp=t_done)
        db_session.flush()
        snap = session_status_snapshot(db_session, underlying="NIFTY", session_date=TODAY, now=t_done + dt.timedelta(minutes=1))
        assert snap.session_state == "COMPLETED"
        assert snap.completion_timestamp == t_done
        assert snap.failure_timestamp is None and snap.failure_detail is None

    def test_failed_event_is_reported_as_failed_with_timestamp_and_safe_detail(self, db_session):
        t_fail = NOON_IST
        log_session_state(db_session, state="FAILED", detail="credential preflight = MISSING: DHAN_ACCESS_TOKEN missing",
                          event_timestamp=t_fail, severity="ERROR")
        db_session.flush()
        snap = session_status_snapshot(db_session, underlying="NIFTY", session_date=TODAY, now=t_fail + dt.timedelta(minutes=1))
        assert snap.session_state == "FAILED"
        assert snap.failure_timestamp == t_fail
        assert "DHAN_ACCESS_TOKEN missing" in snap.failure_detail
        assert snap.completion_timestamp is None

    def test_todays_failed_session_with_yesterdays_recommendation_is_failed_not_live_not_no_trade(self, db_session):
        yesterday = NOON_IST - dt.timedelta(days=1)
        _insert_recommendation(db_session, decision_timestamp=yesterday, status="NO_TRADE")
        log_session_state(db_session, state="FAILED", detail="startup failure during 'live source / config construction': ModuleNotFoundError",
                          event_timestamp=NOON_IST, severity="ERROR")
        db_session.flush()
        snap = session_status_snapshot(db_session, underlying="NIFTY", session_date=TODAY, now=NOON_IST + dt.timedelta(minutes=1))
        assert snap.session_state == "FAILED"
        assert snap.session_state != "LIVE"
        assert snap.latest_recommendation_status == "NO_TRADE"
        assert snap.latest_recommendation_timestamp == yesterday

    def test_yesterdays_recommendation_alone_does_not_make_today_live(self, db_session):
        yesterday = NOON_IST - dt.timedelta(days=1)
        _insert_recommendation(db_session, decision_timestamp=yesterday, status="WAIT")
        snap = session_status_snapshot(db_session, underlying="NIFTY", session_date=TODAY, now=NOON_IST)
        assert snap.session_state == "NOT_STARTED"
        assert snap.latest_recommendation_status == "WAIT"
        assert snap.latest_recommendation_timestamp == yesterday

    def test_todays_wait_recommendation_is_surfaced_as_latest(self, db_session):
        log_session_state(db_session, state="LIVE", detail="polling started", event_timestamp=NOON_IST)
        _insert_recommendation(db_session, decision_timestamp=NOON_IST + dt.timedelta(minutes=5), status="WAIT")
        db_session.flush()
        snap = session_status_snapshot(db_session, underlying="NIFTY", session_date=TODAY, now=NOON_IST + dt.timedelta(minutes=6))
        assert snap.latest_recommendation_status == "WAIT"
        assert snap.session_state == "LIVE"

    def test_todays_no_trade_recommendation_is_surfaced_as_latest(self, db_session):
        log_session_state(db_session, state="LIVE", detail="polling started", event_timestamp=NOON_IST)
        _insert_recommendation(db_session, decision_timestamp=NOON_IST + dt.timedelta(minutes=5), status="NO_TRADE")
        db_session.flush()
        snap = session_status_snapshot(db_session, underlying="NIFTY", session_date=TODAY, now=NOON_IST + dt.timedelta(minutes=6))
        assert snap.latest_recommendation_status == "NO_TRADE"

    def test_cycle_count_reflects_only_todays_recommendations(self, db_session):
        yesterday = NOON_IST - dt.timedelta(days=1)
        _insert_recommendation(db_session, decision_timestamp=yesterday, status="WAIT")
        _insert_recommendation(db_session, decision_timestamp=NOON_IST, status="WAIT")
        _insert_recommendation(db_session, decision_timestamp=NOON_IST + dt.timedelta(minutes=5), status="NO_TRADE")
        db_session.flush()
        snap = session_status_snapshot(db_session, underlying="NIFTY", session_date=TODAY, now=NOON_IST + dt.timedelta(minutes=6))
        assert snap.cycle_count == 2

    def test_never_returns_the_database_url_or_any_credential_in_failure_detail(self, db_session):
        log_session_state(db_session, state="FAILED", detail="database not reachable (check_db_reachable returned False)",
                          event_timestamp=NOON_IST, severity="ERROR")
        db_session.flush()
        snap = session_status_snapshot(db_session, underlying="NIFTY", session_date=TODAY, now=NOON_IST + dt.timedelta(minutes=1))
        for forbidden in ("access_token", "access-token", "DHAN_ACCESS_TOKEN=", "password", "dhanClientId"):
            assert forbidden not in (snap.failure_detail or "")
