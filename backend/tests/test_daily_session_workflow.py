from __future__ import annotations

import datetime as dt
from decimal import Decimal

from sqlalchemy import text

from backend.app.api.queries import latest_recommendation
from backend.app.options import eod_report
from backend.app.options.candidate_generator import CandidateGenerationConfig
from backend.app.options.contracts import ContractIdentity
from backend.app.options.decision_policy import DecisionPolicyThresholds
from backend.app.options.live_quote import LiveQuote
from backend.app.options.live_session import ChainSnapshot, OrchestratorConfig, RunnerConfig, SyntheticLiveChainSource
from backend.app.options.session_scheduler import log_session_state, run_daily_session_workflow, todays_session_state_events

D = dt.date
UTC = dt.timezone.utc
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
FIXTURE = "DEVELOPMENT_FIXTURE"
EXPIRY = D(2026, 10, 6)

TRADING_DAY = D(2026, 9, 24)
WEEKEND_DAY = D(2026, 9, 26)
HOLIDAY_DAY = D(2026, 9, 25)
HOLIDAYS = frozenset({"25-Sep-2026"})

SESSION_START = dt.datetime(2026, 9, 24, 3, 45, tzinfo=UTC)


def opt(strike: int, cp: str, expiry: dt.date = EXPIRY) -> ContractIdentity:
    return ContractIdentity("NIFTY", "IDO", expiry, strike * 100, cp)


def q(strike, cp, last, bid, ask, *, observed_at) -> LiveQuote:
    return LiveQuote.from_raw(
        contract=opt(strike, cp), observed_at=observed_at, last_price=Decimal(last), bid=Decimal(bid), ask=Decimal(ask),
        volume=500, open_interest=1000, underlying_last_price=Decimal("24500"), provider_reported_iv=None,
        provider_reported_greeks=None, source="dhan", source_row_hash=None, provenance_class=FIXTURE,
    )


def synthetic_source() -> SyntheticLiveChainSource:
    def factory(as_of: dt.datetime) -> ChainSnapshot:
        quotes = (q(24500, "CE", "140", "138", "142", observed_at=as_of), q(24500, "PE", "135", "133", "137", observed_at=as_of))
        return ChainSnapshot(source="dhan", underlying="NIFTY", observed_at=as_of, expiry_list_dates=(EXPIRY,), quotes=quotes, raw_response_sha256=None)
    return SyntheticLiveChainSource(factory=factory)


def mkconfig(source) -> OrchestratorConfig:
    return OrchestratorConfig(
        mode="LIVE_PAPER", underlying="NIFTY", source=source,
        candidate_config=CandidateGenerationConfig(lots=1, lot_size=75, lot_size_source="CALLER_SUPPLIED",
                                                   risk_free_rate=Decimal("0.065"), max_quote_age=dt.timedelta(minutes=15),
                                                   max_spread_fraction=Decimal("0.10"), min_volume=None),
        decision_thresholds=DecisionPolicyThresholds(min_probability_positive_net_pnl=Decimal("0.55"), require_verified_forward=False),
        model_version="stage-b-test-v0", strategy_version="stage-b-test-v0",
    )


def _monotonic_clock(start: dt.datetime, step: dt.timedelta):
    state = {"n": 0}
    def clock() -> dt.datetime:
        t = start + state["n"] * step
        state["n"] += 1
        return t
    return clock


def count_recommendations(db) -> int:
    return db.execute(text("SELECT COUNT(*) FROM option_recommendations")).scalar()


def session_categories(db, session_date=TRADING_DAY, now=SESSION_START + dt.timedelta(hours=6)) -> list[str]:
    return [e["category"] for e in todays_session_state_events(db, session_date=session_date, now=now)]


class TestNotATradingDay:
    def test_weekend_produces_no_session_and_no_recommendation(self, db_session):
        result = run_daily_session_workflow(
            db=db_session, session_factory=lambda: db_session, today=WEEKEND_DAY, holiday_dates=HOLIDAYS,
            config=mkconfig(synthetic_source()), runner_config=RunnerConfig(), now_fn=lambda: SESSION_START, sleep_fn=lambda s: None,
        )
        assert result.outcome == "NOT_TRADING_DAY"
        assert count_recommendations(db_session) == 0

    def test_fo_holiday_produces_no_session_and_no_recommendation(self, db_session):
        result = run_daily_session_workflow(
            db=db_session, session_factory=lambda: db_session, today=HOLIDAY_DAY, holiday_dates=HOLIDAYS,
            config=mkconfig(synthetic_source()), runner_config=RunnerConfig(), now_fn=lambda: SESSION_START, sleep_fn=lambda s: None,
        )
        assert result.outcome == "NOT_TRADING_DAY"
        assert count_recommendations(db_session) == 0


class TestTradingDaySession:
    def test_multiple_cycles_persist_and_complete(self, db_session):
        clock = _monotonic_clock(SESSION_START, dt.timedelta(minutes=5))
        stop_at = SESSION_START + dt.timedelta(minutes=17)
        result = run_daily_session_workflow(
            db=db_session, session_factory=lambda: db_session, today=TRADING_DAY, holiday_dates=HOLIDAYS,
            config=mkconfig(synthetic_source()), runner_config=RunnerConfig(polling_interval=dt.timedelta(seconds=1), stop_at=stop_at),
            now_fn=clock, sleep_fn=lambda s: None,
        )
        assert result.outcome == "COMPLETED"
        assert len(result.cycle_results) >= 1
        assert all(r.outcome == "PERSISTED" for r in result.cycle_results)
        assert count_recommendations(db_session) == len(result.cycle_results)

    def test_every_cycle_is_wait_or_no_trade_never_trade(self, db_session):
        clock = _monotonic_clock(SESSION_START, dt.timedelta(minutes=5))
        stop_at = SESSION_START + dt.timedelta(minutes=12)
        result = run_daily_session_workflow(
            db=db_session, session_factory=lambda: db_session, today=TRADING_DAY, holiday_dates=HOLIDAYS,
            config=mkconfig(synthetic_source()), runner_config=RunnerConfig(polling_interval=dt.timedelta(seconds=1), stop_at=stop_at),
            now_fn=clock, sleep_fn=lambda s: None,
        )
        assert len(result.cycle_results) >= 1
        assert all(r.status in ("WAIT", "NO_TRADE") for r in result.cycle_results)

    def test_session_state_events_are_starting_live_completed_in_order(self, db_session):
        clock = _monotonic_clock(SESSION_START, dt.timedelta(minutes=5))
        stop_at = SESSION_START + dt.timedelta(minutes=7)
        run_daily_session_workflow(
            db=db_session, session_factory=lambda: db_session, today=TRADING_DAY, holiday_dates=HOLIDAYS,
            config=mkconfig(synthetic_source()), runner_config=RunnerConfig(polling_interval=dt.timedelta(seconds=1), stop_at=stop_at),
            now_fn=clock, sleep_fn=lambda s: None,
        )
        cats = session_categories(db_session)
        assert cats[0] == "SESSION_STARTING"
        assert cats[1] == "SESSION_LIVE"
        assert cats[-1] == "SESSION_COMPLETED"

    def test_stop_at_is_honoured_through_the_workflow(self, db_session):
        clock = _monotonic_clock(SESSION_START, dt.timedelta(minutes=1))
        stop_at = SESSION_START + dt.timedelta(minutes=30)
        result = run_daily_session_workflow(
            db=db_session, session_factory=lambda: db_session, today=TRADING_DAY, holiday_dates=HOLIDAYS,
            config=mkconfig(synthetic_source()), runner_config=RunnerConfig(polling_interval=dt.timedelta(seconds=1), stop_at=stop_at),
            now_fn=clock, sleep_fn=lambda s: None,
        )
        assert result.outcome == "COMPLETED"
        assert 1 <= len(result.cycle_results) <= 30


class TestPersistenceReadableViaApi:
    def test_the_latest_persisted_recommendation_is_readable_through_the_same_query_the_dashboard_uses(self, db_session):
        clock = _monotonic_clock(SESSION_START, dt.timedelta(minutes=1))
        stop_at = SESSION_START + dt.timedelta(minutes=15)
        run_daily_session_workflow(
            db=db_session, session_factory=lambda: db_session, today=TRADING_DAY, holiday_dates=HOLIDAYS,
            config=mkconfig(synthetic_source()), runner_config=RunnerConfig(polling_interval=dt.timedelta(seconds=1), stop_at=stop_at),
            now_fn=clock, sleep_fn=lambda s: None,
        )
        snapshot = latest_recommendation(db_session, underlying="NIFTY")
        assert snapshot is not None
        assert snapshot["metadata"]["underlying"] == "NIFTY"
        assert snapshot["metadata"]["status"] in ("WAIT", "NO_TRADE")
        assert snapshot["probability"]["predictive_probability_status"] == "NOT_ESTABLISHED"


class TestRestartResume:
    def test_a_live_but_not_yet_completed_session_resumes_without_a_second_starting_event(self, db_session):
        log_session_state(db_session, state="STARTING", detail="prior process starting", event_timestamp=SESSION_START)
        log_session_state(db_session, state="LIVE", detail="prior process polling", event_timestamp=SESSION_START)
        db_session.commit()

        clock = _monotonic_clock(SESSION_START + dt.timedelta(minutes=10), dt.timedelta(minutes=5))
        stop_at = SESSION_START + dt.timedelta(minutes=20)
        result = run_daily_session_workflow(
            db=db_session, session_factory=lambda: db_session, today=TRADING_DAY, holiday_dates=HOLIDAYS,
            config=mkconfig(synthetic_source()), runner_config=RunnerConfig(polling_interval=dt.timedelta(seconds=1), stop_at=stop_at),
            now_fn=clock, sleep_fn=lambda s: None,
        )
        assert result.outcome == "COMPLETED"
        cats = session_categories(db_session)
        assert cats.count("SESSION_STARTING") == 1
        assert cats.count("SESSION_LIVE") == 2
        assert cats[-1] == "SESSION_COMPLETED"

    def test_resume_does_not_duplicate_recommendations_from_the_prior_partial_run(self, db_session):
        clock = _monotonic_clock(SESSION_START, dt.timedelta(minutes=1))
        stop_at_first = SESSION_START + dt.timedelta(minutes=15)
        first = run_daily_session_workflow(
            db=db_session, session_factory=lambda: db_session, today=TRADING_DAY, holiday_dates=HOLIDAYS,
            config=mkconfig(synthetic_source()), runner_config=RunnerConfig(polling_interval=dt.timedelta(seconds=1), stop_at=stop_at_first),
            now_fn=clock, sleep_fn=lambda s: None,
        )
        assert first.outcome == "COMPLETED"
        after_first = count_recommendations(db_session)
        assert after_first >= 1

        clock2 = _monotonic_clock(SESSION_START, dt.timedelta(minutes=1))
        second = run_daily_session_workflow(
            db=db_session, session_factory=lambda: db_session, today=TRADING_DAY, holiday_dates=HOLIDAYS,
            config=mkconfig(synthetic_source()), runner_config=RunnerConfig(polling_interval=dt.timedelta(seconds=1), stop_at=stop_at_first),
            now_fn=clock2, sleep_fn=lambda s: None, force=True,
        )
        assert second.outcome == "COMPLETED"
        assert count_recommendations(db_session) == after_first


class TestDuplicateInvocation:
    def test_a_second_invocation_after_completion_refuses(self, db_session):
        clock = _monotonic_clock(SESSION_START, dt.timedelta(minutes=1))
        stop_at = SESSION_START + dt.timedelta(minutes=15)
        first = run_daily_session_workflow(
            db=db_session, session_factory=lambda: db_session, today=TRADING_DAY, holiday_dates=HOLIDAYS,
            config=mkconfig(synthetic_source()), runner_config=RunnerConfig(polling_interval=dt.timedelta(seconds=1), stop_at=stop_at),
            now_fn=clock, sleep_fn=lambda s: None,
        )
        assert first.outcome == "COMPLETED"
        count_after_first = count_recommendations(db_session)

        second = run_daily_session_workflow(
            db=db_session, session_factory=lambda: db_session, today=TRADING_DAY, holiday_dates=HOLIDAYS,
            config=mkconfig(synthetic_source()), runner_config=RunnerConfig(polling_interval=dt.timedelta(seconds=1), stop_at=stop_at),
            now_fn=lambda: SESSION_START + dt.timedelta(hours=3), sleep_fn=lambda s: None,
        )
        assert second.outcome == "REFUSE_ALREADY_COMPLETED"
        assert count_recommendations(db_session) == count_after_first


class TestFailureHandling:
    def test_an_exception_during_run_session_logs_failed_and_creates_no_recommendation(self, db_session):
        class RaisingSessionFactory:
            def __call__(self):
                raise RuntimeError("simulated catastrophic failure (e.g. DB pool exhausted)")

        result = run_daily_session_workflow(
            db=db_session, session_factory=RaisingSessionFactory(), today=TRADING_DAY, holiday_dates=HOLIDAYS,
            config=mkconfig(synthetic_source()), runner_config=RunnerConfig(polling_interval=dt.timedelta(seconds=1)),
            now_fn=lambda: SESSION_START, sleep_fn=lambda s: None,
        )
        assert result.outcome == "FAILED"
        assert count_recommendations(db_session) == 0
        assert session_categories(db_session)[-1] == "SESSION_FAILED"

    def test_consecutive_fetch_failure_streak_logs_failed_not_completed(self, db_session):

        def always_raises(as_of: dt.datetime):
            raise RuntimeError("expiry_list did not return status=success: {'status': 'failure', 'remarks': {'error_code': None, 'error_type': None, 'error_message': None}}")

        source = SyntheticLiveChainSource(factory=always_raises)
        clock = _monotonic_clock(SESSION_START, dt.timedelta(minutes=5))
        stop_at = SESSION_START + dt.timedelta(hours=6)
        result = run_daily_session_workflow(
            db=db_session, session_factory=lambda: db_session, today=TRADING_DAY, holiday_dates=HOLIDAYS,
            config=mkconfig(source),
            runner_config=RunnerConfig(polling_interval=dt.timedelta(seconds=1), stop_at=stop_at, max_consecutive_fetch_failures=6),
            now_fn=clock, sleep_fn=lambda s: None,
        )
        assert result.outcome == "FAILED"
        assert len(result.cycle_results) == 6
        assert all(r.outcome == "FETCH_FAILURE" for r in result.cycle_results)
        assert count_recommendations(db_session) == 0
        assert session_categories(db_session)[-1] == "SESSION_FAILED"


class TestEodReportReadsAWorkflowRunSession:
    def test_session_run_through_the_workflow_is_correctly_summarized(self, db_session):
        clock = _monotonic_clock(SESSION_START, dt.timedelta(minutes=5))
        stop_at = SESSION_START + dt.timedelta(minutes=12)
        result = run_daily_session_workflow(
            db=db_session, session_factory=lambda: db_session, today=TRADING_DAY, holiday_dates=HOLIDAYS,
            config=mkconfig(synthetic_source()), runner_config=RunnerConfig(polling_interval=dt.timedelta(seconds=1), stop_at=stop_at),
            now_fn=clock, sleep_fn=lambda s: None,
        )
        assert result.outcome == "COMPLETED"

        report = eod_report.build_eod_report(db_session, underlying="NIFTY", session_date=TRADING_DAY, generated_at=SESSION_START + dt.timedelta(hours=7))
        assert report.successful_cycles == len(result.cycle_results)
        assert report.predictive_probability_status == "NOT_ESTABLISHED"
        assert report.realized_pnl_status == "NOT_ESTABLISHED"
        assert report.paper_entry_count == 0
