from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal

import pytest
from sqlalchemy import text

from backend.app.options import decision_policy as dp
from backend.app.options.candidate_generator import CandidateGenerationConfig
from backend.app.options.contracts import ContractIdentity
from backend.app.options.decision_policy import DecisionPolicyThresholds
from backend.app.options.eod_report import build_eod_report, render_markdown, session_window_utc
from backend.app.options.live_quote import LiveQuote
from backend.app.options.live_session import ChainSnapshot, OrchestratorConfig, SyntheticLiveChainSource, run_one_cycle

D = dt.date
UTC = dt.timezone.utc
FIXTURE = "DEVELOPMENT_FIXTURE"
SESSION_DATE = D(2026, 9, 23)
EXPIRY = D(2026, 10, 6)
NOW = dt.datetime(2026, 9, 23, 8, 0, tzinfo=UTC)


def opt(strike, cp, expiry=EXPIRY):
    return ContractIdentity("NIFTY", "IDO", expiry, strike * 100, cp)


def fut(expiry=EXPIRY):
    return ContractIdentity("NIFTY", "IDF", expiry, None, None)


def q(strike, cp, last, bid, ask, *, underlying_last_price=Decimal("24500"), observed_at=NOW, expiry=EXPIRY):
    return LiveQuote.from_raw(
        contract=opt(strike, cp, expiry), observed_at=observed_at, last_price=Decimal(last), bid=Decimal(bid),
        ask=Decimal(ask), volume=500, open_interest=1000, underlying_last_price=underlying_last_price,
        provider_reported_iv=None, provider_reported_greeks=None, source="dhan", source_row_hash="eod-test",
        provenance_class=FIXTURE,
    )


def realistic_chain(observed_at=NOW):
    rows = [(24400, "115", "113", "117", "80", "78", "82"), (24500, "78", "76", "80", "108", "106", "110"),
            (24600, "50", "48", "52", "145", "143", "147")]
    quotes = []
    for strike, ce_l, ce_b, ce_a, pe_l, pe_b, pe_a in rows:
        quotes.append(q(strike, "CE", ce_l, ce_b, ce_a, observed_at=observed_at))
        quotes.append(q(strike, "PE", pe_l, pe_b, pe_a, observed_at=observed_at))
    return tuple(quotes)


def favorable_chain(observed_at=NOW):
    return (q(100, "CE", "910", "905", "915", underlying_last_price=Decimal("1000"), observed_at=observed_at),
            q(100, "PE", "5", "4", "6", underlying_last_price=Decimal("1000"), observed_at=observed_at))


def mkconfig(source, *, min_probability=Decimal("0.5"), require_verified_forward=False):
    return OrchestratorConfig(
        mode="SYNTHETIC", underlying="NIFTY", source=source,
        candidate_config=CandidateGenerationConfig(lots=1, lot_size=75, lot_size_source="CALLER_SUPPLIED",
                                                   risk_free_rate=Decimal("0.065"), max_quote_age=dt.timedelta(minutes=30),
                                                   max_spread_fraction=Decimal("0.20"), min_volume=None),
        decision_thresholds=DecisionPolicyThresholds(min_probability_positive_net_pnl=min_probability, require_verified_forward=require_verified_forward),
        model_version="eod-report-test-v0", strategy_version="eod-report-test-v0",
    )


def snapshot_source(quotes, expiry_list_dates=(EXPIRY,)):
    return SyntheticLiveChainSource(factory=lambda as_of: ChainSnapshot(
        source="dhan", underlying="NIFTY", observed_at=as_of, expiry_list_dates=tuple(expiry_list_dates),
        quotes=tuple(quotes), raw_response_sha256="eod" * 16))


def run_normal_cycle(db_session, *, now=NOW, quotes=None):
    source = snapshot_source(quotes if quotes is not None else realistic_chain())
    return run_one_cycle(db=db_session, config=mkconfig(source), now=now)


def run_trade_cycle(db_session, monkeypatch, *, now=NOW):
    monkeypatch.setattr(dp, "PREDICTIVE_PROBABILITY_STATUS", "ESTABLISHED")
    source = snapshot_source(favorable_chain(now))
    return run_one_cycle(db=db_session, config=mkconfig(source, min_probability=Decimal("0.01")), now=now)


def insert_exit_event(db_session, recommendation_id, *, gross_pnl="1500", net_pnl="1420", costs_total="80", exit_reason="TARGET_EXIT", event_timestamp=None):
    ts = event_timestamp or (NOW + dt.timedelta(hours=1))
    db_session.execute(text(
        "INSERT INTO option_paper_trade_events (id, recommendation_id, event_type, event_timestamp, payload) "
        "VALUES (:id, :rid, 'EXIT', :ts, CAST(:payload AS JSONB))"
    ), {"id": uuid.uuid4(), "rid": str(recommendation_id), "ts": ts,
        "payload": f'{{"gross_pnl": "{gross_pnl}", "net_pnl": "{net_pnl}", "costs_total": "{costs_total}", "exit_reason": "{exit_reason}"}}'})


def insert_session_event(db_session, *, category="PROVIDER_FETCH_FAILURE", severity="ERROR", event_timestamp=None):
    ts = event_timestamp or NOW
    db_session.execute(text(
        "INSERT INTO option_session_events (id, event_timestamp, severity, category, detail) VALUES (:id, :ts, :sev, :cat, 'synthetic test event')"
    ), {"id": uuid.uuid4(), "ts": ts, "sev": severity, "cat": category})


class TestEmptySession:
    def test_empty_session_does_not_crash_and_reports_zero_not_unavailable(self, db_session):
        report = build_eod_report(db_session, underlying="NIFTY", session_date=SESSION_DATE, generated_at=NOW)
        assert report.successful_cycles == 0 and report.failed_cycles == 0
        assert report.status_counts == {}
        assert report.predictive_probability_status == "NO_RECOMMENDATIONS_IN_WINDOW"
        assert report.realized_pnl_status == "NOT_ESTABLISHED"
        assert report.paper_entry_count == 0
        md = render_markdown(report)
        assert "AEVORYN EOD Paper Session Report" in md


class TestOnlyWaitOrNoTrade:
    def test_status_counts_reflect_wait_or_no_trade_zero_trade(self, db_session):
        result = run_normal_cycle(db_session)
        assert result.status != "TRADE"
        report = build_eod_report(db_session, underlying="NIFTY", session_date=SESSION_DATE, generated_at=NOW)
        assert report.status_counts.get("TRADE", 0) == 0
        assert report.status_counts.get(result.status, 0) == 1
        assert report.paper_entry_count == 0
        assert report.trade_recommendation_details == ()


class TestRecommendationsAndSessionEvents:
    def test_both_are_reflected_independently(self, db_session):
        run_normal_cycle(db_session)
        insert_session_event(db_session, category="STALE_QUOTE_WARNING", severity="WARNING")
        report = build_eod_report(db_session, underlying="NIFTY", session_date=SESSION_DATE, generated_at=NOW)
        assert report.successful_cycles == 1
        assert report.session_event_counts_by_category.get("STALE_QUOTE_WARNING") == 1


class TestPaperEntryPresent:
    def test_paper_entry_counted_and_traceable(self, db_session, monkeypatch):
        result = run_trade_cycle(db_session, monkeypatch)
        assert result.status == "TRADE"
        report = build_eod_report(db_session, underlying="NIFTY", session_date=SESSION_DATE, generated_at=NOW)
        assert report.paper_entry_count == 1
        assert report.open_position_count == 1
        assert len(report.trade_recommendation_details) == 1
        assert report.trade_recommendation_details[0].recommendation_id == str(result.recommendation_id)


class TestClosedPaperTradeWithValidOutcome:
    def test_realized_pnl_computed_from_a_recognized_exit_payload(self, db_session, monkeypatch):
        result = run_trade_cycle(db_session, monkeypatch)
        insert_exit_event(db_session, result.recommendation_id, gross_pnl="1500", net_pnl="1420", costs_total="80")
        report = build_eod_report(db_session, underlying="NIFTY", session_date=SESSION_DATE, generated_at=NOW)
        assert report.realized_pnl_status == "AVAILABLE"
        assert report.realized_pnl_summary["closed_trades"] == 1
        assert report.realized_pnl_summary["total_net_pnl"] == "1420"
        assert report.realized_pnl_summary["positive_outcomes"] == 1
        assert report.closed_position_count == 1 and report.open_position_count == 0

    def test_incomplete_exit_payload_excluded_from_pnl_but_counted_separately(self, db_session, monkeypatch):
        result = run_trade_cycle(db_session, monkeypatch)
        db_session.execute(text(
            "INSERT INTO option_paper_trade_events (id, recommendation_id, event_type, event_timestamp, payload) "
            "VALUES (:id, :rid, 'EXIT', :ts, CAST('{\"exit_reason\": \"AMBIGUOUS_UNAVAILABLE\"}' AS JSONB))"
        ), {"id": uuid.uuid4(), "rid": str(result.recommendation_id), "ts": NOW + dt.timedelta(hours=1)})
        report = build_eod_report(db_session, underlying="NIFTY", session_date=SESSION_DATE, generated_at=NOW)
        assert report.realized_pnl_status == "NOT_ESTABLISHED"
        assert report.incomplete_exit_count == 1
        assert report.exit_count == 1


class TestGrossOnlyExitNeverExcludedNeverFakeNet:
    def test_costs_not_modeled_exit_still_counts_toward_gross_pnl(self, db_session, monkeypatch):
        result = run_trade_cycle(db_session, monkeypatch)
        insert_exit_event(db_session, result.recommendation_id, gross_pnl="600", net_pnl="COSTS_NOT_MODELED", costs_total="0")
        report = build_eod_report(db_session, underlying="NIFTY", session_date=SESSION_DATE, generated_at=NOW)
        assert report.realized_pnl_status == "AVAILABLE"
        s = report.realized_pnl_summary
        assert s["closed_trades"] == 1
        assert s["total_gross_pnl"] == "600"
        assert s["positive_outcomes_gross"] == 1
        assert s["net_pnl_status"] == "COSTS_NOT_MODELED"
        assert s["trades_with_net_pnl"] == 0
        assert s["total_net_pnl"] is None
        assert s["positive_outcomes"] is None
        assert report.closed_position_count == 1

    def test_mixed_costs_modeled_and_not_modeled_exits_reported_as_partial(self, db_session, monkeypatch):
        monkeypatch.setattr(dp, "PREDICTIVE_PROBABILITY_STATUS", "ESTABLISHED")
        first = run_trade_cycle(db_session, monkeypatch, now=NOW)
        second = run_trade_cycle(db_session, monkeypatch, now=NOW + dt.timedelta(minutes=5))
        insert_exit_event(db_session, first.recommendation_id, gross_pnl="600", net_pnl="COSTS_NOT_MODELED", costs_total="0", event_timestamp=NOW + dt.timedelta(hours=1))
        insert_exit_event(db_session, second.recommendation_id, gross_pnl="400", net_pnl="380", costs_total="20", event_timestamp=NOW + dt.timedelta(hours=1))
        report = build_eod_report(db_session, underlying="NIFTY", session_date=SESSION_DATE, generated_at=NOW)
        s = report.realized_pnl_summary
        assert s["closed_trades"] == 2
        assert s["total_gross_pnl"] == "1000"
        assert s["net_pnl_status"] == "PARTIAL"
        assert s["trades_with_net_pnl"] == 1
        assert s["total_net_pnl"] is None

    def test_render_markdown_states_costs_not_modeled_honestly(self, db_session, monkeypatch):
        result = run_trade_cycle(db_session, monkeypatch)
        insert_exit_event(db_session, result.recommendation_id, gross_pnl="600", net_pnl="COSTS_NOT_MODELED", costs_total="0")
        report = build_eod_report(db_session, underlying="NIFTY", session_date=SESSION_DATE, generated_at=NOW)
        md = render_markdown(report)
        assert "COSTS_NOT_MODELED" in md
        assert "Total gross P&L: 600" in md
        assert "not modeled" in md.lower()


class TestOpenUnresolvedPaperTrade:
    def test_open_position_reported_pnl_remains_not_established(self, db_session, monkeypatch):
        run_trade_cycle(db_session, monkeypatch)
        report = build_eod_report(db_session, underlying="NIFTY", session_date=SESSION_DATE, generated_at=NOW)
        assert report.open_position_count == 1 and report.closed_position_count == 0
        assert report.realized_pnl_status == "NOT_ESTABLISHED"


class TestMixedSuccessfulAndFailedCycles:
    def test_counts_reflect_both(self, db_session):
        run_normal_cycle(db_session, now=NOW)
        run_normal_cycle(db_session, now=NOW + dt.timedelta(minutes=10))

        def raise_fetch(as_of):
            raise RuntimeError("simulated fetch failure")
        failing_source = SyntheticLiveChainSource(factory=raise_fetch)
        run_one_cycle(db=db_session, config=mkconfig(failing_source), now=NOW + dt.timedelta(minutes=20))

        report = build_eod_report(db_session, underlying="NIFTY", session_date=SESSION_DATE, generated_at=NOW)
        assert report.successful_cycles == 2
        assert report.failed_cycles == 1
        assert report.total_cycles_inferred == 3

    def test_every_classified_fetch_failure_category_is_counted(self, db_session):
        from backend.app.options.pipeline_reliability import FETCH_FAILURE_CATEGORIES
        run_normal_cycle(db_session, now=NOW)
        for i, category in enumerate(FETCH_FAILURE_CATEGORIES):
            insert_session_event(db_session, category=category, event_timestamp=NOW + dt.timedelta(minutes=10 * (i + 1)))
        report = build_eod_report(db_session, underlying="NIFTY", session_date=SESSION_DATE, generated_at=NOW)
        assert report.successful_cycles == 1
        assert report.failed_cycles == len(FETCH_FAILURE_CATEGORIES)


class TestForwardResolutionReporting:
    def test_spot_proxy_and_verified_futures_and_not_applicable_all_distinguished(self, db_session):
        run_normal_cycle(db_session, now=NOW)

        future_q = LiveQuote.from_raw(
            contract=fut(), observed_at=NOW + dt.timedelta(minutes=5), last_price=Decimal("24550"), bid=Decimal("24548"),
            ask=Decimal("24552"), volume=1000, open_interest=5000, underlying_last_price=Decimal("24500"),
            provider_reported_iv=None, provider_reported_greeks=None, source="dhan", source_row_hash="eod-test", provenance_class=FIXTURE,
        )
        run_normal_cycle(db_session, now=NOW + dt.timedelta(minutes=5), quotes=realistic_chain(NOW + dt.timedelta(minutes=5)) + (future_q,))

        run_one_cycle(db=db_session, config=mkconfig(snapshot_source(())), now=NOW + dt.timedelta(minutes=10))

        report = build_eod_report(db_session, underlying="NIFTY", session_date=SESSION_DATE, generated_at=NOW)
        assert report.forward_resolution_counts.get("SPOT_PROXY_UNVERIFIED") == 1
        assert report.forward_resolution_counts.get("VERIFIED_LIVE_FUTURES") == 1
        assert report.forward_resolution_counts.get("NOT_APPLICABLE") == 1


class TestPredictiveProbabilityStatusReporting:
    def test_not_established_is_the_default_real_state(self, db_session):
        run_normal_cycle(db_session)
        report = build_eod_report(db_session, underlying="NIFTY", session_date=SESSION_DATE, generated_at=NOW)
        assert report.predictive_probability_status == "NOT_ESTABLISHED"

    def test_established_reported_when_the_test_seam_was_used(self, db_session, monkeypatch):
        run_trade_cycle(db_session, monkeypatch)
        report = build_eod_report(db_session, underlying="NIFTY", session_date=SESSION_DATE, generated_at=NOW)
        assert report.predictive_probability_status == "ESTABLISHED"

    def test_mixed_states_within_one_window_reported_explicitly_not_silently_picked(self, db_session, monkeypatch):
        run_normal_cycle(db_session, now=NOW)
        run_trade_cycle(db_session, monkeypatch, now=NOW + dt.timedelta(minutes=5))
        report = build_eod_report(db_session, underlying="NIFTY", session_date=SESSION_DATE, generated_at=NOW)
        assert report.predictive_probability_status == "MIXED"

    def test_risk_neutral_probability_is_never_labeled_predictive(self, db_session):
        report_text = render_markdown(build_eod_report_with_data(db_session))
        assert "risk-neutral" in report_text.lower()
        assert "MODEL BASELINE" in report_text


def build_eod_report_with_data(db_session):
    run_normal_cycle(db_session)
    return build_eod_report(db_session, underlying="NIFTY", session_date=SESSION_DATE, generated_at=NOW)


class TestReportReproducibility:
    def test_same_db_state_same_generated_at_produces_identical_report(self, db_session):
        run_normal_cycle(db_session)
        r1 = build_eod_report(db_session, underlying="NIFTY", session_date=SESSION_DATE, generated_at=NOW)
        r2 = build_eod_report(db_session, underlying="NIFTY", session_date=SESSION_DATE, generated_at=NOW)
        assert r1 == r2
        assert render_markdown(r1) == render_markdown(r2)


class TestNoMutationDuringReporting:
    def test_recommendation_and_event_counts_unchanged_after_reporting(self, db_session, monkeypatch):
        run_trade_cycle(db_session, monkeypatch)
        before_recs = db_session.execute(text("SELECT count(*) FROM option_recommendations")).scalar()
        before_events = db_session.execute(text("SELECT count(*) FROM option_paper_trade_events")).scalar()
        build_eod_report(db_session, underlying="NIFTY", session_date=SESSION_DATE, generated_at=NOW)
        build_eod_report(db_session, underlying="NIFTY", session_date=SESSION_DATE, generated_at=NOW)
        after_recs = db_session.execute(text("SELECT count(*) FROM option_recommendations")).scalar()
        after_events = db_session.execute(text("SELECT count(*) FROM option_paper_trade_events")).scalar()
        assert before_recs == after_recs and before_events == after_events


class TestZeroVsUnavailableVsNotEstablished:
    def test_three_distinct_semantics_never_collapsed(self, db_session):
        run_one_cycle(db=db_session, config=mkconfig(snapshot_source(())), now=NOW)
        report = build_eod_report(db_session, underlying="NIFTY", session_date=SESSION_DATE, generated_at=NOW)
        assert report.paper_entry_count == 0
        assert report.session_event_counts_by_category == {}
        assert report.forward_resolution_counts.get("NOT_APPLICABLE") == 1
        assert report.realized_pnl_status == "NOT_ESTABLISHED"
        assert report.realized_pnl_summary is None


class TestSessionWindow:
    def test_window_spans_the_full_ist_calendar_day(self):
        start, end = session_window_utc(SESSION_DATE)
        assert (end - start) == dt.timedelta(days=1)
        assert start.tzinfo is not None and end.tzinfo is not None
