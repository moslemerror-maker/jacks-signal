from __future__ import annotations

import datetime as dt

import pytest

from backend.app.data.live import events

D = dt.date
UTC = dt.timezone.utc


class TestLogEvent:
    def test_rejects_an_invalid_severity(self, db_session):
        with pytest.raises(ValueError, match="severity"):
            events.log_event(db_session, trading_date=D(2026, 9, 29), severity="CRITICAL", category="X", detail="x")

    def test_writes_a_retrievable_row(self, db_session):
        events.log_event(
            db_session, trading_date=D(2026, 9, 29), severity="INFO", category=events.CYCLE_START,
            detail="starting", symbols_expected=299,
        )
        db_session.flush()
        rows = events.events_for_date(db_session, trading_date=D(2026, 9, 29))
        assert len(rows) == 1
        assert rows[0].category == events.CYCLE_START
        assert rows[0].symbols_expected == 299

    def test_does_not_commit_the_caller_controls_the_transaction(self, db_session):
        events.log_event(db_session, trading_date=D(2026, 9, 29), severity="INFO", category=events.CYCLE_START, detail="x")
        assert db_session.new

    def test_never_receives_a_credential_in_detail_by_construction(self, db_session):
        events.log_event(db_session, trading_date=D(2026, 9, 29), severity="INFO", category=events.CYCLE_START, detail="299 symbols expected")
        db_session.flush()
        row = events.events_for_date(db_session, trading_date=D(2026, 9, 29))[0]
        assert "access-token" not in row.detail.lower()
        assert "dhanclientid" not in row.detail.lower()


class TestEventsForDate:
    def test_orders_by_event_timestamp_ascending(self, db_session):
        events.log_event(db_session, trading_date=D(2026, 9, 29), severity="INFO", category="B", detail="second", event_timestamp=dt.datetime(2026, 9, 29, 7, 0, tzinfo=UTC))
        events.log_event(db_session, trading_date=D(2026, 9, 29), severity="INFO", category="A", detail="first", event_timestamp=dt.datetime(2026, 9, 29, 6, 0, tzinfo=UTC))
        db_session.flush()
        rows = events.events_for_date(db_session, trading_date=D(2026, 9, 29))
        assert [r.category for r in rows] == ["A", "B"]

    def test_only_returns_events_for_the_requested_date(self, db_session):
        events.log_event(db_session, trading_date=D(2026, 9, 28), severity="INFO", category="OTHER_DAY", detail="x")
        events.log_event(db_session, trading_date=D(2026, 9, 29), severity="INFO", category="TODAY", detail="x")
        db_session.flush()
        rows = events.events_for_date(db_session, trading_date=D(2026, 9, 29))
        assert [r.category for r in rows] == ["TODAY"]


class TestIsSessionComplete:
    def test_false_when_no_session_complete_event_exists(self, db_session):
        events.log_event(db_session, trading_date=D(2026, 9, 29), severity="INFO", category=events.CYCLE_START, detail="x")
        db_session.flush()
        assert events.is_session_complete(db_session, trading_date=D(2026, 9, 29)) is False

    def test_true_once_a_session_complete_event_is_logged(self, db_session):
        events.log_event(db_session, trading_date=D(2026, 9, 29), severity="INFO", category=events.SESSION_COMPLETE, detail="done")
        db_session.flush()
        assert events.is_session_complete(db_session, trading_date=D(2026, 9, 29)) is True
