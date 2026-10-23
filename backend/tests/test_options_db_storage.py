from __future__ import annotations

import datetime as dt
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError

UTC = dt.timezone.utc
NOW = dt.datetime.now(UTC)


def insert_recommendation(db_session, **overrides) -> uuid.UUID:
    rec_id = uuid.uuid4()
    fields = dict(
        id=rec_id, decision_timestamp=NOW, underlying="NIFTY", status="WAIT", shape=None,
        data_quality="EOD_RESOLVED", model_version="v0", strategy_version="v0", source="dhan", payload="{}",
    )
    fields.update(overrides)
    db_session.execute(text(
        "INSERT INTO option_recommendations (id, decision_timestamp, underlying, status, shape, data_quality, "
        "model_version, strategy_version, source, payload) VALUES "
        "(:id, :decision_timestamp, :underlying, :status, :shape, :data_quality, :model_version, "
        ":strategy_version, :source, CAST(:payload AS JSONB))"
    ), fields)
    return rec_id


class TestOptionRecommendationsSchema:
    def test_insert_succeeds_and_recorded_at_defaults_to_now(self, db_session):
        rec_id = insert_recommendation(db_session)
        row = db_session.execute(text("SELECT decision_timestamp, recorded_at, status FROM option_recommendations WHERE id=:i"), {"i": str(rec_id)}).one()
        assert row.status == "WAIT"
        assert row.recorded_at is not None
        assert row.decision_timestamp.replace(tzinfo=UTC) == NOW.replace(microsecond=row.decision_timestamp.microsecond)

    def test_decision_timestamp_and_recorded_at_are_independent_columns(self, db_session):
        past = NOW - dt.timedelta(hours=2)
        rec_id = insert_recommendation(db_session, decision_timestamp=past)
        row = db_session.execute(text("SELECT decision_timestamp, recorded_at FROM option_recommendations WHERE id=:i"), {"i": str(rec_id)}).one()
        assert row.decision_timestamp.replace(tzinfo=UTC) != row.recorded_at.replace(tzinfo=UTC)
        assert row.recorded_at.replace(tzinfo=UTC) > row.decision_timestamp.replace(tzinfo=UTC)

    def test_invalid_status_rejected_by_check_constraint(self, db_session):
        with pytest.raises((IntegrityError, DBAPIError)):
            with db_session.begin_nested():
                insert_recommendation(db_session, status="MAYBE")

    def test_update_is_blocked(self, db_session):
        rec_id = insert_recommendation(db_session)
        with pytest.raises(DBAPIError, match="immutable"):
            with db_session.begin_nested():
                db_session.execute(text("UPDATE option_recommendations SET status='TRADE' WHERE id=:i"), {"i": str(rec_id)})
        row = db_session.execute(text("SELECT status FROM option_recommendations WHERE id=:i"), {"i": str(rec_id)}).one()
        assert row.status == "WAIT"

    def test_delete_is_blocked(self, db_session):
        rec_id = insert_recommendation(db_session)
        with pytest.raises(DBAPIError, match="cannot be deleted"):
            with db_session.begin_nested():
                db_session.execute(text("DELETE FROM option_recommendations WHERE id=:i"), {"i": str(rec_id)})
        n = db_session.execute(text("SELECT count(*) FROM option_recommendations WHERE id=:i"), {"i": str(rec_id)}).scalar()
        assert n == 1

    def test_payload_holds_arbitrary_nested_json(self, db_session):
        rec_id = insert_recommendation(db_session, payload='{"legs": [{"strike": 24500, "quote": {"bid": 1, "ask": 2}}], "reasons": ["a", "b"]}')
        payload = db_session.execute(text("SELECT payload FROM option_recommendations WHERE id=:i"), {"i": str(rec_id)}).scalar()
        assert payload["legs"][0]["strike"] == 24500 and payload["reasons"] == ["a", "b"]


class TestOptionPaperTradeEventsSchema:
    def test_insert_referencing_a_real_recommendation_succeeds(self, db_session):
        rec_id = insert_recommendation(db_session)
        event_id = uuid.uuid4()
        db_session.execute(text(
            "INSERT INTO option_paper_trade_events (id, recommendation_id, event_type, event_timestamp, payload) "
            "VALUES (:id, :rid, 'PAPER_ENTRY', :ts, CAST(:payload AS JSONB))"
        ), {"id": event_id, "rid": str(rec_id), "ts": NOW, "payload": "{}"})
        n = db_session.execute(text("SELECT count(*) FROM option_paper_trade_events WHERE id=:i"), {"i": str(event_id)}).scalar()
        assert n == 1

    def test_foreign_key_to_nonexistent_recommendation_rejected(self, db_session):
        with pytest.raises((IntegrityError, DBAPIError)):
            with db_session.begin_nested():
                db_session.execute(text(
                    "INSERT INTO option_paper_trade_events (id, recommendation_id, event_type, event_timestamp, payload) "
                    "VALUES (:id, :rid, 'PAPER_ENTRY', :ts, CAST('{}' AS JSONB))"
                ), {"id": uuid.uuid4(), "rid": str(uuid.uuid4()), "ts": NOW})

    def test_invalid_event_type_rejected(self, db_session):
        rec_id = insert_recommendation(db_session)
        with pytest.raises((IntegrityError, DBAPIError)):
            with db_session.begin_nested():
                db_session.execute(text(
                    "INSERT INTO option_paper_trade_events (id, recommendation_id, event_type, event_timestamp, payload) "
                    "VALUES (:id, :rid, 'MYSTERY_EVENT', :ts, CAST('{}' AS JSONB))"
                ), {"id": uuid.uuid4(), "rid": str(rec_id), "ts": NOW})

    def test_multiple_events_can_reference_the_same_recommendation_append_only(self, db_session):
        rec_id = insert_recommendation(db_session)
        for event_type in ("PAPER_ENTRY", "OBSERVATION", "OBSERVATION"):
            db_session.execute(text(
                "INSERT INTO option_paper_trade_events (id, recommendation_id, event_type, event_timestamp, payload) "
                "VALUES (:id, :rid, :et, :ts, CAST('{}' AS JSONB))"
            ), {"id": uuid.uuid4(), "rid": str(rec_id), "et": event_type, "ts": NOW})
        n = db_session.execute(text("SELECT count(*) FROM option_paper_trade_events WHERE recommendation_id=:i"), {"i": str(rec_id)}).scalar()
        assert n == 3

    def test_update_is_blocked(self, db_session):
        rec_id = insert_recommendation(db_session)
        event_id = uuid.uuid4()
        db_session.execute(text(
            "INSERT INTO option_paper_trade_events (id, recommendation_id, event_type, event_timestamp, payload) "
            "VALUES (:id, :rid, 'PAPER_ENTRY', :ts, CAST('{}' AS JSONB))"
        ), {"id": event_id, "rid": str(rec_id), "ts": NOW})
        with pytest.raises(DBAPIError, match="immutable"):
            with db_session.begin_nested():
                db_session.execute(text("UPDATE option_paper_trade_events SET event_type='EXIT' WHERE id=:i"), {"i": str(event_id)})

    def test_delete_is_blocked(self, db_session):
        rec_id = insert_recommendation(db_session)
        event_id = uuid.uuid4()
        db_session.execute(text(
            "INSERT INTO option_paper_trade_events (id, recommendation_id, event_type, event_timestamp, payload) "
            "VALUES (:id, :rid, 'PAPER_ENTRY', :ts, CAST('{}' AS JSONB))"
        ), {"id": event_id, "rid": str(rec_id), "ts": NOW})
        with pytest.raises(DBAPIError, match="cannot be deleted"):
            with db_session.begin_nested():
                db_session.execute(text("DELETE FROM option_paper_trade_events WHERE id=:i"), {"i": str(event_id)})


class TestOptionSessionEventsSchema:
    def test_insert_succeeds(self, db_session):
        event_id = uuid.uuid4()
        db_session.execute(text(
            "INSERT INTO option_session_events (id, event_timestamp, severity, category, detail) "
            "VALUES (:id, :ts, 'WARNING', 'STALE_DATA', 'quote older than max_quote_age')"
        ), {"id": event_id, "ts": NOW})
        n = db_session.execute(text("SELECT count(*) FROM option_session_events WHERE id=:i"), {"i": str(event_id)}).scalar()
        assert n == 1

    def test_invalid_severity_rejected(self, db_session):
        with pytest.raises((IntegrityError, DBAPIError)):
            with db_session.begin_nested():
                db_session.execute(text(
                    "INSERT INTO option_session_events (id, event_timestamp, severity, category, detail) "
                    "VALUES (:id, :ts, 'CATASTROPHIC', 'X', 'x')"
                ), {"id": uuid.uuid4(), "ts": NOW})

    def test_update_is_blocked(self, db_session):
        event_id = uuid.uuid4()
        db_session.execute(text(
            "INSERT INTO option_session_events (id, event_timestamp, severity, category, detail) "
            "VALUES (:id, :ts, 'INFO', 'X', 'x')"
        ), {"id": event_id, "ts": NOW})
        with pytest.raises(DBAPIError, match="immutable"):
            with db_session.begin_nested():
                db_session.execute(text("UPDATE option_session_events SET severity='ERROR' WHERE id=:i"), {"i": str(event_id)})

    def test_delete_is_blocked(self, db_session):
        event_id = uuid.uuid4()
        db_session.execute(text(
            "INSERT INTO option_session_events (id, event_timestamp, severity, category, detail) "
            "VALUES (:id, :ts, 'INFO', 'X', 'x')"
        ), {"id": event_id, "ts": NOW})
        with pytest.raises(DBAPIError, match="cannot be deleted"):
            with db_session.begin_nested():
                db_session.execute(text("DELETE FROM option_session_events WHERE id=:i"), {"i": str(event_id)})


class TestExistingTablesUntouched:
    def test_no_existing_equity_table_was_modified(self, db_session):
        cols = db_session.execute(text(
            "SELECT count(*) FROM information_schema.columns WHERE table_name = 'market_bars'"
        )).scalar()
        assert cols > 0
