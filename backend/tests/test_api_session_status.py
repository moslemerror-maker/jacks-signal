from __future__ import annotations

import datetime as dt
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from backend.app.api.app import create_app
from backend.app.api.dependencies import current_time, get_db
from backend.app.options.session_scheduler import log_session_state

D = dt.date
UTC = dt.timezone.utc
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
TODAY = D(2026, 9, 25)
NOON_IST = dt.datetime.combine(TODAY, dt.time(12, 0), tzinfo=IST).astimezone(UTC)
SERVER_NOW = NOON_IST + dt.timedelta(hours=7)


def make_client(session, *, now: dt.datetime = SERVER_NOW) -> TestClient:
    app = create_app()

    def _db_override():
        yield session

    app.dependency_overrides[get_db] = _db_override
    app.dependency_overrides[current_time] = lambda: now
    return TestClient(app)


class _BrokenSession:

    def execute(self, *args, **kwargs):
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))


def insert_recommendation(db, *, decision_timestamp, status="WAIT", underlying="NIFTY", source="dhan", data_quality="EOD_RESOLVED"):
    rec_id = uuid.uuid4()
    db.execute(text(
        "INSERT INTO option_recommendations (id, decision_timestamp, underlying, status, data_quality, source, payload) "
        "VALUES (:id, :dt, :u, :status, :dq, :src, CAST('{}' AS JSONB))"
    ), {"id": rec_id, "dt": decision_timestamp, "u": underlying, "status": status, "dq": data_quality, "src": source})
    return rec_id


class TestSessionStatusEndpoint:
    def test_no_session_today_and_no_recommendation_ever_is_not_started(self, db_session):
        client = make_client(db_session)
        r = client.get("/api/session/status")
        assert r.status_code == 200
        body = r.json()
        assert body["session_state"] == "NOT_STARTED"
        assert body["latest_recommendation_timestamp"] is None
        assert body["latest_recommendation_status"] is None
        assert body["cycle_count"] == 0

    def test_starting_state_is_reported(self, db_session):
        log_session_state(db_session, state="STARTING", detail="automatic daily session starting", event_timestamp=NOON_IST)
        db_session.flush()
        client = make_client(db_session)
        body = client.get("/api/session/status").json()
        assert body["session_state"] == "STARTING"
        assert body["session_start_timestamp"] is not None

    def test_live_state_is_reported(self, db_session):
        log_session_state(db_session, state="STARTING", detail="starting", event_timestamp=NOON_IST)
        log_session_state(db_session, state="LIVE", detail="polling started", event_timestamp=NOON_IST + dt.timedelta(seconds=3))
        db_session.flush()
        client = make_client(db_session)
        body = client.get("/api/session/status").json()
        assert body["session_state"] == "LIVE"

    def test_completed_state_is_reported(self, db_session):
        log_session_state(db_session, state="STARTING", detail="starting", event_timestamp=NOON_IST)
        log_session_state(db_session, state="COMPLETED", detail="cycles=15 status_counts={}", event_timestamp=NOON_IST + dt.timedelta(hours=6))
        db_session.flush()
        client = make_client(db_session)
        body = client.get("/api/session/status").json()
        assert body["session_state"] == "COMPLETED"
        assert body["completion_timestamp"] is not None

    def test_failed_state_is_reported(self, db_session):
        log_session_state(db_session, state="FAILED", detail="credential preflight = MISSING: DHAN_ACCESS_TOKEN missing",
                          event_timestamp=NOON_IST, severity="ERROR")
        db_session.flush()
        client = make_client(db_session)
        body = client.get("/api/session/status").json()
        assert body["session_state"] == "FAILED"
        assert body["failure_timestamp"] is not None
        assert "DHAN_ACCESS_TOKEN" in body["failure_detail"]

    def test_todays_failed_session_with_yesterdays_recommendation_is_failed_not_live_not_no_trade(self, db_session):
        yesterday = NOON_IST - dt.timedelta(days=1)
        insert_recommendation(db_session, decision_timestamp=yesterday, status="NO_TRADE")
        log_session_state(db_session, state="FAILED", detail="startup failure: ModuleNotFoundError", event_timestamp=NOON_IST, severity="ERROR")
        db_session.flush()
        client = make_client(db_session)
        body = client.get("/api/session/status").json()
        assert body["session_state"] == "FAILED"
        assert body["session_state"] != "LIVE"
        assert body["latest_recommendation_status"] == "NO_TRADE"

    def test_todays_wait_recommendation_is_the_latest(self, db_session):
        log_session_state(db_session, state="LIVE", detail="polling started", event_timestamp=NOON_IST)
        insert_recommendation(db_session, decision_timestamp=NOON_IST + dt.timedelta(minutes=5), status="WAIT")
        db_session.flush()
        client = make_client(db_session)
        body = client.get("/api/session/status").json()
        assert body["latest_recommendation_status"] == "WAIT"

    def test_todays_no_trade_recommendation_is_the_latest(self, db_session):
        log_session_state(db_session, state="LIVE", detail="polling started", event_timestamp=NOON_IST)
        insert_recommendation(db_session, decision_timestamp=NOON_IST + dt.timedelta(minutes=5), status="NO_TRADE")
        db_session.flush()
        client = make_client(db_session)
        body = client.get("/api/session/status").json()
        assert body["latest_recommendation_status"] == "NO_TRADE"

    def test_yesterdays_recommendation_does_not_make_today_live(self, db_session):
        yesterday = NOON_IST - dt.timedelta(days=1)
        insert_recommendation(db_session, decision_timestamp=yesterday, status="WAIT")
        db_session.flush()
        client = make_client(db_session)
        body = client.get("/api/session/status").json()
        assert body["session_state"] == "NOT_STARTED"
        assert body["latest_recommendation_status"] == "WAIT"

    def test_database_unavailable_reports_503_never_no_trade_never_completed(self):
        client = make_client(_BrokenSession())
        r = client.get("/api/session/status")
        assert r.status_code == 503
        body_text = r.text
        assert "NO_TRADE" not in body_text
        assert "COMPLETED" not in body_text
        assert "LIVE" not in body_text

    def test_unsupported_underlying_is_rejected(self, db_session):
        client = make_client(db_session)
        r = client.get("/api/session/status", params={"underlying": "BANKNIFTY"})
        assert r.status_code == 400

    def test_defaults_to_nifty(self, db_session):
        client = make_client(db_session)
        default_r = client.get("/api/session/status")
        explicit_r = client.get("/api/session/status", params={"underlying": "NIFTY"})
        assert default_r.json() == explicit_r.json()

    def test_response_is_exactly_the_documented_shape(self, db_session):
        client = make_client(db_session)
        body = client.get("/api/session/status").json()
        assert set(body) == {
            "underlying", "trading_date", "session_state", "session_start_timestamp", "last_event_timestamp",
            "completion_timestamp", "failure_timestamp", "failure_detail", "cycle_count",
            "latest_recommendation_timestamp", "latest_recommendation_status", "latest_recommendation_id",
            "data_quality", "source",
        }

    def test_never_exposes_credentials_or_env_details(self, db_session):
        log_session_state(db_session, state="FAILED", detail="database not reachable (check_db_reachable returned False)",
                          event_timestamp=NOON_IST, severity="ERROR")
        db_session.flush()
        client = make_client(db_session)
        body_text = client.get("/api/session/status").text.lower()
        for forbidden in ("access_token", "access-token", "client_id", "database_url", "password", "secret", ".env"):
            assert forbidden not in body_text

    def test_endpoint_makes_no_database_writes(self, db_session):
        log_session_state(db_session, state="LIVE", detail="polling started", event_timestamp=NOON_IST)
        db_session.flush()
        before = db_session.execute(text("SELECT COUNT(*) FROM option_session_events")).scalar()
        client = make_client(db_session)
        client.get("/api/session/status")
        client.get("/api/session/status")
        after_events = db_session.execute(text("SELECT COUNT(*) FROM option_session_events")).scalar()
        after_recs = db_session.execute(text("SELECT COUNT(*) FROM option_recommendations")).scalar()
        assert after_events == before
        assert after_recs == 0

    def test_endpoint_never_invokes_dhan_or_live_session_execution(self):
        import ast
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "app" / "api" / "routers" / "session.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
                names.update(f"{node.module}.{alias.name}" for alias in node.names)
        forbidden = ("dhan_client", "dhan_optionchain_client", "dhan_live_chain_source", "dhan_live_quote",
                    "provider_credentials", "credential_preflight", "run_one_cycle", "run_session", "DhanLiveChainSource")
        for f in forbidden:
            assert not any(f in n for n in names), f"forbidden import {f!r} found in session.py"
