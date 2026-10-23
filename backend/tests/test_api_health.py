from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from backend.app.api.app import create_app
from backend.app.api.dependencies import get_db


def make_client(session) -> TestClient:
    app = create_app()

    def _override():
        yield session

    app.dependency_overrides[get_db] = _override
    return TestClient(app)


class _BrokenSession:

    def execute(self, *args, **kwargs):
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))


class TestHealth:
    def test_success_reports_ok_and_reachable(self, db_session):
        client = make_client(db_session)
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json() == {"status": "OK", "database_reachable": True}

    def test_database_unavailable_reports_503_never_crashes(self):
        client = make_client(_BrokenSession())
        r = client.get("/api/health")
        assert r.status_code == 503
        assert r.json() == {"status": "DB_UNAVAILABLE", "database_reachable": False}

    def test_never_exposes_credentials_or_env_details(self, db_session):
        client = make_client(db_session)
        body_text = client.get("/api/health").text.lower()
        for forbidden in ("dhan", "access_token", "client_id", "database_url", "password", "secret", ".env"):
            assert forbidden not in body_text

    def test_response_is_exactly_the_documented_shape(self, db_session):
        client = make_client(db_session)
        assert set(client.get("/api/health").json()) == {"status", "database_reachable"}
