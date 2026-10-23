from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.api import app as app_module
from backend.app.api.app import create_app
from backend.app.api.dependencies import get_db


def make_client(session, app=None) -> TestClient:
    app = app if app is not None else create_app()

    def _override():
        yield session

    app.dependency_overrides[get_db] = _override
    return TestClient(app)


class TestAllowedOriginsLoadsDotenv:
    def test_calls_load_dotenv_before_reading_the_env_var(self, monkeypatch):
        monkeypatch.delenv("MONITORING_UI_ORIGINS", raising=False)

        def fake_load_dotenv():
            monkeypatch.setenv("MONITORING_UI_ORIGINS", "http://127.0.0.1:4173,http://localhost:4173")

        monkeypatch.setattr(app_module, "load_dotenv", fake_load_dotenv)
        assert app_module._allowed_origins() == ["http://127.0.0.1:4173", "http://localhost:4173"]

    def test_127_0_0_1_4173_is_accepted(self, monkeypatch):
        monkeypatch.setenv("MONITORING_UI_ORIGINS", "http://127.0.0.1:4173,http://localhost:4173")
        monkeypatch.setattr(app_module, "load_dotenv", lambda: None)
        assert "http://127.0.0.1:4173" in app_module._allowed_origins()

    def test_localhost_4173_is_accepted(self, monkeypatch):
        monkeypatch.setenv("MONITORING_UI_ORIGINS", "http://127.0.0.1:4173,http://localhost:4173")
        monkeypatch.setattr(app_module, "load_dotenv", lambda: None)
        assert "http://localhost:4173" in app_module._allowed_origins()

    def test_existing_default_origins_are_not_removed_when_unset(self, monkeypatch):
        monkeypatch.delenv("MONITORING_UI_ORIGINS", raising=False)
        monkeypatch.setattr(app_module, "load_dotenv", lambda: None)
        assert app_module._allowed_origins() == [
            "http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173",
        ]

    def test_api_still_works_normally_without_the_setting_present(self, monkeypatch, db_session):
        monkeypatch.delenv("MONITORING_UI_ORIGINS", raising=False)
        monkeypatch.setattr(app_module, "load_dotenv", lambda: None)
        client = make_client(db_session)
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json() == {"status": "OK", "database_reachable": True}


class TestCorsHeaderOnARealRequest:

    def test_allowed_origin_receives_the_cors_header(self, monkeypatch, db_session):
        monkeypatch.setenv("MONITORING_UI_ORIGINS", "http://127.0.0.1:4173,http://localhost:4173")
        monkeypatch.setattr(app_module, "load_dotenv", lambda: None)
        client = make_client(db_session)
        r = client.get("/api/health", headers={"Origin": "http://127.0.0.1:4173"})
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == "http://127.0.0.1:4173"

    def test_the_other_configured_origin_also_receives_the_header(self, monkeypatch, db_session):
        monkeypatch.setenv("MONITORING_UI_ORIGINS", "http://127.0.0.1:4173,http://localhost:4173")
        monkeypatch.setattr(app_module, "load_dotenv", lambda: None)
        client = make_client(db_session)
        r = client.get("/api/health", headers={"Origin": "http://localhost:4173"})
        assert r.headers.get("access-control-allow-origin") == "http://localhost:4173"

    def test_disallowed_origin_receives_no_cors_header(self, monkeypatch, db_session):
        monkeypatch.setenv("MONITORING_UI_ORIGINS", "http://127.0.0.1:4173,http://localhost:4173")
        monkeypatch.setattr(app_module, "load_dotenv", lambda: None)
        client = make_client(db_session)
        r = client.get("/api/health", headers={"Origin": "http://evil.example.com"})
        assert "access-control-allow-origin" not in r.headers

    def test_still_works_for_session_status_endpoint_too(self, monkeypatch, db_session):
        monkeypatch.setenv("MONITORING_UI_ORIGINS", "http://127.0.0.1:4173,http://localhost:4173")
        monkeypatch.setattr(app_module, "load_dotenv", lambda: None)
        client = make_client(db_session)
        r = client.get("/api/session/status?underlying=NIFTY", headers={"Origin": "http://127.0.0.1:4173"})
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == "http://127.0.0.1:4173"
