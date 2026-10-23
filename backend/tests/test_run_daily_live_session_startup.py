from __future__ import annotations

import dataclasses
import datetime as dt
import signal as signal_module

import pytest
from sqlalchemy import text

from backend.app.data.sources.credential_preflight import CredentialPreflightResult
from backend.app.options.session_scheduler import DailySessionResult
from scripts import run_daily_live_session as rdls

UTC = dt.timezone.utc
TODAY = dt.date(2026, 9, 24)
NOW = dt.datetime(2026, 9, 24, 3, 45, tzinfo=UTC)


class _FakeDb:
    def close(self) -> None:
        pass


def _patch_happy_path(monkeypatch, *, log_calls=None, db=None):
    db = db if db is not None else _FakeDb()
    monkeypatch.setattr("backend.app.db.session.get_session", lambda: db)
    monkeypatch.setattr("backend.app.options.session_scheduler.check_db_reachable", lambda _db: True)

    def _log(_db, *, state, detail, event_timestamp, severity="INFO"):
        if log_calls is not None:
            log_calls.append({"state": state, "detail": detail, "severity": severity})
    monkeypatch.setattr("backend.app.options.session_scheduler.log_session_state", _log)
    monkeypatch.setattr("backend.app.options.trading_calendar.load_fo_holiday_dates", lambda path: frozenset())
    monkeypatch.setattr(
        "backend.app.data.sources.credential_preflight.dhan_credential_preflight",
        lambda: CredentialPreflightResult("dhan", "OK", "present and well-formed"),
    )
    monkeypatch.setattr("backend.app.options.session_schedule.ist_today", lambda now: TODAY)
    monkeypatch.setattr("backend.app.options.session_schedule.session_stop_utc", lambda today: NOW + dt.timedelta(hours=6))
    monkeypatch.setattr(rdls, "_live_source", lambda underlying: object())
    monkeypatch.setattr("backend.app.options.session_scheduler.windows_safe_stop_flag", lambda **kw: (lambda: False))
    return db


def _patch_workflow_outcome(monkeypatch, outcome: str, detail: str = "x", capture: dict | None = None):
    def _workflow(**kwargs):
        if capture is not None:
            capture.update(kwargs)
        return DailySessionResult(outcome, detail)
    monkeypatch.setattr("backend.app.options.session_scheduler.run_daily_session_workflow", _workflow)


def _argv(tmp_path, **extra):
    argv = ["--confirm-live", "--holiday-file", str(tmp_path / "holidays.json")]
    for k, v in extra.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return argv


class TestSuccessfulStartup:
    def test_reaches_run_daily_session_workflow(self, monkeypatch, tmp_path):
        _patch_happy_path(monkeypatch)
        captured: dict = {}
        _patch_workflow_outcome(monkeypatch, "COMPLETED", "cycles=0 status_counts={}", capture=captured)
        rc = rdls.main(_argv(tmp_path, stop_file=tmp_path / "stop"))
        assert rc == 0
        assert captured.get("today") == TODAY

    @pytest.mark.parametrize(
        "outcome,expected_rc",
        [
            ("NOT_TRADING_DAY", 0),
            ("REFUSE_ALREADY_COMPLETED", 0),
            ("REFUSE_ALREADY_FAILED", 2),
            ("FAILED", 2),
            ("COMPLETED", 0),
        ],
    )
    def test_outcome_to_exit_code_mapping_unchanged(self, monkeypatch, tmp_path, outcome, expected_rc):
        _patch_happy_path(monkeypatch)
        _patch_workflow_outcome(monkeypatch, outcome)
        rc = rdls.main(_argv(tmp_path, stop_file=tmp_path / "stop"))
        assert rc == expected_rc

    def test_stop_at_still_computed_via_session_stop_utc(self, monkeypatch, tmp_path):
        _patch_happy_path(monkeypatch)
        expected_stop_at = NOW + dt.timedelta(hours=6)
        captured: dict = {}
        _patch_workflow_outcome(monkeypatch, "COMPLETED", capture=captured)
        rdls.main(_argv(tmp_path, stop_file=tmp_path / "stop"))
        assert captured["runner_config"].stop_at == expected_stop_at

    def test_windows_stop_file_is_cleared_at_startup(self, monkeypatch, tmp_path):
        _patch_happy_path(monkeypatch)
        _patch_workflow_outcome(monkeypatch, "COMPLETED")
        stop_file = tmp_path / "stop"
        stop_file.parent.mkdir(parents=True, exist_ok=True)
        stop_file.write_text("stale request from a previous run")
        rdls.main(_argv(tmp_path, stop_file=stop_file))
        assert not stop_file.exists()


class TestStartupFailureCapture:

    def test_exception_during_live_source_construction_is_captured(self, monkeypatch, tmp_path):
        _patch_happy_path(monkeypatch)
        monkeypatch.setattr(rdls, "_live_source", lambda underlying: (_ for _ in ()).throw(ValueError("boom: live source")))
        log = tmp_path / "failures.log"
        monkeypatch.setattr(rdls, "STARTUP_FAILURE_LOG", log)
        rc = rdls.main(_argv(tmp_path, stop_file=tmp_path / "stop"))
        assert rc == 2
        assert log.exists()
        text_ = log.read_text()
        assert "phase='live source / config construction'" in text_
        assert "ValueError" in text_ and "boom: live source" in text_

    def test_exception_during_orchestrator_config_construction_is_captured(self, monkeypatch, tmp_path):
        _patch_happy_path(monkeypatch)
        monkeypatch.setattr(
            "backend.app.options.live_session.OrchestratorConfig",
            lambda *a, **kw: (_ for _ in ()).throw(TypeError("boom: orchestrator config")),
        )
        log = tmp_path / "failures.log"
        monkeypatch.setattr(rdls, "STARTUP_FAILURE_LOG", log)
        rc = rdls.main(_argv(tmp_path, stop_file=tmp_path / "stop"))
        assert rc == 2
        assert "boom: orchestrator config" in log.read_text()

    def test_exception_during_runner_config_construction_is_captured(self, monkeypatch, tmp_path):
        _patch_happy_path(monkeypatch)
        monkeypatch.setattr(
            "backend.app.options.live_session.RunnerConfig",
            lambda *a, **kw: (_ for _ in ()).throw(TypeError("boom: runner config")),
        )
        log = tmp_path / "failures.log"
        monkeypatch.setattr(rdls, "STARTUP_FAILURE_LOG", log)
        rc = rdls.main(_argv(tmp_path, stop_file=tmp_path / "stop"))
        assert rc == 2
        assert "boom: runner config" in log.read_text()

    def test_exception_during_signal_handler_registration_is_captured(self, monkeypatch, tmp_path):
        _patch_happy_path(monkeypatch)

        def _raising_signal(signum, handler):
            raise RuntimeError("boom: signal registration")
        monkeypatch.setattr(signal_module, "signal", _raising_signal)
        log = tmp_path / "failures.log"
        monkeypatch.setattr(rdls, "STARTUP_FAILURE_LOG", log)
        rc = rdls.main(_argv(tmp_path, stop_file=tmp_path / "stop"))
        assert rc == 2
        text_ = log.read_text()
        assert "phase='signal handler registration'" in text_
        assert "boom: signal registration" in text_

    def test_exception_during_stop_file_directory_creation_is_captured(self, monkeypatch, tmp_path):
        _patch_happy_path(monkeypatch)
        blocking_file = tmp_path / "not_a_directory"
        blocking_file.write_text("x")
        bad_stop_file = blocking_file / "subdir" / "stop"
        log = tmp_path / "failures.log"
        monkeypatch.setattr(rdls, "STARTUP_FAILURE_LOG", log)
        rc = rdls.main(_argv(tmp_path, stop_file=bad_stop_file))
        assert rc == 2
        text_ = log.read_text()
        assert "phase='stop-file setup'" in text_

    def test_diagnostic_traceback_is_persisted_with_required_fields(self, monkeypatch, tmp_path):
        _patch_happy_path(monkeypatch)
        monkeypatch.setattr(rdls, "_live_source", lambda underlying: (_ for _ in ()).throw(ValueError("boom: traceback fields")))
        log = tmp_path / "failures.log"
        monkeypatch.setattr(rdls, "STARTUP_FAILURE_LOG", log)
        rdls.main(_argv(tmp_path, stop_file=tmp_path / "stop"))
        text_ = log.read_text()
        assert "timestamp=" in text_
        assert "script=run_daily_live_session.py" in text_
        assert "exception_type=ValueError" in text_
        assert "exception_message=boom: traceback fields" in text_
        assert "traceback:" in text_
        assert "Traceback (most recent call last)" in text_

    def test_secrets_are_never_written_to_the_diagnostic_log(self, monkeypatch, tmp_path):
        secret_token = "sk-super-secret-dhan-token-value-12345"
        monkeypatch.setenv("DHAN_ACCESS_TOKEN", secret_token)
        monkeypatch.setenv("DHAN_CLIENT_ID", "9999999999")
        _patch_happy_path(monkeypatch)
        monkeypatch.setattr(
            "backend.app.options.live_session.OrchestratorConfig",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom: unrelated failure, no credential value here")),
        )
        log = tmp_path / "failures.log"
        monkeypatch.setattr(rdls, "STARTUP_FAILURE_LOG", log)
        rdls.main(_argv(tmp_path, stop_file=tmp_path / "stop"))
        assert secret_token not in log.read_text()


class TestStartupFailureNeverBecomesATradingDecision:
    def test_startup_failure_creates_no_recommendation_or_paper_trade_event(self, monkeypatch, db_session):
        monkeypatch.setattr("backend.app.db.session.get_session", lambda: db_session)
        monkeypatch.setattr("backend.app.options.trading_calendar.load_fo_holiday_dates", lambda path: frozenset())
        monkeypatch.setattr(
            "backend.app.data.sources.credential_preflight.dhan_credential_preflight",
            lambda: CredentialPreflightResult("dhan", "OK", "present and well-formed"),
        )
        monkeypatch.setattr("backend.app.options.session_schedule.ist_today", lambda now: TODAY)
        monkeypatch.setattr("backend.app.options.session_schedule.session_stop_utc", lambda today: NOW + dt.timedelta(hours=6))
        monkeypatch.setattr(rdls, "_live_source", lambda underlying: (_ for _ in ()).throw(ValueError("boom: no trading decision")))
        rc = rdls.main(["--confirm-live"])
        db_session.flush()
        assert rc == 2
        rec_count = db_session.execute(text("SELECT COUNT(*) FROM option_recommendations")).scalar_one()
        paper_count = db_session.execute(text("SELECT COUNT(*) FROM option_paper_trade_events")).scalar_one()
        assert rec_count == 0
        assert paper_count == 0
        row = db_session.execute(text("SELECT category FROM option_session_events ORDER BY recorded_at DESC LIMIT 1")).mappings().one()
        assert row["category"] == "SESSION_FAILED"

    def test_startup_failure_state_is_failed_never_wait_or_no_trade(self, monkeypatch, tmp_path):
        log_calls: list = []
        _patch_happy_path(monkeypatch, log_calls=log_calls)
        monkeypatch.setattr(rdls, "_live_source", lambda underlying: (_ for _ in ()).throw(ValueError("boom: state check")))
        log = tmp_path / "failures.log"
        monkeypatch.setattr(rdls, "STARTUP_FAILURE_LOG", log)
        rdls.main(_argv(tmp_path, stop_file=tmp_path / "stop"))
        assert log_calls, "expected a best-effort SESSION_FAILED log_session_state call"
        states = {c["state"] for c in log_calls}
        assert states == {"FAILED"}
        assert "WAIT" not in states and "NO_TRADE" not in states and "TRADE" not in states

    def test_db_unreachable_still_writes_diagnostic_log_and_exits_nonzero(self, monkeypatch, tmp_path):
        db = _patch_happy_path(monkeypatch)
        monkeypatch.setattr("backend.app.options.session_scheduler.check_db_reachable", lambda _db: False)
        log = tmp_path / "failures.log"
        monkeypatch.setattr(rdls, "STARTUP_FAILURE_LOG", log)
        rc = rdls.main(_argv(tmp_path, stop_file=tmp_path / "stop"))
        assert rc == 2
        assert log.exists()
        assert "database not reachable" in log.read_text()
