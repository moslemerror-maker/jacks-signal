from __future__ import annotations

import dataclasses
import datetime as dt
import subprocess
from unittest.mock import patch

from backend.app.options.live_session import log_session_event
from backend.app.options.pipeline_health import health_concerns, pipeline_health_snapshot, scheduled_task_state
from backend.app.options.session_scheduler import log_session_state

D = dt.date
UTC = dt.timezone.utc
TRADING_DAY = D(2026, 9, 24)
WEEKEND_DAY = D(2026, 9, 26)
HOLIDAYS = frozenset({"25-Sep-2026"})
SESSION_START = dt.datetime(2026, 9, 24, 3, 45, tzinfo=UTC)


class TestPipelineHealthSnapshotTradingDay:
    def test_weekend_is_not_a_trading_day_and_absence_of_a_session_is_not_a_concern(self, db_session):
        snap = pipeline_health_snapshot(db_session, underlying="NIFTY", today=WEEKEND_DAY, now=dt.datetime(2026, 9, 26, 6, 0, tzinfo=UTC), holiday_dates=HOLIDAYS)
        assert snap.is_trading_day is False
        assert health_concerns(snap) == []

    def test_trading_day_with_no_session_yet_within_grace_period_is_not_a_concern(self, db_session):
        snap = pipeline_health_snapshot(db_session, underlying="NIFTY", today=TRADING_DAY, now=SESSION_START + dt.timedelta(minutes=5), holiday_dates=HOLIDAYS)
        assert snap.session_started is False
        assert health_concerns(snap) == []

    def test_trading_day_with_no_session_past_grace_period_is_a_concern(self, db_session):
        snap = pipeline_health_snapshot(db_session, underlying="NIFTY", today=TRADING_DAY, now=SESSION_START + dt.timedelta(minutes=30), holiday_dates=HOLIDAYS)
        assert snap.session_started is False
        concerns = health_concerns(snap)
        assert any("no session has started" in c for c in concerns)


class TestPipelineHealthSnapshotSilence:
    def test_live_session_silent_too_long_is_flagged(self, db_session):
        log_session_state(db_session, state="STARTING", detail="x", event_timestamp=SESSION_START)
        log_session_state(db_session, state="LIVE", detail="x", event_timestamp=SESSION_START)
        db_session.commit()
        now = SESSION_START + dt.timedelta(hours=3)
        snap = pipeline_health_snapshot(db_session, underlying="NIFTY", today=TRADING_DAY, now=now, holiday_dates=HOLIDAYS)
        assert snap.is_silent_too_long is True
        assert any("silent" in c for c in health_concerns(snap))

    def test_recently_active_session_is_not_silent(self, db_session):
        log_session_state(db_session, state="STARTING", detail="x", event_timestamp=SESSION_START)
        log_session_state(db_session, state="LIVE", detail="x", event_timestamp=SESSION_START)
        db_session.commit()
        now = SESSION_START + dt.timedelta(minutes=2)
        snap = pipeline_health_snapshot(db_session, underlying="NIFTY", today=TRADING_DAY, now=now, holiday_dates=HOLIDAYS)
        assert snap.is_silent_too_long is False

    def test_completed_session_is_never_flagged_as_silent(self, db_session):
        log_session_state(db_session, state="STARTING", detail="x", event_timestamp=SESSION_START)
        log_session_state(db_session, state="LIVE", detail="x", event_timestamp=SESSION_START)
        log_session_state(db_session, state="COMPLETED", detail="x", event_timestamp=SESSION_START + dt.timedelta(minutes=5))
        db_session.commit()
        now = SESSION_START + dt.timedelta(hours=3)
        snap = pipeline_health_snapshot(db_session, underlying="NIFTY", today=TRADING_DAY, now=now, holiday_dates=HOLIDAYS)
        assert snap.is_silent_too_long is False
        assert snap.session_completed is True


class TestPipelineHealthSnapshotFailures:
    def test_auth_failure_suspected_events_are_surfaced(self, db_session):
        log_session_state(db_session, state="STARTING", detail="x", event_timestamp=SESSION_START)
        log_session_state(db_session, state="LIVE", detail="x", event_timestamp=SESSION_START)
        log_session_event(db_session, event_timestamp=SESSION_START, severity="ERROR", category="AUTH_FAILURE_SUSPECTED",
                          detail="expiry_list did not return status=success: {'status': 'failure', 'remarks': {'error_code': None, 'error_type': None, 'error_message': None}}")
        db_session.commit()
        now = SESSION_START + dt.timedelta(minutes=2)
        snap = pipeline_health_snapshot(db_session, underlying="NIFTY", today=TRADING_DAY, now=now, holiday_dates=HOLIDAYS)
        assert snap.auth_failure_suspected_today is True
        assert snap.fetch_failure_counts_today == {"AUTH_FAILURE_SUSPECTED": 1}
        assert any("token most likely needs to be refreshed" in c for c in health_concerns(snap))

    def test_session_failed_is_surfaced_as_a_concern(self, db_session):
        log_session_state(db_session, state="STARTING", detail="x", event_timestamp=SESSION_START)
        log_session_state(db_session, state="FAILED", detail="stopped after 6 consecutive failures", event_timestamp=SESSION_START, severity="ERROR")
        db_session.commit()
        now = SESSION_START + dt.timedelta(minutes=2)
        snap = pipeline_health_snapshot(db_session, underlying="NIFTY", today=TRADING_DAY, now=now, holiday_dates=HOLIDAYS)
        assert snap.session_failed is True
        assert any("FAILED" in c for c in health_concerns(snap))

    def test_crash_events_are_counted(self, db_session):
        log_session_state(db_session, state="STARTING", detail="x", event_timestamp=SESSION_START)
        log_session_state(db_session, state="LIVE", detail="x", event_timestamp=SESSION_START)
        log_session_event(db_session, event_timestamp=SESSION_START, severity="ERROR", category="CYCLE_CRASH", detail="boom")
        db_session.commit()
        now = SESSION_START + dt.timedelta(minutes=2)
        snap = pipeline_health_snapshot(db_session, underlying="NIFTY", today=TRADING_DAY, now=now, holiday_dates=HOLIDAYS)
        assert snap.crash_events_today == 1
        assert any("CYCLE_CRASH" in c for c in health_concerns(snap))


class TestScheduledTaskState:
    def test_parses_status_from_schtasks_csv_output(self):
        fake_stdout = '"HostName","TaskName","Next Run Time","Status"\n"MYPC","\\\\AEVORYN-Daily-Live-Session","10/5/2026 9:15:00 AM","Disabled"\n'
        with patch("subprocess.run", return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=fake_stdout, stderr="")):
            assert scheduled_task_state("AEVORYN-Daily-Live-Session") == "Disabled"

    def test_returns_none_on_nonzero_exit_task_not_found(self):
        with patch("subprocess.run", return_value=subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="ERROR: not found")):
            assert scheduled_task_state("DOES-NOT-EXIST") is None

    def test_never_raises_when_schtasks_is_unavailable(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("schtasks not found")):
            assert scheduled_task_state("ANYTHING") is None

    def test_never_raises_on_timeout(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="schtasks", timeout=5)):
            assert scheduled_task_state("ANYTHING") is None

    def test_health_concerns_flags_a_disabled_task(self, db_session):
        snap = pipeline_health_snapshot(db_session, underlying="NIFTY", today=WEEKEND_DAY, now=dt.datetime(2026, 9, 26, 6, 0, tzinfo=UTC), holiday_dates=HOLIDAYS)
        snap = dataclasses.replace(snap, scheduled_task_state="Disabled")
        assert any("Disabled" in c for c in health_concerns(snap))


class TestHealthConcernsCleanState:
    def test_no_concerns_on_a_healthy_weekend_snapshot(self, db_session):
        snap = pipeline_health_snapshot(db_session, underlying="NIFTY", today=WEEKEND_DAY, now=dt.datetime(2026, 9, 26, 6, 0, tzinfo=UTC), holiday_dates=HOLIDAYS)
        assert health_concerns(snap) == []
