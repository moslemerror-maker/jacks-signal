from __future__ import annotations

import datetime as dt
import json

from backend.app.data.backfill.executor import ExecutorConfig
from backend.app.data.live import events
from backend.app.data.live.finalize import finalize_trading_day
from backend.app.data.raw.store import RawStore
from backend.app.data.sources import dhan_equity_historical as deh
from backend.app.data.sources.provider_credentials import DhanCredentials

D = dt.date
UTC = dt.timezone.utc
CREDS = DhanCredentials(client_id="1100000000", access_token="secret")
FAST = ExecutorConfig(pace_seconds=0, backoff_base_seconds=0, jitter_max_seconds=0, sleep_fn=lambda s: None)


def full_session_body(n=360, base_epoch=1790653500):
    return json.dumps({
        "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n, "close": [100.5] * n, "volume": [1000] * n,
        "timestamp": [base_epoch + i * 60 for i in range(n)],
    }).encode()


def short_session_body(n=200, base_epoch=1790653500):
    return json.dumps({
        "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n, "close": [100.5] * n, "volume": [1000] * n,
        "timestamp": [base_epoch + i * 60 for i in range(n)],
    }).encode()


class FakeResponse:
    def __init__(self, status=200, content=b""):
        self.status_code, self.content, self.headers = status, content, {"Content-Type": "application/json"}

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self, items):
        self.items, self.calls = list(items), 0

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls += 1
        return self.items[min(self.calls - 1, len(self.items) - 1)]


def _adapter(items):
    return deh.DhanEquityHistoricalAdapter(CREDS, session=FakeSession(items))


class TestFinalizeTradingDay:
    def test_full_coverage_for_every_symbol_marks_the_session_complete(self, db_session, tmp_path):
        store = RawStore(tmp_path / "raw")
        adapter = _adapter([FakeResponse(200, full_session_body()), FakeResponse(200, full_session_body())])
        report = finalize_trading_day(
            db_session, store, adapter, securities=[(1, "A"), (2, "B")], trading_date=D(2026, 9, 29),
            now_utc_fn=lambda: dt.datetime(2026, 9, 29, 15, 20, tzinfo=UTC), config=FAST,
        )
        assert report.is_session_complete is True
        assert report.symbols_complete == 2
        assert report.symbols_failed == 0
        db_session.flush()
        assert events.is_session_complete(db_session, trading_date=D(2026, 9, 29)) is True

    def test_short_coverage_for_a_symbol_marks_the_session_incomplete_not_error(self, db_session, tmp_path):
        store = RawStore(tmp_path / "raw")
        adapter = _adapter([FakeResponse(200, full_session_body()), FakeResponse(200, short_session_body())])
        report = finalize_trading_day(
            db_session, store, adapter, securities=[(1, "A"), (2, "B")], trading_date=D(2026, 9, 29),
            now_utc_fn=lambda: dt.datetime(2026, 9, 29, 15, 20, tzinfo=UTC), config=FAST,
        )
        assert report.is_session_complete is False
        assert report.symbols_complete == 1
        assert report.symbols_partial == 1
        db_session.flush()
        assert events.is_session_complete(db_session, trading_date=D(2026, 9, 29)) is False
        rows = events.events_for_date(db_session, trading_date=D(2026, 9, 29))
        assert any(r.category == events.RECONCILIATION_COMPLETE and r.severity == "WARNING" for r in rows)

    def test_logs_a_cycle_start_event_before_polling(self, db_session, tmp_path):
        store = RawStore(tmp_path / "raw")
        adapter = _adapter([FakeResponse(200, full_session_body())])
        finalize_trading_day(
            db_session, store, adapter, securities=[(1, "A")], trading_date=D(2026, 9, 29),
            now_utc_fn=lambda: dt.datetime(2026, 9, 29, 15, 20, tzinfo=UTC), config=FAST,
        )
        db_session.flush()
        rows = events.events_for_date(db_session, trading_date=D(2026, 9, 29))
        assert rows[0].category == events.CYCLE_START
