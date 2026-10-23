from __future__ import annotations

import datetime as dt
import json

import pytest
import requests
from sqlalchemy import text

from backend.app.data.backfill.executor import ChunkOutcome, ExecutorConfig
from backend.app.data.canonical.intraday_equity_bar import CanonicalIntradayEquityBar, IntradayProvider
from backend.app.data.live.live_poller import bars_up_to_now, poll_symbol_once, poll_universe_once
from backend.app.data.raw.store import RawStore
from backend.app.data.sources import dhan_equity_historical as deh
from backend.app.data.sources.provider_credentials import DhanCredentials

D = dt.date
UTC = dt.timezone.utc
CREDS = DhanCredentials(client_id="1100000000", access_token="secret")
FAST = ExecutorConfig(pace_seconds=0, backoff_base_seconds=0, jitter_max_seconds=0, sleep_fn=lambda s: None)


def _bar(security_id: int, bar_start_utc: dt.datetime, *, symbol: str = "TEST") -> CanonicalIntradayEquityBar:
    return CanonicalIntradayEquityBar(
        provider=IntradayProvider.DHAN, exchange_segment="NSE_EQ", security_id=security_id, trading_symbol=symbol,
        raw_timestamp_epoch=int(bar_start_utc.timestamp()), bar_start_utc=bar_start_utc,
        bar_start_ist=bar_start_utc.astimezone(dt.timezone(dt.timedelta(hours=5, minutes=30))),
        trading_date=bar_start_utc.date(), interval_minutes=1,
        open=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0,
        source_provider="dhan", source_dataset="nse_eq_intraday_1m", source_key="TEST-KEY",
        source_sha256="x" * 64, normalizer_version="test/1",
    )


class TestBarsUpToNowAdversarial:

    def test_a_bar_strictly_before_the_current_minute_is_kept(self):
        now = dt.datetime(2026, 9, 29, 6, 17, 24, tzinfo=UTC)
        completed = _bar(1, dt.datetime(2026, 9, 29, 6, 16, 0, tzinfo=UTC))
        assert bars_up_to_now([completed], now_utc=now) == [completed]

    def test_a_bar_in_the_currently_forming_minute_is_discarded(self):
        now = dt.datetime(2026, 9, 29, 6, 17, 24, tzinfo=UTC)
        forming = _bar(1, dt.datetime(2026, 9, 29, 6, 17, 0, tzinfo=UTC))
        assert bars_up_to_now([forming], now_utc=now) == []

    def test_a_bar_exactly_at_the_current_minute_boundary_is_discarded(self):
        now = dt.datetime(2026, 9, 29, 6, 17, 0, tzinfo=UTC)
        forming = _bar(1, dt.datetime(2026, 9, 29, 6, 17, 0, tzinfo=UTC))
        assert bars_up_to_now([forming], now_utc=now) == []

    def test_a_clock_skewed_future_bar_is_also_discarded_never_just_the_current_minute(self):
        now = dt.datetime(2026, 9, 29, 6, 17, 24, tzinfo=UTC)
        future = _bar(1, dt.datetime(2026, 9, 29, 6, 25, 0, tzinfo=UTC))
        assert bars_up_to_now([future], now_utc=now) == []

    def test_mixed_batch_keeps_only_the_strictly_completed_bars(self):
        now = dt.datetime(2026, 9, 29, 9, 20, 0, tzinfo=UTC)
        bars = [_bar(1, dt.datetime(2026, 9, 29, 9, 15 + i, tzinfo=UTC)) for i in range(6)]
        kept = bars_up_to_now(bars, now_utc=now)
        assert [b.bar_start_utc.minute for b in kept] == [15, 16, 17, 18, 19]

    def test_appending_a_future_bar_never_changes_the_result_for_earlier_bars(self):
        now = dt.datetime(2026, 9, 29, 9, 20, 0, tzinfo=UTC)
        earlier = [_bar(1, dt.datetime(2026, 9, 29, 9, 15 + i, tzinfo=UTC)) for i in range(5)]
        without_future = bars_up_to_now(earlier, now_utc=now)
        with_future = bars_up_to_now(earlier + [_bar(1, dt.datetime(2026, 9, 29, 9, 25, tzinfo=UTC))], now_utc=now)
        assert without_future == with_future

    def test_empty_input_returns_empty(self):
        assert bars_up_to_now([], now_utc=dt.datetime(2026, 9, 29, 9, 20, tzinfo=UTC)) == []


def good_body(n=3, base_epoch=1790653500):
    return json.dumps({
        "open": [100.0 + i for i in range(n)], "high": [101.0 + i for i in range(n)], "low": [99.0 + i for i in range(n)],
        "close": [100.5 + i for i in range(n)], "volume": [1000 + i for i in range(n)],
        "timestamp": [base_epoch + i * 60 for i in range(n)],
    }).encode()


class QueuedFakeResponse:
    def __init__(self, status=200, content=b""):
        self.status_code, self.content, self.headers = status, content, {"Content-Type": "application/json"}

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError(f"{self.status_code} error")
            err.response = self
            raise err


class QueuedFakeSession:
    def __init__(self, items):
        self.items = list(items)
        self.calls = 0

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls += 1
        item = self.items[min(self.calls - 1, len(self.items) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


def _adapter(items):
    return deh.DhanEquityHistoricalAdapter(CREDS, session=QueuedFakeSession(items))


@pytest.fixture
def store(tmp_path):
    return RawStore(tmp_path / "raw")


class TestPollSymbolOnce:
    def test_success_normalizes_and_persists_completed_bars_only(self, db_session, store):
        now = dt.datetime(2026, 9, 29, 9, 20, 0, tzinfo=UTC)
        result = poll_symbol_once(
            db_session, store, _adapter([QueuedFakeResponse(200, good_body(n=3))]),
            security_id=25, trading_symbol="ADANIENT", today=D(2026, 9, 29), now_utc=now, config=FAST,
        )
        assert result.outcome is ChunkOutcome.SUCCESS
        assert result.bars_fetched == 3
        assert result.bars_persisted == 3
        db_session.flush()
        n = db_session.execute(text("SELECT COUNT(*) FROM intraday_equity_bars WHERE security_id = 25")).scalar_one()
        assert n == 3

    def test_a_bar_in_the_currently_forming_minute_is_fetched_but_not_persisted(self, db_session, store):
        now = dt.datetime(2026, 9, 29, 3, 47, 30, tzinfo=UTC)
        result = poll_symbol_once(
            db_session, store, _adapter([QueuedFakeResponse(200, good_body(n=3))]),
            security_id=26, trading_symbol="X", today=D(2026, 9, 29), now_utc=now, config=FAST,
        )
        assert result.bars_fetched == 3
        assert result.bars_persisted == 2

    def test_401_is_reported_and_nothing_is_persisted(self, db_session, store):
        result = poll_symbol_once(
            db_session, store, _adapter([QueuedFakeResponse(401, b'{"error":"unauthorized"}')]),
            security_id=27, trading_symbol="X", today=D(2026, 9, 29), now_utc=dt.datetime(2026, 9, 29, 9, 20, tzinfo=UTC), config=FAST,
        )
        assert result.outcome is ChunkOutcome.HTTP_401_AUTHENTICATION
        assert result.bars_persisted == 0

    def test_repolling_the_same_symbol_with_more_bars_does_not_duplicate_earlier_ones(self, db_session, store):
        now = dt.datetime(2026, 9, 29, 9, 25, 0, tzinfo=UTC)
        poll_symbol_once(
            db_session, store, _adapter([QueuedFakeResponse(200, good_body(n=3))]),
            security_id=28, trading_symbol="X", today=D(2026, 9, 29), now_utc=now, config=FAST,
        )
        db_session.flush()
        poll_symbol_once(
            db_session, store, _adapter([QueuedFakeResponse(200, good_body(n=5))]),
            security_id=28, trading_symbol="X", today=D(2026, 9, 29), now_utc=now, config=FAST,
        )
        db_session.flush()
        n = db_session.execute(text("SELECT COUNT(*) FROM intraday_equity_bars WHERE security_id = 28")).scalar_one()
        assert n == 5


class TestPollUniverseOnce:
    def test_one_symbols_failure_does_not_block_the_others(self, db_session, store):
        no_retry = ExecutorConfig(pace_seconds=0, backoff_base_seconds=0, jitter_max_seconds=0, max_retries=0, sleep_fn=lambda s: None)
        adapter = _adapter([QueuedFakeResponse(500, b"server error"), QueuedFakeResponse(200, good_body(n=2))])
        results = poll_universe_once(
            db_session, store, adapter, securities=[(29, "BAD"), (30, "GOOD")], today=D(2026, 9, 29),
            now_utc_fn=lambda: dt.datetime(2026, 9, 29, 9, 20, tzinfo=UTC), config=no_retry,
        )
        assert results[0].outcome is ChunkOutcome.HTTP_5XX_PROVIDER
        assert results[1].outcome is ChunkOutcome.SUCCESS
        assert results[1].bars_persisted == 2

    def test_401_stops_the_sweep_but_keeps_prior_results(self, db_session, store):
        adapter = _adapter([QueuedFakeResponse(200, good_body(n=2)), QueuedFakeResponse(401, b"{}")])
        results = poll_universe_once(
            db_session, store, adapter, securities=[(31, "FIRST"), (32, "SECOND"), (33, "THIRD")], today=D(2026, 9, 29),
            now_utc_fn=lambda: dt.datetime(2026, 9, 29, 9, 20, tzinfo=UTC), config=FAST,
        )
        assert len(results) == 2
        assert results[0].outcome is ChunkOutcome.SUCCESS
        assert results[1].outcome is ChunkOutcome.HTTP_401_AUTHENTICATION
