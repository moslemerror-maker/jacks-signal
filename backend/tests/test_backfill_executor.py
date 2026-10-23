from __future__ import annotations

import datetime as dt
import json

import pytest
import requests

from backend.app.data.backfill.chunk_planner import BackfillChunk
from backend.app.data.backfill.executor import ChunkOutcome, ExecutorConfig, execute_plan
from backend.app.data.raw.store import RawStore
from backend.app.data.sources import dhan_equity_historical as deh
from backend.app.data.sources.provider_credentials import DhanCredentials

D = dt.date
CREDS = DhanCredentials(client_id="1100000000", access_token="secret")
FAST = ExecutorConfig(pace_seconds=0, backoff_base_seconds=0, jitter_max_seconds=0, sleep_fn=lambda s: None)


def good_body(n=3, base_epoch=1756188900):
    return json.dumps({
        "open": [100.0 + i for i in range(n)], "high": [101.0 + i for i in range(n)], "low": [99.0 + i for i in range(n)],
        "close": [100.5 + i for i in range(n)], "volume": [1000 + i for i in range(n)], "timestamp": [base_epoch + i * 60 for i in range(n)],
    }).encode()


def empty_body():
    return json.dumps({k: [] for k in ("open", "high", "low", "close", "volume", "timestamp")}).encode()


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


def chunk(security_id=2885, symbol="RELIANCE", from_date=D(2026, 8, 26), to_date=D(2026, 9, 25)):
    return BackfillChunk(security_id, symbol, from_date, to_date)


class TestSuccess:
    def test_single_successful_chunk(self, store):
        result = execute_plan(store, _adapter([QueuedFakeResponse(200, good_body())]), [chunk()], config=FAST)
        assert len(result.results) == 1
        r = result.results[0]
        assert r.outcome is ChunkOutcome.SUCCESS and r.bar_count == 3 and r.attempts == 1
        assert result.stopped_early is False

    def test_multiple_successful_chunks_across_securities(self, store):
        chunks = [chunk(2885, "RELIANCE"), chunk(11536, "TCS")]
        adapter = _adapter([QueuedFakeResponse(200, good_body()), QueuedFakeResponse(200, good_body())])
        result = execute_plan(store, adapter, chunks, config=FAST)
        assert [r.outcome for r in result.results] == [ChunkOutcome.SUCCESS, ChunkOutcome.SUCCESS]
        assert result.summary["SUCCESS"] == 2 and result.summary["total_planned"] == 2


class TestSkippedExistingAndResume:
    def test_already_stored_chunk_is_skipped_with_zero_network_calls(self, store):
        c = chunk()
        key = deh.source_key_for(c.security_id, c.from_date, c.to_date)
        store.put(provider="dhan", dataset=deh.DATASET, source_key=key, content=good_body(), source_url=deh._ENDPOINT,
                  http_status=200, response_headers={}, fetcher_version="test/1", license_status="UNCLEAR")
        adapter = _adapter([])
        result = execute_plan(store, adapter, [c], config=FAST)
        assert result.results[0].outcome is ChunkOutcome.SKIPPED_EXISTING
        assert result.results[0].bar_count == 3

    def test_crash_and_rerun_only_hits_network_for_remaining_chunks(self, store):
        chunks = [chunk(2885, "RELIANCE", D(2026, 1, 1), D(2026, 3, 31)), chunk(11536, "TCS", D(2026, 1, 1), D(2026, 3, 31))]
        session = QueuedFakeSession([QueuedFakeResponse(200, good_body())])
        adapter = deh.DhanEquityHistoricalAdapter(CREDS, session=session)
        execute_plan(store, adapter, chunks[:1], config=FAST)
        assert session.calls == 1

        session2 = QueuedFakeSession([QueuedFakeResponse(200, good_body())])
        adapter2 = deh.DhanEquityHistoricalAdapter(CREDS, session=session2)
        result = execute_plan(store, adapter2, chunks, config=FAST)
        assert result.results[0].outcome is ChunkOutcome.SKIPPED_EXISTING
        assert result.results[1].outcome is ChunkOutcome.SUCCESS
        assert session2.calls == 1

    def test_rerun_never_duplicates_raw_artifacts(self, store):
        c = chunk()
        adapter1 = _adapter([QueuedFakeResponse(200, good_body())])
        execute_plan(store, adapter1, [c], config=FAST)
        adapter2 = _adapter([])
        execute_plan(store, adapter2, [c], config=FAST)
        key = deh.source_key_for(c.security_id, c.from_date, c.to_date)
        assert len(store.versions("dhan", deh.DATASET, key)) == 1


class TestAuthenticationStop:
    def test_401_stops_the_plan_immediately_with_no_retry(self, store):
        chunks = [chunk(2885, "RELIANCE"), chunk(11536, "TCS")]
        adapter = _adapter([QueuedFakeResponse(401, b"unauthorized")])
        result = execute_plan(store, adapter, chunks, config=FAST)
        assert result.stopped_early is True and "401" in result.stop_reason
        assert len(result.results) == 1
        assert result.results[0].outcome is ChunkOutcome.HTTP_401_AUTHENTICATION
        assert result.results[0].attempts == 1

    def test_401_preserves_prior_successful_work(self, store):
        chunks = [chunk(2885, "RELIANCE"), chunk(11536, "TCS")]
        adapter = _adapter([QueuedFakeResponse(200, good_body()), QueuedFakeResponse(401, b"unauthorized")])
        result = execute_plan(store, adapter, chunks, config=FAST)
        assert result.results[0].outcome is ChunkOutcome.SUCCESS
        assert result.results[1].outcome is ChunkOutcome.HTTP_401_AUTHENTICATION
        assert result.stopped_early is True
        key = deh.source_key_for(2885, D(2026, 8, 26), D(2026, 9, 25))
        assert store.exists("dhan", deh.DATASET, key) is True


class TestRateLimitRetry:
    def test_429_then_success_is_a_bounded_retry_that_succeeds(self, store):
        adapter = _adapter([QueuedFakeResponse(429, b"rate limited"), QueuedFakeResponse(200, good_body())])
        result = execute_plan(store, adapter, [chunk()], config=FAST)
        r = result.results[0]
        assert r.outcome is ChunkOutcome.SUCCESS and r.attempts == 2

    def test_429_exhausts_retries_and_is_classified_failed_not_crashing_the_plan(self, store):
        config = ExecutorConfig(pace_seconds=0, backoff_base_seconds=0, jitter_max_seconds=0, sleep_fn=lambda s: None, max_retries=3)
        chunks = [chunk(2885, "RELIANCE"), chunk(11536, "TCS")]
        adapter = _adapter([QueuedFakeResponse(429, b"rate limited")] * (1 + config.max_retries) + [QueuedFakeResponse(200, good_body())])
        result = execute_plan(store, adapter, chunks, config=config)
        assert result.results[0].outcome is ChunkOutcome.HTTP_429_RATE_LIMIT
        assert result.results[0].attempts == 1 + config.max_retries
        assert result.stopped_early is False
        assert result.results[1].outcome is ChunkOutcome.SUCCESS


class TestServerErrorRetry:
    def test_5xx_retries_then_succeeds(self, store):
        adapter = _adapter([QueuedFakeResponse(503, b"unavailable"), QueuedFakeResponse(200, good_body())])
        result = execute_plan(store, adapter, [chunk()], config=FAST)
        assert result.results[0].outcome is ChunkOutcome.SUCCESS and result.results[0].attempts == 2

    def test_5xx_exhaustion_is_classified_and_plan_continues(self, store):
        config = ExecutorConfig(pace_seconds=0, backoff_base_seconds=0, jitter_max_seconds=0, sleep_fn=lambda s: None, max_retries=2)
        adapter = _adapter([QueuedFakeResponse(500, b"error")] * 10)
        result = execute_plan(store, adapter, [chunk()], config=config)
        assert result.results[0].outcome is ChunkOutcome.HTTP_5XX_PROVIDER
        assert result.results[0].attempts == 1 + config.max_retries
        assert result.stopped_early is False


class TestNetworkFailureRetry:
    def test_timeout_retries_then_succeeds(self, store):
        adapter = _adapter([requests.Timeout("timed out"), QueuedFakeResponse(200, good_body())])
        result = execute_plan(store, adapter, [chunk()], config=FAST)
        assert result.results[0].outcome is ChunkOutcome.SUCCESS and result.results[0].attempts == 2

    def test_connection_error_retries_then_succeeds(self, store):
        adapter = _adapter([requests.ConnectionError("network down"), QueuedFakeResponse(200, good_body())])
        result = execute_plan(store, adapter, [chunk()], config=FAST)
        assert result.results[0].outcome is ChunkOutcome.SUCCESS and result.results[0].attempts == 2

    def test_timeout_exhaustion_is_classified_network_timeout(self, store):
        config = ExecutorConfig(pace_seconds=0, backoff_base_seconds=0, jitter_max_seconds=0, sleep_fn=lambda s: None, max_retries=2)
        adapter = _adapter([requests.Timeout("timed out")] * 10)
        result = execute_plan(store, adapter, [chunk()], config=config)
        assert result.results[0].outcome is ChunkOutcome.NETWORK_TIMEOUT
        assert result.results[0].attempts == 1 + config.max_retries


class TestMalformedAndEmpty:
    def test_malformed_response_is_quarantined_not_retried_within_this_run(self, store):
        adapter = _adapter([QueuedFakeResponse(200, b"<html>not json</html>")])
        result = execute_plan(store, adapter, [chunk()], config=FAST)
        r = result.results[0]
        assert r.outcome is ChunkOutcome.MALFORMED_RESPONSE and r.attempts == 1

    def test_empty_but_well_shaped_response_is_classified_empty_not_success(self, store):
        adapter = _adapter([QueuedFakeResponse(200, empty_body())])
        result = execute_plan(store, adapter, [chunk()], config=FAST)
        r = result.results[0]
        assert r.outcome is ChunkOutcome.EMPTY_RESPONSE and r.bar_count == 0

    def test_malformed_chunk_does_not_block_the_rest_of_the_plan(self, store):
        chunks = [chunk(2885, "RELIANCE"), chunk(11536, "TCS")]
        adapter = _adapter([QueuedFakeResponse(200, b"garbage"), QueuedFakeResponse(200, good_body())])
        result = execute_plan(store, adapter, chunks, config=FAST)
        assert result.results[0].outcome is ChunkOutcome.MALFORMED_RESPONSE
        assert result.results[1].outcome is ChunkOutcome.SUCCESS
        assert result.stopped_early is False


class TestSequentialOnly:
    def test_calls_happen_in_plan_order_one_at_a_time(self, store):
        chunks = [chunk(2885, "RELIANCE"), chunk(11536, "TCS"), chunk(1333, "HDFCBANK")]
        session = QueuedFakeSession([QueuedFakeResponse(200, good_body())] * 3)
        adapter = deh.DhanEquityHistoricalAdapter(CREDS, session=session)
        execute_plan(store, adapter, chunks, config=FAST)
        assert session.calls == 3

    def test_default_config_pace_is_1_second_not_zero(self):
        assert ExecutorConfig().pace_seconds == 1.0


class TestPlanSummary:
    def test_summary_counts_every_outcome_class(self, store):
        chunks = [chunk(2885, "A"), chunk(11536, "B")]
        adapter = _adapter([QueuedFakeResponse(200, good_body()), QueuedFakeResponse(200, empty_body())])
        result = execute_plan(store, adapter, chunks, config=FAST)
        assert result.summary["SUCCESS"] == 1 and result.summary["EMPTY_RESPONSE"] == 1 and result.summary["total_planned"] == 2
