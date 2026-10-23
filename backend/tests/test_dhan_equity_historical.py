from __future__ import annotations

import datetime as dt
import hashlib
import json

import pytest

from backend.app.data.raw.adapter import FetchedContent, IngestStatus, ProviderAdapter, ingest
from backend.app.data.raw.artifact import RawIntegrityError
from backend.app.data.raw.store import RawStore
from backend.app.data.sources import dhan_equity_historical as deh
from backend.app.data.sources.errors import InvalidContentError
from backend.app.data.sources.provider_credentials import DhanCredentials

D = dt.date
T0 = dt.datetime(2026, 9, 25, 10, 0, tzinfo=dt.timezone.utc)
CREDS = DhanCredentials(client_id="1100000000", access_token="super-secret-jwt-value")

FROM, TO = D(2026, 8, 26), D(2026, 9, 25)
KEY = deh.source_key_for(2885, FROM, TO)


def good_body(n: int = 3) -> bytes:
    base_epoch = 1756188900
    data = {
        "open": [100.0 + i for i in range(n)], "high": [101.0 + i for i in range(n)], "low": [99.0 + i for i in range(n)],
        "close": [100.5 + i for i in range(n)], "volume": [1000 + i for i in range(n)],
        "timestamp": [base_epoch + i * 60 for i in range(n)],
    }
    return json.dumps(data).encode()


class FakeResponse:
    def __init__(self, status=200, content=b"", headers=None):
        self.status_code, self.content = status, content
        self.headers = headers or {"Content-Type": "application/json", "Content-Length": str(len(content)), "Date": "Fri, 25 Sep 2026 10:00:00 GMT"}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self, response):
        self.response, self.requests = response, []

    def post(self, url, json=None, headers=None, timeout=None):
        self.requests.append((url, json, headers, timeout))
        return self.response


class TestSourceKey:
    def test_round_trip(self):
        k = deh.source_key_for(2885, FROM, TO)
        assert k == "NSE_EQ-2885-1m-20260826-20260925"
        parsed = deh.parse_source_key(k)
        assert (parsed.security_id, parsed.from_date, parsed.to_date, parsed.interval) == (2885, FROM, TO, 1)

    def test_malformed_key_rejected(self):
        with pytest.raises(ValueError):
            deh.parse_source_key("not-a-valid-key")


class TestAdapter:
    def test_satisfies_the_protocol(self):
        assert isinstance(deh.DhanEquityHistoricalAdapter(CREDS), ProviderAdapter)

    def test_fetch_posts_the_expected_body_and_auth_headers(self):
        s = FakeSession(FakeResponse(200, good_body()))
        got = deh.DhanEquityHistoricalAdapter(CREDS, session=s).fetch(deh.DATASET, KEY)
        url, body, headers, timeout = s.requests[0]
        assert url == deh._ENDPOINT and timeout == 30
        assert body == {"securityId": "2885", "exchangeSegment": "NSE_EQ", "instrument": "EQUITY", "interval": 1, "oi": False,
                         "fromDate": "2026-08-26", "toDate": "2026-09-25"}
        assert headers["access-token"] == CREDS.access_token and headers["dhanClientId"] == CREDS.client_id
        assert got.content == good_body() and got.http_status == 200 and got.source_url == deh._ENDPOINT and got.media_type == "application/json"

    def test_only_whitelisted_response_headers_are_kept(self):
        s = FakeSession(FakeResponse(200, good_body(), headers={"Content-Type": "application/json", "Set-Cookie": "session=abc"}))
        got = deh.DhanEquityHistoricalAdapter(CREDS, session=s).fetch(deh.DATASET, KEY)
        assert "Set-Cookie" not in got.headers and got.headers["Content-Type"] == "application/json"

    def test_404_is_none_and_5xx_raises(self):
        assert deh.DhanEquityHistoricalAdapter(CREDS, session=FakeSession(FakeResponse(404))).fetch(deh.DATASET, KEY) is None
        with pytest.raises(RuntimeError):
            deh.DhanEquityHistoricalAdapter(CREDS, session=FakeSession(FakeResponse(500))).fetch(deh.DATASET, KEY)

    def test_unknown_dataset_rejected(self):
        a = deh.DhanEquityHistoricalAdapter(CREDS, session=FakeSession(FakeResponse(200, good_body())))
        with pytest.raises(ValueError):
            a.fetch("nope", KEY)
        with pytest.raises(ValueError):
            a.source_url("nope", KEY)

    @pytest.mark.parametrize("bad, why", [
        (b"not json at all", "not valid JSON"),
        (b"[]", "not a JSON object"),
        (json.dumps({"open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]}).encode(), "missing required fields"),
        (json.dumps({"open": 1, "high": [1], "low": [1], "close": [1], "volume": [1], "timestamp": [1]}).encode(), "not lists"),
        (json.dumps({"open": [1, 2], "high": [1], "low": [1], "close": [1], "volume": [1], "timestamp": [1]}).encode(), "inconsistent lengths"),
        (json.dumps({"status": "failure", "remarks": {"error_code": "DH-905"}, "data": ""}).encode(), "missing required fields"),
    ])
    def test_malformed_responses_are_rejected(self, bad, why):
        with pytest.raises(InvalidContentError, match=why):
            deh.DhanEquityHistoricalAdapter(CREDS).validate(deh.DATASET, bad)

    def test_well_shaped_but_empty_response_is_valid(self):
        empty = json.dumps({k: [] for k in ("open", "high", "low", "close", "volume", "timestamp")}).encode()
        deh.DhanEquityHistoricalAdapter(CREDS).validate(deh.DATASET, empty)


class TestRawStoreIntegration:
    @pytest.fixture
    def store(self, tmp_path):
        return RawStore(tmp_path / "raw")

    def _adapter(self, content=None):
        return deh.DhanEquityHistoricalAdapter(CREDS, session=FakeSession(FakeResponse(200, content if content is not None else good_body())))

    def test_round_trip_exact_bytes(self, store):
        body = good_body()
        a = self._adapter(body)
        r = ingest(store, a, deh.DATASET, KEY, now=T0)
        assert r.status is IngestStatus.CREATED
        recovered = store.read_bytes("dhan", deh.DATASET, KEY)
        assert recovered == body
        assert json.loads(recovered) == json.loads(body)

    def test_content_identity_matches_stdlib_sha256(self, store):
        body = good_body()
        r = ingest(store, self._adapter(body), deh.DATASET, KEY, now=T0)
        assert r.sha256 == hashlib.sha256(body).hexdigest()

    def test_identical_content_is_a_noop_on_refresh(self, store):
        body = good_body()
        ingest(store, self._adapter(body), deh.DATASET, KEY, now=T0)
        r2 = ingest(store, self._adapter(body), deh.DATASET, KEY, refresh=True, now=T0)
        assert r2.status is IngestStatus.UNCHANGED
        assert len(store.versions("dhan", deh.DATASET, KEY)) == 1

    def test_different_content_produces_different_identity_and_a_new_version(self, store):
        first = good_body(3)
        ingest(store, self._adapter(first), deh.DATASET, KEY, now=T0)
        second = good_body(5)
        r = ingest(store, self._adapter(second), deh.DATASET, KEY, refresh=True, now=T0)
        assert r.status is IngestStatus.NEW_VERSION and r.version == 2
        assert r.sha256 != hashlib.sha256(first).hexdigest() and r.sha256 == hashlib.sha256(second).hexdigest()
        assert store.read_bytes("dhan", deh.DATASET, KEY, version=1) == first
        assert store.read_bytes("dhan", deh.DATASET, KEY, version=2) == second

    def test_immutability_a_tampered_artifact_refuses_further_writes(self, store):
        body = good_body()
        ingest(store, self._adapter(body), deh.DATASET, KEY, now=T0)
        store.artifact_path("dhan", deh.DATASET, KEY).write_bytes(b"tampered raw bytes")
        with pytest.raises(RawIntegrityError):
            ingest(store, self._adapter(good_body(5)), deh.DATASET, KEY, refresh=True, now=T0)
        with pytest.raises(RawIntegrityError):
            store.read_bytes("dhan", deh.DATASET, KEY)

    def test_required_provider_and_request_metadata_is_retained(self, store):
        ingest(store, self._adapter(), deh.DATASET, KEY, now=T0)
        m = store.latest("dhan", deh.DATASET, KEY)
        assert m.provider == "dhan" and m.dataset == deh.DATASET and m.source_key == KEY
        assert m.source_url == deh._ENDPOINT and m.fetcher_version == "dhan_equity_historical/1"
        assert m.license_status == "UNCLEAR" and m.fetched_at == T0.isoformat() and m.http_status == 200
        assert m.media_type == "application/json"
        parsed_key = deh.parse_source_key(m.source_key)
        assert (parsed_key.security_id, parsed_key.from_date, parsed_key.to_date, parsed_key.interval) == (2885, FROM, TO, 1)

    def test_malformed_response_is_quarantined_and_the_key_is_retried_next_time(self, store):
        bad = b"<html>error</html>"
        r = ingest(store, self._adapter(bad), deh.DATASET, KEY, now=T0)
        assert r.status is IngestStatus.QUARANTINED
        assert store.exists("dhan", deh.DATASET, KEY) is False
        assert list((store.root / "_quarantine").rglob("*"))
        r2 = ingest(store, self._adapter(good_body()), deh.DATASET, KEY, now=T0)
        assert r2.status is IngestStatus.CREATED

    def test_raw_timestamp_array_is_preserved_exactly_no_conversion(self, store):
        body = good_body()
        original_timestamps = json.loads(body)["timestamp"]
        ingest(store, self._adapter(body), deh.DATASET, KEY, now=T0)
        stored = json.loads(store.read_bytes("dhan", deh.DATASET, KEY))
        assert stored["timestamp"] == original_timestamps

    def test_credentials_never_appear_in_stored_metadata_or_content(self, store):
        ingest(store, self._adapter(), deh.DATASET, KEY, now=T0)
        m = store.latest("dhan", deh.DATASET, KEY)
        sidecar_text = m.to_json()
        content_text = store.read_bytes("dhan", deh.DATASET, KEY).decode()
        for secret in (CREDS.access_token, CREDS.client_id):
            assert secret not in sidecar_text
            assert secret not in content_text
            assert secret not in json.dumps(m.response_headers)
