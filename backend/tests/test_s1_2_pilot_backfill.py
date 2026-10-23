from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from backend.app.data.raw.store import RawStore
from backend.app.data.sources import dhan_equity_historical as deh
from backend.app.data.sources import dhan_security_master as dsm
from backend.app.data.sources.provider_credentials import DhanCredentials
from scripts import run_s1_2_dhan_equity_pilot_backfill as script

CREDS = DhanCredentials(client_id="1100000000", access_token="secret")

_MASTER = pd.DataFrame([
    ("NSE", 2885, "EQUITY", "RELIANCE", "EQ"), ("NSE", 11536, "EQUITY", "TCS", "EQ"), ("NSE", 14366, "EQUITY", "IDEA", "EQ"),
], columns=["SEM_EXM_EXCH_ID", "SEM_SMST_SECURITY_ID", "SEM_INSTRUMENT_NAME", "SEM_TRADING_SYMBOL", "SEM_SERIES"])


def _body(n=3):
    return json.dumps({k: list(range(n)) for k in ("open", "high", "low", "close", "volume", "timestamp")}).encode()


class FakeAdapter:

    provider_id = "dhan"
    fetcher_version = "fake/1"
    license_status = "UNCLEAR"

    def __init__(self, by_security_id):
        self.by_security_id, self.calls = by_security_id, []

    def fetch(self, dataset, source_key):
        self.calls.append(source_key)
        parsed = deh.parse_source_key(source_key)
        v = self.by_security_id.get(parsed.security_id, _body())
        if isinstance(v, Exception):
            raise v
        return type("F", (), {"content": v, "http_status": 200, "source_url": deh._ENDPOINT, "headers": {}, "media_type": "application/json"})()

    def validate(self, dataset, content):
        return deh.validate_intraday_json_shape(content)

    def datasets(self):
        return (deh.DATASET,)


class TestRunOrchestration:
    @pytest.fixture
    def store(self, tmp_path):
        return RawStore(tmp_path / "raw")

    def test_all_symbols_resolve_and_acquire(self, store):
        report = script.run(CREDS, store=store, symbols=("RELIANCE", "TCS", "IDEA"), master=_MASTER, adapter=FakeAdapter({}), pace_seconds=0)
        assert report["resolution_failures"] == []
        assert len(report["acquisitions"]) == 3 and report["failures"] == []
        assert report["summary"]["successful_acquisitions"] == 3 and report["summary"]["total_raw_artifacts"] == 3
        assert all(a["sha256"] for a in report["acquisitions"])

    def test_unresolvable_symbol_is_reported_separately_and_others_still_proceed(self, store):
        report = script.run(CREDS, store=store, symbols=("RELIANCE", "NOTREAL"), master=_MASTER, adapter=FakeAdapter({}), pace_seconds=0)
        assert report["resolution_failures"] == ["NOTREAL"]
        assert len(report["acquisitions"]) == 1 and report["acquisitions"][0]["symbol"] == "RELIANCE"

    def test_malformed_response_is_a_reported_failure_not_a_crash(self, store):
        adapter = FakeAdapter({2885: b"<html>bad</html>"})
        report = script.run(CREDS, store=store, symbols=("RELIANCE", "TCS"), master=_MASTER, adapter=adapter, pace_seconds=0)
        assert report["summary"]["failed_acquisitions"] == 1 and report["summary"]["successful_acquisitions"] == 1
        assert report["failures"][0]["symbol"] == "RELIANCE" and report["failures"][0]["status"] == "QUARANTINED"

    def test_transport_error_is_a_reported_failure_not_a_crash_and_not_retried(self, store):
        adapter = FakeAdapter({2885: ConnectionError("network down")})
        report = script.run(CREDS, store=store, symbols=("RELIANCE", "TCS"), master=_MASTER, adapter=adapter, pace_seconds=0)
        assert report["summary"]["failed_acquisitions"] == 1 and report["failures"][0]["status"] == "ERROR"
        assert adapter.calls.count(deh.source_key_for(2885, script.FROM_DATE, script.TO_DATE)) == 1

    def test_summary_storage_arithmetic(self, store):
        report = script.run(CREDS, store=store, symbols=("RELIANCE", "TCS"), master=_MASTER, adapter=FakeAdapter({}), pace_seconds=0)
        sizes = [a["raw_response_size_bytes"] for a in report["acquisitions"]]
        s = report["summary"]
        assert s["total_storage_bytes"] == sum(sizes) and s["largest_artifact_bytes"] == max(sizes) and s["smallest_artifact_bytes"] == min(sizes)
        assert s["average_artifact_size_bytes"] == sum(sizes) / len(sizes)

    def test_no_store_mode_fetches_and_validates_but_writes_nothing(self, tmp_path):
        store = RawStore(tmp_path / "raw")
        report = script.run(CREDS, store=None, symbols=("RELIANCE",), master=_MASTER, adapter=FakeAdapter({}), pace_seconds=0)
        assert report["acquisitions"][0]["status"] == "FETCHED_NOT_STORED"
        assert store.manifest_entries() == []


class TestMainWithoutCredentials:
    def test_exits_2_and_prints_which_vars_are_missing(self, monkeypatch, capsys):
        monkeypatch.delenv("DHAN_CLIENT_ID", raising=False)
        monkeypatch.delenv("DHAN_ACCESS_TOKEN", raising=False)
        monkeypatch.setattr(script, "load_dotenv", lambda: None)
        rc = script.main([])
        out = capsys.readouterr().out
        assert rc == 2 and "NOT CONFIGURED" in out and ".env.example" in out
