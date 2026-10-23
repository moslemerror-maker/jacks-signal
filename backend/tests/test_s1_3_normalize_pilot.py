from __future__ import annotations

import datetime as dt
import json

import pytest

from backend.app.data.raw.store import RawStore
from backend.app.data.sources import dhan_equity_historical as deh
from scripts import run_s1_3_dhan_intraday_normalize_pilot as script

UTC = dt.timezone.utc
D = dt.date
FROM, TO = D(2026, 8, 26), D(2026, 8, 26)
E0, E1, E2 = 1787715900, 1787715960, 1787716020


def _clean(n=3):
    return {"open": [100.0]*n, "high": [101.0]*n, "low": [99.0]*n, "close": [100.5]*n, "volume": [1000]*n, "timestamp": [E0, E1, E2][:n]}


@pytest.fixture
def store(tmp_path):
    return RawStore(tmp_path / "raw")


def _seed(store, security_id, data):
    key = deh.source_key_for(security_id, FROM, TO)
    store.put(provider="dhan", dataset=deh.DATASET, source_key=key, content=json.dumps(data).encode(), source_url=deh._ENDPOINT,
              http_status=200, response_headers={}, fetcher_version="test/1", license_status="UNCLEAR", fetched_at=dt.datetime(2026, 9, 25, tzinfo=UTC))


class TestRunOrchestration:
    def test_partial_session_has_zero_rejections_but_is_flagged_incomplete(self, store):
        _seed(store, 2885, _clean())
        _seed(store, 11536, _clean())
        report = script.run(store, {"RELIANCE": 2885, "TCS": 11536}, from_date=FROM, to_date=TO)
        assert report["summary"]["total_raw_rows"] == 6 and report["summary"]["total_normalized_rows"] == 6
        assert report["symbols"]["RELIANCE"]["security_id"] == 2885
        assert report["symbols"]["RELIANCE"]["rejected_row_count"] == 0
        assert report["symbols"]["RELIANCE"]["reconciliation"]["discrepancies"] == []
        assert report["summary"]["all_clean"] is False

    def test_full_360_bar_session_is_all_clean(self, store):
        n = 360
        full = {"open": [100.0]*n, "high": [101.0]*n, "low": [99.0]*n, "close": [100.5]*n, "volume": [1000]*n, "timestamp": [E0 + i*60 for i in range(n)]}
        _seed(store, 2885, full)
        report = script.run(store, {"RELIANCE": 2885}, from_date=FROM, to_date=TO)
        assert report["summary"]["all_clean"] is True
        assert report["symbols"]["RELIANCE"]["complete_360_sessions"] == 1

    def test_rejected_rows_flip_all_clean_to_false(self, store):
        raw = _clean()
        raw["timestamp"][1] = raw["timestamp"][0]
        _seed(store, 2885, raw)
        report = script.run(store, {"RELIANCE": 2885}, from_date=FROM, to_date=TO)
        assert report["summary"]["all_clean"] is False
        assert report["symbols"]["RELIANCE"]["rejected_row_count"] == 2

    def test_missing_s1_2_report_exits_2(self, tmp_path, monkeypatch):
        monkeypatch.setattr(script, "S1_2_REPORT", tmp_path / "nope.json")
        rc = script.main([])
        assert rc == 2
