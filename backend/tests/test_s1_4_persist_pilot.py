from __future__ import annotations

import datetime as dt
import json

import pytest

from backend.app.data.raw.store import RawStore
from backend.app.data.sources import dhan_equity_historical as deh
from backend.app.db.models import IntradayEquityBar
from scripts import run_s1_4_persist_intraday_pilot as script

UTC = dt.timezone.utc
D = dt.date
FROM, TO = D(2026, 8, 26), D(2026, 8, 26)
E0, E1, E2 = 1787715900, 1787715960, 1787716020


def _clean():
    return {"open": [100.0]*3, "high": [101.0]*3, "low": [99.0]*3, "close": [100.5]*3, "volume": [1000]*3, "timestamp": [E0, E1, E2]}


@pytest.fixture
def raw_store(tmp_path):
    return RawStore(tmp_path / "raw")


def _seed(store, security_id, data):
    key = deh.source_key_for(security_id, FROM, TO)
    store.put(provider="dhan", dataset=deh.DATASET, source_key=key, content=json.dumps(data).encode(), source_url=deh._ENDPOINT,
              http_status=200, response_headers={}, fetcher_version="test/1", license_status="UNCLEAR", fetched_at=dt.datetime(2026, 9, 25, tzinfo=UTC))


class TestRunOrchestration:
    def test_persists_all_normalized_bars(self, db_session, raw_store):
        _seed(raw_store, 2885, _clean())
        report = script.run(db_session, raw_store, {"RELIANCE": 2885}, from_date=FROM, to_date=TO)
        db_session.commit()
        assert report["summary"]["total_persisted_rows"] == 3
        assert db_session.query(IntradayEquityBar).count() == 3

    def test_running_twice_is_idempotent(self, db_session, raw_store):
        _seed(raw_store, 2885, _clean())
        script.run(db_session, raw_store, {"RELIANCE": 2885}, from_date=FROM, to_date=TO)
        db_session.commit()
        script.run(db_session, raw_store, {"RELIANCE": 2885}, from_date=FROM, to_date=TO)
        db_session.commit()
        assert db_session.query(IntradayEquityBar).count() == 3

    def test_missing_s1_2_report_exits_2(self, tmp_path, monkeypatch):
        monkeypatch.setattr(script, "S1_2_REPORT", tmp_path / "nope.json")
        rc = script.main([])
        assert rc == 2
