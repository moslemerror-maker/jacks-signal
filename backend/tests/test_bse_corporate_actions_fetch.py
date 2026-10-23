from __future__ import annotations

import json
from datetime import date

import pytest
import requests

from backend.app.data.sources import bse_corporate_actions as bse
from backend.app.data.sources.bse_corporate_actions import (
    fetch_bse_corporate_actions,
    fetch_bse_corporate_actions_snapshot,
)
from backend.app.data.sources.fetch_result import FetchResult

_ROWS = [
    {"scrip_code": 500325, "short_name": "RELIND", "Ex_date": "07 Sep 2017", "Purpose": "Bonus issue 1:1",
     "RD_Date": "09 Sep 2017", "BCRD_FROM": "", "BCRD_TO": "", "ND_START_DATE": "", "ND_END_DATE": "", "payment_date": ""},
    {"scrip_code": 500325, "short_name": "RELIND", "Ex_date": "16 Aug 2019", "Purpose": "Dividend - Rs. - 6.5000",
     "RD_Date": "17 Aug 2019"},
]


def _resp(status=200, content: bytes | None = None, url="http://fake/bse"):
    r = requests.Response()
    r.status_code = status
    r._content = json.dumps(_ROWS).encode() if content is None else content
    r.url = url
    return r


class _Session:
    def __init__(self, *responses):
        self._responses, self.calls = list(responses), []

    def get(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]


def _data_files(d):
    return [p for p in d.glob("*.json") if not p.name.endswith(".meta.json")]


class TestRequestAndSnapshot:
    def test_the_window_is_sent_as_fdate_and_tdate_in_yyyymmdd_with_the_scrip_code(self, tmp_path):
        s = _Session(_resp())
        fetch_bse_corporate_actions_snapshot("500325", "RELIANCE", tmp_path, date(2015, 1, 1), date(2026, 9, 20), session=s)
        p = s.calls[0]["params"]
        assert (p["Fdate"], p["TDate"], p["scripcode"]) == ("20150101", "20260920", "500325")
        assert s.calls[0]["url"] == bse._BSE_ACTIONS_URL

    def test_it_sends_no_isin_and_asks_for_no_identity_mapping(self, tmp_path):
        s = _Session(_resp())
        fetch_bse_corporate_actions_snapshot("500325", "RELIANCE", tmp_path, date(2015, 1, 1), date(2026, 9, 20), session=s)
        assert not any("isin" in k.lower() for k in s.calls[0]["params"])

    def test_a_snapshot_and_sidecar_are_written_with_full_provenance(self, tmp_path):
        r = fetch_bse_corporate_actions_snapshot(
            "500325", "RELIANCE", tmp_path, date(2015, 1, 1), date(2018, 12, 31), purpose="c1-part-1of3", session=_Session(_resp()))
        assert isinstance(r, FetchResult) and r.created and r.record_count == 2
        assert r.path.name.startswith("RELIANCE_20150101-20181231_")
        meta = json.loads(r.path.with_suffix(r.path.suffix + ".meta.json").read_text())
        assert meta["sha256"] == r.sha256 and meta["scrip_code"] == "500325" and meta["purpose"] == "c1-part-1of3"
        assert meta["query_window"] == {"from_date": "2015-01-01", "to_date": "2018-12-31"}
        assert meta["query_params"]["Fdate"] == "20150101" and meta["fetched_at"] == r.fetched_at
        assert "validation evidence only" in meta["note"] and "OUR clock" in meta["note"]

    def test_the_stored_bytes_are_exactly_the_response(self, tmp_path):
        body = json.dumps(_ROWS, indent=1).encode()
        r = fetch_bse_corporate_actions_snapshot("500325", "RELIANCE", tmp_path, date(2015, 1, 1), date(2026, 9, 20),
                                                 session=_Session(_resp(content=body)))
        assert r.path.read_bytes() == body

    def test_an_identical_refetch_of_the_same_window_is_a_no_op(self, tmp_path):
        s = _Session(_resp())
        a = fetch_bse_corporate_actions_snapshot("500325", "RELIANCE", tmp_path, date(2015, 1, 1), date(2026, 9, 20), session=s)
        before = a.path.read_bytes()
        b = fetch_bse_corporate_actions_snapshot("500325", "RELIANCE", tmp_path, date(2015, 1, 1), date(2026, 9, 20), session=s)
        assert (a.created, b.created) == (True, False) and b.path == a.path and a.path.read_bytes() == before
        assert len(_data_files(tmp_path / "RELIANCE")) == 1

    def test_a_changed_response_for_the_same_window_is_a_new_snapshot_and_the_old_one_survives(self, tmp_path):
        a = fetch_bse_corporate_actions_snapshot("500325", "RELIANCE", tmp_path, date(2015, 1, 1), date(2026, 9, 20), session=_Session(_resp()))
        changed = json.dumps(_ROWS[:1]).encode()
        b = fetch_bse_corporate_actions_snapshot("500325", "RELIANCE", tmp_path, date(2015, 1, 1), date(2026, 9, 20),
                                                 session=_Session(_resp(content=changed)))
        assert b.created and b.path != a.path and a.path.exists()
        assert json.loads(a.path.read_text()) == _ROWS

    def test_identical_bytes_for_two_different_windows_are_two_recorded_snapshots(self, tmp_path):
        a = fetch_bse_corporate_actions_snapshot("500325", "RELIANCE", tmp_path, date(2015, 1, 1), date(2018, 12, 31), session=_Session(_resp()))
        b = fetch_bse_corporate_actions_snapshot("500325", "RELIANCE", tmp_path, date(2019, 1, 1), date(2022, 12, 31), session=_Session(_resp()))
        assert a.created and b.created and a.path != b.path and a.sha256 == b.sha256

    def test_a_reversed_window_is_refused_before_any_request(self, tmp_path):
        s = _Session(_resp())
        with pytest.raises(ValueError):
            fetch_bse_corporate_actions_snapshot("500325", "RELIANCE", tmp_path, date(2020, 1, 1), date(2019, 1, 1), session=s)
        assert s.calls == []

    def test_an_empty_list_is_a_valid_recorded_snapshot_not_an_error(self, tmp_path):
        r = fetch_bse_corporate_actions_snapshot("500325", "RELIANCE", tmp_path, date(2015, 1, 1), date(2015, 2, 1),
                                                 session=_Session(_resp(content=b"[]")))
        assert r.created and r.record_count == 0


class TestMalformedAndFailedResponses:
    @pytest.mark.parametrize("body", [b"<html>blocked</html>", b'{"Table": []}', b'[1, 2]', b""])
    def test_a_malformed_body_is_quarantined_and_never_stored_as_a_snapshot(self, tmp_path, body):
        with pytest.raises(bse.InvalidContentError):
            fetch_bse_corporate_actions_snapshot("500325", "RELIANCE", tmp_path, date(2015, 1, 1), date(2026, 9, 20),
                                                 session=_Session(_resp(content=body)))
        assert _data_files(tmp_path / "RELIANCE") == []
        quarantined = [p for p in (tmp_path / "quarantine").glob("*.json") if not p.name.endswith(".meta.json")]
        assert len(quarantined) == 1 and quarantined[0].read_bytes() == body
        meta = json.loads(quarantined[0].with_suffix(".json.meta.json").read_text())
        assert meta["reason"] and meta["http_status"] == 200

    def test_an_http_error_status_raises_and_writes_nothing(self, tmp_path):
        with pytest.raises(requests.HTTPError):
            fetch_bse_corporate_actions_snapshot("500325", "RELIANCE", tmp_path, date(2015, 1, 1), date(2026, 9, 20),
                                                 session=_Session(_resp(status=503)))
        assert _data_files(tmp_path / "RELIANCE") == []


class TestBackwardCompatibleWrapper:
    def test_wrapper_returns_the_path_when_new_and_none_when_identical(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bse, "_new_session", lambda: _Session(_resp()))
        first = fetch_bse_corporate_actions("500325", "RELIANCE", tmp_path, date(2015, 1, 1), date(2026, 9, 20))
        second = fetch_bse_corporate_actions("500325", "RELIANCE", tmp_path, date(2015, 1, 1), date(2026, 9, 20))
        assert first is not None and second is None


class TestBseIsNeverAFirstClassSource:
    def test_the_module_does_not_touch_the_database_or_the_events_tables(self):
        src = open(bse.__file__, encoding="utf-8").read()
        assert "sqlalchemy" not in src and "Session(" not in src.replace("requests.Session(", "")
        assert "CorporateActionEvent" not in src and "InstrumentIsin" not in src
