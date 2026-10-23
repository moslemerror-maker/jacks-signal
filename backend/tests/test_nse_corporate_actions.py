from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest
import requests

from backend.app.data.sources.errors import InvalidContentError
from backend.app.data.sources.nse_corporate_actions import fetch_corporate_actions


def _fake_response(status_code: int, content: bytes, url: str = "http://fake/corpActions") -> requests.Response:
    resp = requests.Response()
    resp.status_code = status_code
    resp._content = content
    resp.url = url
    return resp


class _FakeSession:
    def __init__(self, responses: list[requests.Response]):
        self._responses = list(responses)
        self.call_count = 0
        self.calls: list[dict] = []

    def get(self, *args, **kwargs):
        idx = min(self.call_count, len(self._responses) - 1)
        self.call_count += 1
        self.calls.append(kwargs)
        return self._responses[idx]


def _patch_session(monkeypatch, *responses: requests.Response) -> _FakeSession:
    fake = _FakeSession(list(responses))
    monkeypatch.setattr(
        "backend.app.data.sources.nse_corporate_actions._new_session", lambda: fake
    )
    return fake


def _data_files(directory) -> list:
    return [p for p in directory.glob("*.json") if not p.name.endswith(".meta.json")]


_RELIANCE_SAMPLE = [
    {
        "symbol": "RELIANCE", "series": "EQ", "comp": "Reliance Industries Limited",
        "isin": "INE002A01018", "faceVal": "10", "subject": "Dividend - Rs 6 Per Share",
        "exDate": "05-Jun-2026", "recDate": "05-Jun-2026", "bcStartDate": "-",
        "bcEndDate": "-", "ndStartDate": "-", "ndEndDate": "-",
        "caBroadcastDate": None, "ind": "-",
    },
    {
        "symbol": "RELIANCE", "series": "EQ", "comp": "Reliance Industries Limited",
        "isin": "INE002A01018", "faceVal": "10", "subject": "Bonus 1:1",
        "exDate": "28-Oct-2024", "recDate": "28-Oct-2024", "bcStartDate": "-",
        "bcEndDate": "-", "ndStartDate": "-", "ndEndDate": "-",
        "caBroadcastDate": None, "ind": "-",
    },
]


class TestExactPayloadPreservation:
    def test_written_bytes_are_byte_identical_to_the_response(self, tmp_path, monkeypatch):
        raw_body = (
            b'[{"symbol":"RELIANCE",  "unknownField": 123, "subject":"Bonus 1:1",\n'
            b'   "nested": {"x": [1,2,3]}}]'
        )
        _patch_session(monkeypatch, _fake_response(200, raw_body))
        path = fetch_corporate_actions("RELIANCE", tmp_path)
        assert path is not None
        assert path.read_bytes() == raw_body

    def test_unknown_fields_are_not_discarded(self, tmp_path, monkeypatch):
        raw_body = json.dumps([{"symbol": "X", "totallyUnknownField": "kept-verbatim"}]).encode()
        _patch_session(monkeypatch, _fake_response(200, raw_body))
        path = fetch_corporate_actions("X", tmp_path)
        stored = json.loads(path.read_text())
        assert stored[0]["totallyUnknownField"] == "kept-verbatim"


class TestChecksumCorrectness:
    def test_meta_sha256_matches_actual_content_hash(self, tmp_path, monkeypatch):
        import hashlib

        body = json.dumps(_RELIANCE_SAMPLE).encode()
        _patch_session(monkeypatch, _fake_response(200, body))
        path = fetch_corporate_actions("RELIANCE", tmp_path)
        meta = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text())
        assert meta["sha256"] == hashlib.sha256(body).hexdigest()
        assert meta["sha256"][:16] in path.name


class TestProvenanceSidecar:
    def test_all_required_fields_present(self, tmp_path, monkeypatch):
        body = json.dumps(_RELIANCE_SAMPLE).encode()
        _patch_session(monkeypatch, _fake_response(200, body))
        path = fetch_corporate_actions("RELIANCE", tmp_path)
        meta = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text())
        for field in (
            "symbol", "provider", "source_url", "query_params", "fetched_at",
            "sha256", "http_status", "content_length", "record_count",
            "announcement_date_observed", "parsing_applied", "schema_confirmed",
        ):
            assert field in meta, f"missing provenance field: {field}"
        assert meta["parsing_applied"] is False
        assert meta["schema_confirmed"] is False
        assert meta["record_count"] == len(_RELIANCE_SAMPLE)

    def test_query_params_reflect_what_was_actually_sent(self, tmp_path, monkeypatch):
        body = json.dumps(_RELIANCE_SAMPLE).encode()
        _patch_session(monkeypatch, _fake_response(200, body))
        path = fetch_corporate_actions(
            "RELIANCE", tmp_path, from_date=date(2015, 1, 1), to_date=date(2026, 9, 11)
        )
        meta = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text())
        assert meta["query_params"] == {
            "index": "equities", "symbol": "RELIANCE",
            "from_date": "01-01-2015", "to_date": "11-09-2026",
        }
        assert meta["fetch_mode"] == "windowed"
        assert meta["query_window"] == {"from_date": "2015-01-01", "to_date": "2026-09-11"}

    def test_no_date_params_omits_from_to_keys(self, tmp_path, monkeypatch):
        body = json.dumps(_RELIANCE_SAMPLE).encode()
        _patch_session(monkeypatch, _fake_response(200, body))
        path = fetch_corporate_actions("RELIANCE", tmp_path)
        meta = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text())
        assert meta["query_params"] == {"index": "equities", "symbol": "RELIANCE"}
        assert "from_date" not in meta["query_params"]
        assert "to_date" not in meta["query_params"]
        assert meta["fetch_mode"] == "default" and meta["query_window"] is None


class TestAnnouncementDateVsIngestionTimestamp:
    def test_fetched_at_is_our_clock_not_derived_from_any_source_date(self, tmp_path, monkeypatch):
        body = json.dumps(_RELIANCE_SAMPLE).encode()
        _patch_session(monkeypatch, _fake_response(200, body))
        before = datetime.now(timezone.utc)
        path = fetch_corporate_actions("RELIANCE", tmp_path)
        after = datetime.now(timezone.utc)
        meta = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text())
        fetched_at = datetime.fromisoformat(meta["fetched_at"])
        assert before <= fetched_at <= after
        for row in _RELIANCE_SAMPLE:
            assert row["exDate"] not in meta["fetched_at"]

    def test_announcement_date_observed_false_when_every_record_has_null_broadcast_date(
        self, tmp_path, monkeypatch
    ):
        body = json.dumps(_RELIANCE_SAMPLE).encode()
        _patch_session(monkeypatch, _fake_response(200, body))
        path = fetch_corporate_actions("RELIANCE", tmp_path)
        meta = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text())
        assert meta["announcement_date_observed"] is False

    def test_announcement_date_observed_true_when_any_record_has_a_broadcast_date(
        self, tmp_path, monkeypatch
    ):
        rows = [dict(_RELIANCE_SAMPLE[0]), dict(_RELIANCE_SAMPLE[1])]
        rows[1]["caBroadcastDate"] = "27-Oct-2024 18:30:00"
        body = json.dumps(rows).encode()
        _patch_session(monkeypatch, _fake_response(200, body))
        path = fetch_corporate_actions("RELIANCE", tmp_path)
        meta = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text())
        assert meta["announcement_date_observed"] is True

    def test_no_announcement_date_is_ever_invented(self, tmp_path, monkeypatch):
        body = json.dumps(_RELIANCE_SAMPLE).encode()
        _patch_session(monkeypatch, _fake_response(200, body))
        path = fetch_corporate_actions("RELIANCE", tmp_path)
        meta = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text())
        assert "announcement_date" not in meta
        assert "announcement_at" not in meta
        assert "publication_date" not in meta


class TestIdempotentRefetch:
    def test_byte_identical_refetch_is_a_true_no_op(self, tmp_path, monkeypatch):
        body = json.dumps(_RELIANCE_SAMPLE).encode()
        _patch_session(monkeypatch, _fake_response(200, body))
        first = fetch_corporate_actions("RELIANCE", tmp_path)
        assert first is not None

        _patch_session(monkeypatch, _fake_response(200, body))
        second = fetch_corporate_actions("RELIANCE", tmp_path)
        assert second is None

        files = _data_files(tmp_path / "RELIANCE")
        assert len(files) == 1


class TestCorrectedResponseCreatesNewSnapshotNeverOverwrites:
    def test_different_content_writes_a_new_snapshot_leaving_the_old_one_untouched(
        self, tmp_path, monkeypatch
    ):
        original_body = json.dumps(_RELIANCE_SAMPLE).encode()
        _patch_session(monkeypatch, _fake_response(200, original_body))
        first_path = fetch_corporate_actions("RELIANCE", tmp_path)
        original_bytes_on_disk = first_path.read_bytes()

        corrected = list(_RELIANCE_SAMPLE) + [
            {
                "symbol": "RELIANCE", "series": "EQ", "comp": "Reliance Industries Limited",
                "isin": "INE002A01018", "faceVal": "10",
                "subject": "Dividend - Rs 5.5 Per Share (CORRECTED)",
                "exDate": "14-Aug-2025", "recDate": "14-Aug-2025", "bcStartDate": "-",
                "bcEndDate": "-", "ndStartDate": "-", "ndEndDate": "-",
                "caBroadcastDate": None, "ind": "-",
            }
        ]
        corrected_body = json.dumps(corrected).encode()
        _patch_session(monkeypatch, _fake_response(200, corrected_body))
        second_path = fetch_corporate_actions("RELIANCE", tmp_path)

        assert second_path is not None
        assert second_path != first_path
        assert first_path.exists()
        assert first_path.read_bytes() == original_bytes_on_disk
        assert second_path.read_bytes() == corrected_body

        files = _data_files(tmp_path / "RELIANCE")
        assert len(files) == 2


class TestMalformedResponseHandling:
    def test_non_json_body_is_quarantined_not_committed(self, tmp_path, monkeypatch):
        bad_body = b"<html>captcha challenge</html>"
        _patch_session(monkeypatch, _fake_response(200, bad_body))
        with pytest.raises(InvalidContentError):
            fetch_corporate_actions("RELIANCE", tmp_path)
        assert _data_files(tmp_path / "RELIANCE") == []
        assert (tmp_path / "quarantine").exists()

    def test_json_object_instead_of_list_is_rejected(self, tmp_path, monkeypatch):
        bad_body = json.dumps({"data": _RELIANCE_SAMPLE}).encode()
        _patch_session(monkeypatch, _fake_response(200, bad_body))
        with pytest.raises(InvalidContentError, match="list"):
            fetch_corporate_actions("RELIANCE", tmp_path)

    def test_list_of_non_objects_is_rejected(self, tmp_path, monkeypatch):
        bad_body = json.dumps(["not", "objects"]).encode()
        _patch_session(monkeypatch, _fake_response(200, bad_body))
        with pytest.raises(InvalidContentError, match="object"):
            fetch_corporate_actions("RELIANCE", tmp_path)

    def test_quarantine_preserves_the_bad_content_for_inspection(self, tmp_path, monkeypatch):
        bad_body = b"not json at all"
        _patch_session(monkeypatch, _fake_response(200, bad_body))
        with pytest.raises(InvalidContentError):
            fetch_corporate_actions("RELIANCE", tmp_path)
        quarantined = _data_files(tmp_path / "quarantine")
        assert len(quarantined) == 1
        assert quarantined[0].read_bytes() == bad_body


class TestErrorBehavior:
    def test_non_2xx_response_raises_httperror_and_writes_nothing(self, tmp_path, monkeypatch):
        _patch_session(monkeypatch, _fake_response(500, b"server error"))
        with pytest.raises(requests.HTTPError):
            fetch_corporate_actions("RELIANCE", tmp_path)
        assert not (tmp_path / "RELIANCE").exists() or not list(
            (tmp_path / "RELIANCE").glob("*.json")
        )

    def test_404_raises_httperror(self, tmp_path, monkeypatch):
        _patch_session(monkeypatch, _fake_response(404, b"not found"))
        with pytest.raises(requests.HTTPError):
            fetch_corporate_actions("UNKNOWNSYMBOL", tmp_path)


class TestWindowedFetchC1:

    def test_the_parameter_names_actually_sent_are_from_date_and_to_date_never_from_to(
        self, tmp_path, monkeypatch
    ):
        from backend.app.data.sources.nse_corporate_actions import fetch_corporate_actions_snapshot

        fake = _patch_session(monkeypatch, _fake_response(200, json.dumps(_RELIANCE_SAMPLE).encode()))
        fetch_corporate_actions_snapshot(
            "RELIANCE", tmp_path, from_date=date(2015, 1, 1), to_date=date(2026, 9, 20),
        )
        sent = fake.calls[0]["params"]
        assert sent["from_date"] == "01-01-2015" and sent["to_date"] == "20-09-2026"
        assert "from" not in sent and "to" not in sent

    def test_default_mode_sends_no_window_parameters_at_all(self, tmp_path, monkeypatch):
        from backend.app.data.sources.nse_corporate_actions import fetch_corporate_actions_snapshot

        fake = _patch_session(monkeypatch, _fake_response(200, json.dumps(_RELIANCE_SAMPLE).encode()))
        fetch_corporate_actions_snapshot("RELIANCE", tmp_path)
        assert fake.calls[0]["params"] == {"index": "equities", "symbol": "RELIANCE"}

    def test_window_is_part_of_the_file_name_and_provenance(self, tmp_path, monkeypatch):
        from backend.app.data.sources.nse_corporate_actions import fetch_corporate_actions_snapshot

        _patch_session(monkeypatch, _fake_response(200, json.dumps(_RELIANCE_SAMPLE).encode()))
        r = fetch_corporate_actions_snapshot(
            "RELIANCE", tmp_path, from_date=date(2015, 1, 1), to_date=date(2018, 12, 31),
            purpose="c1-part-1/3",
        )
        assert r.created and "RELIANCE_20150101-20181231_" in r.path.name
        meta = json.loads(r.path.with_suffix(r.path.suffix + ".meta.json").read_text())
        assert meta["purpose"] == "c1-part-1/3"
        assert meta["sha256"] == r.sha256 and meta["record_count"] == len(_RELIANCE_SAMPLE)

    def test_same_bytes_for_two_different_windows_are_two_separately_recorded_snapshots(
        self, tmp_path, monkeypatch
    ):
        from backend.app.data.sources.nse_corporate_actions import fetch_corporate_actions_snapshot

        body = json.dumps(_RELIANCE_SAMPLE).encode()
        _patch_session(monkeypatch, _fake_response(200, body))
        whole = fetch_corporate_actions_snapshot("RELIANCE", tmp_path, from_date=date(2015, 1, 1), to_date=date(2026, 9, 20))
        part = fetch_corporate_actions_snapshot("RELIANCE", tmp_path, from_date=date(2019, 1, 1), to_date=date(2026, 9, 20))
        assert whole.created and part.created and whole.path != part.path
        assert whole.sha256 == part.sha256
        assert len(_data_files(tmp_path / "RELIANCE")) == 2

    def test_repeating_the_same_window_with_identical_bytes_is_a_no_op_that_still_returns_the_path(
        self, tmp_path, monkeypatch
    ):
        from backend.app.data.sources.nse_corporate_actions import fetch_corporate_actions_snapshot

        body = json.dumps(_RELIANCE_SAMPLE).encode()
        _patch_session(monkeypatch, _fake_response(200, body))
        first = fetch_corporate_actions_snapshot("RELIANCE", tmp_path, from_date=date(2015, 1, 1), to_date=date(2026, 9, 20))
        original_meta = first.path.with_suffix(first.path.suffix + ".meta.json").read_text()
        second = fetch_corporate_actions_snapshot("RELIANCE", tmp_path, from_date=date(2015, 1, 1), to_date=date(2026, 9, 20))
        assert not second.created and second.path == first.path
        assert first.path.with_suffix(first.path.suffix + ".meta.json").read_text() == original_meta

    def test_a_changed_response_for_the_same_window_is_a_new_snapshot(self, tmp_path, monkeypatch):
        from backend.app.data.sources.nse_corporate_actions import fetch_corporate_actions_snapshot

        _patch_session(monkeypatch, _fake_response(200, json.dumps(_RELIANCE_SAMPLE).encode()))
        a = fetch_corporate_actions_snapshot("RELIANCE", tmp_path, from_date=date(2015, 1, 1), to_date=date(2026, 9, 20))
        _patch_session(monkeypatch, _fake_response(200, json.dumps(_RELIANCE_SAMPLE[:1]).encode()))
        b = fetch_corporate_actions_snapshot("RELIANCE", tmp_path, from_date=date(2015, 1, 1), to_date=date(2026, 9, 20))
        assert a.created and b.created and a.path != b.path
        assert a.path.exists() and b.path.exists()

    def test_half_a_window_or_an_inverted_window_is_rejected_before_any_request(self, tmp_path, monkeypatch):
        from backend.app.data.sources.nse_corporate_actions import fetch_corporate_actions_snapshot

        fake = _patch_session(monkeypatch, _fake_response(200, b"[]"))
        with pytest.raises(ValueError, match="together"):
            fetch_corporate_actions_snapshot("RELIANCE", tmp_path, from_date=date(2015, 1, 1))
        with pytest.raises(ValueError, match=">="):
            fetch_corporate_actions_snapshot("RELIANCE", tmp_path, from_date=date(2016, 1, 1), to_date=date(2015, 1, 1))
        assert fake.call_count == 0
