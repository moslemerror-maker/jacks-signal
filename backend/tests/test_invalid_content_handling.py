from __future__ import annotations

import http.server
import json
import threading
import zipfile
from datetime import date
from pathlib import Path

import pytest

from backend.app.data.sources.errors import InvalidContentError
from backend.app.data.sources.nse_bhavcopy import (
    _validate_bhavcopy_content,
    fetch_bhavcopy,
)
from backend.app.data.sources.nse_index_history import (
    _validate_index_history_content,
    fetch_index_history,
)


def _write_zip_bytes(members: dict[str, bytes]) -> bytes:
    import io

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return buf.getvalue()


class _OneShotHandler(http.server.BaseHTTPRequestHandler):
    status = 200
    body = b""

    def do_GET(self):
        self.send_response(type(self).status)
        self.send_header("Content-Length", str(len(type(self).body)))
        self.end_headers()
        self.wfile.write(type(self).body)

    def log_message(self, format, *args):  # noqa: A002
        pass


def _start_server(status: int, body: bytes):
    handler = type("Handler", (_OneShotHandler,), {"status": status, "body": body})
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


class TestValidateBhavcopyContent:
    def test_valid_zip_passes(self):
        content = _write_zip_bytes({"data.csv": b"SYMBOL,SERIES\nRELIANCE,EQ\n"})
        _validate_bhavcopy_content(content)

    def test_non_zip_content_raises(self):
        with pytest.raises(Exception):
            _validate_bhavcopy_content(b"this is an html error page, not a zip")

    def test_truncated_zip_raises(self):
        full = _write_zip_bytes({"data.csv": b"x" * 1000})
        with pytest.raises(Exception):
            _validate_bhavcopy_content(full[: len(full) // 2])

    def test_zip_with_wrong_member_count_raises(self):
        content = _write_zip_bytes({"a.csv": b"x", "b.csv": b"y"})
        with pytest.raises(Exception):
            _validate_bhavcopy_content(content)


class TestValidateIndexHistoryContent:
    def test_valid_json_passes(self):
        content = json.dumps({"data": [{"EOD_INDEX_NAME": "NIFTY 50"}]}).encode()
        _validate_index_history_content(content)

    def test_invalid_json_raises(self):
        with pytest.raises(InvalidContentError, match="not valid JSON"):
            _validate_index_history_content(b"{not json,,,")

    def test_missing_data_key_raises(self):
        content = json.dumps({"unexpected": "shape"}).encode()
        with pytest.raises(InvalidContentError, match='no "data" key'):
            _validate_index_history_content(content)

    def test_data_not_a_list_raises(self):
        content = json.dumps({"data": "error message string"}).encode()
        with pytest.raises(InvalidContentError, match="not a list"):
            _validate_index_history_content(content)

    def test_html_error_page_raises(self):
        with pytest.raises(InvalidContentError):
            _validate_index_history_content(b"<html><body>Access Denied</body></html>")


class TestFetchBhavcopyQuarantine:
    def test_invalid_content_is_quarantined_not_committed(self, tmp_path, monkeypatch):
        bad_body = b"<html>not a bhavcopy</html>"
        server, port = _start_server(200, bad_body)
        monkeypatch.setattr(
            "backend.app.data.sources.nse_bhavcopy.bhavcopy_url",
            lambda trade_date: f"http://127.0.0.1:{port}/bhav.zip",
        )
        try:
            with pytest.raises(InvalidContentError):
                fetch_bhavcopy(date(2024, 1, 1), tmp_path)
        finally:
            server.shutdown()

        assert not (tmp_path / "bhav.zip").exists()
        assert not (tmp_path / "bhav.zip.meta.json").exists()

        q_file = tmp_path / "quarantine" / "bhav.zip"
        q_meta = tmp_path / "quarantine" / "bhav.zip.meta.json"
        assert q_file.read_bytes() == bad_body
        meta = json.loads(q_meta.read_text())
        assert meta["trade_date"] == "2024-01-01"
        assert meta["http_status"] == 200
        assert "reason" in meta and meta["reason"]

    def test_quarantined_date_is_retried_on_next_run(self, tmp_path, monkeypatch):
        call_count = {"n": 0}
        good_content = _write_zip_bytes({"data.csv": b"SYMBOL,SERIES\nRELIANCE,EQ\n"})

        def fake_get(url, trade_date=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return 200, b"<html>bad</html>"
            return 200, good_content

        server1, port1 = _start_server(200, b"<html>bad</html>")
        monkeypatch.setattr(
            "backend.app.data.sources.nse_bhavcopy.bhavcopy_url",
            lambda trade_date: f"http://127.0.0.1:{port1}/bhav.zip",
        )
        try:
            with pytest.raises(InvalidContentError):
                fetch_bhavcopy(date(2024, 1, 1), tmp_path)
        finally:
            server1.shutdown()

        assert not (tmp_path / "bhav.zip").exists()

        server2, port2 = _start_server(200, good_content)
        monkeypatch.setattr(
            "backend.app.data.sources.nse_bhavcopy.bhavcopy_url",
            lambda trade_date: f"http://127.0.0.1:{port2}/bhav.zip",
        )
        try:
            result_path = fetch_bhavcopy(date(2024, 1, 1), tmp_path)
        finally:
            server2.shutdown()

        assert result_path.exists()
        assert result_path.read_bytes() == good_content
        assert (tmp_path / "bhav.zip.meta.json").exists()

    def test_valid_content_is_not_quarantined(self, tmp_path, monkeypatch):
        good_content = _write_zip_bytes({"data.csv": b"SYMBOL,SERIES\nRELIANCE,EQ\n"})
        server, port = _start_server(200, good_content)
        monkeypatch.setattr(
            "backend.app.data.sources.nse_bhavcopy.bhavcopy_url",
            lambda trade_date: f"http://127.0.0.1:{port}/bhav.zip",
        )
        try:
            fetch_bhavcopy(date(2024, 1, 1), tmp_path)
        finally:
            server.shutdown()

        assert (tmp_path / "bhav.zip").exists()
        assert not (tmp_path / "quarantine").exists()


class TestFetchIndexHistoryQuarantine:
    def test_invalid_content_is_quarantined_not_committed(self, tmp_path, monkeypatch):
        bad_body = b"<html>captcha challenge</html>"

        class FakeSession:
            def get(self, *a, **kw):
                import requests as req

                resp = req.Response()
                resp.status_code = 200
                resp._content = bad_body
                resp.url = "http://fake/indicesHistory"
                return resp

        monkeypatch.setattr(
            "backend.app.data.sources.nse_index_history._new_session",
            lambda: FakeSession(),
        )

        with pytest.raises(InvalidContentError):
            fetch_index_history("NIFTY 50", date(2024, 1, 1), date(2024, 1, 31), tmp_path)

        filename = "NIFTY_50_2024-01-01_2024-01-31.json"
        assert not (tmp_path / filename).exists()
        q_file = tmp_path / "quarantine" / filename
        q_meta = tmp_path / "quarantine" / f"{filename}.meta.json"
        assert q_file.read_bytes() == bad_body
        meta = json.loads(q_meta.read_text())
        assert meta["reason"]

    def test_quarantined_chunk_is_retried_on_next_run(self, tmp_path, monkeypatch):
        good_content = json.dumps({"data": []}).encode()

        class BadSession:
            def get(self, *a, **kw):
                import requests as req

                resp = req.Response()
                resp.status_code = 200
                resp._content = b"not json at all"
                resp.url = "http://fake/indicesHistory"
                return resp

        class GoodSession:
            def get(self, *a, **kw):
                import requests as req

                resp = req.Response()
                resp.status_code = 200
                resp._content = good_content
                resp.url = "http://fake/indicesHistory"
                return resp

        monkeypatch.setattr(
            "backend.app.data.sources.nse_index_history._new_session",
            lambda: BadSession(),
        )
        with pytest.raises(InvalidContentError):
            fetch_index_history("NIFTY 50", date(2024, 1, 1), date(2024, 1, 31), tmp_path)

        filename = "NIFTY_50_2024-01-01_2024-01-31.json"
        assert not (tmp_path / filename).exists()

        monkeypatch.setattr(
            "backend.app.data.sources.nse_index_history._new_session",
            lambda: GoodSession(),
        )
        result_path = fetch_index_history(
            "NIFTY 50", date(2024, 1, 1), date(2024, 1, 31), tmp_path
        )
        assert result_path.exists()
        assert result_path.read_bytes() == good_content
