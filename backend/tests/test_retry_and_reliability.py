from __future__ import annotations

import http.server
import io
import json
import socket
import threading
import time
import zipfile
from pathlib import Path

import pytest
import requests

from backend.app.data.sources.nse_bhavcopy import _get_session, fetch_bhavcopy


def _valid_bhavcopy_zip_bytes(content: bytes = b"SYMBOL,SERIES\nRELIANCE,EQ\n") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("data.csv", content)
    return buf.getvalue()


class _ScriptedHandler(http.server.BaseHTTPRequestHandler):

    responses: list[tuple[int, bytes]] = []
    request_count = 0

    def do_GET(self):
        idx = min(type(self).request_count, len(type(self).responses) - 1)
        type(self).request_count += 1
        status, body = type(self).responses[idx]
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        pass


def _start_server(responses: list[tuple[int, bytes]]):
    handler = type("Handler", (_ScriptedHandler,), {"responses": responses, "request_count": 0})
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    return server, handler, port


@pytest.fixture
def fast_backoff(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda seconds: None)


class TestRetryOn5xx:
    def test_retries_and_eventually_succeeds(self, fast_backoff):
        server, handler, port = _start_server(
            [(503, b"unavailable"), (503, b"unavailable"), (200, b"ok-body")]
        )
        try:
            response = _get_session().get(f"http://127.0.0.1:{port}/x", timeout=5)
        finally:
            server.shutdown()

        assert response.status_code == 200
        assert response.content == b"ok-body"
        assert handler.request_count == 3

    def test_exhausts_retries_and_raises(self, fast_backoff):
        server, handler, port = _start_server([(503, b"unavailable")])
        try:
            with pytest.raises(requests.exceptions.RetryError):
                _get_session().get(f"http://127.0.0.1:{port}/x", timeout=5)
        finally:
            server.shutdown()

        assert handler.request_count == 6


class TestNoRetryOn404:
    def test_404_is_not_retried(self, fast_backoff):
        server, handler, port = _start_server([(404, b"not found")])
        try:
            response = _get_session().get(f"http://127.0.0.1:{port}/x", timeout=5)
        finally:
            server.shutdown()

        assert response.status_code == 404
        assert handler.request_count == 1

    def test_fetch_bhavcopy_raises_httperror_on_404(self, tmp_path, monkeypatch):
        server, handler, port = _start_server([(404, b"not found")])
        monkeypatch.setattr(
            "backend.app.data.sources.nse_bhavcopy.bhavcopy_url",
            lambda trade_date: f"http://127.0.0.1:{port}/bhav.zip",
        )
        try:
            from datetime import date

            with pytest.raises(requests.HTTPError) as exc_info:
                fetch_bhavcopy(date(2024, 1, 1), tmp_path)
        finally:
            server.shutdown()

        assert exc_info.value.response.status_code == 404
        assert not (tmp_path / "bhav.zip").exists()


class TestConnectionRefused:
    def test_retries_connection_refused(self, fast_backoff):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        with pytest.raises(requests.exceptions.ConnectionError):
            _get_session().get(f"http://127.0.0.1:{port}/x", timeout=2)


class TestImmutability:
    def test_existing_raw_file_is_never_refetched_or_overwritten(self, tmp_path, monkeypatch):
        from datetime import date

        raw_dir = tmp_path
        existing_content = b"ORIGINAL-CONTENT-MUST-SURVIVE"

        monkeypatch.setattr(
            "backend.app.data.sources.nse_bhavcopy.bhavcopy_url",
            lambda trade_date: "http://127.0.0.1:1/should-never-be-requested",
        )
        url = "http://127.0.0.1:1/should-never-be-requested"
        filename = url.rsplit("/", 1)[-1]
        dest = raw_dir / filename
        dest.write_bytes(existing_content)
        dest.with_suffix(dest.suffix + ".meta.json").write_text("{}")

        result_path = fetch_bhavcopy(date(2024, 1, 1), raw_dir)

        assert result_path == dest
        assert dest.read_bytes() == existing_content

    def test_missing_meta_json_triggers_refetch_even_if_data_file_exists(
        self, tmp_path, monkeypatch
    ):
        from datetime import date

        fresh_content = _valid_bhavcopy_zip_bytes(b"SYMBOL,SERIES\nFRESH,EQ\n")
        server, handler, port = _start_server([(200, fresh_content)])
        monkeypatch.setattr(
            "backend.app.data.sources.nse_bhavcopy.bhavcopy_url",
            lambda trade_date: f"http://127.0.0.1:{port}/bhav.zip",
        )
        (tmp_path / "bhav.zip").write_bytes(b"STALE-PARTIAL-CONTENT")

        try:
            fetch_bhavcopy(date(2024, 1, 1), tmp_path)
        finally:
            server.shutdown()

        assert handler.request_count == 1
        assert (tmp_path / "bhav.zip").read_bytes() == fresh_content


class TestProvenance:
    def test_meta_json_records_expected_fields_with_correct_checksum(
        self, tmp_path, monkeypatch
    ):
        import hashlib
        from datetime import date

        content = _valid_bhavcopy_zip_bytes()
        server, handler, port = _start_server([(200, content)])
        monkeypatch.setattr(
            "backend.app.data.sources.nse_bhavcopy.bhavcopy_url",
            lambda trade_date: f"http://127.0.0.1:{port}/bhav.zip",
        )
        try:
            fetch_bhavcopy(date(2024, 3, 15), tmp_path)
        finally:
            server.shutdown()

        meta = json.loads((tmp_path / "bhav.zip.meta.json").read_text())
        assert meta["trade_date"] == "2024-03-15"
        assert meta["source_url"] == f"http://127.0.0.1:{port}/bhav.zip"
        assert meta["http_status"] == 200
        assert meta["content_length"] == len(content)
        assert meta["sha256"] == hashlib.sha256(content).hexdigest()
        assert "fetched_at" in meta
