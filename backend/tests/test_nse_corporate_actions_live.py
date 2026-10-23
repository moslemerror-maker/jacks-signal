from __future__ import annotations

import pytest
import requests

from backend.app.data.sources.nse_corporate_actions import fetch_corporate_actions


def _live_source_available() -> bool:
    try:
        import requests as req

        session = req.Session()
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
                )
            }
        )
        session.get("https://www.nseindia.com/", timeout=10)
        return True
    except requests.exceptions.RequestException:
        return False


@pytest.fixture(scope="module")
def live_source_guard():
    if not _live_source_available():
        pytest.skip("live NSE source unreachable from this environment")


class TestLiveFetchEstablishesRealResponseBehavior:
    def test_fetching_reliance_succeeds_and_produces_a_valid_snapshot(
        self, tmp_path, live_source_guard
    ):
        path = fetch_corporate_actions("RELIANCE", tmp_path)
        assert path is not None
        assert path.exists()
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        assert meta_path.exists()

        import json

        payload = json.loads(path.read_text())
        assert isinstance(payload, list)
        assert len(payload) > 0
        assert all(row.get("symbol") == "RELIANCE" for row in payload)

    def test_refetching_the_same_symbol_immediately_is_a_true_no_op_or_a_new_snapshot(
        self, tmp_path, live_source_guard
    ):
        first = fetch_corporate_actions("RELIANCE", tmp_path)
        first_bytes = first.read_bytes()

        second = fetch_corporate_actions("RELIANCE", tmp_path)

        assert first.read_bytes() == first_bytes
        if second is None:
            pass
        else:
            assert second.exists()
            assert second != first
