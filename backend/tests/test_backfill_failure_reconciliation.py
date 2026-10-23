from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import requests

from backend.app.db.models import MarketBar
from scripts import backfill_phase1

FIXTURES = Path(__file__).parent / "fixtures" / "bhavcopy"
UDIFF_FILE = FIXTURES / "BhavCopy_NSE_CM_0_0_0_20250915_F_0000.csv.zip"


def _http_error(status: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f"{status} error", response=response)


@pytest.fixture
def isolated_backfill(monkeypatch, db_session):
    monkeypatch.setattr(backfill_phase1, "get_session", lambda: db_session)
    monkeypatch.setattr(
        backfill_phase1,
        "_fetch_reference_bhavcopy",
        lambda end, max_lookback_days=10: (UDIFF_FILE, date(2025, 9, 15)),
    )
    return db_session


class TestReconcileOrRaise:
    def test_matching_counts_does_not_raise(self):
        backfill_phase1._reconcile_or_raise(
            expected_count=10, accounted_count=10, label="test", buckets={"a": 10}
        )

    def test_mismatched_counts_raises(self):
        with pytest.raises(RuntimeError, match="reconciliation failed"):
            backfill_phase1._reconcile_or_raise(
                expected_count=10, accounted_count=7, label="test", buckets={"a": 7}
            )


class TestEquityFailedDatesAreRecordedNotSilent:
    def test_non_404_failure_raises_after_processing_all_days(
        self, isolated_backfill, monkeypatch
    ):
        attempted = []

        def fake_fetch(trade_date, raw_dir):
            attempted.append(trade_date)
            raise _http_error(403)

        monkeypatch.setattr(backfill_phase1, "fetch_bhavcopy", fake_fetch)

        with pytest.raises(RuntimeError, match="did not complete successfully"):
            backfill_phase1.backfill_equities(date(2024, 7, 1), date(2024, 7, 5))

        assert len(attempted) == 5

    def test_mix_of_404_and_failure_still_raises_for_the_failure(
        self, isolated_backfill, monkeypatch
    ):

        def fake_fetch(trade_date, raw_dir):
            if trade_date == date(2024, 7, 3):
                raise _http_error(403)
            raise _http_error(404)

        monkeypatch.setattr(backfill_phase1, "fetch_bhavcopy", fake_fetch)

        with pytest.raises(RuntimeError, match="1.*date.*failed"):
            backfill_phase1.backfill_equities(date(2024, 7, 1), date(2024, 7, 5))

    def test_all_404s_completes_successfully(self, isolated_backfill, monkeypatch):

        def fake_fetch(trade_date, raw_dir):
            raise _http_error(404)

        monkeypatch.setattr(backfill_phase1, "fetch_bhavcopy", fake_fetch)

        backfill_phase1.backfill_equities(date(2024, 7, 1), date(2024, 7, 5))
        assert isolated_backfill.query(MarketBar).count() == 0


class TestIndexFailedChunksAreRecordedNotSilent:
    def test_non_404_failure_raises_after_processing_all_chunks(self, db_session, monkeypatch):
        monkeypatch.setattr(backfill_phase1, "get_session", lambda: db_session)
        attempted = []

        def fake_fetch(index_symbol, chunk_start, chunk_end, raw_dir):
            attempted.append((index_symbol, chunk_start, chunk_end))
            raise _http_error(500)

        monkeypatch.setattr(backfill_phase1, "fetch_index_history", fake_fetch)

        with pytest.raises(RuntimeError, match="did not complete successfully"):
            backfill_phase1.backfill_indices(date(2024, 1, 1), date(2024, 2, 28))

        assert len(attempted) == 4
