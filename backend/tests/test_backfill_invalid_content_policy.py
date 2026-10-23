from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from backend.app.data.sources.errors import InvalidContentError
from backend.app.db.models import MarketBar
from scripts import backfill_phase1

FIXTURES = Path(__file__).parent / "fixtures" / "bhavcopy"
UDIFF_FILE = FIXTURES / "BhavCopy_NSE_CM_0_0_0_20250915_F_0000.csv.zip"


@pytest.fixture
def isolated_backfill(monkeypatch, db_session):
    monkeypatch.setattr(backfill_phase1, "get_session", lambda: db_session)
    monkeypatch.setattr(
        backfill_phase1,
        "_fetch_reference_bhavcopy",
        lambda end, max_lookback_days=10: (UDIFF_FILE, date(2025, 9, 15)),
    )
    return db_session


class TestOnInvalidContentSkip:
    def test_skip_continues_past_quarantined_day_and_completes(
        self, isolated_backfill, monkeypatch
    ):
        def always_invalid(trade_date, raw_dir):
            raise InvalidContentError(f"fake invalid content for {trade_date}")

        monkeypatch.setattr(backfill_phase1, "fetch_bhavcopy", always_invalid)

        backfill_phase1.backfill_equities(
            date(2024, 7, 1), date(2024, 7, 5), on_invalid_content="skip"
        )

        assert isolated_backfill.query(MarketBar).count() == 0


class TestOnInvalidContentFail:
    def test_fail_raises_on_first_quarantined_day(self, isolated_backfill, monkeypatch):
        call_dates = []

        def always_invalid(trade_date, raw_dir):
            call_dates.append(trade_date)
            raise InvalidContentError(f"fake invalid content for {trade_date}")

        monkeypatch.setattr(backfill_phase1, "fetch_bhavcopy", always_invalid)

        with pytest.raises(InvalidContentError):
            backfill_phase1.backfill_equities(
                date(2024, 7, 1), date(2024, 7, 5), on_invalid_content="fail"
            )

        assert call_dates == [date(2024, 7, 1)]

    def test_skip_processes_all_days_before_completing(self, isolated_backfill, monkeypatch):
        call_dates = []

        def always_invalid(trade_date, raw_dir):
            call_dates.append(trade_date)
            raise InvalidContentError(f"fake invalid content for {trade_date}")

        monkeypatch.setattr(backfill_phase1, "fetch_bhavcopy", always_invalid)

        backfill_phase1.backfill_equities(
            date(2024, 7, 1), date(2024, 7, 5), on_invalid_content="skip"
        )

        assert len(call_dates) == 5
