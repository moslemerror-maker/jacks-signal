from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.exc import ProgrammingError

from backend.app.data.instruments.corporate_actions import (
    latest_coverage_status,
    normalize_snapshot,
    record_coverage_assessment,
)
from backend.app.db.models import (
    AssetType,
    CorporateActionCoverageAssessment,
    CorporateActionCoverageStatus,
    CorporateActionEvent,
    CorporateActionVerificationStatus,
    Instrument,
)


@pytest.fixture
def instrument(db_session) -> Instrument:
    inst = Instrument(exchange="NSE", asset_type=AssetType.EQUITY)
    db_session.add(inst)
    db_session.flush()
    return inst


def _write_snapshot(tmp_path, symbol: str, rows: list[dict]):
    snap_dir = tmp_path / symbol
    snap_dir.mkdir(parents=True, exist_ok=True)
    body = json.dumps(rows).encode()
    import hashlib

    sha = hashlib.sha256(body).hexdigest()
    path = snap_dir / f"{symbol}_{sha[:16]}.json"
    path.write_bytes(body)
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(
            {
                "symbol": symbol,
                "provider": "NSE (nseindia.com, unofficial JSON API)",
                "source_url": f"http://fake/corpActions?symbol={symbol}",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "sha256": sha,
            }
        )
    )
    return path


class TestCompletenessStatusIncludedInProvenance:
    def test_stored_assessment_carries_status_reasoning_and_sources(self, db_session, instrument):
        assessment_id = record_coverage_assessment(
            db_session, instrument.id, date(2015, 1, 1), date(2026, 9, 11),
            CorporateActionCoverageStatus.PARTIAL,
            "NSE record_count=20 meets known cap",
            [{"provider": "NSE", "record_count": 20, "known_page_size_cap": 20}],
        )
        db_session.flush()

        row = db_session.get(CorporateActionCoverageAssessment, assessment_id)
        assert row.status == CorporateActionCoverageStatus.PARTIAL
        assert "cap" in row.reasoning
        assert row.sources_considered[0]["provider"] == "NSE"
        assert row.assessed_at is not None
        assert row.assessed_by == "stage3.9-phase-b1"


class TestLatestCoverageStatusLookup:
    def test_returns_unknown_when_nothing_was_ever_assessed(self, db_session, instrument):
        status = latest_coverage_status(db_session, instrument.id, date(2015, 1, 1), date(2026, 9, 11))
        assert status == CorporateActionCoverageStatus.UNKNOWN

    def test_returns_the_most_recently_assessed_status(self, db_session, instrument):
        record_coverage_assessment(
            db_session, instrument.id, date(2015, 1, 1), date(2026, 9, 11),
            CorporateActionCoverageStatus.PARTIAL, "first pass, NSE only", [{"provider": "NSE"}],
        )
        db_session.flush()
        status = latest_coverage_status(db_session, instrument.id, date(2015, 1, 1), date(2026, 9, 11))
        assert status == CorporateActionCoverageStatus.PARTIAL


class TestLaterAssessmentNeverMutatesEarlierRowsOrRawSnapshots:
    def test_a_second_assessment_creates_a_new_row_the_first_untouched(self, db_session, instrument):
        first_id = record_coverage_assessment(
            db_session, instrument.id, date(2015, 1, 1), date(2026, 9, 11),
            CorporateActionCoverageStatus.PARTIAL, "NSE only - truncated", [{"provider": "NSE"}],
        )
        db_session.flush()
        first_reasoning_before = db_session.get(CorporateActionCoverageAssessment, first_id).reasoning

        second_id = record_coverage_assessment(
            db_session, instrument.id, date(2015, 1, 1), date(2026, 9, 11),
            CorporateActionCoverageStatus.CONFLICTED, "NSE + BSE compared and disagreed",
            [{"provider": "NSE"}, {"provider": "BSE"}],
        )
        db_session.flush()

        assert second_id != first_id
        first_row = db_session.get(CorporateActionCoverageAssessment, first_id)
        assert first_row.reasoning == first_reasoning_before
        assert first_row.status == CorporateActionCoverageStatus.PARTIAL

        all_rows = db_session.scalars(
            select(CorporateActionCoverageAssessment).where(
                CorporateActionCoverageAssessment.instrument_id == instrument.id
            )
        ).all()
        assert len(all_rows) == 2

        current = latest_coverage_status(db_session, instrument.id, date(2015, 1, 1), date(2026, 9, 11))
        assert current == CorporateActionCoverageStatus.CONFLICTED

    def test_normalizing_an_additional_snapshot_does_not_touch_earlier_raw_files(
        self, db_session, tmp_path, instrument
    ):
        rows = [
            {
                "symbol": "TESTCO", "isin": "INE_X", "faceVal": "10",
                "subject": "Dividend - Rs 1 Per Share", "exDate": "05-Jun-2020",
                "recDate": "05-Jun-2020", "caBroadcastDate": None,
            }
        ]
        path1 = _write_snapshot(tmp_path, "TESTCO", rows)
        original_bytes = path1.read_bytes()
        normalize_snapshot(db_session, path1, instrument.id)
        db_session.flush()

        more_rows = rows + [
            {
                "symbol": "TESTCO", "isin": "INE_X", "faceVal": "10",
                "subject": "Bonus 1:1", "exDate": "05-Jun-2021",
                "recDate": "05-Jun-2021", "caBroadcastDate": None,
            }
        ]
        path2 = _write_snapshot(tmp_path, "TESTCO2", more_rows)
        normalize_snapshot(db_session, path2, instrument.id)
        db_session.flush()

        assert path1.read_bytes() == original_bytes
        events = db_session.scalars(
            select(CorporateActionEvent).where(CorporateActionEvent.instrument_id == instrument.id)
        ).all()
        assert len(events) == 2


class TestImmutability:
    def test_assessment_rows_cannot_be_updated_at_all_not_even_status(self, db_session, instrument):
        assessment_id = record_coverage_assessment(
            db_session, instrument.id, date(2015, 1, 1), date(2026, 9, 11),
            CorporateActionCoverageStatus.UNKNOWN, "initial", [{"provider": "NSE"}],
        )
        db_session.flush()
        row = db_session.get(CorporateActionCoverageAssessment, assessment_id)
        row.status = CorporateActionCoverageStatus.COMPLETE
        with pytest.raises(ProgrammingError, match="immutable"):
            db_session.flush()

    def test_assessment_rows_cannot_be_deleted(self, db_session, instrument):
        assessment_id = record_coverage_assessment(
            db_session, instrument.id, date(2015, 1, 1), date(2026, 9, 11),
            CorporateActionCoverageStatus.UNKNOWN, "initial", [{"provider": "NSE"}],
        )
        db_session.flush()
        row = db_session.get(CorporateActionCoverageAssessment, assessment_id)
        db_session.delete(row)
        with pytest.raises(ProgrammingError, match="cannot be deleted"):
            db_session.flush()


class TestUnresolvedEventsCannotEnterAnAdjustmentCalculation:
    def test_unverified_event_combined_with_incomplete_coverage_is_doubly_blocked(
        self, db_session, tmp_path, instrument
    ):
        from backend.app.data.instruments.corporate_actions import assess_adjustment_eligibility

        rows = [
            {
                "symbol": "TESTCO", "isin": "INE_X", "faceVal": "10",
                "subject": "Bonus 1:1", "exDate": "05-Jun-2020",
                "recDate": "05-Jun-2020", "caBroadcastDate": None,
            }
        ]
        path = _write_snapshot(tmp_path, "TESTCO", rows)
        normalize_snapshot(db_session, path, instrument.id)
        db_session.flush()

        event = db_session.scalars(select(CorporateActionEvent)).one()
        assert event.verification_status == CorporateActionVerificationStatus.UNVERIFIED

        window_status = latest_coverage_status(
            db_session, instrument.id, date(2015, 1, 1), date(2026, 9, 11)
        )
        assert window_status == CorporateActionCoverageStatus.UNKNOWN

        result = assess_adjustment_eligibility(db_session, instrument.id, date(2015, 1, 1), date(2026, 9, 11))
        assert result.eligible is False
        codes = {b.code for b in result.blockers}
        assert "COVERAGE_NOT_COMPLETE" in codes and "EVENT_NOT_MANUALLY_VERIFIED" in codes
