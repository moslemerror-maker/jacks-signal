from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.exc import ProgrammingError

from backend.app.data.instruments.corporate_actions import (
    normalize_snapshot,
    promote_verification_status,
)
from backend.app.db.models import (
    AssetType,
    CorporateActionComponent,
    CorporateActionEvent,
    CorporateActionVerificationStatus,
    Instrument,
    InstrumentIsin,
)


@pytest.fixture
def instrument(db_session) -> Instrument:
    inst = Instrument(exchange="NSE", asset_type=AssetType.EQUITY)
    db_session.add(inst)
    db_session.flush()
    db_session.add(
        InstrumentIsin(
            instrument_id=inst.id, isin="INE000TEST01", effective_from=date(2015, 1, 1)
        )
    )
    db_session.flush()
    return inst


def _write_snapshot(tmp_path, symbol: str, rows: list[dict], fetched_at: str | None = None):
    snap_dir = tmp_path / symbol
    snap_dir.mkdir(parents=True, exist_ok=True)
    body = json.dumps(rows).encode()
    import hashlib

    sha = hashlib.sha256(body).hexdigest()
    path = snap_dir / f"{symbol}_{sha[:16]}.json"
    path.write_bytes(body)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "symbol": symbol,
                "provider": "NSE (nseindia.com, unofficial JSON API)",
                "source_url": f"http://fake/corpActions?symbol={symbol}",
                "fetched_at": fetched_at or datetime.now(timezone.utc).isoformat(),
                "sha256": sha,
            }
        )
    )
    return path


_SIMPLE_ROWS = [
    {
        "symbol": "TESTCO", "isin": "INE_TEST_0001", "faceVal": "10",
        "subject": "Dividend - Rs 6 Per Share", "exDate": "05-Jun-2026",
        "recDate": "05-Jun-2026", "caBroadcastDate": None,
    },
]

_COMPOUND_ROWS = [
    {
        "symbol": "TESTCO", "isin": "INE_TEST_0001", "faceVal": "2",
        "subject": (
            "Bonus 1:1/Face Value Split (Sub-Division) - From Rs 10/- "
            "Per Share To Rs 2/- Per Share"
        ),
        "exDate": "08-Sep-2016", "recDate": "08-Sep-2016", "caBroadcastDate": None,
    },
]


class TestNormalizeCreatesEventsAndComponents:
    def test_simple_dividend_creates_one_event_one_component(self, db_session, tmp_path, instrument):
        path = _write_snapshot(tmp_path, "TESTCO", _SIMPLE_ROWS)
        written = normalize_snapshot(db_session, path, instrument.id)
        db_session.flush()
        assert written == 1

        events = db_session.scalars(
            select(CorporateActionEvent).where(CorporateActionEvent.instrument_id == instrument.id)
        ).all()
        assert len(events) == 1
        event = events[0]
        assert event.raw_subject_text == "Dividend - Rs 6 Per Share"
        assert event.verification_status == CorporateActionVerificationStatus.UNVERIFIED
        assert event.source_isin == "INE_TEST_0001"

        components = db_session.scalars(
            select(CorporateActionComponent).where(CorporateActionComponent.event_id == event.id)
        ).all()
        assert len(components) == 1
        assert components[0].component_type == "DIVIDEND"
        assert components[0].cash_amount_per_share == 6.0
        assert components[0].verification_status == CorporateActionVerificationStatus.MACHINE_PARSED

    def test_compound_subject_creates_two_ordered_components_not_flattened(
        self, db_session, tmp_path, instrument
    ):
        path = _write_snapshot(tmp_path, "TESTCO", _COMPOUND_ROWS)
        normalize_snapshot(db_session, path, instrument.id)
        db_session.flush()

        event = db_session.scalars(
            select(CorporateActionEvent).where(CorporateActionEvent.instrument_id == instrument.id)
        ).one()
        components = db_session.scalars(
            select(CorporateActionComponent)
            .where(CorporateActionComponent.event_id == event.id)
            .order_by(CorporateActionComponent.component_order)
        ).all()
        assert len(components) == 2
        assert components[0].component_type == "BONUS"
        assert components[0].ratio_new == 1 and components[0].ratio_existing == 1
        assert components[1].component_type == "SPLIT"
        assert components[1].face_value_from == 10.0 and components[1].face_value_to == 2.0
        assert not hasattr(event, "ratio_new")


class TestExactSourceTextPreservation:
    def test_raw_subject_text_matches_source_byte_for_byte(self, db_session, tmp_path, instrument):
        weird_subject = "  Dividend - Rs 6 Per Share  "
        rows = [dict(_SIMPLE_ROWS[0], subject=weird_subject)]
        path = _write_snapshot(tmp_path, "TESTCO", rows)
        normalize_snapshot(db_session, path, instrument.id)
        db_session.flush()
        event = db_session.scalars(select(CorporateActionEvent)).one()
        assert event.raw_subject_text == weird_subject.strip()
        component = db_session.scalars(select(CorporateActionComponent)).one()
        assert "Dividend" in component.raw_text_fragment


class TestIdempotentAndCorrectedSnapshots:
    def test_renormalizing_the_identical_snapshot_does_not_duplicate(
        self, db_session, tmp_path, instrument
    ):
        path = _write_snapshot(tmp_path, "TESTCO", _SIMPLE_ROWS)
        first = normalize_snapshot(db_session, path, instrument.id)
        db_session.flush()
        second = normalize_snapshot(db_session, path, instrument.id)
        db_session.flush()

        assert first == 1
        assert second == 0
        events = db_session.scalars(select(CorporateActionEvent)).all()
        assert len(events) == 1

    def test_a_corrected_snapshot_creates_a_new_event_not_an_overwrite(
        self, db_session, tmp_path, instrument
    ):
        path1 = _write_snapshot(tmp_path, "TESTCO", _SIMPLE_ROWS)
        normalize_snapshot(db_session, path1, instrument.id)
        db_session.flush()

        corrected_rows = [dict(_SIMPLE_ROWS[0], subject="Dividend - Rs 6.5 Per Share (CORRECTED)")]
        path2 = _write_snapshot(tmp_path, "TESTCO2", corrected_rows)
        written = normalize_snapshot(db_session, path2, instrument.id)
        db_session.flush()

        assert written == 1
        events = db_session.scalars(
            select(CorporateActionEvent).order_by(CorporateActionEvent.raw_subject_text)
        ).all()
        assert len(events) == 2
        subjects = {e.raw_subject_text for e in events}
        assert "Dividend - Rs 6 Per Share" in subjects
        assert "Dividend - Rs 6.5 Per Share (CORRECTED)" in subjects


class TestImmutability:
    def test_content_field_update_is_rejected(self, db_session, tmp_path, instrument):
        path = _write_snapshot(tmp_path, "TESTCO", _SIMPLE_ROWS)
        normalize_snapshot(db_session, path, instrument.id)
        db_session.flush()

        event = db_session.scalars(select(CorporateActionEvent)).one()
        event.raw_subject_text = "sneaky overwrite"
        with pytest.raises(ProgrammingError, match="immutable"):
            db_session.flush()

    def test_component_content_field_update_is_rejected(self, db_session, tmp_path, instrument):
        path = _write_snapshot(tmp_path, "TESTCO", _SIMPLE_ROWS)
        normalize_snapshot(db_session, path, instrument.id)
        db_session.flush()

        component = db_session.scalars(select(CorporateActionComponent)).one()
        component.cash_amount_per_share = 999.0
        with pytest.raises(ProgrammingError, match="immutable"):
            db_session.flush()

    def test_delete_is_rejected(self, db_session, tmp_path, instrument):
        path = _write_snapshot(tmp_path, "TESTCO", _SIMPLE_ROWS)
        normalize_snapshot(db_session, path, instrument.id)
        db_session.flush()

        event = db_session.scalars(select(CorporateActionEvent)).one()
        db_session.delete(event)
        with pytest.raises(ProgrammingError, match="immutable"):
            db_session.flush()

    def test_delete_of_an_event_with_no_components_is_rejected_by_its_own_trigger(
        self, db_session, tmp_path, instrument
    ):
        path = _write_snapshot(tmp_path, "TESTCO", _SIMPLE_ROWS)
        normalize_snapshot(db_session, path, instrument.id)
        db_session.flush()
        from backend.app.db.models import CorporateActionEvent as Event
        from backend.app.db.models import CorporateActionVerificationStatus
        import uuid as _uuid
        from datetime import date as _date

        bare = Event(
            id=_uuid.uuid4(), instrument_id=instrument.id, source_symbol="TESTCO",
            ex_date=_date(2030, 1, 1), raw_subject_text="Dividend - Rs 1 Per Share",
            source_provider="x", source_url="x", source_snapshot_path="x",
            source_snapshot_sha256="x", source_fetched_at=datetime.now(timezone.utc),
            verification_status=CorporateActionVerificationStatus.UNVERIFIED,
            created_by="test",
        )
        db_session.add(bare)
        db_session.flush()

        db_session.delete(bare)
        with pytest.raises(ProgrammingError, match="cannot be deleted"):
            db_session.flush()


class TestVerificationStatusPromotion:
    def test_promote_verification_status_updates_in_place(self, db_session, tmp_path, instrument):
        path = _write_snapshot(tmp_path, "TESTCO", _SIMPLE_ROWS)
        normalize_snapshot(db_session, path, instrument.id)
        db_session.flush()

        event = db_session.scalars(select(CorporateActionEvent)).one()
        promote_verification_status(
            db_session, event.id, CorporateActionVerificationStatus.MANUALLY_VERIFIED,
            "cross-checked against real InstrumentIsin history",
        )
        db_session.flush()
        db_session.expire_all()

        refreshed = db_session.get(CorporateActionEvent, event.id)
        assert refreshed.verification_status == CorporateActionVerificationStatus.MANUALLY_VERIFIED
        assert "cross-checked" in refreshed.verification_notes

    def test_normalize_snapshot_never_auto_promotes_to_manually_verified(
        self, db_session, tmp_path, instrument
    ):
        path = _write_snapshot(tmp_path, "TESTCO", _COMPOUND_ROWS)
        normalize_snapshot(db_session, path, instrument.id)
        db_session.flush()

        events = db_session.scalars(select(CorporateActionEvent)).all()
        components = db_session.scalars(select(CorporateActionComponent)).all()
        for row in [*events, *components]:
            assert row.verification_status != CorporateActionVerificationStatus.MANUALLY_VERIFIED


class TestIdentityChainNeverMutated:
    def test_normalizing_events_does_not_touch_instrument_isin_history(
        self, db_session, tmp_path, instrument
    ):
        before = db_session.scalars(
            select(InstrumentIsin).where(InstrumentIsin.instrument_id == instrument.id)
        ).all()
        before_snapshot = [(r.isin, r.effective_from, r.effective_to) for r in before]

        path = _write_snapshot(tmp_path, "TESTCO", _COMPOUND_ROWS)
        normalize_snapshot(db_session, path, instrument.id)
        db_session.flush()

        after = db_session.scalars(
            select(InstrumentIsin).where(InstrumentIsin.instrument_id == instrument.id)
        ).all()
        after_snapshot = [(r.isin, r.effective_from, r.effective_to) for r in after]
        assert before_snapshot == after_snapshot


class TestIncompleteHistoryDetectedNotSilentlyAccepted:
    def test_missing_ex_date_is_skipped_not_guessed(self, db_session, tmp_path, instrument):
        rows = [dict(_SIMPLE_ROWS[0], exDate="-")]
        path = _write_snapshot(tmp_path, "TESTCO", rows)
        written = normalize_snapshot(db_session, path, instrument.id)
        db_session.flush()
        assert written == 0
        assert db_session.scalars(select(CorporateActionEvent)).all() == []

    def test_empty_subject_is_skipped_not_guessed(self, db_session, tmp_path, instrument):
        rows = [dict(_SIMPLE_ROWS[0], subject="")]
        path = _write_snapshot(tmp_path, "TESTCO", rows)
        written = normalize_snapshot(db_session, path, instrument.id)
        db_session.flush()
        assert written == 0
