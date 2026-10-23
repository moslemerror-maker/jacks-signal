from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError, ProgrammingError

from backend.app.data.instruments.corporate_actions import (
    AmbiguousEventError,
    EventNotFoundError,
    capture_baseline,
    current_components,
    find_event,
    normalize_snapshot,
    plan_reparse,
    promote_component_verification_status,
    promote_verification_status,
    reparse_event_components,
)
from backend.app.data.normalizers.corporate_actions import PARSER_VERSION
from backend.app.db.models import (
    AssetType,
    CorporateActionComponent,
    CorporateActionEvent,
    CorporateActionVerificationLog,
    CorporateActionVerificationStatus as Status,
    Instrument,
)

_BAJ_BONUS = "Bonus 4:1"
_BAJ_SPLIT = "Face Value Split (Sub-Division) - From Rs 2/- Per Share To Re 1/- Per Share"
_JSW = "Fv Splt Frm Rs 10 To Re 1"


@pytest.fixture
def instrument(db_session) -> Instrument:
    inst = Instrument(exchange="NSE", asset_type=AssetType.EQUITY)
    db_session.add(inst)
    db_session.flush()
    return inst


def _event(db_session, instrument, ex_date, subject, status=Status.UNVERIFIED) -> CorporateActionEvent:
    e = CorporateActionEvent(
        id=uuid.uuid4(), instrument_id=instrument.id, source_symbol="TESTCO", ex_date=ex_date,
        raw_subject_text=subject, source_provider="test", source_url="http://fake",
        source_snapshot_path="x", source_snapshot_sha256="x",
        source_fetched_at=datetime.now(timezone.utc), verification_status=status,
        created_by="test",
    )
    db_session.add(e)
    db_session.flush()
    return e


def _legacy_component(db_session, event, ctype, fragment, status, **fields) -> CorporateActionComponent:
    c = CorporateActionComponent(
        id=uuid.uuid4(), event_id=event.id, component_order=1, component_type=ctype,
        raw_text_fragment=fragment, verification_status=status, **fields,
    )
    db_session.add(c)
    db_session.flush()
    return c


def _log_rows(db_session, target_id):
    return db_session.scalars(
        select(CorporateActionVerificationLog)
        .where(CorporateActionVerificationLog.target_id == target_id)
        .order_by(CorporateActionVerificationLog.seq)
    ).all()


class TestSameDateEventsAreNeverAmbiguouslySelected:
    def _baj(self, db_session, instrument):
        bonus = _event(db_session, instrument, date(2025, 6, 16), _BAJ_BONUS)
        split = _event(db_session, instrument, date(2025, 6, 16), _BAJ_SPLIT)
        return bonus, split

    def test_bare_instrument_date_lookup_raises_instead_of_picking_one(self, db_session, instrument):
        self._baj(db_session, instrument)
        with pytest.raises(AmbiguousEventError, match="2 events match"):
            find_event(db_session, instrument.id, date(2025, 6, 16))

    def test_subject_disambiguates_to_exactly_the_requested_event(self, db_session, instrument):
        bonus, split = self._baj(db_session, instrument)
        assert find_event(db_session, instrument.id, date(2025, 6, 16), _BAJ_BONUS).id == bonus.id
        assert find_event(db_session, instrument.id, date(2025, 6, 16), f"  {_BAJ_SPLIT} ").id == split.id

    def test_single_event_on_a_date_needs_no_subject(self, db_session, instrument):
        e = _event(db_session, instrument, date(2019, 9, 19), "Face Value Split From Rs.2 To Rs.1")
        assert find_event(db_session, instrument.id, date(2019, 9, 19)).id == e.id

    def test_no_match_raises_not_found(self, db_session, instrument):
        with pytest.raises(EventNotFoundError):
            find_event(db_session, instrument.id, date(2000, 1, 1))
        _event(db_session, instrument, date(2025, 6, 16), _BAJ_BONUS)
        with pytest.raises(EventNotFoundError):
            find_event(db_session, instrument.id, date(2025, 6, 16), "some other subject")

    def test_promoting_one_same_date_event_leaves_the_other_untouched(self, db_session, instrument):
        bonus, split = self._baj(db_session, instrument)
        target = find_event(db_session, instrument.id, date(2025, 6, 16), _BAJ_SPLIT)
        promote_verification_status(db_session, target.id, Status.MANUALLY_VERIFIED, "verified the split", actor="t")
        db_session.expire_all()
        assert db_session.get(CorporateActionEvent, split.id).verification_status == Status.MANUALLY_VERIFIED
        assert db_session.get(CorporateActionEvent, bonus.id).verification_status == Status.UNVERIFIED


class TestVerificationAuditTrail:
    def test_promotion_logs_previous_state_before_overwriting(self, db_session, instrument):
        e = _event(db_session, instrument, date(2025, 6, 16), _BAJ_BONUS)
        promote_verification_status(db_session, e.id, Status.MANUALLY_VERIFIED, "first notes", actor="alice")
        promote_verification_status(db_session, e.id, Status.UNVERIFIED, "second notes", actor="bob",
                                    action="CORRECTION")
        db_session.expire_all()

        assert db_session.get(CorporateActionEvent, e.id).verification_notes == "second notes"
        first, second = _log_rows(db_session, e.id)
        assert (first.from_status, first.to_status, first.from_notes, first.to_notes, first.actor) == (
            "UNVERIFIED", "MANUALLY_VERIFIED", None, "first notes", "alice")
        assert (second.from_status, second.to_status, second.from_notes, second.to_notes, second.action) == (
            "MANUALLY_VERIFIED", "UNVERIFIED", "first notes", "second notes", "CORRECTION")
        assert first.seq < second.seq

    def test_component_promotions_are_logged_too(self, db_session, instrument):
        e = _event(db_session, instrument, date(2025, 6, 16), _BAJ_BONUS)
        c = _legacy_component(db_session, e, "BONUS", _BAJ_BONUS, Status.MACHINE_PARSED,
                              ratio_new=4, ratio_existing=1)
        promote_component_verification_status(db_session, c.id, Status.DISPUTED, "n", actor="t")
        (row,) = _log_rows(db_session, c.id)
        assert (row.target_kind, row.from_status, row.to_status) == ("COMPONENT", "MACHINE_PARSED", "DISPUTED")

    def test_baseline_capture_records_current_state_without_changing_it(self, db_session, instrument):
        e = _event(db_session, instrument, date(2025, 6, 16), _BAJ_BONUS, status=Status.MANUALLY_VERIFIED)
        capture_baseline(db_session, "EVENT", e.id, actor="t", note="pre-remediation")
        db_session.expire_all()
        assert db_session.get(CorporateActionEvent, e.id).verification_status == Status.MANUALLY_VERIFIED
        (row,) = _log_rows(db_session, e.id)
        assert row.action == "BASELINE" and row.to_status == "MANUALLY_VERIFIED"
        assert row.payload == {"note": "pre-remediation"}

    def test_a_wrong_verification_can_be_corrected_without_losing_the_evidence_of_the_error(
        self, db_session, instrument
    ):
        wrong = _event(db_session, instrument, date(2025, 6, 16), _BAJ_BONUS, status=Status.MANUALLY_VERIFIED)
        capture_baseline(db_session, "EVENT", wrong.id, actor="c0")
        promote_verification_status(
            db_session, wrong.id, Status.UNVERIFIED, "reverted: notes described the split", actor="c0",
            action="CORRECTION",
        )
        db_session.expire_all()
        assert db_session.get(CorporateActionEvent, wrong.id).verification_status == Status.UNVERIFIED
        actions = [r.action for r in _log_rows(db_session, wrong.id)]
        assert actions == ["BASELINE", "CORRECTION"]

    def test_promoting_an_unknown_id_raises_and_writes_no_log_row(self, db_session):
        with pytest.raises(EventNotFoundError):
            promote_verification_status(db_session, uuid.uuid4(), Status.MANUALLY_VERIFIED, "n")
        assert db_session.scalars(select(CorporateActionVerificationLog)).all() == []

    def test_log_rows_cannot_be_updated(self, db_session, instrument):
        e = _event(db_session, instrument, date(2025, 6, 16), _BAJ_BONUS)
        promote_verification_status(db_session, e.id, Status.MANUALLY_VERIFIED, "n", actor="t")
        with pytest.raises(ProgrammingError, match="append-only"):
            db_session.execute(text("UPDATE corporate_action_verification_log SET actor = 'tampered'"))

    def test_log_rows_cannot_be_deleted(self, db_session, instrument):
        e = _event(db_session, instrument, date(2025, 6, 16), _BAJ_BONUS)
        promote_verification_status(db_session, e.id, Status.MANUALLY_VERIFIED, "n", actor="t")
        with pytest.raises(ProgrammingError, match="cannot be deleted"):
            db_session.execute(text("DELETE FROM corporate_action_verification_log"))


class TestReparseRecordsCorrectedReadingWithoutMutation:
    def _legacy_jsw(self, db_session, instrument):
        e = _event(db_session, instrument, date(2017, 1, 4), _JSW, status=Status.MANUALLY_VERIFIED)
        c = _legacy_component(db_session, e, "OTHER", _JSW, Status.MANUALLY_VERIFIED)
        return e, c

    def test_reparse_adds_a_new_revision_and_leaves_the_old_one_untouched(self, db_session, instrument):
        e, old = self._legacy_jsw(db_session, instrument)
        assert reparse_event_components(db_session, e.id, actor="t") == 2
        db_session.expire_all()

        old_after = db_session.get(CorporateActionComponent, old.id)
        assert (old_after.component_type.value, old_after.parse_revision, old_after.parser_version,
                old_after.face_value_from) == ("OTHER", 1, "stage3.9-phase-b-v1", None)
        (new,) = current_components(db_session, e.id)
        assert (new.component_type.value, new.parse_revision, new.parser_version) == ("SPLIT", 2, PARSER_VERSION)
        assert (new.face_value_from, new.face_value_to) == (10.0, 1.0)

    def test_new_revision_never_inherits_manually_verified(self, db_session, instrument):
        e, _ = self._legacy_jsw(db_session, instrument)
        reparse_event_components(db_session, e.id, actor="t")
        (new,) = current_components(db_session, e.id)
        assert new.verification_status == Status.MACHINE_PARSED

    def test_reparse_is_idempotent(self, db_session, instrument):
        e, _ = self._legacy_jsw(db_session, instrument)
        assert reparse_event_components(db_session, e.id, actor="t") == 2
        assert plan_reparse(db_session, e.id) is None
        assert reparse_event_components(db_session, e.id, actor="t") is None
        total = db_session.scalars(
            select(CorporateActionComponent).where(CorporateActionComponent.event_id == e.id)
        ).all()
        assert len(total) == 2

    def test_unchanged_event_is_not_reparsed_and_keeps_its_verification(self, db_session, instrument):
        e = _event(db_session, instrument, date(2019, 9, 19),
                   "Face Value Split (Sub-Division) - From Rs 2 Per Share To Rs 1 Per Share",
                   status=Status.MANUALLY_VERIFIED)
        _legacy_component(db_session, e, "SPLIT", e.raw_subject_text, Status.MANUALLY_VERIFIED,
                          face_value_from=2.0, face_value_to=1.0)
        assert reparse_event_components(db_session, e.id, actor="t") is None
        (c,) = current_components(db_session, e.id)
        assert c.verification_status == Status.MANUALLY_VERIFIED and c.parse_revision == 1

    def test_reparse_is_logged_with_old_and_new_readings(self, db_session, instrument):
        e, _ = self._legacy_jsw(db_session, instrument)
        reparse_event_components(db_session, e.id, actor="c0")
        (row,) = _log_rows(db_session, e.id)
        assert row.action == "REPARSE"
        assert row.payload["old"][0]["type"] == "OTHER" and row.payload["new"][0]["type"] == "SPLIT"
        assert (row.payload["old_revision"], row.payload["new_revision"]) == (1, 2)

    def test_stale_rights_reading_is_downgraded_to_unverified_on_reparse(self, db_session, instrument):
        subject = "Rights 1:15 @ Premium Rs 1247"
        e = _event(db_session, instrument, date(2020, 5, 13), subject)
        _legacy_component(db_session, e, "RIGHTS", subject, Status.MACHINE_PARSED,
                          ratio_new=1, ratio_existing=15)
        assert reparse_event_components(db_session, e.id, actor="t") == 2
        (new,) = current_components(db_session, e.id)
        assert new.verification_status == Status.UNVERIFIED
        assert "issue price" in new.verification_notes


class TestRevisionColumnsAreImmutableAndUnique:
    def test_parse_revision_cannot_be_rewritten(self, db_session, instrument):
        e = _event(db_session, instrument, date(2017, 1, 4), _JSW)
        c = _legacy_component(db_session, e, "OTHER", _JSW, Status.UNVERIFIED)
        c.parse_revision = 9
        with pytest.raises(ProgrammingError, match="immutable"):
            db_session.flush()

    def test_parser_version_cannot_be_rewritten(self, db_session, instrument):
        e = _event(db_session, instrument, date(2017, 1, 4), _JSW)
        c = _legacy_component(db_session, e, "OTHER", _JSW, Status.UNVERIFIED)
        c.parser_version = "forged"
        with pytest.raises(ProgrammingError, match="immutable"):
            db_session.flush()

    def test_duplicate_event_revision_order_is_rejected(self, db_session, instrument):
        e = _event(db_session, instrument, date(2017, 1, 4), _JSW)
        _legacy_component(db_session, e, "OTHER", _JSW, Status.UNVERIFIED)
        with pytest.raises(IntegrityError):
            _legacy_component(db_session, e, "SPLIT", _JSW, Status.UNVERIFIED)


class TestNormalizeSnapshotUsesTheHardenedParser:
    def _snapshot(self, tmp_path, rows):
        d = tmp_path / "TESTCO"
        d.mkdir()
        body = json.dumps(rows).encode()
        import hashlib
        sha = hashlib.sha256(body).hexdigest()
        p = d / f"TESTCO_{sha[:16]}.json"
        p.write_bytes(body)
        p.with_suffix(p.suffix + ".meta.json").write_text(json.dumps({
            "provider": "NSE", "source_url": "http://fake", "sha256": sha,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }))
        return p

    def test_new_events_record_parser_version_and_revision_one(self, db_session, tmp_path, instrument):
        p = self._snapshot(tmp_path, [{"symbol": "TESTCO", "subject": _JSW, "exDate": "04-Jan-2017",
                                       "recDate": "05-Jan-2017", "isin": "X", "faceVal": "1"}])
        assert normalize_snapshot(db_session, p, instrument.id) == 1
        e = find_event(db_session, instrument.id, date(2017, 1, 4))
        (c,) = current_components(db_session, e.id)
        assert (c.component_type.value, c.parse_revision, c.parser_version) == ("SPLIT", 1, PARSER_VERSION)
        assert (c.face_value_from, c.face_value_to) == (10.0, 1.0)

    def test_two_same_date_records_become_two_distinct_events(self, db_session, tmp_path, instrument):
        base = {"symbol": "TESTCO", "exDate": "16-Jun-2025", "recDate": "16-Jun-2025", "isin": "X", "faceVal": "1"}
        p = self._snapshot(tmp_path, [dict(base, subject=_BAJ_BONUS), dict(base, subject=_BAJ_SPLIT)])
        assert normalize_snapshot(db_session, p, instrument.id) == 2
        assert len(db_session.scalars(select(CorporateActionEvent)).all()) == 2
        with pytest.raises(AmbiguousEventError):
            find_event(db_session, instrument.id, date(2025, 6, 16))
