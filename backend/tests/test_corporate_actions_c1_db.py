from __future__ import annotations

import copy
import inspect
import uuid
from datetime import date, datetime, timezone

import pytest
from sqlalchemy import null, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import DBAPIError, IntegrityError

from backend.app.data.instruments.corporate_actions import (
    EvidenceRejected,
    coverage_status_for_window,
    latest_coverage_status,
    nse_event_verification_lookup,
    record_coverage_assessment,
    record_derived_coverage_assessment,
)
from backend.app.db.models import (
    AssetType,
    CorporateActionComponent,
    CorporateActionCoverageAssessment as Assessment,
    CorporateActionCoverageStatus as Status,
    CorporateActionEvent,
    CorporateActionUnresolvedEvent as Unresolved,
    CorporateActionVerificationStatus as Verification,
    Instrument,
)
from backend.app.research.corporate_actions_coverage import DERIVED_BY, REQUIRED_CHECKS, evidence_sha256
from backend.tests.c1_helpers import BASE_BSE, BASE_NSE, D, bse_row, derive, nse_row

FULL = (date(2015, 1, 1), date(2026, 9, 11))

ZERO_PATHS = [
    ("reconciliation", "conflict_count"),
    ("reconciliation", "one_sided_count"),
    ("reconciliation", "unclassified_count"),
    ("reconciliation", "duplicate_price_affecting_count"),
    ("reconciliation", "ambiguous_same_date_count"),
    ("reconciliation", "unparseable_row_count"),
    ("revisions", "blocking_count"),
    ("diagnostics", "jump_scan", "blocking_count"),
    ("diagnostics", "isin_transitions", "uncorroborated_count"),
]


@pytest.fixture
def instrument(db_session) -> Instrument:
    inst = Instrument(exchange="NSE", asset_type=AssetType.EQUITY)
    db_session.add(inst)
    db_session.flush()
    return inst


def _complete_evidence(instrument, **kw) -> dict:
    ev = derive(**kw)
    assert ev["verdict"] == "COMPLETE"
    ev["instrument"]["instrument_id"] = str(instrument.id)
    return ev


def _raw_insert(session, instrument, status, evidence, *, sha="a" * 64, derived_by=DERIVED_BY, window=FULL):
    session.execute(
        pg_insert(Assessment).values(
            id=uuid.uuid4(), instrument_id=instrument.id, window_start=window[0], window_end=window[1],
            status=status, assessed_at=datetime.now(timezone.utc), reasoning="raw", sources_considered=[],
            assessed_by="raw-test", evidence=null() if evidence is None else evidence, evidence_sha256=sha if evidence is not None else None,
            derived_by=derived_by if evidence is not None else None,
        )
    )


def _rejected(session, fn, match):
    with pytest.raises(DBAPIError, match=match):
        with session.begin_nested():
            fn()


def _set(ev, path, value):
    node = ev
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value


class TestTheDerivedWritePath:
    def test_a_derived_complete_persists_with_evidence_hash_and_derived_by(self, db_session, instrument):
        ev = _complete_evidence(instrument)
        aid = record_derived_coverage_assessment(db_session, instrument.id, ev, [])
        db_session.flush()
        row = db_session.get(Assessment, aid)
        assert row.status == Status.COMPLETE
        assert row.evidence["schema"] == "coverage-evidence/1"
        assert row.evidence_sha256 == evidence_sha256(ev) and len(row.evidence_sha256) == 64
        assert row.derived_by == DERIVED_BY and row.assessed_by == "stage3.9-c1"
        assert (row.window_start, row.window_end) == FULL

    def test_the_stored_evidence_names_every_source_snapshot_hash(self, db_session, instrument):
        ev = _complete_evidence(instrument)
        row = db_session.get(Assessment, record_derived_coverage_assessment(db_session, instrument.id, ev, []))
        db_session.flush()
        for src in row.sources_considered:
            assert src["whole_snapshot_sha256"] and len(src["part_snapshot_sha256"]) == 3

    def test_the_status_comes_from_the_verdict_and_there_is_no_status_parameter(self):
        params = inspect.signature(record_derived_coverage_assessment).parameters
        assert "status" not in params and "verdict" not in params

    def test_a_non_complete_verdict_is_recorded_as_that_verdict(self, db_session, instrument):
        bse = [(d, bse_row(d, "Bonus issue 3:1")) if d == D(2017, 9, 7) else (d, r) for d, r in BASE_BSE]
        ev = derive(bse_dated=bse)
        aid = record_derived_coverage_assessment(db_session, instrument.id, ev, [])
        db_session.flush()
        assert db_session.get(Assessment, aid).status == Status.CONFLICTED

    def test_evidence_claiming_complete_with_a_failed_check_is_rejected_before_the_database(self, db_session, instrument):
        ev = _complete_evidence(instrument)
        ev["checks"]["boundary_semantics_verified"] = False
        with pytest.raises(EvidenceRejected, match="boundary_semantics_verified"):
            record_derived_coverage_assessment(db_session, instrument.id, ev, [])

    def test_evidence_with_no_recognised_verdict_is_rejected(self, db_session, instrument):
        ev = _complete_evidence(instrument)
        ev["verdict"] = "PROBABLY_COMPLETE"
        with pytest.raises(EvidenceRejected):
            record_derived_coverage_assessment(db_session, instrument.id, ev, [])

    def test_the_legacy_recording_path_refuses_complete_and_writes_nothing(self, db_session, instrument):
        with pytest.raises(ValueError, match="COMPLETE cannot be asserted directly"):
            record_coverage_assessment(db_session, instrument.id, *FULL, Status.COMPLETE, "trust me", [{"provider": "NSE"}])
        assert db_session.scalars(select(Assessment).where(Assessment.instrument_id == instrument.id)).all() == []

    @pytest.mark.parametrize("status", [Status.PARTIAL, Status.UNKNOWN, Status.CONFLICTED])
    def test_the_legacy_path_still_records_non_complete_assessments_without_evidence(self, db_session, instrument, status):
        aid = record_coverage_assessment(db_session, instrument.id, *FULL, status, "manual note", [{"provider": "NSE"}])
        db_session.flush()
        row = db_session.get(Assessment, aid)
        assert row.status == status and row.evidence is None and row.derived_by is None and row.evidence_sha256 is None


class TestTheDatabaseRejectsAnIncompleteComplete:
    def test_control_a_genuine_bundle_inserts_raw(self, db_session, instrument):
        _raw_insert(db_session, instrument, Status.COMPLETE, _complete_evidence(instrument))
        db_session.flush()

    def test_complete_with_no_evidence_is_rejected(self, db_session, instrument):
        _rejected(db_session, lambda: _raw_insert(db_session, instrument, Status.COMPLETE, None),
                  "requires an evidence bundle")

    @pytest.mark.parametrize("status", [Status.COMPLETE, Status.PARTIAL])
    def test_json_null_evidence_is_not_a_way_around_the_no_evidence_rule(self, db_session, instrument, status):
        def go():
            db_session.execute(pg_insert(Assessment).values(
                id=uuid.uuid4(), instrument_id=instrument.id, window_start=FULL[0], window_end=FULL[1],
                status=status, assessed_at=datetime.now(timezone.utc), reasoning="r", sources_considered=[],
                assessed_by="x", evidence=None, evidence_sha256="a" * 64, derived_by=DERIVED_BY))
        _rejected(db_session, go, "does not equal the evidence-derived verdict")

    def test_evidence_without_its_hash_and_derived_by_is_rejected(self, db_session, instrument):
        def go():
            db_session.execute(pg_insert(Assessment).values(
                id=uuid.uuid4(), instrument_id=instrument.id, window_start=FULL[0], window_end=FULL[1],
                status=Status.COMPLETE, assessed_at=datetime.now(timezone.utc), reasoning="r", sources_considered=[],
                assessed_by="x", evidence=_complete_evidence(instrument), evidence_sha256=None, derived_by=None))
        _rejected(db_session, go, "ck_coverage_evidence_columns_together|ck_coverage_complete_requires_derivation")

    @pytest.mark.parametrize("check", REQUIRED_CHECKS)
    def test_every_required_check_is_individually_enforced_by_the_database(self, db_session, instrument, check):
        ev = _complete_evidence(instrument)
        ev["checks"][check] = False
        _rejected(db_session, lambda: _raw_insert(db_session, instrument, Status.COMPLETE, ev),
                  f"evidence check {check} is not recorded as passed")

    @pytest.mark.parametrize("check", REQUIRED_CHECKS)
    def test_a_check_missing_from_the_bundle_is_not_treated_as_passed(self, db_session, instrument, check):
        ev = _complete_evidence(instrument)
        del ev["checks"][check]
        _rejected(db_session, lambda: _raw_insert(db_session, instrument, Status.COMPLETE, ev), "is not recorded as passed")

    @pytest.mark.parametrize("path", ZERO_PATHS, ids=[".".join(p) for p in ZERO_PATHS])
    def test_a_nonzero_blocking_count_is_rejected_even_if_every_check_says_true(self, db_session, instrument, path):
        ev = _complete_evidence(instrument)
        _set(ev, path, 1)
        _rejected(db_session, lambda: _raw_insert(db_session, instrument, Status.COMPLETE, ev),
                  f"{'.'.join(path)}|must be recorded as 0, found 1")

    @pytest.mark.parametrize("path", ZERO_PATHS, ids=[".".join(p) for p in ZERO_PATHS])
    def test_an_absent_blocking_count_is_rejected_not_read_as_zero(self, db_session, instrument, path):
        ev = _complete_evidence(instrument)
        parent = ev
        for k in path[:-1]:
            parent = parent[k]
        del parent[path[-1]]
        _rejected(db_session, lambda: _raw_insert(db_session, instrument, Status.COMPLETE, ev), "found -1")

    @pytest.mark.parametrize("missing", ["NSE", "BSE"])
    def test_a_bundle_lacking_either_providers_whole_snapshot_is_rejected(self, db_session, instrument, missing):
        ev = _complete_evidence(instrument)
        for s in ev["sources"]:
            if s["provider"] == missing:
                s["whole"] = None
        _rejected(db_session, lambda: _raw_insert(db_session, instrument, Status.COMPLETE, ev),
                  "both NSE and BSE must have a whole-window snapshot")

    def test_a_bundle_with_no_sources_is_rejected(self, db_session, instrument):
        ev = _complete_evidence(instrument)
        ev["sources"] = []
        _rejected(db_session, lambda: _raw_insert(db_session, instrument, Status.COMPLETE, ev), "both NSE and BSE")

    @pytest.mark.parametrize("status", [Status.PARTIAL, Status.UNKNOWN, Status.CONFLICTED])
    def test_status_cannot_disagree_with_the_evidence_verdict(self, db_session, instrument, status):
        _rejected(db_session, lambda: _raw_insert(db_session, instrument, status, _complete_evidence(instrument)),
                  "does not equal the evidence-derived verdict")

    def test_complete_status_over_a_non_complete_verdict_is_rejected(self, db_session, instrument):
        ev = _complete_evidence(instrument)
        ev["verdict"] = "PARTIAL"
        _rejected(db_session, lambda: _raw_insert(db_session, instrument, Status.COMPLETE, ev),
                  "status cannot be asserted independently of its evidence")

    @pytest.mark.parametrize("who", ["a-human", "coverage-derivations/x", "", "manual"])
    def test_derived_by_must_be_a_derivation_engine_id(self, db_session, instrument, who):
        _rejected(db_session, lambda: _raw_insert(db_session, instrument, Status.COMPLETE,
                  _complete_evidence(instrument), derived_by=who), "derived_by must be a coverage-derivation engine id")

    def test_a_non_complete_row_cannot_carry_a_forged_derived_by_either(self, db_session, instrument):
        ev = derive(providers={})
        _rejected(db_session, lambda: _raw_insert(db_session, instrument, Status.UNKNOWN, ev, derived_by="human"),
                  "derived_by must be")

    def test_an_unrecognised_evidence_schema_is_rejected(self, db_session, instrument):
        ev = _complete_evidence(instrument)
        ev["schema"] = "coverage-evidence/99"
        _rejected(db_session, lambda: _raw_insert(db_session, instrument, Status.COMPLETE, ev), "unrecognised coverage evidence schema")

    def test_a_hand_written_status_insert_with_no_evidence_cannot_be_complete_but_can_be_partial(self, db_session, instrument):
        _rejected(db_session, lambda: _raw_insert(db_session, instrument, Status.COMPLETE, None), "requires an evidence bundle")
        _raw_insert(db_session, instrument, Status.PARTIAL, None)
        db_session.flush()


class TestImmutableHistory:
    def _make(self, db_session, instrument, register=()):
        aid = record_derived_coverage_assessment(db_session, instrument.id, _complete_evidence(instrument), list(register))
        db_session.flush()
        return aid

    def test_a_derived_assessment_cannot_be_updated(self, db_session, instrument):
        aid = self._make(db_session, instrument)
        row = db_session.get(Assessment, aid)
        row.reasoning = "rewritten"
        with pytest.raises(DBAPIError, match="immutable"):
            db_session.flush()

    def test_a_derived_assessment_cannot_be_deleted(self, db_session, instrument):
        aid = self._make(db_session, instrument)
        db_session.delete(db_session.get(Assessment, aid))
        with pytest.raises(DBAPIError, match="cannot be deleted"):
            db_session.flush()

    def test_later_evidence_adds_a_new_row_and_leaves_the_earlier_one_intact(self, db_session, instrument):
        first = self._make(db_session, instrument)
        before = db_session.get(Assessment, first)
        snapshot = (before.status, before.evidence_sha256, before.reasoning)
        bse = [(d, bse_row(d, "Bonus issue 3:1")) if d == D(2017, 9, 7) else (d, r) for d, r in BASE_BSE]
        second = record_derived_coverage_assessment(db_session, instrument.id, derive(bse_dated=bse), [])
        db_session.flush()
        db_session.expire_all()
        assert second != first
        row = db_session.get(Assessment, first)
        assert (row.status, row.evidence_sha256, row.reasoning) == snapshot
        assert len(db_session.scalars(select(Assessment).where(Assessment.instrument_id == instrument.id)).all()) == 2

    def test_an_identical_re_derivation_is_a_new_row_not_an_overwrite(self, db_session, instrument):
        a = self._make(db_session, instrument)
        b = self._make(db_session, instrument)
        assert a != b
        assert len(db_session.scalars(select(Assessment).where(Assessment.instrument_id == instrument.id)).all()) == 2


class TestUnresolvedEventRegister:
    ITEMS = [
        {"ex_date": date(2020, 5, 13), "action_class": "RIGHTS", "reason_code": "UNSUPPORTED_ACTION_TYPE",
         "severity": "BLOCKS_ADJUSTMENT", "detail": {"why": "rights unsupported"}},
        {"ex_date": date(2016, 3, 16), "action_class": "REVISION", "reason_code": "REVISED_NON_PRICE_AFFECTING",
         "severity": "INFO", "detail": {"note": "dividend revised"}},
    ]

    def _register(self, db_session, instrument):
        aid = record_derived_coverage_assessment(db_session, instrument.id, _complete_evidence(instrument), self.ITEMS)
        db_session.flush()
        return aid

    def test_items_are_stored_linked_to_their_assessment(self, db_session, instrument):
        aid = self._register(db_session, instrument)
        rows = db_session.scalars(select(Unresolved).where(Unresolved.assessment_id == aid)).all()
        assert sorted(r.reason_code for r in rows) == ["REVISED_NON_PRICE_AFFECTING", "UNSUPPORTED_ACTION_TYPE"]
        assert all(r.instrument_id == instrument.id and r.registered_at is not None for r in rows)

    def test_a_complete_window_and_unresolved_events_coexist(self, db_session, instrument):
        aid = self._register(db_session, instrument)
        assert db_session.get(Assessment, aid).status == Status.COMPLETE
        blockers = db_session.scalars(select(Unresolved).where(Unresolved.severity == "BLOCKS_ADJUSTMENT")).all()
        assert len(blockers) == 1 and blockers[0].action_class == "RIGHTS"

    def test_severity_is_constrained(self, db_session, instrument):
        aid = self._register(db_session, instrument)
        def go():
            db_session.execute(pg_insert(Unresolved).values(
                id=uuid.uuid4(), instrument_id=instrument.id, assessment_id=aid, action_class="X",
                reason_code="Y", severity="SORT_OF_BAD", detail={}))
        with pytest.raises(IntegrityError, match="ck_unresolved_event_severity"):
            with db_session.begin_nested():
                go()

    def test_register_rows_cannot_be_updated(self, db_session, instrument):
        aid = self._register(db_session, instrument)
        row = db_session.scalars(select(Unresolved).where(Unresolved.assessment_id == aid)).first()
        row.severity = "INFO"
        with pytest.raises(DBAPIError, match="append-only"):
            db_session.flush()

    def test_register_rows_cannot_be_deleted(self, db_session, instrument):
        aid = self._register(db_session, instrument)
        db_session.delete(db_session.scalars(select(Unresolved).where(Unresolved.assessment_id == aid)).first())
        with pytest.raises(DBAPIError, match="cannot be deleted"):
            db_session.flush()

    def test_a_register_item_must_reference_a_real_assessment(self, db_session, instrument):
        def go():
            db_session.execute(pg_insert(Unresolved).values(
                id=uuid.uuid4(), instrument_id=instrument.id, assessment_id=uuid.uuid4(), action_class="X",
                reason_code="Y", severity="INFO", detail={}))
        with pytest.raises(IntegrityError):
            with db_session.begin_nested():
                go()

    def test_a_derived_assessment_and_its_register_are_written_together(self, db_session, instrument):
        bad = [{"ex_date": None, "action_class": "X", "reason_code": "Y", "severity": "NOPE", "detail": {}}]
        with pytest.raises(IntegrityError):
            with db_session.begin_nested():
                record_derived_coverage_assessment(db_session, instrument.id, _complete_evidence(instrument), bad)
                db_session.flush()
        assert db_session.scalars(select(Assessment).where(Assessment.instrument_id == instrument.id)).all() == []


class TestWindowContainmentStatus:
    def _derived(self, db_session, instrument, evidence, window):
        evidence = copy.deepcopy(evidence)
        evidence["window"] = {"from": window[0].isoformat(), "to": window[1].isoformat()}
        record_derived_coverage_assessment(db_session, instrument.id, evidence, [])
        db_session.flush()

    def _conflicted(self):
        bse = [(d, bse_row(d, "Bonus issue 3:1")) if d == D(2017, 9, 7) else (d, r) for d, r in BASE_BSE]
        return derive(bse_dated=bse)

    def test_no_assessment_is_unknown_never_complete(self, db_session, instrument):
        assert coverage_status_for_window(db_session, instrument.id, *FULL) == Status.UNKNOWN

    def test_a_complete_superset_window_covers_a_subset_query(self, db_session, instrument):
        self._derived(db_session, instrument, _complete_evidence(instrument), FULL)
        assert coverage_status_for_window(db_session, instrument.id, date(2020, 1, 1), date(2022, 12, 31)) == Status.COMPLETE

    def test_the_exact_window_is_covered(self, db_session, instrument):
        self._derived(db_session, instrument, _complete_evidence(instrument), FULL)
        assert coverage_status_for_window(db_session, instrument.id, *FULL) == Status.COMPLETE

    def test_a_query_reaching_outside_the_assessed_window_is_unknown(self, db_session, instrument):
        self._derived(db_session, instrument, _complete_evidence(instrument), FULL)
        assert coverage_status_for_window(db_session, instrument.id, date(2010, 1, 1), date(2020, 1, 1)) == Status.UNKNOWN
        assert coverage_status_for_window(db_session, instrument.id, date(2020, 1, 1), date(2026, 12, 31)) == Status.UNKNOWN

    def test_a_later_overlapping_non_complete_assessment_withdraws_the_complete(self, db_session, instrument):
        self._derived(db_session, instrument, _complete_evidence(instrument), FULL)
        self._derived(db_session, instrument, self._conflicted(), (date(2019, 1, 1), date(2021, 12, 31)))
        assert coverage_status_for_window(db_session, instrument.id, date(2020, 1, 1), date(2020, 12, 31)) == Status.CONFLICTED
        assert coverage_status_for_window(db_session, instrument.id, date(2018, 6, 1), date(2019, 6, 1)) == Status.CONFLICTED

    def test_a_later_non_overlapping_assessment_does_not_withdraw_it(self, db_session, instrument):
        self._derived(db_session, instrument, _complete_evidence(instrument), FULL)
        self._derived(db_session, instrument, self._conflicted(), (date(2019, 1, 1), date(2021, 12, 31)))
        assert coverage_status_for_window(db_session, instrument.id, date(2023, 1, 1), date(2023, 12, 31)) == Status.COMPLETE

    def test_an_earlier_non_complete_assessment_does_not_override_a_later_complete(self, db_session, instrument):
        self._derived(db_session, instrument, self._conflicted(), FULL)
        self._derived(db_session, instrument, _complete_evidence(instrument), FULL)
        assert coverage_status_for_window(db_session, instrument.id, *FULL) == Status.COMPLETE

    def test_a_later_complete_for_a_narrower_window_does_not_restore_an_earlier_withdrawal_outside_it(self, db_session, instrument):
        self._derived(db_session, instrument, _complete_evidence(instrument), FULL)
        self._derived(db_session, instrument, self._conflicted(), FULL)
        self._derived(db_session, instrument, _complete_evidence(instrument), (date(2020, 1, 1), date(2020, 12, 31)))
        assert coverage_status_for_window(db_session, instrument.id, date(2016, 1, 1), date(2016, 12, 31)) == Status.CONFLICTED
        assert coverage_status_for_window(db_session, instrument.id, date(2020, 3, 1), date(2020, 4, 1)) == Status.COMPLETE

    def test_other_instruments_assessments_never_leak(self, db_session, instrument):
        other = Instrument(exchange="NSE", asset_type=AssetType.EQUITY)
        db_session.add(other)
        db_session.flush()
        self._derived(db_session, other, _complete_evidence(other), FULL)
        assert coverage_status_for_window(db_session, instrument.id, *FULL) == Status.UNKNOWN

    def test_exact_window_lookup_is_unchanged_and_does_not_use_containment(self, db_session, instrument):
        self._derived(db_session, instrument, _complete_evidence(instrument), FULL)
        assert latest_coverage_status(db_session, instrument.id, *FULL) == Status.COMPLETE
        assert latest_coverage_status(db_session, instrument.id, date(2020, 1, 1), date(2020, 12, 31)) == Status.UNKNOWN


class TestVerifiedEventVersusCompleteWindow:
    def _verified_bonus(self, db_session, instrument):
        e = CorporateActionEvent(
            id=uuid.uuid4(), instrument_id=instrument.id, source_symbol="TESTCO", ex_date=date(2017, 9, 7),
            raw_subject_text="Bonus 1:1", source_provider="test", source_url="http://fake", source_snapshot_path="x",
            source_snapshot_sha256="x", source_fetched_at=datetime.now(timezone.utc),
            verification_status=Verification.MANUALLY_VERIFIED, created_by="test",
        )
        db_session.add(e)
        db_session.flush()
        db_session.add(CorporateActionComponent(
            id=uuid.uuid4(), event_id=e.id, component_order=1, component_type="BONUS", raw_text_fragment="Bonus 1:1",
            ratio_new=1, ratio_existing=1, verification_status=Verification.MANUALLY_VERIFIED))
        db_session.flush()
        return e

    def test_a_manually_verified_event_does_not_make_its_window_complete(self, db_session, instrument):
        self._verified_bonus(db_session, instrument)
        assert coverage_status_for_window(db_session, instrument.id, *FULL) == Status.UNKNOWN
        assert latest_coverage_status(db_session, instrument.id, *FULL) == Status.UNKNOWN

    def test_the_verification_lookup_reports_the_event_but_it_is_a_separate_fact_from_coverage(self, db_session, instrument):
        e = self._verified_bonus(db_session, instrument)
        (entry,) = nse_event_verification_lookup(db_session, instrument.id)[(date(2017, 9, 7), "BONUS")]
        assert entry["event_id"] == e.id and entry["status"] == "MANUALLY_VERIFIED" and entry["terms"] == (1, 1)

    def test_a_complete_window_does_not_verify_any_event(self, db_session, instrument):
        e = CorporateActionEvent(
            id=uuid.uuid4(), instrument_id=instrument.id, source_symbol="TESTCO", ex_date=date(2017, 9, 7),
            raw_subject_text="Bonus 1:1", source_provider="test", source_url="http://fake", source_snapshot_path="x",
            source_snapshot_sha256="x", source_fetched_at=datetime.now(timezone.utc),
            verification_status=Verification.UNVERIFIED, created_by="test")
        db_session.add(e)
        db_session.flush()
        record_derived_coverage_assessment(db_session, instrument.id, _complete_evidence(instrument), [])
        db_session.flush()
        db_session.expire_all()
        assert coverage_status_for_window(db_session, instrument.id, *FULL) == Status.COMPLETE
        assert db_session.get(CorporateActionEvent, e.id).verification_status == Verification.UNVERIFIED


class TestMigrationArtifactsExist:
    def test_the_trigger_and_check_constraints_are_installed(self, db_session):
        triggers = {r[0] for r in db_session.execute(text(
            "SELECT tgname FROM pg_trigger WHERE NOT tgisinternal"))}
        assert {"trg_coverage_assessment_derived", "trg_unresolved_events_immutable", "trg_unresolved_events_no_delete"} <= triggers
        checks = {r[0] for r in db_session.execute(text(
            "SELECT conname FROM pg_constraint WHERE contype = 'c'"))}
        assert {"ck_coverage_complete_requires_derivation", "ck_coverage_evidence_columns_together",
                "ck_unresolved_event_severity"} <= checks
