from __future__ import annotations

import copy
import hashlib
import inspect
import json
import re
import uuid
from datetime import date as D, datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import DBAPIError, IntegrityError

from backend.app.data.instruments import term_verification as tvdb
from backend.app.data.instruments.corporate_actions import (
    assess_adjustment_eligibility,
    nse_event_verification_lookup,
    promote_component_verification_status,
    promote_verification_status,
    record_derived_coverage_assessment,
)
from backend.app.data.instruments.term_verification import (
    ComponentNotFound,
    NoActiveVerification,
    NotVerifiable,
    PrimarySourceSpec,
    SheetHashMismatch,
    TermsAttestationMismatch,
    VerificationRefused,
    build_component_sheet,
    legacy_verification_inventory,
    revoke_component_terms,
    verify_component_terms,
)
from backend.app.db.models import (
    AssetType,
    CorporateActionComponent as Cmp,
    CorporateActionEvent as E,
    CorporateActionTermVerification as V,
    CorporateActionVerificationLog as L,
    CorporateActionVerificationStatus as VS,
    Instrument,
    InstrumentSymbol,
)
from backend.app.research import adjustment_eligibility as ae
from backend.app.research import term_verification as tv
from backend.tests.c1_helpers import derive
from backend.tests.c2_helpers import HUMAN, insert_c2_record, valid_evidence

EX = D(2020, 1, 1)
FULL = (D(2015, 1, 1), D(2026, 9, 11))
MIGRATION = Path(__file__).resolve().parents[1] / "migrations" / "versions" / "0ec974920efd_phase_3_stage_3_9_c2_corporate_action_.py"


@pytest.fixture
def instrument(db_session) -> Instrument:
    inst = Instrument(exchange="NSE", asset_type=AssetType.EQUITY)
    db_session.add(inst)
    db_session.flush()
    db_session.add(InstrumentSymbol(instrument_id=inst.id, symbol="TESTCO", effective_from=D(2010, 1, 1), effective_to=None))
    db_session.flush()
    return inst


def make_event(db, inst, ex=EX, subject="Bonus 1:1", comps=(("BONUS", VS.MACHINE_PARSED, dict(ratio_new=1, ratio_existing=1), "Bonus 1:1"),),
               status=VS.UNVERIFIED):
    e = E(id=uuid.uuid4(), instrument_id=inst.id, source_symbol="TESTCO", ex_date=ex, record_date=ex, raw_subject_text=subject,
          source_provider="test", source_url="http://fake", source_snapshot_path="x", source_snapshot_sha256="x",
          source_fetched_at=datetime.now(timezone.utc), verification_status=status, created_by="test")
    db.add(e)
    db.flush()
    ids = []
    for i, spec in enumerate(comps, 1):
        ctype, cstatus, fields, frag, *rev = spec
        cid = uuid.uuid4()
        db.add(Cmp(id=cid, event_id=e.id, component_order=i, component_type=ctype, raw_text_fragment=frag,
                   verification_status=cstatus, parse_revision=rev[0] if rev else 1, **fields))
        db.flush()
        ids.append(cid)
    return e, ids


def bonus_comp(ratio=(1, 1), status=VS.MACHINE_PARSED, frag=None, rev=1):
    return ("BONUS", status, dict(ratio_new=ratio[0], ratio_existing=ratio[1]), frag or f"Bonus {ratio[0]}:{ratio[1]}", rev)


def split_comp(fv=(10.0, 2.0), status=VS.MACHINE_PARSED, frag=None, rev=1):
    return ("SPLIT", status, dict(face_value_from=fv[0], face_value_to=fv[1]),
            frag or f"Face Value Split (Sub-Division) - From Rs {int(fv[0])}/- Per Share To Rs {int(fv[1])}/- Per Share", rev)


def _rejected(db, fn, match):
    with pytest.raises(DBAPIError, match=match):
        with db.begin_nested():
            fn()


def n_records(db):
    return db.scalar(select(func.count()).select_from(V))


def n_log(db):
    return db.scalar(select(func.count()).select_from(L))


def status_of(db, model, id_):
    return db.execute(select(model.verification_status).where(model.id == id_)).scalar().value


class TestRecordsAreAppendOnly:
    def test_control_a_valid_record_inserts(self, db_session, instrument):
        _, (cid,) = make_event(db_session, instrument)
        insert_c2_record(db_session, cid, "BONUS", 1, 1)
        assert n_records(db_session) == 1

    def test_update_is_refused(self, db_session, instrument):
        _, (cid,) = make_event(db_session, instrument)
        rid = insert_c2_record(db_session, cid, "BONUS", 1, 1)
        _rejected(db_session, lambda: db_session.execute(update(V).where(V.id == rid).values(tier=2)), "append-only")

    def test_delete_is_refused(self, db_session, instrument):
        _, (cid,) = make_event(db_session, instrument)
        rid = insert_c2_record(db_session, cid, "BONUS", 1, 1)
        _rejected(db_session, lambda: db_session.execute(V.__table__.delete().where(V.id == rid)), "append-only")

    def test_a_revocation_is_a_new_row_and_the_original_is_untouched(self, db_session, instrument):
        _, (cid,) = make_event(db_session, instrument)
        rid = insert_c2_record(db_session, cid, "BONUS", 1, 1)
        insert_c2_record(db_session, cid, "BONUS", 1, 1, action="REVOKED", revokes_id=rid)
        rows = db_session.execute(select(V.action, V.seq).where(V.component_id == cid).order_by(V.seq)).all()
        assert [r.action for r in rows] == ["VERIFIED", "REVOKED"]


class TestAttesterMustBeANamedHuman:
    @pytest.mark.parametrize("name", ["system", "Assistant Code", "gpt-5-agent", "stage3.9-c1", "automation", "Model 5", "pytest", "test",
                                      "unspecified", "bot", "root", "  system  ", "ab", "123", "human"])
    def test_the_database_itself_rejects_reserved_and_system_identities(self, db_session, instrument, name):
        _, (cid,) = make_event(db_session, instrument)
        _rejected(db_session, lambda: insert_c2_record(db_session, cid, "BONUS", 1, 1, attested_by=name),
                  "ck_term_verification_attested_by_human")

    @pytest.mark.parametrize("name", ["Dr Ada Lovelace", "Botha Nel", "Testa Rossi", "Autumn Rao", "R. Sharma"])
    def test_named_humans_are_accepted(self, db_session, instrument, name):
        _, (cid,) = make_event(db_session, instrument)
        insert_c2_record(db_session, cid, "BONUS", 1, 1, attested_by=name)

    def test_the_database_pattern_is_identical_to_the_python_constant(self, db_session):
        (definition,) = db_session.execute(text(
            "SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = 'ck_term_verification_attested_by_human'")).one()
        assert tv.RESERVED_ATTESTER_PATTERN in definition

    def test_python_and_database_agree_on_which_names_are_reserved(self, db_session, instrument):
        _, (cid,) = make_event(db_session, instrument)
        for name in ["system", "Assistant", "stage3.9", "Automation Bot", "Botha Nel", "Dr Ada Lovelace", "Autumn Rao", "tests-run", "gpt-5"]:
            try:
                tv.validate_attester(name)
                py_ok = True
            except tv.AttesterRejected:
                py_ok = False
            try:
                with db_session.begin_nested():
                    insert_c2_record(db_session, cid, "BONUS", 1, 1, attested_by=name)
                    raise _Rollback()
            except _Rollback:
                db_ok = True
            except DBAPIError:
                db_ok = False
            assert py_ok == db_ok, name


class _Rollback(Exception):
    pass


class TestARecordBindsToTheExactComponentAndTerms:
    def test_evidence_naming_a_different_component_is_refused(self, db_session, instrument):
        _, (cid,) = make_event(db_session, instrument)
        _, (other,) = make_event(db_session, instrument, ex=D(2021, 1, 1), subject="Bonus 1:1 (b)")
        ev = valid_evidence(other, "BONUS", 1, 1)
        _rejected(db_session, lambda: insert_c2_record(db_session, cid, "BONUS", 1, 1, evidence=ev), "different component")

    def test_attested_terms_that_differ_from_the_components_own_terms_are_refused(self, db_session, instrument):
        _, (cid,) = make_event(db_session, instrument, comps=[bonus_comp((4, 1))])
        _rejected(db_session, lambda: insert_c2_record(db_session, cid, "BONUS", 1, 4), "do not equal the component")
        _rejected(db_session, lambda: insert_c2_record(db_session, cid, "BONUS", 2, 1), "do not equal the component")

    def test_a_split_record_must_match_the_stored_face_values(self, db_session, instrument):
        _, (cid,) = make_event(db_session, instrument, comps=[split_comp((10.0, 2.0))])
        _rejected(db_session, lambda: insert_c2_record(db_session, cid, "SPLIT", 10, 5), "do not equal the component")
        insert_c2_record(db_session, cid, "SPLIT", 10, 2)

    def test_a_fractional_face_value_is_matched_exactly(self, db_session, instrument):
        _, (cid,) = make_event(db_session, instrument, comps=[split_comp((2.5, 0.5))])
        insert_c2_record(db_session, cid, "SPLIT", "2.5", "0.5")

    def test_kind_must_match_the_component_type(self, db_session, instrument):
        _, (cid,) = make_event(db_session, instrument, comps=[bonus_comp((2, 1))])
        _rejected(db_session, lambda: insert_c2_record(db_session, cid, "SPLIT", 2, 1), "does not match the component type")

    @pytest.mark.parametrize("ctype,fields", [("RIGHTS", dict(ratio_new=1, ratio_existing=15)), ("DEMERGER", {}), ("DIVIDEND", dict(cash_amount_per_share=5.0)),
                                              ("OTHER", {}), ("MERGER", {})])
    def test_unsupported_component_types_can_never_be_verified(self, db_session, instrument, ctype, fields):
        _, (cid,) = make_event(db_session, instrument, comps=[(ctype, VS.UNVERIFIED, fields, "x")])
        _rejected(db_session, lambda: insert_c2_record(db_session, cid, "BONUS", 1, 15), "does not match the component type")

    def test_an_unknown_component_is_refused(self, db_session):
        with pytest.raises(DBAPIError):
            with db_session.begin_nested():
                insert_c2_record(db_session, uuid.uuid4(), "BONUS", 1, 1)

    def test_a_reparsed_component_is_not_current_and_cannot_be_verified(self, db_session, instrument):
        e, (old,) = make_event(db_session, instrument)
        db_session.add(Cmp(id=uuid.uuid4(), event_id=e.id, component_order=1, component_type="BONUS", raw_text_fragment="Bonus 1:1",
                           verification_status=VS.MACHINE_PARSED, parse_revision=2, ratio_new=1, ratio_existing=1))
        db_session.flush()
        _rejected(db_session, lambda: insert_c2_record(db_session, old, "BONUS", 1, 1), "re-parse creates a new component")

    def test_one_records_verification_never_applies_to_a_sibling_component(self, db_session, instrument):
        e, (c1, c2) = make_event(db_session, instrument, comps=[bonus_comp((1, 1)), split_comp((10.0, 2.0))])
        insert_c2_record(db_session, c1, "BONUS", 1, 1)
        _rejected(db_session, lambda: promote_component_verification_status(db_session, c2, VS.MANUALLY_VERIFIED, "n", actor="t"),
                  "MANUALLY_VERIFIED requires an active C2")

    def test_a_second_active_record_for_the_same_component_is_refused(self, db_session, instrument):
        _, (cid,) = make_event(db_session, instrument)
        insert_c2_record(db_session, cid, "BONUS", 1, 1)
        _rejected(db_session, lambda: insert_c2_record(db_session, cid, "BONUS", 1, 1), "already has an active verification")

    def test_after_a_revocation_the_component_can_be_verified_again(self, db_session, instrument):
        _, (cid,) = make_event(db_session, instrument)
        rid = insert_c2_record(db_session, cid, "BONUS", 1, 1)
        insert_c2_record(db_session, cid, "BONUS", 1, 1, action="REVOKED", revokes_id=rid)
        insert_c2_record(db_session, cid, "BONUS", 1, 1)


class TestEvidenceMustBeAReadySheetForThisComponent:
    @pytest.mark.parametrize("mutate,why", [
        (lambda e: e.update(schema="something-else"), "not a corporate-action-term-evidence-sheet"),
        (lambda e: e.update(pit_status="ESTABLISHED"), "pit_status NOT_ESTABLISHED"),
        (lambda e: e["readiness"].update(result="BLOCKED"), "not READY with zero blockers"),
        (lambda e: e["readiness"].update(result="READY_TIER2_NEEDS_PRIMARY"), "not READY with zero blockers"),
        (lambda e: e["readiness"].update(verifiable=False), "not READY with zero blockers"),
        (lambda e: e["readiness"].update(blockers=[{"code": "TERMS_CONFLICT", "detail": {}}]), "not READY with zero blockers"),
        (lambda e: e["readiness"].pop("blockers"), "not READY with zero blockers"),
        (lambda e: e["readiness"].update(tier_required=2), "tier does not equal"),
        (lambda e: e["terms"].update(term_a="2"), "terms differ from the attested terms"),
        (lambda e: e["terms"].update(kind="SPLIT"), "terms differ from the attested terms"),
        (lambda e: e.pop("terms"), "terms differ from the attested terms"),
    ])
    def test_a_sheet_that_is_not_ready_or_disagrees_with_the_record_is_refused(self, db_session, instrument, mutate, why):
        _, (cid,) = make_event(db_session, instrument)
        ev = valid_evidence(cid, "BONUS", 1, 1)
        mutate(ev)
        _rejected(db_session, lambda: insert_c2_record(db_session, cid, "BONUS", 1, 1, evidence=ev), why)

    def test_tier2_requires_a_valid_hashed_primary_source(self, db_session, instrument):
        _, (cid,) = make_event(db_session, instrument, comps=[bonus_comp((4, 1))])
        ev = valid_evidence(cid, "BONUS", 4, 1, tier=2)
        ev["primary_source"] = {"valid": False, "sha256": "a" * 64}
        _rejected(db_session, lambda: insert_c2_record(db_session, cid, "BONUS", 4, 1, tier=2, evidence=ev), "Tier 2 requires")
        ev["primary_source"] = {"valid": True, "sha256": "short"}
        _rejected(db_session, lambda: insert_c2_record(db_session, cid, "BONUS", 4, 1, tier=2, evidence=ev), "Tier 2 requires")
        insert_c2_record(db_session, cid, "BONUS", 4, 1, tier=2)

    @pytest.mark.parametrize("kind,a,b,frag_terms", [
        ("BONUS", "1.5", 1, dict(ratio_new=1, ratio_existing=1)), ("BONUS", 0, 1, dict(ratio_new=1, ratio_existing=1)),
        ("BONUS", -1, 1, dict(ratio_new=1, ratio_existing=1)),
    ])
    def test_malformed_terms_are_refused_by_check_constraints(self, db_session, instrument, kind, a, b, frag_terms):
        _, (cid,) = make_event(db_session, instrument)
        with pytest.raises((IntegrityError, DBAPIError)):
            with db_session.begin_nested():
                insert_c2_record(db_session, cid, kind, a, b)

    def test_a_noop_split_and_bad_hash_and_blank_note_are_refused(self, db_session, instrument):
        _, (cid,) = make_event(db_session, instrument, comps=[split_comp((10.0, 2.0))])
        with pytest.raises((IntegrityError, DBAPIError)):
            with db_session.begin_nested():
                db_session.execute(pg_insert(V).values(
                    id=uuid.uuid4(), component_id=cid, action="VERIFIED", action_kind="SPLIT", term_a=10, term_b=2, tier=1,
                    evidence=valid_evidence(cid, "SPLIT", 10, 2), evidence_sha256="NOTAHASH", attested_by=HUMAN,
                    attestation_note="ok reviewed", pit_status="NOT_ESTABLISHED"))
        with pytest.raises((IntegrityError, DBAPIError)):
            with db_session.begin_nested():
                insert_c2_record(db_session, cid, "SPLIT", 10, 2, note="   ")


class TestRevocationRules:
    def test_a_revocation_must_reference_an_active_verified_record_of_the_same_component_and_terms(self, db_session, instrument):
        _, (c1,) = make_event(db_session, instrument)
        _, (c2,) = make_event(db_session, instrument, ex=D(2021, 1, 1), subject="Bonus 1:1 (b)")
        rid = insert_c2_record(db_session, c1, "BONUS", 1, 1)
        insert_c2_record(db_session, c2, "BONUS", 1, 1)
        _rejected(db_session, lambda: insert_c2_record(db_session, c2, "BONUS", 1, 1, action="REVOKED", revokes_id=rid),
                  "must carry the component, kind and exact terms")
        _rejected(db_session, lambda: insert_c2_record(db_session, c1, "BONUS", 1, 1, action="REVOKED", revokes_id=None), "must reference a VERIFIED record")

    def test_a_record_cannot_be_revoked_twice(self, db_session, instrument):
        _, (cid,) = make_event(db_session, instrument)
        rid = insert_c2_record(db_session, cid, "BONUS", 1, 1)
        insert_c2_record(db_session, cid, "BONUS", 1, 1, action="REVOKED", revokes_id=rid)
        _rejected(db_session, lambda: insert_c2_record(db_session, cid, "BONUS", 1, 1, action="REVOKED", revokes_id=rid), "already been revoked")

    def test_a_revocation_cannot_revoke_a_revocation(self, db_session, instrument):
        _, (cid,) = make_event(db_session, instrument)
        rid = insert_c2_record(db_session, cid, "BONUS", 1, 1)
        rev = insert_c2_record(db_session, cid, "BONUS", 1, 1, action="REVOKED", revokes_id=rid)
        _rejected(db_session, lambda: insert_c2_record(db_session, cid, "BONUS", 1, 1, action="REVOKED", revokes_id=rev), "must reference a VERIFIED record")

    def test_a_revocation_must_state_a_reason(self, db_session, instrument):
        _, (cid,) = make_event(db_session, instrument)
        rid = insert_c2_record(db_session, cid, "BONUS", 1, 1)
        _rejected(db_session, lambda: insert_c2_record(db_session, cid, "BONUS", 1, 1, action="REVOKED", revokes_id=rid,
                  evidence={"revocation": {"reason": "  "}}), "must state a reason")


class TestBareStatusPromotionCannotBypassTheRecord:

    def _flip(self, db, cid):
        db.execute(update(Cmp).where(Cmp.id == cid).values(verification_status=VS.MANUALLY_VERIFIED))

    def test_a_bare_sql_update_is_refused(self, db_session, instrument):
        _, (cid,) = make_event(db_session, instrument)
        _rejected(db_session, lambda: self._flip(db_session, cid), "MANUALLY_VERIFIED requires an active C2")
        assert status_of(db_session, Cmp, cid) == "MACHINE_PARSED"

    @pytest.mark.parametrize("start", [VS.UNVERIFIED, VS.MACHINE_PARSED, VS.DISPUTED])
    def test_from_every_status(self, db_session, instrument, start):
        _, (cid,) = make_event(db_session, instrument, comps=[bonus_comp(status=start)])
        _rejected(db_session, lambda: self._flip(db_session, cid), "MANUALLY_VERIFIED requires an active C2")

    def test_the_application_helper_is_refused_too_and_writes_no_log_row(self, db_session, instrument):
        _, (cid,) = make_event(db_session, instrument)
        before = n_log(db_session)
        _rejected(db_session, lambda: promote_component_verification_status(db_session, cid, VS.MANUALLY_VERIFIED, "n", actor="t"),
                  "MANUALLY_VERIFIED requires an active C2")
        assert n_log(db_session) == before

    def test_with_an_active_matching_record_it_is_allowed(self, db_session, instrument):
        _, (cid,) = make_event(db_session, instrument)
        insert_c2_record(db_session, cid, "BONUS", 1, 1)
        self._flip(db_session, cid)
        assert status_of(db_session, Cmp, cid) == "MANUALLY_VERIFIED"

    def test_a_revoked_record_no_longer_permits_it(self, db_session, instrument):
        _, (cid,) = make_event(db_session, instrument)
        rid = insert_c2_record(db_session, cid, "BONUS", 1, 1)
        insert_c2_record(db_session, cid, "BONUS", 1, 1, action="REVOKED", revokes_id=rid)
        _rejected(db_session, lambda: self._flip(db_session, cid), "MANUALLY_VERIFIED requires an active C2")

    def test_another_components_record_does_not_count(self, db_session, instrument):
        _, (c1,) = make_event(db_session, instrument)
        _, (c2,) = make_event(db_session, instrument, ex=D(2021, 1, 1), subject="Bonus 1:1 (b)")
        insert_c2_record(db_session, c1, "BONUS", 1, 1)
        _rejected(db_session, lambda: self._flip(db_session, c2), "MANUALLY_VERIFIED requires an active C2")

    @pytest.mark.parametrize("ctype,fields", [("RIGHTS", dict(ratio_new=1, ratio_existing=15)), ("DEMERGER", {}), ("DIVIDEND", dict(cash_amount_per_share=5.0)), ("OTHER", {})])
    def test_unsupported_types_can_never_become_manually_verified_by_update(self, db_session, instrument, ctype, fields):
        _, (cid,) = make_event(db_session, instrument, comps=[(ctype, VS.UNVERIFIED, fields, "x")])
        _rejected(db_session, lambda: self._flip(db_session, cid), "MANUALLY_VERIFIED requires an active C2")

    def test_demotion_is_always_allowed(self, db_session, instrument):
        _, (cid,) = make_event(db_session, instrument, comps=[bonus_comp(status=VS.MANUALLY_VERIFIED)])
        db_session.execute(update(Cmp).where(Cmp.id == cid).values(verification_status=VS.DISPUTED))
        assert status_of(db_session, Cmp, cid) == "DISPUTED"

    def test_notes_only_updates_and_other_transitions_are_untouched(self, db_session, instrument):
        _, (cid,) = make_event(db_session, instrument)
        db_session.execute(update(Cmp).where(Cmp.id == cid).values(verification_notes="a note"))
        db_session.execute(update(Cmp).where(Cmp.id == cid).values(verification_status=VS.UNVERIFIED))

    def test_a_legacy_manually_verified_row_is_left_exactly_as_it_is(self, db_session, instrument):
        _, (cid,) = make_event(db_session, instrument, comps=[bonus_comp(status=VS.MANUALLY_VERIFIED)])
        db_session.execute(update(Cmp).where(Cmp.id == cid).values(verification_notes="annotation"))
        assert status_of(db_session, Cmp, cid) == "MANUALLY_VERIFIED"

    def test_insert_time_is_deliberately_not_guarded_so_existing_fixtures_keep_working(self, db_session, instrument):
        make_event(db_session, instrument, comps=[bonus_comp(status=VS.MANUALLY_VERIFIED)])


class TestMigrationArtifacts:
    def test_triggers_and_table_exist(self, db_session):
        trig = {r[0] for r in db_session.execute(text("SELECT tgname FROM pg_trigger WHERE NOT tgisinternal"))}
        assert {"trg_term_verification_append_only", "trg_term_verification_insert_guard", "trg_component_manual_verification_guard"} <= trig

    def test_the_migration_revises_c1_and_its_downgrade_refuses_when_records_exist(self):
        src = MIGRATION.read_text(encoding="utf-8")
        assert "down_revision: Union[str, Sequence[str], None] = 'fdd64860370f'" in src
        down = src[src.index("def downgrade"):]
        assert "refusing to downgrade" in down and down.index("refusing to downgrade") < down.index("DROP") and "docs/32" in down

    def test_the_downgrade_really_refuses_when_audit_records_exist(self, db_session, instrument):
        import importlib.util

        from alembic.migration import MigrationContext
        from alembic.operations import Operations

        _, (cid,) = make_event(db_session, instrument)
        insert_c2_record(db_session, cid, "BONUS", 1, 1)
        spec = importlib.util.spec_from_file_location("c2_migration_under_test", MIGRATION)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        conn = db_session.connection()
        with Operations.context(MigrationContext.configure(conn)):
            with pytest.raises(RuntimeError, match="refusing to downgrade: 1 term-verification record"):
                module.downgrade()
        assert db_session.scalar(text("SELECT count(*) FROM corporate_action_term_verifications")) == 1

    def test_the_downgrade_is_allowed_when_no_records_exist_and_removes_exactly_what_the_upgrade_added(self, db_session):
        import importlib.util

        from alembic.migration import MigrationContext
        from alembic.operations import Operations

        spec = importlib.util.spec_from_file_location("c2_migration_under_test2", MIGRATION)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with Operations.context(MigrationContext.configure(db_session.connection())):
            module.downgrade()
        assert db_session.scalar(text("SELECT count(*) FROM information_schema.tables WHERE table_name = 'corporate_action_term_verifications'")) == 0
        assert db_session.scalar(text("SELECT count(*) FROM pg_trigger WHERE tgname IN ('trg_component_manual_verification_guard', "
                                      "'trg_term_verification_insert_guard', 'trg_term_verification_append_only')")) == 0

    def test_the_migration_writes_no_data(self):
        src = MIGRATION.read_text(encoding="utf-8").lower()
        up = src[src.index("def upgrade"):src.index("def downgrade")]
        assert "insert into" not in up and "update corporate_action" not in up and "delete from" not in up


class World:

    def __init__(self, db, inst, root: Path, *, subject="Bonus 1:1", bse_purpose="Bonus issue 1:1", comps=None, ex=EX,
                 fetched="2026-09-20T10:00:00+00:00", nse_symbol="TESTCO", bse_short="TESTCO"):
        self.db, self.inst, self.root, self.ex = db, inst, root, ex
        self.roots = {"NSE": root / "nse", "BSE": root / "bse"}
        self.event, self.ids = make_event(db, inst, ex=ex, subject=subject, comps=comps or [bonus_comp()])
        self.nse_rows = [self.nse_row(subject)]
        self.bse_rows = [self.bse_row(bse_purpose, short=bse_short)]
        self.write("NSE", self.nse_rows, fetched=fetched, symbol=nse_symbol)
        self.write("BSE", self.bse_rows, fetched=fetched)

    def nse_row(self, subject, ex=None, rec=None, symbol="TESTCO"):
        ex = ex or self.ex
        return {"symbol": symbol, "series": "EQ", "comp": "Testco Limited", "isin": "INE000A01018", "subject": subject,
                "exDate": ex.strftime("%d-%b-%Y"), "recDate": (rec or ex).strftime("%d-%b-%Y"), "faceVal": "1", "caBroadcastDate": None}

    def bse_row(self, purpose, ex=None, rec=None, short="TESTCO"):
        ex = ex or self.ex
        return {"scrip_code": 500001, "short_name": short, "long_name": "Testco Limited", "Purpose": purpose,
                "Ex_date": ex.strftime("%d %b %Y"), "RD_Date": (rec or ex).strftime("%d %b %Y")}

    def write(self, provider, rows, *, fetched="2026-09-20T10:00:00+00:00", symbol="TESTCO", name=None):
        d = self.roots[provider] / symbol
        d.mkdir(parents=True, exist_ok=True)
        body = json.dumps(rows).encode()
        sha = hashlib.sha256(body).hexdigest()
        path = d / (name or f"{symbol}_{sha[:16]}.json")
        path.write_bytes(body)
        meta = {"provider": "NSE (nseindia.com)" if provider == "NSE" else "BSE (api.bseindia.com)", "symbol": symbol,
                "sha256": sha, "fetched_at": fetched, "record_count": len(rows), "fetch_mode": "windowed", "purpose": "c1-whole",
                "query_window": {"from_date": "2015-01-01", "to_date": "2026-09-20"},
                "query_params": {"scripcode": "500001"} if provider == "BSE" else {}}
        path.with_suffix(".json.meta.json").write_text(json.dumps(meta))
        return path

    def replace(self, provider, rows, **kw):
        for f in (self.roots[provider] / kw.get("symbol", "TESTCO")).glob("*"):
            f.unlink()
        return self.write(provider, rows, **kw)

    @property
    def cid(self):
        return self.ids[0]

    def sheet(self, cid=None, **kw):
        return build_component_sheet(self.db, cid or self.cid, raw_roots=self.roots, **kw)

    def verify(self, cid=None, *, sheet=None, terms=("BONUS", 1, 1), by=HUMAN, note="I reviewed the evidence sheet", **kw):
        cid = cid or self.cid
        sheet = sheet or self.sheet(cid, **({"primary_source": kw["primary_source"]} if "primary_source" in kw else {}))
        return verify_component_terms(self.db, cid, sheet_sha256=sheet.sha256, attested_by=by, attested_terms=terms,
                                      attestation_note=note, raw_roots=self.roots, **kw)


@pytest.fixture
def world(db_session, instrument, tmp_path):
    return World(db_session, instrument, tmp_path)


def counts(db):
    return (n_records(db), n_log(db), db.scalar(select(func.count()).select_from(E)), db.scalar(select(func.count()).select_from(Cmp)))


class TestBuildingTheSheetIsReadOnlyAndDeterministic:
    def test_a_normal_bonus_is_ready_tier1_and_nothing_is_written(self, db_session, world):
        before = counts(db_session)
        s = world.sheet()
        assert s.readiness == tv.READY_TIER1 and s.content["verification_state"] == tv.STATE_NOT_VERIFIED
        assert counts(db_session) == before and not db_session.new and not db_session.dirty
        assert status_of(db_session, Cmp, world.cid) == "MACHINE_PARSED"

    def test_the_hash_is_deterministic_across_rebuilds(self, world):
        assert world.sheet().sha256 == world.sheet().sha256

    def test_the_sheet_carries_the_database_side_facts(self, world):
        c = world.sheet().content
        assert c["instrument"]["symbol"] == "TESTCO" and c["component"]["component_id"] == str(world.cid) and c["event"]["event_id"] == str(world.event.id)
        assert c["pit_status"] == "NOT_ESTABLISHED"

    def test_an_unknown_component_id_raises(self, db_session, world):
        with pytest.raises(ComponentNotFound):
            build_component_sheet(db_session, uuid.uuid4(), raw_roots=world.roots)

    def test_a_legacy_status_is_identified_on_the_sheet(self, db_session, instrument, tmp_path):
        w = World(db_session, instrument, tmp_path, comps=[bonus_comp(status=VS.MANUALLY_VERIFIED)])
        assert w.sheet().content["verification_state"] == tv.STATE_LEGACY

    def test_other_instruments_snapshots_are_never_used(self, db_session, instrument, tmp_path):
        w = World(db_session, instrument, tmp_path, nse_symbol="OTHERCO")
        assert "EVIDENCE_UNAVAILABLE" in {b["code"] for b in w.sheet().blockers}


class TestVerifyHappyPath:
    def test_it_creates_one_immutable_record_and_promotes_with_an_audit_log(self, db_session, world):
        sheet = world.sheet()
        rid = world.verify(sheet=sheet)
        rec = db_session.execute(select(V).where(V.id == rid)).scalar_one()
        assert (rec.action, rec.action_kind, rec.tier, rec.attested_by) == ("VERIFIED", "BONUS", 1, HUMAN)
        assert rec.evidence_sha256 == sheet.sha256 and rec.evidence["readiness"]["result"] == "READY_TIER1" and rec.pit_status == "NOT_ESTABLISHED"
        assert (rec.term_a, rec.term_b) == (1, 1)
        assert status_of(db_session, Cmp, world.cid) == "MANUALLY_VERIFIED" and status_of(db_session, E, world.event.id) == "MANUALLY_VERIFIED"
        logs = db_session.execute(select(L.action, L.from_status, L.to_status, L.actor, L.payload).where(L.target_id == world.cid)).all()
        assert [(l.action, l.from_status, l.to_status, l.actor) for l in logs] == [("STATUS_CHANGE", "MACHINE_PARSED", "MANUALLY_VERIFIED", HUMAN)]
        assert logs[0].payload["c2_record_id"] == str(rid) and logs[0].payload["sheet_sha256"] == sheet.sha256

    def test_after_verification_the_component_is_no_longer_a_candidate(self, world):
        world.verify()
        assert world.sheet().readiness == tv.NOT_A_CANDIDATE

    def test_verifying_the_same_component_twice_is_refused(self, db_session, world):
        world.verify()
        with pytest.raises(NotVerifiable):
            world.verify()

    def test_a_split_and_its_exact_terms(self, db_session, instrument, tmp_path):
        w = World(db_session, instrument, tmp_path, subject="Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share",
                  bse_purpose="Stock  Split From Rs.10/- to Rs.2/-", comps=[split_comp((10.0, 2.0))])
        rid = w.verify(terms=("SPLIT", 10, 2))
        rec = db_session.execute(select(V).where(V.id == rid)).scalar_one()
        assert (rec.action_kind, rec.term_a, rec.term_b) == ("SPLIT", 10, 2)


class TestExplicitAbbreviatedSplitWording:

    JSW_NSE = "Fv Splt Frm Rs 10 To Re 1"

    def _world(self, db, inst, tmp_path, nse_text=None, terms=(10.0, 1.0)):
        text_ = nse_text or self.JSW_NSE
        return World(db, inst, tmp_path, subject=" " + text_, bse_purpose="Stock  Split From Rs.10/- to Rs.1/-",
                     comps=[("SPLIT", VS.MACHINE_PARSED, dict(face_value_from=terms[0], face_value_to=terms[1]), text_)])

    def test_it_is_ready_tier1_and_can_be_verified_with_no_primary_document(self, db_session, instrument, tmp_path):
        w = self._world(db_session, instrument, tmp_path)
        sheet = w.sheet()
        assert sheet.readiness == tv.READY_TIER1 and sheet.content["readiness"]["notes"][0]["code"] == "NON_TEMPLATE_WORDING_READ_EXPLICITLY"
        rid = w.verify(sheet=sheet, terms=("SPLIT", 10, 1))
        rec = db_session.execute(select(V).where(V.id == rid)).scalar_one()
        assert rec.tier == 1 and rec.evidence["primary_source"] is None
        assert rec.evidence["wording"]["NSE"]["wording_class"] == "EXPLICIT_NON_TEMPLATE"

    def test_an_ambiguous_abbreviation_is_still_refused_and_writes_nothing(self, db_session, instrument, tmp_path):
        w = self._world(db_session, instrument, tmp_path, nse_text="Fv Splt 10 1")
        before = counts(db_session)
        assert w.sheet().readiness != tv.READY_TIER1
        with pytest.raises(NotVerifiable):
            w.verify(terms=("SPLIT", 10, 1))
        assert counts(db_session) == before and status_of(db_session, Cmp, w.cid) == "MACHINE_PARSED"

    def test_explicit_wording_that_disagrees_with_the_stored_terms_is_refused(self, db_session, instrument, tmp_path):
        w = self._world(db_session, instrument, tmp_path, terms=(10.0, 5.0))
        before = counts(db_session)
        with pytest.raises(NotVerifiable):
            w.verify(terms=("SPLIT", 10, 5))
        assert counts(db_session) == before


class TestVerifyRefusesAndWritesNothing:
    def _refused(self, db, world, exc, **kw):
        before = counts(db)
        with pytest.raises(exc):
            world.verify(**kw)
        assert counts(db) == before and status_of(db, Cmp, world.cid) == "MACHINE_PARSED"

    def test_a_wrong_sheet_hash(self, db_session, world):
        with pytest.raises(SheetHashMismatch):
            verify_component_terms(db_session, world.cid, sheet_sha256="0" * 64, attested_by=HUMAN, attested_terms=("BONUS", 1, 1),
                                   attestation_note="I reviewed it", raw_roots=world.roots)

    def test_a_sheet_reviewed_for_a_different_component(self, db_session, instrument, tmp_path):
        w = World(db_session, instrument, tmp_path)
        _, (other,) = make_event(db_session, instrument, ex=D(2021, 1, 1), subject="Bonus 1:1 (b)")
        other_sheet = build_component_sheet(db_session, other, raw_roots=w.roots)
        self._refused(db_session, w, SheetHashMismatch, sheet=other_sheet)

    def test_evidence_that_changed_after_review(self, db_session, world):
        reviewed = world.sheet()
        world.replace("BSE", [world.bse_row("Bonus issue 1:1", rec=D(2020, 1, 5))], fetched="2026-09-21T10:00:00+00:00")
        self._refused(db_session, world, SheetHashMismatch, sheet=reviewed)

    def test_the_component_changing_after_review_changes_the_hash(self, db_session, world):
        reviewed = world.sheet()
        promote_component_verification_status(db_session, world.cid, VS.DISPUTED, "second source disagrees", actor="someone")
        with pytest.raises(SheetHashMismatch):
            world.verify(sheet=reviewed)

    @pytest.mark.parametrize("terms", [("BONUS", 1, 4), ("BONUS", 2, 1), ("SPLIT", 1, 1), ("BONUS", 0, 1), ("BONUS", "x", 1), ("BONUS", 1.5, 1)])
    def test_attested_terms_that_are_not_the_components_exact_terms(self, db_session, world, terms):
        self._refused(db_session, world, TermsAttestationMismatch, terms=terms)

    @pytest.mark.parametrize("bad", [None, "BONUS", ("BONUS",), ("BONUS", 1), 5])
    def test_malformed_attested_terms(self, db_session, world, bad):
        self._refused(db_session, world, TermsAttestationMismatch, terms=bad)

    @pytest.mark.parametrize("name", ["system", "Assistant", "bot-code", "stage3.9-c1", "automation", "unspecified", "pytest", "", "  ", "ab", None, 7])
    def test_reserved_system_and_non_string_identities(self, db_session, world, name):
        self._refused(db_session, world, tv.AttesterRejected, by=name)

    @pytest.mark.parametrize("note", ["", "   ", "short", None])
    def test_a_missing_attestation_note(self, db_session, world, note):
        self._refused(db_session, world, VerificationRefused, note=note)

    def test_conflicting_sources(self, db_session, instrument, tmp_path):
        w = World(db_session, instrument, tmp_path, bse_purpose="Bonus issue 2:1")
        self._refused(db_session, w, NotVerifiable)

    def test_class_only_agreement(self, db_session, instrument, tmp_path):
        w = World(db_session, instrument, tmp_path, subject="Face Value Split - details awaited", bse_purpose="Stock  Split From Rs.10/- to Rs.2/-",
                  comps=[split_comp((10.0, 2.0))])
        self._refused(db_session, w, NotVerifiable, terms=("SPLIT", 10, 2))

    @pytest.mark.parametrize("ctype,fields,subject,purpose", [
        ("RIGHTS", dict(ratio_new=1, ratio_existing=15), "Rights 1:15 @ Premium Rs 1247", "Right Issue of Equity Shares "),
        ("DEMERGER", {}, "Demerger", "Spin Off "),
        ("MERGER", {}, "Amalgamation", "Amalgamation"),
        ("OTHER", {}, "Some Novel Event", "Some Novel Event"),
    ])
    def test_unsupported_and_unknown_types_even_with_agreeing_evidence_and_an_attestation(self, db_session, instrument, tmp_path, ctype, fields, subject, purpose):
        w = World(db_session, instrument, tmp_path, subject=subject, bse_purpose=purpose, comps=[(ctype, VS.UNVERIFIED, fields, subject)])
        before = counts(db_session)
        with pytest.raises(NotVerifiable) as exc:
            verify_component_terms(db_session, w.cid, sheet_sha256=w.sheet().sha256, attested_by=HUMAN, attested_terms=("BONUS", 1, 15),
                                   attestation_note="I reviewed it", raw_roots=w.roots)
        assert w.sheet().readiness == tv.BLOCKED and counts(db_session) == before

    def test_stale_evidence_fetched_before_the_ex_date(self, db_session, instrument, tmp_path):
        w = World(db_session, instrument, tmp_path, fetched="2019-12-01T00:00:00+00:00")
        self._refused(db_session, w, NotVerifiable)

    def test_evidence_from_the_wrong_instrument(self, db_session, instrument, tmp_path):
        w = World(db_session, instrument, tmp_path, bse_short="OTHERCO")
        self._refused(db_session, w, NotVerifiable)

    def test_a_reparsed_component_that_is_no_longer_current(self, db_session, world):
        db_session.add(Cmp(id=uuid.uuid4(), event_id=world.event.id, component_order=1, component_type="BONUS", raw_text_fragment="Bonus 1:1",
                           verification_status=VS.MACHINE_PARSED, parse_revision=2, ratio_new=1, ratio_existing=1))
        db_session.flush()
        self._refused(db_session, world, NotVerifiable)

    def test_a_disputed_component(self, db_session, instrument, tmp_path):
        w = World(db_session, instrument, tmp_path, comps=[bonus_comp(status=VS.DISPUTED)])
        before = counts(db_session)
        with pytest.raises(NotVerifiable):
            w.verify()
        assert counts(db_session) == before

    def test_an_unknown_component_id(self, db_session, world):
        with pytest.raises(ComponentNotFound):
            verify_component_terms(db_session, uuid.uuid4(), sheet_sha256="0" * 64, attested_by=HUMAN, attested_terms=("BONUS", 1, 1),
                                   attestation_note="I reviewed it", raw_roots=world.roots)

    def test_two_same_class_twins_on_one_date(self, db_session, instrument, tmp_path):
        w = World(db_session, instrument, tmp_path, comps=[bonus_comp(), bonus_comp()])
        before = counts(db_session)
        with pytest.raises(NotVerifiable):
            w.verify(cid=w.ids[0])
        assert counts(db_session) == before


class TestThereIsNoBulkVerification:
    def test_the_api_takes_exactly_one_component_id(self):
        params = inspect.signature(verify_component_terms).parameters
        assert list(params)[:2] == ["session", "component_id"]
        assert not [n for n in dir(tvdb) if re.search(r"bulk|verify_all|verify_many|verify_events|verify_batch|verify_components", n)]

    @pytest.mark.parametrize("bad", ["list", "tuple", "event_id_str", "none"])
    def test_anything_but_one_uuid_is_refused_before_the_database_is_touched(self, db_session, world, bad):
        cid = {"list": [world.cid, world.cid], "tuple": (world.cid,), "event_id_str": str(world.event.id), "none": None}[bad]
        before = counts(db_session)
        for call in (
            lambda: verify_component_terms(db_session, cid, sheet_sha256="0" * 64, attested_by=HUMAN, attested_terms=("BONUS", 1, 1),
                                           attestation_note="I reviewed it", raw_roots=world.roots),
            lambda: revoke_component_terms(db_session, cid, attested_by=HUMAN, reason="r", attestation_note="I withdrew my attestation"),
            lambda: build_component_sheet(db_session, cid, raw_roots=world.roots),
        ):
            with pytest.raises(VerificationRefused, match="per component"):
                call()
        assert counts(db_session) == before and status_of(db_session, Cmp, world.cid) == "MACHINE_PARSED"

    def test_verifying_one_component_leaves_its_neighbours_untouched(self, db_session, instrument, tmp_path):
        w = World(db_session, instrument, tmp_path)
        _, (other,) = make_event(db_session, instrument, ex=D(2021, 1, 1), subject="Bonus 1:1 (b)")
        w.verify()
        assert status_of(db_session, Cmp, other) == "MACHINE_PARSED"


class TestTier2NeedsARealHashedPrimarySource:
    @pytest.fixture
    def w4(self, db_session, instrument, tmp_path):
        return World(db_session, instrument, tmp_path, subject="Bonus 4:1", bse_purpose="Bonus issue 4:1", comps=[bonus_comp((4, 1))])

    @staticmethod
    def _doc(tmp_path, quote="the Board approved a bonus issue of 4 equity shares for every 1 equity share held", body=b"%PDF-1.4 outcome of board meeting"):
        root = tmp_path / "primary"
        root.mkdir(exist_ok=True)
        f = root / "outcome.pdf"
        f.write_bytes(body)
        return root, PrimarySourceSpec("Outcome of Board Meeting", str(f), hashlib.sha256(body).hexdigest(), quote)

    def test_a_directional_bonus_without_a_primary_source_is_refused(self, db_session, w4):
        assert w4.sheet().readiness == tv.READY_TIER2_NEEDS_PRIMARY
        before = counts(db_session)
        with pytest.raises(NotVerifiable):
            w4.verify(terms=("BONUS", 4, 1))
        assert counts(db_session) == before

    def test_with_a_real_hashed_document_it_is_verified_at_tier2(self, db_session, w4, tmp_path):
        root, spec = self._doc(tmp_path)
        sheet = w4.sheet(primary_source=spec, primary_root=root)
        assert sheet.readiness == tv.READY_TIER2
        rid = verify_component_terms(db_session, w4.cid, sheet_sha256=sheet.sha256, attested_by=HUMAN, attested_terms=("BONUS", 4, 1),
                                     attestation_note="I read the board outcome", raw_roots=w4.roots, primary_source=spec, primary_root=root)
        rec = db_session.execute(select(V).where(V.id == rid)).scalar_one()
        assert rec.tier == 2 and rec.evidence["primary_source"]["sha256"] == spec.sha256 and rec.evidence["primary_source"]["valid"] is True

    def test_a_wrong_hash_a_missing_file_or_a_file_outside_the_store_is_not_available(self, db_session, w4, tmp_path):
        root, spec = self._doc(tmp_path)
        for bad in (PrimarySourceSpec(spec.description, spec.path, "0" * 64, spec.quoted_text),
                    PrimarySourceSpec(spec.description, str(root / "missing.pdf"), spec.sha256, spec.quoted_text)):
            s = w4.sheet(primary_source=bad, primary_root=root)
            assert s.readiness == tv.BLOCKED and "PRIMARY_SOURCE_INVALID" in {b["code"] for b in s.blockers}
        outside = tmp_path / "elsewhere.pdf"
        outside.write_bytes(b"x")
        s = w4.sheet(primary_source=PrimarySourceSpec("d", str(outside), hashlib.sha256(b"x").hexdigest(), spec.quoted_text), primary_root=root)
        assert s.readiness == tv.BLOCKED

    def test_a_quote_that_does_not_contain_the_terms_is_refused(self, db_session, w4, tmp_path):
        root, spec = self._doc(tmp_path, quote="the Board approved a bonus issue of 7 equity shares for every 9 equity share held")
        assert w4.sheet(primary_source=spec, primary_root=root).readiness == tv.BLOCKED

    def test_the_file_changing_after_review_changes_the_sheet_hash(self, db_session, w4, tmp_path):
        root, spec = self._doc(tmp_path)
        reviewed = w4.sheet(primary_source=spec, primary_root=root)
        Path(spec.path).write_bytes(b"tampered")
        with pytest.raises(SheetHashMismatch):
            verify_component_terms(db_session, w4.cid, sheet_sha256=reviewed.sha256, attested_by=HUMAN, attested_terms=("BONUS", 4, 1),
                                   attestation_note="I read the board outcome", raw_roots=w4.roots, primary_source=spec, primary_root=root)


class TestEventPromotion:
    def test_a_compound_event_is_promoted_only_after_every_supported_component_is_verified(self, db_session, instrument, tmp_path):
        subject = "Bonus 1:1/Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share"
        w = World(db_session, instrument, tmp_path, subject=subject, bse_purpose="Bonus issue 1:1",
                  comps=[bonus_comp((1, 1)), split_comp((10.0, 2.0))])
        w.replace("BSE", [w.bse_row("Bonus issue 1:1"), w.bse_row("Stock  Split From Rs.10/- to Rs.2/-")])
        bonus_id, split_id = w.ids
        w.verify(cid=bonus_id, terms=("BONUS", 1, 1))
        assert status_of(db_session, Cmp, bonus_id) == "MANUALLY_VERIFIED" and status_of(db_session, E, w.event.id) == "UNVERIFIED"
        w.verify(cid=split_id, terms=("SPLIT", 10, 2))
        assert status_of(db_session, E, w.event.id) == "MANUALLY_VERIFIED"

    def test_an_event_with_an_unsupported_component_is_never_promoted(self, db_session, instrument, tmp_path):
        w = World(db_session, instrument, tmp_path, subject="Bonus 1:1/Demerger", bse_purpose="Bonus issue 1:1",
                  comps=[bonus_comp((1, 1)), ("DEMERGER", VS.UNVERIFIED, {}, "Demerger")])
        w.replace("BSE", [w.bse_row("Bonus issue 1:1"), w.bse_row("Spin Off ")])
        w.verify(cid=w.ids[0], terms=("BONUS", 1, 1))
        assert status_of(db_session, Cmp, w.ids[0]) == "MANUALLY_VERIFIED" and status_of(db_session, E, w.event.id) == "UNVERIFIED"

    def test_a_benign_agm_and_a_dividend_do_not_prevent_event_promotion(self, db_session, instrument, tmp_path):
        w = World(db_session, instrument, tmp_path, subject="Annual General Meeting/Bonus 1:1/Dividend - Rs 5 Per Share", bse_purpose="Bonus issue 1:1",
                  comps=[("OTHER", VS.UNVERIFIED, {}, "Annual General Meeting"), bonus_comp((1, 1)),
                         ("DIVIDEND", VS.MACHINE_PARSED, dict(cash_amount_per_share=5.0), "Dividend - Rs 5 Per Share")])
        w.replace("BSE", [w.bse_row("Bonus issue 1:1"), w.bse_row("Final Dividend - Rs. - 5.0000")])
        w.verify(cid=w.ids[1], terms=("BONUS", 1, 1))
        assert status_of(db_session, E, w.event.id) == "MANUALLY_VERIFIED"


class TestLegacyPhaseBComponentsGoThroughTheSameWorkflow:
    def test_a_legacy_component_is_ready_and_verifying_it_adds_the_record_and_logs_an_annotation(self, db_session, instrument, tmp_path):
        w = World(db_session, instrument, tmp_path, comps=[bonus_comp(status=VS.MANUALLY_VERIFIED)])
        promote_verification_status(db_session, w.event.id, VS.MANUALLY_VERIFIED, "phase b", actor="phase-b-script")
        sheet = w.sheet()
        assert sheet.content["verification_state"] == tv.STATE_LEGACY and sheet.readiness == tv.READY_TIER1
        rid = w.verify(sheet=sheet)
        assert w.sheet().content["verification_state"] == tv.STATE_C2_VERIFIED and rid
        actions = [l.action for l in db_session.execute(select(L.action).where(L.target_id == w.cid).order_by(L.seq))]
        assert actions == ["ANNOTATION"]

    def test_the_legacy_history_stays_in_the_sheet_and_the_log(self, db_session, instrument, tmp_path):
        w = World(db_session, instrument, tmp_path, comps=[bonus_comp(status=VS.MANUALLY_VERIFIED)])
        from backend.app.data.instruments.corporate_actions import capture_baseline

        capture_baseline(db_session, "COMPONENT", w.cid, actor="stage3.9-c0-remediation", note="baseline of a legacy verification")
        w.verify()
        hist = w.sheet().content["prior_verification_history"]
        assert [h["action"] for h in hist][:2] == ["BASELINE", "ANNOTATION"] and hist[0]["actor"] == "stage3.9-c0-remediation"
        assert hist[0]["target_kind"] == "COMPONENT" and hist[-1]["target_kind"] == "EVENT"

    def test_the_inventory_lists_only_supported_manually_verified_components_without_an_active_record(self, db_session, instrument, tmp_path):
        legacy = World(db_session, instrument, tmp_path / "a", comps=[bonus_comp(status=VS.MANUALLY_VERIFIED)])
        _, (done,) = make_event(db_session, instrument, ex=D(2021, 1, 1), subject="Bonus 1:1 (b)", comps=[bonus_comp(status=VS.MACHINE_PARSED)])
        insert_c2_record(db_session, done, "BONUS", 1, 1)
        db_session.execute(update(Cmp).where(Cmp.id == done).values(verification_status=VS.MANUALLY_VERIFIED))
        make_event(db_session, instrument, ex=D(2022, 1, 1), subject="Bonus 2:1", comps=[bonus_comp((2, 1), status=VS.MACHINE_PARSED)])
        make_event(db_session, instrument, ex=D(2023, 1, 1), subject="Rights", comps=[("RIGHTS", VS.MANUALLY_VERIFIED, dict(ratio_new=1, ratio_existing=15), "Rights")])
        inv = legacy_verification_inventory(db_session)
        assert [i["component_id"] for i in inv] == [str(legacy.cid)] and inv[0]["state"] == "LEGACY_MANUALLY_VERIFIED_WITHOUT_C2_EVIDENCE"


class TestRevoke:
    def test_it_appends_a_revoked_record_demotes_and_keeps_the_history(self, db_session, world):
        rid = world.verify()
        rrid = revoke_component_terms(db_session, world.cid, attested_by=HUMAN, reason="BSE later corrected the ratio", attestation_note="I withdrew my attestation")
        rows = db_session.execute(select(V.action, V.revokes_id, V.id).where(V.component_id == world.cid).order_by(V.seq)).all()
        assert [r.action for r in rows] == ["VERIFIED", "REVOKED"] and rows[1].revokes_id == rid and rows[1].id == rrid
        assert status_of(db_session, Cmp, world.cid) == "MACHINE_PARSED" and status_of(db_session, E, world.event.id) == "UNVERIFIED"
        assert db_session.execute(select(V.attested_by).where(V.id == rid)).scalar() == HUMAN
        acts = [l.action for l in db_session.execute(select(L.action).where(L.target_id == world.cid).order_by(L.seq))]
        assert acts == ["STATUS_CHANGE", "CORRECTION"]

    def test_after_a_revocation_the_component_is_a_candidate_again_and_can_be_reverified(self, db_session, world):
        world.verify()
        revoke_component_terms(db_session, world.cid, attested_by=HUMAN, reason="r", attestation_note="I withdrew my attestation")
        assert world.sheet().content["verification_state"] == tv.STATE_C2_REVOKED and world.sheet().readiness == tv.READY_TIER1
        world.verify()
        assert status_of(db_session, Cmp, world.cid) == "MANUALLY_VERIFIED"

    def test_nothing_to_revoke(self, db_session, world):
        before = counts(db_session)
        with pytest.raises(NoActiveVerification):
            revoke_component_terms(db_session, world.cid, attested_by=HUMAN, reason="r", attestation_note="I withdrew my attestation")
        assert counts(db_session) == before

    @pytest.mark.parametrize("kwargs,exc", [
        (dict(attested_by="system", reason="r", attestation_note="I withdrew my attestation"), tv.AttesterRejected),
        (dict(attested_by="Assistant", reason="r", attestation_note="I withdrew my attestation"), tv.AttesterRejected),
        (dict(attested_by=HUMAN, reason="  ", attestation_note="I withdrew my attestation"), VerificationRefused),
        (dict(attested_by=HUMAN, reason="r", attestation_note="x"), VerificationRefused),
    ])
    def test_reserved_identity_missing_reason_or_note(self, db_session, world, kwargs, exc):
        world.verify()
        before = counts(db_session)
        with pytest.raises(exc):
            revoke_component_terms(db_session, world.cid, **kwargs)
        assert counts(db_session) == before and status_of(db_session, Cmp, world.cid) == "MANUALLY_VERIFIED"

    def test_an_unknown_component(self, db_session):
        with pytest.raises(ComponentNotFound):
            revoke_component_terms(db_session, uuid.uuid4(), attested_by=HUMAN, reason="r", attestation_note="I withdrew my attestation")


def _complete(db, inst, register=()):
    ev = copy.deepcopy(derive())
    ev["instrument"]["instrument_id"] = str(inst.id)
    ev["window"] = {"from": FULL[0].isoformat(), "to": FULL[1].isoformat()}
    aid = record_derived_coverage_assessment(db, inst.id, ev, list(register))
    db.flush()
    return aid


class TestTheGateRequiresTheActiveRecord:
    def test_a_legacy_phase_b_component_stays_blocked_even_with_complete_coverage(self, db_session, instrument, tmp_path):
        w = World(db_session, instrument, tmp_path, comps=[bonus_comp(status=VS.MANUALLY_VERIFIED)])
        promote_verification_status(db_session, w.event.id, VS.MANUALLY_VERIFIED, "phase b", actor="phase-b-script")
        _complete(db_session, instrument)
        r = assess_adjustment_eligibility(db_session, instrument.id, *FULL)
        assert not r.eligible and {b.code for b in r.blockers} == {ae.NO_ACTIVE_C2_VERIFICATION}
        assert r.blockers[0].detail["legacy_manually_verified_without_c2_evidence"] is True

    def test_a_valid_c2_verification_makes_it_eligible_subject_to_the_other_gates(self, db_session, instrument, tmp_path):
        w = World(db_session, instrument, tmp_path, comps=[bonus_comp(status=VS.MANUALLY_VERIFIED)])
        promote_verification_status(db_session, w.event.id, VS.MANUALLY_VERIFIED, "phase b", actor="phase-b-script")
        _complete(db_session, instrument)
        assert not assess_adjustment_eligibility(db_session, instrument.id, *FULL).eligible
        w.verify()
        assert assess_adjustment_eligibility(db_session, instrument.id, *FULL).eligible

    def test_the_other_gates_still_apply_after_verification(self, db_session, instrument, tmp_path):
        w = World(db_session, instrument, tmp_path)
        w.verify()
        assert not assess_adjustment_eligibility(db_session, instrument.id, *FULL).eligible
        _complete(db_session, instrument, [{"ex_date": EX, "action_class": "BONUS", "reason_code": "TERMS_NOT_MANUALLY_VERIFIED",
                                            "severity": "BLOCKS_ADJUSTMENT", "detail": {}}])
        r = assess_adjustment_eligibility(db_session, instrument.id, *FULL)
        assert not r.eligible and {b.source for b in r.blockers} == {"REGISTER"}

    def test_a_revoked_verification_blocks_again(self, db_session, instrument, tmp_path):
        w = World(db_session, instrument, tmp_path)
        w.verify()
        _complete(db_session, instrument)
        assert assess_adjustment_eligibility(db_session, instrument.id, *FULL).eligible
        revoke_component_terms(db_session, w.cid, attested_by=HUMAN, reason="withdrawn", attestation_note="I withdrew my attestation")
        r = assess_adjustment_eligibility(db_session, instrument.id, *FULL)
        assert not r.eligible and ae.NO_ACTIVE_C2_VERIFICATION in {b.code for b in r.blockers}

    def test_a_status_flipped_back_to_verified_without_a_record_is_blocked(self, db_session, instrument, tmp_path):
        w = World(db_session, instrument, tmp_path)
        w.verify()
        _complete(db_session, instrument)
        revoke_component_terms(db_session, w.cid, attested_by=HUMAN, reason="withdrawn", attestation_note="I withdrew my attestation")
        promote_verification_status(db_session, w.event.id, VS.MANUALLY_VERIFIED, "restored", actor="someone")
        assert not assess_adjustment_eligibility(db_session, instrument.id, *FULL).eligible

    def test_a_reparsed_component_blocks_again_and_the_old_record_does_not_carry_over(self, db_session, instrument, tmp_path):
        w = World(db_session, instrument, tmp_path)
        w.verify()
        _complete(db_session, instrument)
        assert assess_adjustment_eligibility(db_session, instrument.id, *FULL).eligible
        db_session.add(Cmp(id=uuid.uuid4(), event_id=w.event.id, component_order=1, component_type="BONUS", raw_text_fragment="Bonus 1:1",
                           verification_status=VS.MANUALLY_VERIFIED, parse_revision=2, ratio_new=1, ratio_existing=1))
        db_session.flush()
        r = assess_adjustment_eligibility(db_session, instrument.id, *FULL)
        assert not r.eligible and ae.NO_ACTIVE_C2_VERIFICATION in {b.code for b in r.blockers}
        assert "NOT_CURRENT_PARSE_REVISION" in {b["code"] for b in w.sheet().blockers} or w.sheet().readiness == tv.NOT_A_CANDIDATE

    def test_a_record_for_one_component_never_makes_a_same_date_sibling_eligible(self, db_session, instrument, tmp_path):
        subject = "Bonus 1:1/Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share"
        w = World(db_session, instrument, tmp_path, subject=subject, comps=[bonus_comp((1, 1)), split_comp((10.0, 2.0))])
        w.replace("BSE", [w.bse_row("Bonus issue 1:1"), w.bse_row("Stock  Split From Rs.10/- to Rs.2/-")])
        w.verify(cid=w.ids[0], terms=("BONUS", 1, 1))
        _complete(db_session, instrument)
        r = assess_adjustment_eligibility(db_session, instrument.id, *FULL)
        by_class = {}
        for b in r.blockers:
            by_class.setdefault(b.action_class, set()).add(b.code)
        assert not r.eligible
        assert ae.NO_ACTIVE_C2_VERIFICATION in by_class["SPLIT"]
        assert by_class["BONUS"] == {ae.EVENT_NOT_MANUALLY_VERIFIED}

    def test_unsupported_types_stay_blocked_whatever_records_exist(self, db_session, instrument, tmp_path):
        make_event(db_session, instrument, subject="Rights", comps=[("RIGHTS", VS.MANUALLY_VERIFIED, dict(ratio_new=1, ratio_existing=15), "Rights")],
                   status=VS.MANUALLY_VERIFIED)
        _complete(db_session, instrument)
        r = assess_adjustment_eligibility(db_session, instrument.id, *FULL)
        assert not r.eligible and {b.code for b in r.blockers} == {ae.UNSUPPORTED_ACTION_TYPE}

    def test_the_gate_stays_read_only(self, db_session, instrument, tmp_path):
        World(db_session, instrument, tmp_path).verify()
        _complete(db_session, instrument)
        before = counts(db_session)
        assess_adjustment_eligibility(db_session, instrument.id, *FULL)
        assert counts(db_session) == before and not db_session.new and not db_session.dirty


class TestTheRegisterLookupReportsTheRecord:
    def test_c2_active_is_true_only_with_an_active_matching_record(self, db_session, instrument, tmp_path):
        legacy = World(db_session, instrument, tmp_path / "a", comps=[bonus_comp(status=VS.MANUALLY_VERIFIED)])
        verified = World(db_session, instrument, tmp_path / "b", ex=D(2021, 1, 1))
        verified.verify()
        lookup = nse_event_verification_lookup(db_session, instrument.id)
        assert lookup[(EX, "BONUS")][0]["c2_active"] is False and lookup[(EX, "BONUS")][0]["status"] == "MANUALLY_VERIFIED"
        assert lookup[(D(2021, 1, 1), "BONUS")][0]["c2_active"] is True
        revoke_component_terms(db_session, verified.cid, attested_by=HUMAN, reason="r", attestation_note="I withdrew my attestation")
        assert nse_event_verification_lookup(db_session, instrument.id)[(D(2021, 1, 1), "BONUS")][0]["c2_active"] is False
