from __future__ import annotations

import copy
import uuid
from datetime import date as D, datetime, timezone

import pytest
from sqlalchemy import func, select

from backend.app.data.instruments.corporate_actions import (
    assess_adjustment_eligibility,
    governing_coverage_assessment,
    promote_component_verification_status,
    promote_verification_status,
    record_derived_coverage_assessment,
)
from backend.app.db.models import (
    AssetType,
    CorporateActionComponent,
    CorporateActionCoverageAssessment as Assessment,
    CorporateActionCoverageStatus as Status,
    CorporateActionEvent,
    CorporateActionUnresolvedEvent as Unresolved,
    CorporateActionVerificationLog,
    CorporateActionVerificationStatus as VS,
    Instrument,
)
from backend.app.research import adjustment_eligibility as ae
from backend.tests.c1_helpers import BASE_BSE, BASE_NSE, bse_row, derive
from backend.tests.c2_helpers import insert_c2_record

FULL = (D(2015, 1, 1), D(2026, 9, 11))


@pytest.fixture
def instrument(db_session) -> Instrument:
    inst = Instrument(exchange="NSE", asset_type=AssetType.EQUITY)
    db_session.add(inst)
    db_session.flush()
    return inst


def _instrument(db_session) -> Instrument:
    inst = Instrument(exchange="NSE", asset_type=AssetType.EQUITY)
    db_session.add(inst)
    db_session.flush()
    return inst


def add_event(db, inst, ex, subject, *comps, status=VS.MANUALLY_VERIFIED, legacy=False):
    e = CorporateActionEvent(
        id=uuid.uuid4(), instrument_id=inst.id, source_symbol="TESTCO", ex_date=ex, raw_subject_text=subject,
        source_provider="test", source_url="http://fake", source_snapshot_path="x", source_snapshot_sha256="x",
        source_fetched_at=datetime.now(timezone.utc), verification_status=status, created_by="test",
    )
    db.add(e)
    db.flush()
    for i, spec in enumerate(comps, 1):
        ctype, cstatus, fields, frag, *rev = spec
        cid = uuid.uuid4()
        db.add(CorporateActionComponent(
            id=cid, event_id=e.id, component_order=i, component_type=ctype, raw_text_fragment=frag,
            verification_status=cstatus, parse_revision=rev[0] if rev else 1, **fields))
        db.flush()
        if not legacy and cstatus == VS.MANUALLY_VERIFIED and ctype in ("SPLIT", "BONUS"):
            a, b = ((fields.get("face_value_from"), fields.get("face_value_to")) if ctype == "SPLIT"
                    else (fields.get("ratio_new"), fields.get("ratio_existing")))
            if a is not None and b is not None:
                insert_c2_record(db, cid, ctype, a, b)
    db.flush()
    return e


def SPLIT(status=VS.MANUALLY_VERIFIED, fv=(10.0, 2.0), frag="Face Value Split"):
    return ("SPLIT", status, dict(face_value_from=fv[0], face_value_to=fv[1]), frag)


def BONUS(status=VS.MANUALLY_VERIFIED, ratio=(1, 1), frag="Bonus 1:1"):
    return ("BONUS", status, dict(ratio_new=ratio[0], ratio_existing=ratio[1]), frag)


def AGM():
    return ("OTHER", VS.UNVERIFIED, {}, "Annual General Meeting")


def DIV():
    return ("DIVIDEND", VS.MACHINE_PARSED, dict(cash_amount_per_share=5.0), "Dividend - Rs 5 Per Share")


def assess(db, inst, register=(), window=FULL, evidence=None):
    ev = copy.deepcopy(evidence) if evidence is not None else derive()
    ev["instrument"]["instrument_id"] = str(inst.id)
    ev["window"] = {"from": window[0].isoformat(), "to": window[1].isoformat()}
    aid = record_derived_coverage_assessment(db, inst.id, ev, list(register))
    db.flush()
    return aid


def reg(ex, cls, code="TERMS_NOT_MANUALLY_VERIFIED", sev="BLOCKS_ADJUSTMENT"):
    return {"ex_date": ex, "action_class": cls, "reason_code": code, "severity": sev, "detail": {"t": 1}}


def conflicted():
    bse = [(d, bse_row(d, "Bonus issue 3:1")) if d == D(2017, 9, 7) else (d, r) for d, r in BASE_BSE]
    return derive(bse_dated=bse)


def codes(r):
    return sorted({b.code for b in r.blockers})


def counts(db):
    return tuple(db.scalar(select(func.count()).select_from(m)) for m in
                 (CorporateActionEvent, CorporateActionComponent, Assessment, Unresolved, CorporateActionVerificationLog))


class TestCoverageIsRequiredButNeverSufficient:
    def test_no_assessment_at_all_is_ineligible_with_unknown_coverage(self, db_session, instrument):
        r = assess_adjustment_eligibility(db_session, instrument.id, *FULL)
        assert not r.eligible and r.coverage_status == "UNKNOWN" and r.governing_assessment_id is None
        assert codes(r) == [ae.COVERAGE_NOT_COMPLETE]

    def test_complete_with_no_events_and_an_empty_register_is_eligible(self, db_session, instrument):
        aid = assess(db_session, instrument)
        r = assess_adjustment_eligibility(db_session, instrument.id, *FULL)
        assert r.eligible and r.coverage_status == "COMPLETE" and r.governing_assessment_id == str(aid)

    def test_complete_plus_an_unverified_event_is_ineligible(self, db_session, instrument):
        assess(db_session, instrument)
        add_event(db_session, instrument, D(2020, 1, 1), "Bonus 1:1", BONUS(VS.MACHINE_PARSED), status=VS.UNVERIFIED)
        r = assess_adjustment_eligibility(db_session, instrument.id, *FULL)
        assert not r.eligible and r.coverage_status == "COMPLETE"
        assert {ae.EVENT_NOT_MANUALLY_VERIFIED, ae.COMPONENT_NOT_MANUALLY_VERIFIED} <= set(codes(r))

    @pytest.mark.parametrize("make", [
        lambda: derive(providers={}),
        lambda: conflicted(),
    ])
    def test_a_non_complete_assessment_blocks_even_with_perfect_events(self, db_session, instrument, make):
        assess(db_session, instrument, evidence=make())
        add_event(db_session, instrument, D(2020, 1, 1), "Bonus 1:1", BONUS())
        r = assess_adjustment_eligibility(db_session, instrument.id, *FULL)
        assert not r.eligible and codes(r) == [ae.COVERAGE_NOT_COMPLETE]

    def test_a_later_overlapping_non_complete_assessment_withdraws_eligibility(self, db_session, instrument):
        assess(db_session, instrument)
        assess(db_session, instrument, evidence=conflicted(), window=(D(2019, 1, 1), D(2021, 12, 31)))
        assert not assess_adjustment_eligibility(db_session, instrument.id, D(2020, 1, 1), D(2020, 12, 31)).eligible
        assert assess_adjustment_eligibility(db_session, instrument.id, D(2023, 1, 1), D(2023, 12, 31)).eligible

    def test_an_interval_reaching_past_the_assessed_window_has_unknown_coverage(self, db_session, instrument):
        assess(db_session, instrument)
        r = assess_adjustment_eligibility(db_session, instrument.id, D(2026, 1, 1), D(2026, 12, 31))
        assert not r.eligible and r.coverage_status == "UNKNOWN"

    def test_start_after_end_is_rejected(self, db_session, instrument):
        with pytest.raises(ValueError):
            assess_adjustment_eligibility(db_session, instrument.id, D(2020, 1, 2), D(2020, 1, 1))


class TestKotakbank2015_07_08OnTheRealSchema:
    def _kotak(self, db, inst, register=True):
        add_event(db, inst, D(2015, 6, 18), "Annual General Meeting/ Dividend - Rs 0.6", AGM(), DIV(), status=VS.UNVERIFIED)
        add_event(db, inst, D(2015, 7, 8), "Bonus 1:1", BONUS(VS.MACHINE_PARSED), status=VS.UNVERIFIED)
        add_event(db, inst, D(2017, 7, 12), "Annual General Meeting/ Dividend - Rs 0.75", AGM(), DIV(), status=VS.UNVERIFIED)
        add_event(db, inst, D(2026, 1, 14), "Face Value Split (Sub-Division) - From Rs 5/- To Re 1/-", SPLIT(fv=(5.0, 1.0)))
        return assess(db, inst, [reg(D(2015, 7, 8), "BONUS")] if register else [])

    @pytest.mark.parametrize("start,end,eligible", [
        (D(2015, 1, 1), D(2015, 7, 7), True),
        (D(2015, 1, 1), D(2015, 7, 8), False),
        (D(2015, 7, 7), D(2015, 7, 8), False),
        (D(2015, 7, 8), D(2015, 7, 8), True),
        (D(2015, 7, 8), D(2026, 9, 11), True),
        (D(2015, 7, 9), D(2026, 9, 11), True),
        (D(2015, 7, 7), D(2026, 9, 11), False),
        (D(2015, 1, 1), D(2026, 9, 11), False),
    ])
    def test_boundary_matrix_against_the_database(self, db_session, instrument, start, end, eligible):
        self._kotak(db_session, instrument)
        r = assess_adjustment_eligibility(db_session, instrument.id, start, end)
        assert r.eligible is eligible
        if not eligible:
            assert r.barrier_dates == (D(2015, 7, 8),)

    def test_the_blocker_is_reported_from_live_state_and_from_the_register(self, db_session, instrument):
        self._kotak(db_session, instrument)
        r = assess_adjustment_eligibility(db_session, instrument.id, *FULL)
        assert {(b.source, b.code) for b in r.blockers} == {
            ("EVENT", ae.EVENT_NOT_MANUALLY_VERIFIED), ("EVENT", ae.COMPONENT_NOT_MANUALLY_VERIFIED),
            ("EVENT", ae.NO_ACTIVE_C2_VERIFICATION), ("REGISTER", ae.UNRESOLVED_REGISTER_ITEM),
        }


class TestTheRegisterIsStrictAndHistorical:
    def test_verification_after_the_assessment_does_not_clear_the_register_row(self, db_session, instrument):
        e = add_event(db_session, instrument, D(2020, 1, 1), "Bonus 1:1", BONUS(VS.MACHINE_PARSED), status=VS.UNVERIFIED)
        assess(db_session, instrument, [reg(D(2020, 1, 1), "BONUS")])
        before = assess_adjustment_eligibility(db_session, instrument.id, *FULL)
        assert not before.eligible and {b.source for b in before.blockers} == {"EVENT", "REGISTER"}

        promote_verification_status(db_session, e.id, VS.MANUALLY_VERIFIED, "verified", actor="test")
        comp = db_session.scalars(select(CorporateActionComponent).where(CorporateActionComponent.event_id == e.id)).one()
        insert_c2_record(db_session, comp.id, "BONUS", 1, 1)
        promote_component_verification_status(db_session, comp.id, VS.MANUALLY_VERIFIED, "verified", actor="test")
        db_session.flush()

        after = assess_adjustment_eligibility(db_session, instrument.id, *FULL)
        assert not after.eligible
        assert {(b.source, b.code) for b in after.blockers} == {("REGISTER", ae.UNRESOLVED_REGISTER_ITEM)}

    def test_a_new_assessment_creates_the_new_register_state_and_the_old_rows_survive(self, db_session, instrument):
        add_event(db_session, instrument, D(2020, 1, 1), "Bonus 1:1", BONUS())
        first = assess(db_session, instrument, [reg(D(2020, 1, 1), "BONUS")])
        assert not assess_adjustment_eligibility(db_session, instrument.id, *FULL).eligible

        second = assess(db_session, instrument, [])
        r = assess_adjustment_eligibility(db_session, instrument.id, *FULL)
        assert r.eligible and r.governing_assessment_id == str(second)
        old_rows = db_session.scalars(select(Unresolved).where(Unresolved.assessment_id == first)).all()
        assert len(old_rows) == 1 and old_rows[0].reason_code == "TERMS_NOT_MANUALLY_VERIFIED"

    def test_only_the_governing_assessments_register_applies(self, db_session, instrument):
        assess(db_session, instrument, [reg(D(2016, 6, 1), "BONUS")])
        assess(db_session, instrument, [], window=(D(2016, 1, 1), D(2016, 12, 31)))
        assert assess_adjustment_eligibility(db_session, instrument.id, D(2016, 3, 1), D(2016, 9, 1)).eligible
        assert assess_adjustment_eligibility(db_session, instrument.id, D(2017, 3, 1), D(2017, 9, 1)).eligible
        assert not assess_adjustment_eligibility(db_session, instrument.id, D(2015, 3, 1), D(2016, 9, 1)).eligible

    def test_a_null_dated_blocking_row_blocks_every_interval_in_the_governing_window(self, db_session, instrument):
        assess(db_session, instrument, [{"ex_date": None, "action_class": "UNCLASSIFIED", "reason_code": "UNPARSEABLE_ROW",
                                         "severity": "BLOCKS_COMPLETE", "detail": {}}])
        for s, e in [FULL, (D(2020, 1, 1), D(2020, 1, 1)), (D(2015, 1, 1), D(2015, 1, 2))]:
            r = assess_adjustment_eligibility(db_session, instrument.id, s, e)
            assert not r.eligible and r.blockers[0].detail["global"] is True

    def test_info_rows_never_block(self, db_session, instrument):
        assess(db_session, instrument, [reg(D(2020, 3, 23), "DIAGNOSTIC", "WATCH_JUMP", "INFO"),
                                        reg(D(2016, 3, 16), "REVISION", "REVISED_NON_PRICE_AFFECTING", "INFO"),
                                        {"ex_date": None, "action_class": "DIAGNOSTIC", "reason_code": "WATCH_JUMP",
                                         "severity": "INFO", "detail": {}}])
        assert assess_adjustment_eligibility(db_session, instrument.id, *FULL).eligible

    def test_a_register_only_event_absent_from_the_events_table_still_blocks(self, db_session, instrument):
        assess(db_session, instrument, [reg(D(2020, 1, 1), "BONUS", "NOT_IN_EVENTS_TABLE")])
        r = assess_adjustment_eligibility(db_session, instrument.id, *FULL)
        assert not r.eligible and r.barrier_dates == (D(2020, 1, 1),)


class TestLiveEventState:
    def test_an_event_added_after_the_assessment_is_seen_even_though_the_register_is_clean(self, db_session, instrument):
        assess(db_session, instrument, [])
        assert assess_adjustment_eligibility(db_session, instrument.id, *FULL).eligible
        add_event(db_session, instrument, D(2021, 5, 5), "Bonus 2:1", BONUS(VS.MACHINE_PARSED, (2, 1)), status=VS.UNVERIFIED)
        assert not assess_adjustment_eligibility(db_session, instrument.id, *FULL).eligible

    def test_a_verified_event_later_disputed_becomes_a_blocker_immediately(self, db_session, instrument):
        e = add_event(db_session, instrument, D(2020, 1, 1), "Bonus 1:1", BONUS())
        assess(db_session, instrument, [])
        assert assess_adjustment_eligibility(db_session, instrument.id, *FULL).eligible
        promote_verification_status(db_session, e.id, VS.DISPUTED, "second source disagrees", actor="test")
        db_session.flush()
        r = assess_adjustment_eligibility(db_session, instrument.id, *FULL)
        assert not r.eligible and codes(r) == [ae.EVENT_NOT_MANUALLY_VERIFIED]

    def test_verified_split_and_verified_bonus_are_eligible(self, db_session, instrument):
        add_event(db_session, instrument, D(2016, 3, 1), "Face Value Split", SPLIT())
        add_event(db_session, instrument, D(2018, 3, 1), "Bonus 1:1", BONUS())
        assess(db_session, instrument, [])
        r = assess_adjustment_eligibility(db_session, instrument.id, *FULL)
        assert r.eligible and r.events_considered == 2

    def test_bajfinance_same_date_split_and_bonus_as_two_events_or_one_compound_event(self, db_session, instrument):
        add_event(db_session, instrument, D(2016, 9, 8), "Bonus 1:1/Face Value Split", BONUS(), SPLIT(fv=(10.0, 2.0)))
        add_event(db_session, instrument, D(2025, 6, 16), "Bonus 4:1", BONUS(VS.MACHINE_PARSED, (4, 1)), status=VS.UNVERIFIED)
        add_event(db_session, instrument, D(2025, 6, 16), "Face Value Split 2->1", SPLIT(VS.MACHINE_PARSED, (2.0, 1.0)), status=VS.UNVERIFIED)
        assess(db_session, instrument, [reg(D(2025, 6, 16), "BONUS"), reg(D(2025, 6, 16), "SPLIT")])
        assert assess_adjustment_eligibility(db_session, instrument.id, D(2015, 1, 1), D(2025, 6, 13)).eligible
        assert assess_adjustment_eligibility(db_session, instrument.id, D(2025, 6, 16), D(2026, 9, 11)).eligible
        r = assess_adjustment_eligibility(db_session, instrument.id, *FULL)
        assert not r.eligible and r.barrier_dates == (D(2025, 6, 16),) and {b.action_class for b in r.blockers} == {"BONUS", "SPLIT"}

    @pytest.mark.parametrize("ctype,fragment", [("RIGHTS", "Rights 1:15"), ("DEMERGER", "Demerger"), ("MERGER", "Amalgamation")])
    def test_unsupported_types_block_even_when_verified_at_both_levels(self, db_session, instrument, ctype, fragment):
        add_event(db_session, instrument, D(2020, 5, 13), fragment, (ctype, VS.MANUALLY_VERIFIED, {}, fragment))
        assess(db_session, instrument, [])
        r = assess_adjustment_eligibility(db_session, instrument.id, *FULL)
        assert not r.eligible and codes(r) == [ae.UNSUPPORTED_ACTION_TYPE]
        assert assess_adjustment_eligibility(db_session, instrument.id, D(2020, 5, 13), D(2021, 1, 1)).eligible
        assert assess_adjustment_eligibility(db_session, instrument.id, D(2015, 1, 1), D(2020, 5, 12)).eligible

    def test_agm_only_other_is_exempt_but_a_hidden_split_text_as_other_blocks(self, db_session, instrument):
        add_event(db_session, instrument, D(2018, 7, 11), "AGM/ Dividend", AGM(), DIV(), status=VS.UNVERIFIED)
        assess(db_session, instrument, [])
        assert assess_adjustment_eligibility(db_session, instrument.id, *FULL).eligible
        add_event(db_session, instrument, D(2017, 1, 4), "Fv Splt Frm Rs 10 To Re 1",
                  ("OTHER", VS.MANUALLY_VERIFIED, {}, "Fv Splt Frm Rs 10 To Re 1"))
        r = assess_adjustment_eligibility(db_session, instrument.id, *FULL)
        assert not r.eligible and codes(r) == [ae.UNCLASSIFIED_COMPONENT]

    def test_terms_are_read_from_the_database_fields_and_must_be_complete_and_positive(self, db_session, instrument):
        add_event(db_session, instrument, D(2016, 3, 1), "split with no terms", ("SPLIT", VS.MANUALLY_VERIFIED, {}, "Split"))
        assess(db_session, instrument, [])
        r = assess_adjustment_eligibility(db_session, instrument.id, *FULL)
        assert not r.eligible and codes(r) == [ae.NO_ACTIVE_C2_VERIFICATION, ae.TERMS_MISSING_OR_INVALID]

    def test_only_the_current_parse_revision_is_judged(self, db_session, instrument):
        add_event(db_session, instrument, D(2017, 1, 4), "Fv Splt Frm Rs 10 To Re 1",
                  ("OTHER", VS.UNVERIFIED, {}, "Fv Splt Frm Rs 10 To Re 1", 1),
                  ("SPLIT", VS.MANUALLY_VERIFIED, dict(face_value_from=10.0, face_value_to=1.0), "Fv Splt Frm Rs 10 To Re 1", 2))
        assess(db_session, instrument, [])
        assert assess_adjustment_eligibility(db_session, instrument.id, *FULL).eligible

    def test_a_worse_current_revision_wins_over_a_better_old_one(self, db_session, instrument):
        add_event(db_session, instrument, D(2017, 1, 4), "split",
                  ("SPLIT", VS.MANUALLY_VERIFIED, dict(face_value_from=10.0, face_value_to=1.0), "split", 1),
                  ("SPLIT", VS.UNVERIFIED, dict(face_value_from=10.0, face_value_to=1.0), "split", 2))
        assess(db_session, instrument, [])
        r = assess_adjustment_eligibility(db_session, instrument.id, *FULL)
        assert not r.eligible and codes(r) == [ae.COMPONENT_NOT_MANUALLY_VERIFIED, ae.NO_ACTIVE_C2_VERIFICATION]

    def test_an_event_with_no_components_blocks(self, db_session, instrument):
        add_event(db_session, instrument, D(2020, 1, 1), "mystery")
        assess(db_session, instrument, [])
        assert codes(assess_adjustment_eligibility(db_session, instrument.id, *FULL)) == [ae.EVENT_HAS_NO_COMPONENTS]

    def test_events_outside_the_interval_and_of_other_instruments_are_ignored(self, db_session, instrument):
        other = _instrument(db_session)
        add_event(db_session, other, D(2020, 1, 1), "Bonus 1:1", BONUS(VS.MACHINE_PARSED), status=VS.UNVERIFIED)
        add_event(db_session, instrument, D(2015, 1, 1), "Bonus 1:1", BONUS(VS.MACHINE_PARSED), status=VS.UNVERIFIED)
        assess(db_session, instrument, [])
        r = assess_adjustment_eligibility(db_session, instrument.id, *FULL)
        assert r.eligible and r.events_considered == 0


class TestTheGateIsReadOnly:
    def test_no_row_in_any_corporate_action_table_changes_and_nothing_is_dirty(self, db_session, instrument):
        add_event(db_session, instrument, D(2020, 1, 1), "Bonus 1:1", BONUS(VS.MACHINE_PARSED), status=VS.UNVERIFIED)
        assess(db_session, instrument, [reg(D(2020, 1, 1), "BONUS")])
        db_session.flush()
        before = counts(db_session)
        states = [(e.id, e.verification_status) for e in db_session.scalars(select(CorporateActionEvent))]
        for s, e in [FULL, (D(2015, 1, 1), D(2019, 12, 31)), (D(2020, 1, 1), D(2026, 9, 11))]:
            assess_adjustment_eligibility(db_session, instrument.id, s, e)
        assert not db_session.new and not db_session.dirty and not db_session.deleted
        assert counts(db_session) == before
        assert states == [(e.id, e.verification_status) for e in db_session.scalars(select(CorporateActionEvent))]

    def test_the_governing_assessment_lookup_agrees_with_coverage_status_for_window(self, db_session, instrument):
        from backend.app.data.instruments.corporate_actions import coverage_status_for_window

        assess(db_session, instrument)
        assess(db_session, instrument, evidence=conflicted(), window=(D(2019, 1, 1), D(2021, 12, 31)))
        for w in [FULL, (D(2020, 1, 1), D(2020, 12, 31)), (D(2023, 1, 1), D(2023, 12, 31)), (D(2010, 1, 1), D(2011, 1, 1))]:
            assert governing_coverage_assessment(db_session, instrument.id, *w)[0] == coverage_status_for_window(db_session, instrument.id, *w)
