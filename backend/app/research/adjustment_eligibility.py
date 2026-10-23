from __future__ import annotations

import dataclasses
import datetime
from typing import Sequence

from backend.app.research.corporate_action_reconciliation import (
    PRICE_AFFECTING,
    is_benign_meeting_fragment,
)
from backend.app.research.exact_terms import terms_equal, try_normalize

SUPPORTED_ADJUSTMENT_TYPES = frozenset({"SPLIT", "BONUS"})
MANUALLY_VERIFIED = "MANUALLY_VERIFIED"

COVERAGE_NOT_COMPLETE = "COVERAGE_NOT_COMPLETE"
UNSUPPORTED_ACTION_TYPE = "UNSUPPORTED_ACTION_TYPE"
UNCLASSIFIED_COMPONENT = "UNCLASSIFIED_COMPONENT"
EVENT_HAS_NO_COMPONENTS = "EVENT_HAS_NO_COMPONENTS"
EVENT_NOT_MANUALLY_VERIFIED = "EVENT_NOT_MANUALLY_VERIFIED"
COMPONENT_NOT_MANUALLY_VERIFIED = "COMPONENT_NOT_MANUALLY_VERIFIED"
TERMS_MISSING_OR_INVALID = "TERMS_MISSING_OR_INVALID"
UNRESOLVED_REGISTER_ITEM = "UNRESOLVED_REGISTER_ITEM"
NO_ACTIVE_C2_VERIFICATION = "NO_ACTIVE_C2_VERIFICATION"
C2_VERIFICATION_TERMS_MISMATCH = "C2_VERIFICATION_TERMS_MISMATCH"


@dataclasses.dataclass(frozen=True)
class RecordView:
    record_id: str
    kind: str
    term_a: object
    term_b: object
    tier: int = 1


@dataclasses.dataclass(frozen=True)
class ComponentView:
    component_type: str
    verification_status: str
    raw_text_fragment: str
    terms: tuple | None
    verification_record: RecordView | None = None


@dataclasses.dataclass(frozen=True)
class EventView:
    event_id: str
    ex_date: datetime.date
    verification_status: str
    components: tuple[ComponentView, ...]


@dataclasses.dataclass(frozen=True)
class RegisterItemView:
    ex_date: datetime.date | None
    action_class: str
    reason_code: str
    severity: str
    assessment_id: str | None = None


@dataclasses.dataclass(frozen=True)
class Blocker:
    source: str
    code: str
    ex_date: datetime.date | None
    action_class: str | None
    detail: dict

    def as_dict(self) -> dict:
        return {
            "source": self.source, "code": self.code,
            "ex_date": self.ex_date.isoformat() if self.ex_date else None,
            "action_class": self.action_class, "detail": self.detail,
        }


@dataclasses.dataclass(frozen=True)
class AdjustmentEligibility:
    eligible: bool
    start: datetime.date
    end: datetime.date
    coverage_status: str
    governing_assessment_id: str | None
    blockers: tuple[Blocker, ...]
    barrier_dates: tuple[datetime.date, ...]
    events_considered: int
    note: str = (
        "interval semantics are (start, end] anchored at end; eligibility is a property of the interval "
        "and its events, never of a coverage status alone"
    )

    def as_dict(self) -> dict:
        return {
            "eligible": self.eligible, "start": self.start.isoformat(), "end": self.end.isoformat(),
            "coverage_status": self.coverage_status, "governing_assessment_id": self.governing_assessment_id,
            "blockers": [b.as_dict() for b in self.blockers],
            "barrier_dates": [d.isoformat() for d in self.barrier_dates],
            "events_considered": self.events_considered, "note": self.note,
        }


def in_interval(ex_date: datetime.date, start: datetime.date, end: datetime.date) -> bool:
    return start < ex_date <= end


def _terms_problem(component_type: str, terms: tuple | None) -> str | None:
    if terms is None or len(terms) != 2:
        return "terms absent or incomplete"
    return try_normalize(component_type, terms[0], terms[1])[1]


def evaluate_adjustment_eligibility(
    *,
    start: datetime.date,
    end: datetime.date,
    coverage_status: str,
    governing_assessment_id: str | None,
    events: Sequence[EventView],
    register_items: Sequence[RegisterItemView],
) -> AdjustmentEligibility:
    if start > end:
        raise ValueError(f"start {start} is after end {end}")

    blockers: list[Blocker] = []
    if coverage_status != "COMPLETE":
        blockers.append(Blocker(
            "COVERAGE", COVERAGE_NOT_COMPLETE, None, None,
            {"coverage_status": coverage_status,
             "note": "coverage for the interval is not COMPLETE (an absent assessment is UNKNOWN)"},
        ))

    considered = 0
    for ev in sorted(events, key=lambda e: (e.ex_date, e.event_id)):
        if not in_interval(ev.ex_date, start, end):
            continue
        considered += 1
        if not ev.components:
            blockers.append(Blocker(
                "EVENT", EVENT_HAS_NO_COMPONENTS, ev.ex_date, None,
                {"event_id": ev.event_id, "note": "no current components: the event's content is unknown"},
            ))
            continue
        for c in ev.components:
            ctype = c.component_type
            base = {"event_id": ev.event_id, "fragment": c.raw_text_fragment}
            if ctype == "DIVIDEND":
                continue
            if ctype == "OTHER":
                if is_benign_meeting_fragment(c.raw_text_fragment):
                    continue
                blockers.append(Blocker(
                    "EVENT", UNCLASSIFIED_COMPONENT, ev.ex_date, "OTHER",
                    {**base, "note": "OTHER text that is not an exact AGM/EGM phrase may hide a price-affecting action"},
                ))
                continue
            if ctype not in SUPPORTED_ADJUSTMENT_TYPES:
                blockers.append(Blocker(
                    "EVENT", UNSUPPORTED_ACTION_TYPE, ev.ex_date, ctype,
                    {**base, "price_affecting": ctype in PRICE_AFFECTING,
                     "note": "not a currently supported adjustment type"},
                ))
                continue
            if ev.verification_status != MANUALLY_VERIFIED:
                blockers.append(Blocker(
                    "EVENT", EVENT_NOT_MANUALLY_VERIFIED, ev.ex_date, ctype,
                    {**base, "event_status": ev.verification_status},
                ))
            if c.verification_status != MANUALLY_VERIFIED:
                blockers.append(Blocker(
                    "EVENT", COMPONENT_NOT_MANUALLY_VERIFIED, ev.ex_date, ctype,
                    {**base, "component_status": c.verification_status},
                ))
            problem = _terms_problem(ctype, c.terms)
            if problem:
                blockers.append(Blocker(
                    "EVENT", TERMS_MISSING_OR_INVALID, ev.ex_date, ctype, {**base, "problem": problem},
                ))
            rec = c.verification_record
            if rec is None:
                legacy = ev.verification_status == MANUALLY_VERIFIED and c.verification_status == MANUALLY_VERIFIED
                blockers.append(Blocker(
                    "EVENT", NO_ACTIVE_C2_VERIFICATION, ev.ex_date, ctype,
                    {**base, "legacy_manually_verified_without_c2_evidence": legacy,
                     "note": "no active C2 term-verification record; a MANUALLY_VERIFIED status set before C2 is "
                             "not evidence"},
                ))
            elif problem is None:
                have, _ = try_normalize(ctype, c.terms[0], c.terms[1])
                attested, _ = try_normalize(rec.kind, rec.term_a, rec.term_b)
                if not terms_equal(have, attested):
                    blockers.append(Blocker(
                        "EVENT", C2_VERIFICATION_TERMS_MISMATCH, ev.ex_date, ctype,
                        {**base, "record_id": rec.record_id,
                         "component_terms": have.as_dict() if have else None,
                         "attested_terms": attested.as_dict() if attested else None},
                    ))

    for item in register_items:
        if item.severity == "INFO":
            continue
        if item.ex_date is None:
            blockers.append(Blocker(
                "REGISTER", UNRESOLVED_REGISTER_ITEM, None, item.action_class,
                {"reason_code": item.reason_code, "severity": item.severity, "assessment_id": item.assessment_id,
                 "global": True, "note": "a blocking register entry with no date blocks every interval"},
            ))
        elif in_interval(item.ex_date, start, end):
            blockers.append(Blocker(
                "REGISTER", UNRESOLVED_REGISTER_ITEM, item.ex_date, item.action_class,
                {"reason_code": item.reason_code, "severity": item.severity, "assessment_id": item.assessment_id,
                 "global": False},
            ))

    barrier_dates = tuple(sorted({b.ex_date for b in blockers if b.ex_date is not None}))
    return AdjustmentEligibility(
        eligible=not blockers, start=start, end=end, coverage_status=coverage_status,
        governing_assessment_id=governing_assessment_id, blockers=tuple(blockers),
        barrier_dates=barrier_dates, events_considered=considered,
    )
