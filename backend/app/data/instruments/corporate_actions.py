from __future__ import annotations

import dataclasses
import json
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

from sqlalchemy import exists, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, aliased

from backend.app.data.normalizers.corporate_actions import (
    PARSER_VERSION,
    ParsedComponent,
    parse_subject_components,
)
from backend.app.db.models import (
    CorporateActionComponent,
    CorporateActionCoverageAssessment,
    CorporateActionCoverageStatus,
    CorporateActionEvent,
    CorporateActionTermVerification,
    CorporateActionUnresolvedEvent,
    CorporateActionVerificationLog,
    CorporateActionVerificationStatus,
)


class EventNotFoundError(LookupError):
    pass


class AmbiguousEventError(LookupError):
    pass


def _parse_nse_date(value: str | None) -> date | None:
    if not value or value == "-":
        return None
    try:
        return datetime.strptime(value, "%d-%b-%Y").date()
    except ValueError:
        return None


def _parse_announcement_date(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def normalize_snapshot(
    session: Session,
    snapshot_path: Path,
    instrument_id: uuid.UUID,
    *,
    created_by: str = "stage3.9-phase-b",
) -> int:
    meta_path = snapshot_path.with_suffix(snapshot_path.suffix + ".meta.json")
    meta = json.loads(meta_path.read_text())
    rows = json.loads(snapshot_path.read_text())

    written = 0
    for row in rows:
        ex_date = _parse_nse_date(row.get("exDate"))
        if ex_date is None:
            continue
        subject = (row.get("subject") or "").strip()
        if not subject:
            continue

        event_id = uuid.uuid4()
        stmt = (
            pg_insert(CorporateActionEvent)
            .values(
                id=event_id,
                instrument_id=instrument_id,
                source_symbol=row.get("symbol") or "",
                ex_date=ex_date,
                record_date=_parse_nse_date(row.get("recDate")),
                source_isin=row.get("isin"),
                raw_subject_text=subject,
                raw_face_val=row.get("faceVal"),
                source_provider=meta["provider"],
                source_url=meta["source_url"],
                source_snapshot_path=str(snapshot_path),
                source_snapshot_sha256=meta["sha256"],
                source_fetched_at=datetime.fromisoformat(meta["fetched_at"]),
                announcement_date=_parse_announcement_date(row.get("caBroadcastDate")),
                verification_status=CorporateActionVerificationStatus.UNVERIFIED,
                created_by=created_by,
            )
            .on_conflict_do_nothing(constraint="uq_corporate_action_event_identity")
            .returning(CorporateActionEvent.id)
        )
        if session.execute(stmt).first() is None:
            continue

        written += 1
        _insert_components(
            session, event_id, parse_subject_components(subject),
            parse_revision=1, parser_version=PARSER_VERSION,
        )
    return written


def _insert_components(
    session: Session,
    event_id: uuid.UUID,
    parsed: list[ParsedComponent],
    *,
    parse_revision: int,
    parser_version: str,
) -> None:
    for order, component in enumerate(parsed, start=1):
        session.execute(
            pg_insert(CorporateActionComponent).values(
                id=uuid.uuid4(),
                event_id=event_id,
                parse_revision=parse_revision,
                parser_version=parser_version,
                component_order=order,
                component_type=component.component_type,
                raw_text_fragment=component.raw_text_fragment,
                face_value_from=component.face_value_from,
                face_value_to=component.face_value_to,
                ratio_new=component.ratio_new,
                ratio_existing=component.ratio_existing,
                cash_amount_per_share=component.cash_amount_per_share,
                currency=component.currency,
                verification_status=(
                    CorporateActionVerificationStatus.MACHINE_PARSED
                    if component.confidently_parsed
                    else CorporateActionVerificationStatus.UNVERIFIED
                ),
                verification_notes=component.note,
            )
        )


def find_event(
    session: Session,
    instrument_id: uuid.UUID,
    ex_date: date,
    raw_subject_text: str | None = None,
) -> CorporateActionEvent:
    query = select(CorporateActionEvent).where(
        CorporateActionEvent.instrument_id == instrument_id,
        CorporateActionEvent.ex_date == ex_date,
    )
    if raw_subject_text is not None:
        query = query.where(CorporateActionEvent.raw_subject_text == raw_subject_text.strip())
    matches = session.scalars(query).all()
    if not matches:
        raise EventNotFoundError(
            f"no corporate-action event for instrument_id={instrument_id} ex_date={ex_date}"
            + (f" raw_subject_text={raw_subject_text!r}" if raw_subject_text else "")
        )
    if len(matches) > 1:
        raise AmbiguousEventError(
            f"{len(matches)} events match instrument_id={instrument_id} ex_date={ex_date}: "
            f"{[m.raw_subject_text for m in matches]} - pass raw_subject_text or use the event id"
        )
    return matches[0]


def current_components(session: Session, event_id: uuid.UUID) -> list[CorporateActionComponent]:
    latest = (
        select(CorporateActionComponent.parse_revision)
        .where(CorporateActionComponent.event_id == event_id)
        .order_by(CorporateActionComponent.parse_revision.desc())
        .limit(1)
        .scalar_subquery()
    )
    return list(
        session.scalars(
            select(CorporateActionComponent)
            .where(
                CorporateActionComponent.event_id == event_id,
                CorporateActionComponent.parse_revision == latest,
            )
            .order_by(CorporateActionComponent.component_order)
        )
    )


_LOG_TARGET_KINDS = ("EVENT", "COMPONENT", "COVERAGE_ASSESSMENT")


def log_verification_change(
    session: Session,
    target_kind: str,
    target_id: uuid.UUID,
    action: str,
    *,
    actor: str,
    from_status: str | None = None,
    to_status: str | None = None,
    from_notes: str | None = None,
    to_notes: str | None = None,
    payload: dict | None = None,
) -> None:
    if target_kind not in _LOG_TARGET_KINDS:
        raise ValueError(f"target_kind must be one of {_LOG_TARGET_KINDS}, got {target_kind!r}")
    session.execute(
        pg_insert(CorporateActionVerificationLog).values(
            id=uuid.uuid4(),
            target_kind=target_kind,
            target_id=target_id,
            action=action,
            from_status=from_status,
            to_status=to_status,
            from_notes=from_notes,
            to_notes=to_notes,
            payload=payload,
            actor=actor,
            logged_at=datetime.now(timezone.utc),
        )
    )


def _read_state(session: Session, target_kind: str, target_id: uuid.UUID):
    model = {"EVENT": CorporateActionEvent, "COMPONENT": CorporateActionComponent}[target_kind]
    row = session.execute(
        select(model.verification_status, model.verification_notes).where(model.id == target_id)
    ).one_or_none()
    if row is None:
        raise EventNotFoundError(f"no {target_kind.lower()} with id={target_id}")
    return model, _enum_value(row[0]), row[1]


def _change_status(
    session: Session,
    target_kind: str,
    target_id: uuid.UUID,
    new_status: CorporateActionVerificationStatus,
    notes: str,
    *,
    actor: str,
    action: str,
    payload: dict | None,
) -> None:
    model, from_status, from_notes = _read_state(session, target_kind, target_id)
    session.execute(
        model.__table__.update()
        .where(model.id == target_id)
        .values(verification_status=new_status, verification_notes=notes)
    )
    log_verification_change(
        session, target_kind, target_id, action, actor=actor,
        from_status=from_status, to_status=new_status.value,
        from_notes=from_notes, to_notes=notes, payload=payload,
    )


def promote_verification_status(
    session: Session,
    event_id: uuid.UUID,
    new_status: CorporateActionVerificationStatus,
    notes: str,
    *,
    actor: str = "unspecified",
    action: str = "STATUS_CHANGE",
    payload: dict | None = None,
) -> None:
    _change_status(
        session, "EVENT", event_id, new_status, notes,
        actor=actor, action=action, payload=payload,
    )


def promote_component_verification_status(
    session: Session,
    component_id: uuid.UUID,
    new_status: CorporateActionVerificationStatus,
    notes: str,
    *,
    actor: str = "unspecified",
    action: str = "STATUS_CHANGE",
    payload: dict | None = None,
) -> None:
    _change_status(
        session, "COMPONENT", component_id, new_status, notes,
        actor=actor, action=action, payload=payload,
    )


def capture_baseline(
    session: Session, target_kind: str, target_id: uuid.UUID, *, actor: str,
    note: str | None = None,
) -> None:
    if target_kind == "COVERAGE_ASSESSMENT":
        row = session.get(CorporateActionCoverageAssessment, target_id)
        if row is None:
            raise EventNotFoundError(f"no coverage assessment with id={target_id}")
        log_verification_change(
            session, target_kind, target_id, "BASELINE", actor=actor,
            to_status=row.status.value, to_notes=row.reasoning,
            payload={"note": note} if note else None,
        )
        return
    _, status, notes = _read_state(session, target_kind, target_id)
    log_verification_change(
        session, target_kind, target_id, "BASELINE", actor=actor,
        to_status=status, to_notes=notes,
        payload={"note": note} if note else None,
    )


def _enum_value(x):
    return getattr(x, "value", x)


def _component_signature(
    component_type: str, fragment: str, fv_from, fv_to, ratio_new, ratio_existing,
    amount, currency, machine_parsed: bool,
) -> tuple:
    return (
        component_type, fragment, fv_from, fv_to, ratio_new, ratio_existing,
        amount, currency, machine_parsed,
    )


def plan_reparse(session: Session, event_id: uuid.UUID) -> list[ParsedComponent] | None:
    event = session.get(CorporateActionEvent, event_id)
    if event is None:
        raise EventNotFoundError(f"no event with id={event_id}")
    parsed = parse_subject_components(event.raw_subject_text)
    existing = current_components(session, event_id)

    want = [
        _component_signature(
            p.component_type, p.raw_text_fragment, p.face_value_from, p.face_value_to,
            p.ratio_new, p.ratio_existing, p.cash_amount_per_share, p.currency,
            p.confidently_parsed,
        )
        for p in parsed
    ]
    have = [
        _component_signature(
            _enum_value(c.component_type), c.raw_text_fragment, c.face_value_from, c.face_value_to,
            c.ratio_new, c.ratio_existing, c.cash_amount_per_share, c.currency,
            _enum_value(c.verification_status) != CorporateActionVerificationStatus.UNVERIFIED.value,
        )
        for c in existing
    ]
    return None if want == have else parsed


def reparse_event_components(
    session: Session, event_id: uuid.UUID, *, actor: str,
    parser_version: str = PARSER_VERSION,
) -> int | None:
    parsed = plan_reparse(session, event_id)
    if parsed is None:
        return None
    existing = current_components(session, event_id)
    old_revision = existing[0].parse_revision if existing else 0
    new_revision = old_revision + 1
    _insert_components(
        session, event_id, parsed, parse_revision=new_revision, parser_version=parser_version
    )
    log_verification_change(
        session, "EVENT", event_id, "REPARSE", actor=actor,
        payload={
            "parser_version": parser_version,
            "old_revision": old_revision,
            "new_revision": new_revision,
            "old": [
                {"type": _enum_value(c.component_type), "fragment": c.raw_text_fragment,
                 "status": _enum_value(c.verification_status)}
                for c in existing
            ],
            "new": [
                {"type": p.component_type, "fragment": p.raw_text_fragment,
                 "confident": p.confidently_parsed}
                for p in parsed
            ],
        },
    )
    return new_revision


def record_coverage_assessment(
    session: Session,
    instrument_id: uuid.UUID,
    window_start: date,
    window_end: date,
    status: CorporateActionCoverageStatus,
    reasoning: str,
    sources_considered: list[dict],
    *,
    assessed_by: str = "stage3.9-phase-b1",
) -> uuid.UUID:
    if status == CorporateActionCoverageStatus.COMPLETE:
        raise ValueError(
            "COMPLETE cannot be asserted directly: use record_derived_coverage_assessment, which derives "
            "the verdict from an evidence bundle"
        )
    assessment_id = uuid.uuid4()
    session.execute(
        pg_insert(CorporateActionCoverageAssessment).values(
            id=assessment_id,
            instrument_id=instrument_id,
            window_start=window_start,
            window_end=window_end,
            status=status,
            assessed_at=datetime.now(timezone.utc),
            reasoning=reasoning,
            sources_considered=sources_considered,
            assessed_by=assessed_by,
        )
    )
    return assessment_id


def latest_coverage_status(
    session: Session, instrument_id: uuid.UUID, window_start: date, window_end: date
) -> CorporateActionCoverageStatus:
    from sqlalchemy import select

    row = session.scalar(
        select(CorporateActionCoverageAssessment)
        .where(
            CorporateActionCoverageAssessment.instrument_id == instrument_id,
            CorporateActionCoverageAssessment.window_start == window_start,
            CorporateActionCoverageAssessment.window_end == window_end,
        )
        .order_by(CorporateActionCoverageAssessment.assessed_at.desc())
        .limit(1)
    )
    return row.status if row is not None else CorporateActionCoverageStatus.UNKNOWN


class EvidenceRejected(ValueError):
    pass


def record_derived_coverage_assessment(
    session: Session,
    instrument_id: uuid.UUID,
    evidence: dict,
    register_items: list[dict],
    *,
    assessed_by: str = "stage3.9-c1",
) -> uuid.UUID:
    from backend.app.research.corporate_actions_coverage import (
        DERIVED_BY,
        REQUIRED_CHECKS,
        evidence_sha256,
    )

    verdict = evidence.get("verdict")
    if verdict not in {s.value for s in CorporateActionCoverageStatus}:
        raise EvidenceRejected(f"evidence has no recognised verdict: {verdict!r}")
    if verdict == "COMPLETE":
        failed = [k for k in REQUIRED_CHECKS if evidence.get("checks", {}).get(k) is not True]
        if failed:
            raise EvidenceRejected(f"evidence claims COMPLETE but checks are not all true: {failed}")

    window_start = date.fromisoformat(evidence["window"]["from"])
    window_end = date.fromisoformat(evidence["window"]["to"])
    assessment_id = uuid.uuid4()
    sources_considered = [
        {
            "provider": src["provider"],
            "whole_snapshot_sha256": (src["whole"] or {}).get("sha256"),
            "part_snapshot_sha256": [p["sha256"] for p in src["parts"]],
            "record_count": (src["whole"] or {}).get("record_count"),
        }
        for src in evidence["sources"]
    ]
    session.execute(
        pg_insert(CorporateActionCoverageAssessment).values(
            id=assessment_id,
            instrument_id=instrument_id,
            window_start=window_start,
            window_end=window_end,
            status=CorporateActionCoverageStatus(verdict),
            assessed_at=datetime.now(timezone.utc),
            reasoning=evidence["reasoning"],
            sources_considered=sources_considered,
            assessed_by=assessed_by,
            evidence=evidence,
            evidence_sha256=evidence_sha256(evidence),
            derived_by=DERIVED_BY,
        )
    )
    for item in register_items:
        session.execute(
            pg_insert(CorporateActionUnresolvedEvent).values(
                id=uuid.uuid4(),
                instrument_id=instrument_id,
                assessment_id=assessment_id,
                event_id=item.get("event_id"),
                ex_date=item.get("ex_date"),
                action_class=item["action_class"],
                reason_code=item["reason_code"],
                severity=item["severity"],
                detail=json.loads(json.dumps(item["detail"], default=str)),
            )
        )
    return assessment_id


def governing_coverage_assessment(
    session: Session, instrument_id: uuid.UUID, window_start: date, window_end: date
) -> tuple[CorporateActionCoverageStatus, CorporateActionCoverageAssessment | None]:
    A = CorporateActionCoverageAssessment
    containing = session.scalar(
        select(A)
        .where(A.instrument_id == instrument_id, A.window_start <= window_start, A.window_end >= window_end)
        .order_by(A.assessed_at.desc())
        .limit(1)
    )
    if containing is None:
        return CorporateActionCoverageStatus.UNKNOWN, None
    later_overlapping = session.scalar(
        select(A)
        .where(
            A.instrument_id == instrument_id,
            A.assessed_at > containing.assessed_at,
            A.window_start <= window_end,
            A.window_end >= window_start,
            A.status != CorporateActionCoverageStatus.COMPLETE,
        )
        .order_by(A.assessed_at.desc())
        .limit(1)
    )
    if containing.status == CorporateActionCoverageStatus.COMPLETE and later_overlapping is not None:
        return later_overlapping.status, containing
    return containing.status, containing


def coverage_status_for_window(
    session: Session, instrument_id: uuid.UUID, window_start: date, window_end: date
) -> CorporateActionCoverageStatus:
    return governing_coverage_assessment(session, instrument_id, window_start, window_end)[0]


def active_term_verifications(session: Session, component_ids) -> dict:
    ids = list(component_ids)
    if not ids:
        return {}
    V = CorporateActionTermVerification
    R = aliased(CorporateActionTermVerification)
    rows = session.execute(
        select(V.id, V.component_id, V.action_kind, V.term_a, V.term_b, V.tier)
        .where(
            V.component_id.in_(ids),
            V.action == "VERIFIED",
            ~exists().where(R.action == "REVOKED", R.revokes_id == V.id),
        )
        .order_by(V.seq)
    ).all()
    return {r.component_id: (str(r.id), r.action_kind, r.term_a, r.term_b, r.tier) for r in rows}


@dataclasses.dataclass(frozen=True)
class AdjustmentGateInputs:
    coverage_status: str
    governing_assessment_id: str | None
    events: tuple
    register_items: tuple


def assess_adjustment_eligibility(
    session: Session, instrument_id: uuid.UUID, start: date, end: date
):
    from backend.app.research.adjustment_eligibility import evaluate_adjustment_eligibility

    inputs = load_adjustment_gate_inputs(session, instrument_id, start, end)
    return evaluate_adjustment_eligibility(
        start=start, end=end, coverage_status=inputs.coverage_status,
        governing_assessment_id=inputs.governing_assessment_id,
        events=inputs.events, register_items=inputs.register_items,
    )


def load_adjustment_gate_inputs(
    session: Session, instrument_id: uuid.UUID, start: date, end: date
) -> AdjustmentGateInputs:
    from backend.app.research.adjustment_eligibility import (
        ComponentView,
        EventView,
        RecordView,
        RegisterItemView,
    )

    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    status, assessment = governing_coverage_assessment(session, instrument_id, start, end)

    E, Cmp, U = CorporateActionEvent, CorporateActionComponent, CorporateActionUnresolvedEvent
    event_rows = session.execute(
        select(E.id, E.ex_date, E.verification_status)
        .where(E.instrument_id == instrument_id, E.ex_date > start, E.ex_date <= end)
        .order_by(E.ex_date, E.id)
    ).all()
    by_event: dict = {}
    if event_rows:
        for row in session.execute(
            select(Cmp.id, Cmp.event_id, Cmp.parse_revision, Cmp.component_order, Cmp.component_type,
                   Cmp.verification_status, Cmp.raw_text_fragment, Cmp.face_value_from, Cmp.face_value_to,
                   Cmp.ratio_new, Cmp.ratio_existing)
            .where(Cmp.event_id.in_([r.id for r in event_rows]))
            .order_by(Cmp.event_id, Cmp.parse_revision.desc(), Cmp.component_order)
        ):
            by_event.setdefault(row.event_id, []).append(row)

    current_ids = []
    for er in event_rows:
        rs = by_event.get(er.id, [])
        top_rev = max((r.parse_revision for r in rs), default=None)
        current_ids += [r.id for r in rs if r.parse_revision == top_rev]
    active_records = active_term_verifications(session, current_ids)

    events = []
    for er in event_rows:
        rows = by_event.get(er.id, [])
        top = max((r.parse_revision for r in rows), default=None)
        comps = []
        for c in (r for r in rows if r.parse_revision == top):
            ctype = _enum_value(c.component_type)
            if ctype == "SPLIT":
                terms = (c.face_value_from, c.face_value_to)
            elif ctype == "BONUS":
                terms = (c.ratio_new, c.ratio_existing)
            else:
                terms = None
            rec = active_records.get(c.id)
            comps.append(ComponentView(
                ctype, _enum_value(c.verification_status), c.raw_text_fragment, terms,
                RecordView(rec[0], rec[1], rec[2], rec[3], rec[4]) if rec else None,
            ))
        events.append(EventView(str(er.id), er.ex_date, _enum_value(er.verification_status), tuple(comps)))

    register = []
    if assessment is not None:
        for r in session.execute(
            select(U.ex_date, U.action_class, U.reason_code, U.severity).where(U.assessment_id == assessment.id)
        ):
            register.append(RegisterItemView(r.ex_date, r.action_class, r.reason_code, r.severity, str(assessment.id)))

    return AdjustmentGateInputs(
        coverage_status=_enum_value(status),
        governing_assessment_id=str(assessment.id) if assessment is not None else None,
        events=tuple(events), register_items=tuple(register),
    )


def nse_event_verification_lookup(session: Session, instrument_id: uuid.UUID) -> dict:
    from backend.app.research.exact_terms import terms_equal, try_normalize

    lookup: dict = {}
    events = session.scalars(
        select(CorporateActionEvent).where(CorporateActionEvent.instrument_id == instrument_id)
    ).all()
    comps_by_event = {e.id: current_components(session, e.id) for e in events}
    active = active_term_verifications(session, [c.id for cs in comps_by_event.values() for c in cs])
    for e in events:
        for c in comps_by_event[e.id]:
            cls = _enum_value(c.component_type)
            if cls == "SPLIT":
                terms = (c.face_value_from, c.face_value_to) if c.face_value_from and c.face_value_to else None
            elif cls == "BONUS":
                terms = (c.ratio_new, c.ratio_existing) if c.ratio_new and c.ratio_existing else None
            else:
                continue
            rec = active.get(c.id)
            c2_active = False
            if rec is not None and terms is not None:
                c2_active = terms_equal(try_normalize(cls, *terms)[0], try_normalize(rec[1], rec[2], rec[3])[0])
            lookup.setdefault((e.ex_date, cls), []).append(
                {"event_id": e.id, "status": _enum_value(c.verification_status), "terms": terms, "c2_active": c2_active}
            )
    return lookup
