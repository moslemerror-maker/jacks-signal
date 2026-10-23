from __future__ import annotations

import dataclasses
import hashlib
import json
import uuid
from datetime import date
from pathlib import Path
from typing import Sequence

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from backend.app.data.instruments.corporate_actions import (
    active_term_verifications,
    log_verification_change,
    promote_component_verification_status,
    promote_verification_status,
)
from backend.app.db.models import (
    CorporateActionComponent as Cmp,
    CorporateActionEvent as E,
    CorporateActionTermVerification as V,
    CorporateActionVerificationLog as L,
    CorporateActionVerificationStatus as VS,
    InstrumentSymbol,
)
from backend.app.research.adjustment_eligibility import SUPPORTED_ADJUSTMENT_TYPES
from backend.app.research.corporate_action_evidence import discover_snapshots, snapshot_coverage_window
from backend.app.research.corporate_action_reconciliation import is_benign_meeting_fragment
from backend.app.research.corporate_actions_coverage import canonical_json
from backend.app.research.exact_terms import ExactTerms, terms_equal, try_normalize
from backend.app.research.term_verification import (
    ComponentFacts,
    EvidenceSheet,
    HistoryFact,
    PrimarySourceDoc,
    ProviderEvidence,
    RecordFact,
    SheetInputs,
    SiblingFact,
    build_evidence_sheet,
    validate_attester,
)

DEFAULT_RAW_ROOTS = {"NSE": Path("data/raw/corporate_actions"), "BSE": Path("data/raw/corporate_actions_bse")}
DEFAULT_PRIMARY_ROOT = Path("data/raw/corporate_actions_primary")
_MIN_NOTE = 10


class VerificationRefused(ValueError):
    pass


class ComponentNotFound(LookupError):
    pass


def _single_component_id(component_id) -> uuid.UUID:
    if not isinstance(component_id, uuid.UUID):
        raise VerificationRefused(
            f"verification is per component: component_id must be a single uuid.UUID, got {type(component_id).__name__}")
    return component_id


class SheetHashMismatch(VerificationRefused):
    pass


class NotVerifiable(VerificationRefused):
    def __init__(self, readiness: str, blockers: list):
        self.readiness, self.blockers = readiness, blockers
        super().__init__(f"component is not verifiable: readiness {readiness}, blockers "
                         f"{[b['code'] for b in blockers] or 'none (Tier 2 primary source needed)'}")


class TermsAttestationMismatch(VerificationRefused):
    pass


class NoActiveVerification(VerificationRefused):
    pass


@dataclasses.dataclass(frozen=True)
class PrimarySourceSpec:
    description: str
    path: str
    sha256: str
    quoted_text: str


def load_primary_source(spec: PrimarySourceSpec | None, primary_root: Path | None = None) -> PrimarySourceDoc | None:
    if spec is None:
        return None
    root = (primary_root or DEFAULT_PRIMARY_ROOT).resolve()
    verified = False
    try:
        p = Path(spec.path).resolve()
        if p.is_file() and p.is_relative_to(root):
            verified = hashlib.sha256(p.read_bytes()).hexdigest() == spec.sha256
    except OSError:
        verified = False
    return PrimarySourceDoc(spec.description, spec.path, spec.sha256, verified, spec.quoted_text)


def _current_rows(session: Session, event_ids: Sequence) -> dict:
    ids = list(event_ids)
    if not ids:
        return {}
    out: dict = {}
    allrows: dict = {}
    for r in session.execute(
        select(Cmp.id, Cmp.event_id, Cmp.component_type, Cmp.verification_status, Cmp.parse_revision, Cmp.parser_version,
               Cmp.component_order, Cmp.raw_text_fragment, Cmp.face_value_from, Cmp.face_value_to, Cmp.ratio_new,
               Cmp.ratio_existing)
        .where(Cmp.event_id.in_(ids)).order_by(Cmp.event_id, Cmp.parse_revision.desc(), Cmp.component_order)
    ):
        allrows.setdefault(r.event_id, []).append(r)
    for eid, rows in allrows.items():
        top = max(r.parse_revision for r in rows)
        out[eid] = [r for r in rows if r.parse_revision == top]
    return out


def _ctype(row) -> str:
    return getattr(row.component_type, "value", row.component_type)


def _terms_of(row) -> tuple | None:
    t = _ctype(row)
    if t == "SPLIT":
        return (row.face_value_from, row.face_value_to)
    if t == "BONUS":
        return (row.ratio_new, row.ratio_existing)
    return None


def _status(x) -> str:
    return getattr(x, "value", x)


def _provider_evidence(provider: str, root: Path, symbol: str, ex_date: date) -> ProviderEvidence | None:
    covering = []
    for s in discover_snapshots(root, symbol):
        if s.provider != provider:
            continue
        w = snapshot_coverage_window(s)
        if w is not None and w[0] <= ex_date <= w[1]:
            covering.append(s)
    if not covering:
        return None
    windowed = [s for s in covering if s.fetch_mode == "windowed"]
    primary = max(windowed or covering, key=lambda s: (s.fetched_at, s.sha256))
    return ProviderEvidence(provider, primary, tuple(sorted(covering, key=lambda s: (s.fetched_at, s.sha256))))


def load_sheet_inputs(
    session: Session, component_id: uuid.UUID, *, raw_roots: dict | None = None,
    primary_source: PrimarySourceSpec | None = None, primary_root: Path | None = None,
) -> SheetInputs:
    component_id = _single_component_id(component_id)
    roots = {**DEFAULT_RAW_ROOTS, **(raw_roots or {})}
    comp = session.execute(
        select(Cmp.id, Cmp.event_id, Cmp.component_type, Cmp.verification_status, Cmp.parse_revision, Cmp.parser_version,
               Cmp.raw_text_fragment, Cmp.face_value_from, Cmp.face_value_to, Cmp.ratio_new, Cmp.ratio_existing)
        .where(Cmp.id == component_id)
    ).one_or_none()
    if comp is None:
        raise ComponentNotFound(f"no corporate-action component with id={component_id}")
    ev = session.execute(
        select(E.id, E.instrument_id, E.ex_date, E.record_date, E.raw_subject_text, E.verification_status)
        .where(E.id == comp.event_id)
    ).one()
    current_rev = session.scalar(select(func.max(Cmp.parse_revision)).where(Cmp.event_id == comp.event_id))
    symbol = session.scalar(
        select(InstrumentSymbol.symbol)
        .where(InstrumentSymbol.instrument_id == ev.instrument_id, InstrumentSymbol.effective_to.is_(None))
        .limit(1)
    ) or ""

    facts = ComponentFacts(
        component_id=str(comp.id), event_id=str(ev.id), instrument_id=str(ev.instrument_id), symbol=symbol,
        ex_date=ev.ex_date, record_date=ev.record_date, component_type=_ctype(comp),
        component_status=_status(comp.verification_status), event_status=_status(ev.verification_status),
        parse_revision=comp.parse_revision, current_parse_revision=current_rev, parser_version=comp.parser_version,
        raw_fragment=comp.raw_text_fragment, event_raw_subject=ev.raw_subject_text, raw_terms=_terms_of(comp),
    )

    same_date_event_ids = list(session.scalars(
        select(E.id).where(E.instrument_id == ev.instrument_id, E.ex_date == ev.ex_date)
    ))
    siblings = []
    for eid, rows in _current_rows(session, same_date_event_ids).items():
        for r in rows:
            if r.id == comp.id:
                continue
            siblings.append(SiblingFact(str(eid), str(r.id), _ctype(r), _status(r.verification_status),
                                        r.raw_text_fragment, _terms_of(r)))

    history = [
        HistoryFact(h.seq, h.target_kind, h.action, h.from_status, h.to_status, h.actor, h.logged_at.isoformat(),
                    (h.to_notes or "")[:240])
        for h in session.execute(
            select(L.seq, L.target_kind, L.action, L.from_status, L.to_status, L.actor, L.logged_at, L.to_notes)
            .where(L.target_id.in_([comp.id, ev.id])).order_by(L.seq)
        )
    ]
    records = [
        RecordFact(str(r.id), r.seq, r.action, r.action_kind, r.term_a, r.term_b, r.tier, r.attested_by,
                   r.attested_at.isoformat(), str(r.revokes_id) if r.revokes_id else None)
        for r in session.execute(
            select(V.id, V.seq, V.action, V.action_kind, V.term_a, V.term_b, V.tier, V.attested_by, V.attested_at, V.revokes_id)
            .where(V.component_id == comp.id).order_by(V.seq)
        )
    ]
    return SheetInputs(
        facts=facts, siblings=tuple(siblings), history=tuple(history), records=tuple(records),
        nse=_provider_evidence("NSE", roots["NSE"], symbol, ev.ex_date) if symbol else None,
        bse=_provider_evidence("BSE", roots["BSE"], symbol, ev.ex_date) if symbol else None,
        primary_source=load_primary_source(primary_source, primary_root),
    )


def build_component_sheet(
    session: Session, component_id: uuid.UUID, *, raw_roots: dict | None = None,
    primary_source: PrimarySourceSpec | None = None, primary_root: Path | None = None,
) -> EvidenceSheet:
    return build_evidence_sheet(load_sheet_inputs(
        session, component_id, raw_roots=raw_roots, primary_source=primary_source, primary_root=primary_root))


def _unpack_terms(attested_terms) -> tuple:
    if isinstance(attested_terms, ExactTerms):
        return attested_terms.kind, attested_terms.term_a, attested_terms.term_b
    try:
        kind, a, b = attested_terms
    except (TypeError, ValueError):
        raise TermsAttestationMismatch("attested_terms must be (kind, term_a, term_b), e.g. ('BONUS', 4, 1)") from None
    return kind, a, b


def verify_component_terms(
    session: Session, component_id: uuid.UUID, *, sheet_sha256: str, attested_by: str, attested_terms,
    attestation_note: str, primary_source: PrimarySourceSpec | None = None,
    raw_roots: dict | None = None, primary_root: Path | None = None,
) -> uuid.UUID:
    component_id = _single_component_id(component_id)
    name = validate_attester(attested_by)
    note = (attestation_note or "").strip()
    if len(note) < _MIN_NOTE:
        raise VerificationRefused("attestation_note must record the human's attestation (at least 10 characters)")
    kind, a, b = _unpack_terms(attested_terms)

    inputs = load_sheet_inputs(session, component_id, raw_roots=raw_roots, primary_source=primary_source,
                               primary_root=primary_root)
    sheet = build_evidence_sheet(inputs)
    if sheet.sha256 != sheet_sha256:
        raise SheetHashMismatch(
            f"the evidence sheet for component {component_id} is {sheet.sha256}, not the reviewed {sheet_sha256}: "
            "the evidence, the component or its history changed since review (or the sheet is for another component)")
    if not sheet.content["readiness"]["verifiable"]:
        raise NotVerifiable(sheet.readiness, sheet.blockers)
    terms = sheet.terms
    attested, why = try_normalize(kind, a, b)
    if attested is None or not terms_equal(attested, terms):
        raise TermsAttestationMismatch(
            f"attested terms {kind} {a}:{b} ({why or 'well-formed'}) do not equal the component's exact terms "
            f"{terms.statement() if terms else None}")

    tier = sheet.tier_required
    record_id = uuid.uuid4()
    session.execute(pg_insert(V).values(
        id=record_id, component_id=component_id, action="VERIFIED", action_kind=terms.kind,
        term_a=terms.term_a, term_b=terms.term_b, tier=tier, evidence=json.loads(sheet.canonical),
        evidence_sha256=sheet.sha256, attested_by=name, attestation_note=note, pit_status="NOT_ESTABLISHED",
    ))
    notes = (f"C2 term verification {record_id}: {terms.statement()}; tier {tier}; sheet {sheet.sha256[:16]}; "
             f"attested by {name}")
    payload = {"c2_record_id": str(record_id), "sheet_sha256": sheet.sha256, "tier": tier, "attested_by": name,
               "terms": terms.as_dict(), "pit_status": "NOT_ESTABLISHED"}
    if inputs.facts.component_status != VS.MANUALLY_VERIFIED.value:
        promote_component_verification_status(
            session, component_id, VS.MANUALLY_VERIFIED, notes, actor=name, action="STATUS_CHANGE", payload=payload)
    else:
        log_verification_change(
            session, "COMPONENT", component_id, "ANNOTATION", actor=name, from_status=VS.MANUALLY_VERIFIED.value,
            to_status=VS.MANUALLY_VERIFIED.value, to_notes=notes,
            payload={**payload, "legacy_status_now_evidenced": True})
    _promote_event_if_fully_verified(session, uuid.UUID(inputs.facts.event_id), name, payload)
    return record_id


def _promote_event_if_fully_verified(session: Session, event_id: uuid.UUID, name: str, payload: dict) -> bool:
    rows = _current_rows(session, [event_id]).get(event_id, [])
    supported = [r for r in rows if _ctype(r) in SUPPORTED_ADJUSTMENT_TYPES]
    unresolved_other = [
        r for r in rows
        if _ctype(r) not in SUPPORTED_ADJUSTMENT_TYPES and not (
            _ctype(r) == "DIVIDEND" or (_ctype(r) == "OTHER" and is_benign_meeting_fragment(r.raw_text_fragment)))
    ]
    if not supported or unresolved_other:
        return False
    active = active_term_verifications(session, [r.id for r in supported])
    if not all(r.id in active for r in supported):
        return False
    status = _status(session.scalar(select(E.verification_status).where(E.id == event_id)))
    if status == VS.MANUALLY_VERIFIED.value:
        return False
    promote_verification_status(
        session, event_id, VS.MANUALLY_VERIFIED,
        f"C2: every supported component of this event has an active term verification (attested by {name})",
        actor=name, action="STATUS_CHANGE", payload={**payload, "all_supported_components_verified": True})
    return True


def revoke_component_terms(
    session: Session, component_id: uuid.UUID, *, attested_by: str, reason: str, attestation_note: str,
) -> uuid.UUID:
    component_id = _single_component_id(component_id)
    name = validate_attester(attested_by)
    reason = (reason or "").strip()
    note = (attestation_note or "").strip()
    if not reason:
        raise VerificationRefused("a revocation must state a reason")
    if len(note) < _MIN_NOTE:
        raise VerificationRefused("attestation_note must record the human's attestation (at least 10 characters)")
    if session.scalar(select(Cmp.id).where(Cmp.id == component_id)) is None:
        raise ComponentNotFound(f"no corporate-action component with id={component_id}")
    active = active_term_verifications(session, [component_id]).get(component_id)
    if active is None:
        raise NoActiveVerification(f"component {component_id} has no active C2 verification to revoke")
    active_id, kind, a, b, tier = active
    evidence = {"schema": "corporate-action-term-revocation/1", "pit_status": "NOT_ESTABLISHED",
                "revocation": {"reason": reason, "revokes_id": active_id}}
    record_id = uuid.uuid4()
    session.execute(pg_insert(V).values(
        id=record_id, component_id=component_id, action="REVOKED", action_kind=kind, term_a=a, term_b=b, tier=tier,
        evidence=evidence, evidence_sha256=hashlib.sha256(canonical_json(evidence).encode("utf-8")).hexdigest(),
        attested_by=name, attestation_note=note, pit_status="NOT_ESTABLISHED", revokes_id=uuid.UUID(active_id),
    ))
    payload = {"c2_revocation_record_id": str(record_id), "revokes": active_id, "reason": reason, "attested_by": name}
    event_id = session.scalar(select(Cmp.event_id).where(Cmp.id == component_id))
    comp_status = _status(session.scalar(select(Cmp.verification_status).where(Cmp.id == component_id)))
    if comp_status == VS.MANUALLY_VERIFIED.value:
        promote_component_verification_status(
            session, component_id, VS.MACHINE_PARSED, f"C2 revocation {record_id}: {reason}; attested by {name}",
            actor=name, action="CORRECTION", payload=payload)
    ev_status = _status(session.scalar(select(E.verification_status).where(E.id == event_id)))
    if ev_status == VS.MANUALLY_VERIFIED.value:
        promote_verification_status(
            session, event_id, VS.UNVERIFIED, f"C2 revocation of a component of this event: {reason}; attested by {name}",
            actor=name, action="CORRECTION", payload=payload)
    return record_id


def legacy_verification_inventory(session: Session) -> list[dict]:
    rows = session.execute(
        select(Cmp.id, Cmp.event_id, Cmp.component_type, Cmp.verification_status, Cmp.parse_revision, E.ex_date,
               E.instrument_id, E.verification_status.label("event_status"))
        .join(E, E.id == Cmp.event_id)
        .where(Cmp.verification_status == VS.MANUALLY_VERIFIED)
        .order_by(E.instrument_id, E.ex_date, Cmp.component_order)
    ).all()
    current = _current_rows(session, {r.event_id for r in rows})
    active = active_term_verifications(session, [r.id for r in rows])
    out = []
    for r in rows:
        if r.id not in {c.id for c in current.get(r.event_id, [])}:
            continue
        if _ctype(r) in SUPPORTED_ADJUSTMENT_TYPES and r.id not in active:
            out.append({"component_id": str(r.id), "event_id": str(r.event_id), "ex_date": r.ex_date.isoformat(),
                        "class": _ctype(r), "instrument_id": str(r.instrument_id),
                        "state": "LEGACY_MANUALLY_VERIFIED_WITHOUT_C2_EVIDENCE"})
    return out
