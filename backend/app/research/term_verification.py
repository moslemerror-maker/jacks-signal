from __future__ import annotations

import dataclasses
import datetime
import hashlib
import re
from decimal import Decimal
from typing import Sequence

from backend.app.research.adjustment_eligibility import SUPPORTED_ADJUSTMENT_TYPES
from backend.app.research.corporate_action_evidence import SnapshotRecord, _canonical, _row_date
from backend.app.research.corporate_action_reconciliation import (
    PRICE_AFFECTING,
    bse_components,
    detect_revision,
    is_benign_meeting_fragment,
    nse_components,
    reconcile_economic_events,
)
from backend.app.research.corporate_actions_coverage import canonical_json
from backend.app.research.exact_terms import ExactTerms, canonical_decimal, terms_equal, try_normalize

SHEET_SCHEMA = "corporate-action-term-evidence-sheet/1"
PIT_STATUS = "NOT_ESTABLISHED"
PIT_NOTE = (
    "Point-in-time knowledge is NOT ESTABLISHED. NSE caBroadcastDate is null in every observed row, BSE has no "
    "announcement field, and both sources were fetched long after the event, so this evidence shows the terms "
    "as currently published - not when they became public or what earlier versions said."
)

NOT_A_CANDIDATE = "NOT_A_CANDIDATE"
BLOCKED = "BLOCKED"
READY_TIER1 = "READY_TIER1"
READY_TIER2_NEEDS_PRIMARY = "READY_TIER2_NEEDS_PRIMARY"
READY_TIER2 = "READY_TIER2"
VERIFIABLE_RESULTS = (READY_TIER1, READY_TIER2)

STATE_NOT_VERIFIED = "NOT_VERIFIED"
STATE_C2_VERIFIED = "C2_VERIFIED"
STATE_C2_REVOKED = "C2_REVOKED"
STATE_LEGACY = "LEGACY_MANUALLY_VERIFIED_WITHOUT_C2_EVIDENCE"

MANUALLY_VERIFIED = "MANUALLY_VERIFIED"
DISPUTED = "DISPUTED"


RESERVED_ATTESTER_PATTERN = (
    r"^[[:space:]]*(system|auto|automation|automated|automatic|machine|assistant|model|llm|gpt|bot|"
    r"script|stage|unspecified|unknown|none|null|root|admin|cron|batch|tool|agent|pytest|test|tests|human)"
    r"([^a-zA-Z]|$)"
)
_RESERVED_RE = re.compile(
    RESERVED_ATTESTER_PATTERN.replace("[[:space:]]", r"\s").replace("([^a-zA-Z]|$)", r"(?![a-z])"), re.IGNORECASE
)


class AttesterRejected(ValueError):
    pass


def validate_attester(attested_by) -> str:
    if not isinstance(attested_by, str):
        raise AttesterRejected(f"attested_by must be a string, got {type(attested_by).__name__}")
    name = attested_by.strip()
    if len(name) < 3 or not re.search(r"[A-Za-z]", name):
        raise AttesterRejected("attested_by must be a named human (at least 3 characters including a letter)")
    if _RESERVED_RE.search(name):
        raise AttesterRejected(
            f"attested_by {attested_by!r} is a reserved/system/automation/model identity; "
            "only a named human may attest"
        )
    return name


_TEMPLATES = {
    ("NSE", "BONUS"): re.compile(r"^bonus (\d+):(\d+)$"),
    ("BSE", "BONUS"): re.compile(r"^bonus issue (\d+):(\d+)$"),
    ("NSE", "SPLIT"): re.compile(
        r"^face value split(?: \(sub-division\))?(?: -)? from rs\.? ?(\d+(?:\.\d+)?)(?:/-)?(?: per share)?"
        r" to r[se]\.? ?(\d+(?:\.\d+)?)(?:/-)?(?: per share)?$"
    ),
    ("BSE", "SPLIT"): re.compile(
        r"^stock split from rs\.? ?(\d+(?:\.\d+)?)(?:/-)? to r[se]\.? ?(\d+(?:\.\d+)?)(?:/-)?$"
    ),
}


def normalize_wording(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    t = re.sub(r"\s*:\s*", ":", t)
    t = re.sub(r"\s+/-", "/-", t)
    return t


def wording_is_standard(provider: str, kind: str, fragment: str) -> bool:
    pattern = _TEMPLATES.get((provider, kind))
    return bool(pattern and pattern.match(normalize_wording(fragment)))


_EXPLICIT_LEADING = frozenset({"face", "value", "fv", "split", "splt", "sub-division", "subdivision", "sub", "division", "divn", "stock"})
_SPLIT_MARKERS = frozenset({"split", "splt", "sub-division", "subdivision", "divn"})
_FROM_MARKERS = frozenset({"from", "frm"})
_CURRENCY = frozenset({"rs", "re"})
_PER_SHARE = frozenset({"per", "share", "shares"})
_EXPLICIT_CHARS = re.compile(r"^[a-z0-9 ./()\-]*$")
_TOKEN = re.compile(r"\d+(?:\.\d+)?|[a-z]+(?:-[a-z]+)*")


def explicit_split_terms(fragment: str) -> tuple[Decimal, Decimal] | None:
    text = normalize_wording(fragment)
    if not text or not _EXPLICIT_CHARS.match(text):
        return None
    tokens = _TOKEN.findall(text)
    nums = [i for i, t in enumerate(tokens) if t[0].isdigit()]
    tos = [i for i, t in enumerate(tokens) if t == "to"]
    froms = [i for i, t in enumerate(tokens) if t in _FROM_MARKERS]
    if len(nums) != 2 or len(tos) != 1 or len(froms) != 1:
        return None
    f, t, (n1, n2) = froms[0], tos[0], nums
    if not (f < n1 < t < n2):
        return None
    lead = set(tokens[:f])
    if not lead or not lead <= _EXPLICIT_LEADING or not lead & _SPLIT_MARKERS:
        return None
    if not (set(tokens[f + 1:n1]) <= _CURRENCY and set(tokens[n1 + 1:t]) <= _PER_SHARE
            and set(tokens[t + 1:n2]) <= _CURRENCY and set(tokens[n2 + 1:]) <= _PER_SHARE):
        return None
    return Decimal(tokens[n1]), Decimal(tokens[n2])


@dataclasses.dataclass(frozen=True)
class ComponentFacts:
    component_id: str
    event_id: str
    instrument_id: str
    symbol: str
    ex_date: datetime.date | None
    record_date: datetime.date | None
    component_type: str
    component_status: str
    event_status: str
    parse_revision: int
    current_parse_revision: int
    parser_version: str
    raw_fragment: str
    event_raw_subject: str
    raw_terms: tuple | None


@dataclasses.dataclass(frozen=True)
class SiblingFact:
    event_id: str
    component_id: str
    component_type: str
    component_status: str
    raw_fragment: str
    terms: tuple | None


@dataclasses.dataclass(frozen=True)
class HistoryFact:
    seq: int
    target_kind: str
    action: str
    from_status: str | None
    to_status: str | None
    actor: str
    logged_at: str
    notes: str


@dataclasses.dataclass(frozen=True)
class RecordFact:
    record_id: str
    seq: int
    action: str
    kind: str
    term_a: object
    term_b: object
    tier: int
    attested_by: str
    attested_at: str
    revokes_id: str | None


@dataclasses.dataclass(frozen=True)
class ProviderEvidence:
    provider: str
    primary: SnapshotRecord | None
    observations: tuple[SnapshotRecord, ...]


@dataclasses.dataclass(frozen=True)
class PrimarySourceDoc:
    description: str
    path: str
    sha256: str
    sha256_verified: bool
    quoted_text: str


@dataclasses.dataclass(frozen=True)
class SheetInputs:
    facts: ComponentFacts
    siblings: tuple[SiblingFact, ...]
    history: tuple[HistoryFact, ...]
    records: tuple[RecordFact, ...]
    nse: ProviderEvidence | None
    bse: ProviderEvidence | None
    primary_source: PrimarySourceDoc | None = None


@dataclasses.dataclass(frozen=True)
class EvidenceSheet:
    content: dict
    sha256: str
    canonical: str

    @property
    def readiness(self) -> str:
        return self.content["readiness"]["result"]

    @property
    def tier_required(self) -> int | None:
        return self.content["readiness"]["tier_required"]

    @property
    def blockers(self) -> list[dict]:
        return self.content["readiness"]["blockers"]

    @property
    def terms(self) -> ExactTerms | None:
        t = self.content.get("terms")
        return _terms_from_dict(t) if t else None


def _terms_from_dict(t: dict) -> ExactTerms | None:
    from decimal import Decimal

    terms, _ = try_normalize(t["kind"], Decimal(t["term_a"]), Decimal(t["term_b"]))
    return terms


def active_record(records: Sequence[RecordFact]) -> RecordFact | None:
    revoked = {r.revokes_id for r in records if r.action == "REVOKED" and r.revokes_id}
    active = [r for r in records if r.action == "VERIFIED" and r.record_id not in revoked]
    return sorted(active, key=lambda r: r.seq)[-1] if active else None


def _blocker(code: str, **detail) -> dict:
    return {"code": code, "detail": detail}


def _rows_at_date(provider: str, snap: SnapshotRecord, day: datetime.date) -> list[dict]:
    return [r for r in snap.rows if _row_date(provider, r) == day]


def _canon_rows(rows: Sequence[dict]) -> list[str]:
    return sorted(_canonical(r) for r in rows)


def _rec_dates(provider: str, rows: Sequence[dict]) -> set:
    field, fmt = ("recDate", "%d-%b-%Y") if provider == "NSE" else ("RD_Date", "%d %b %Y")
    out = set()
    for r in rows:
        v = r.get(field)
        if v in (None, "", "-"):
            continue
        try:
            out.add(datetime.datetime.strptime(v, fmt).date())
        except ValueError:
            out.add(v)
    return out


def _norm_name(x) -> str:
    return re.sub(r"[^a-z0-9]", "", str(x or "").lower())


def _provider_section(provider: str, ev: ProviderEvidence | None, facts: ComponentFacts, blockers: list) -> dict:
    if ev is None or ev.primary is None:
        blockers.append(_blocker("EVIDENCE_UNAVAILABLE", provider=provider,
                                 note="no snapshot of this provider covers the ex-date"))
        return {"provider": provider, "available": False}
    day = facts.ex_date
    prim = ev.primary
    obs_rows = [(_canon_rows(_rows_at_date(provider, s, day)) if day else []) for s in ev.observations]
    stable = all(o == obs_rows[0] for o in obs_rows) if obs_rows else False
    integrity = all(s.integrity_ok for s in ev.observations) and prim.integrity_ok
    if not integrity:
        blockers.append(_blocker("EVIDENCE_INTEGRITY_FAILED", provider=provider,
                                 failed=[s.sha256[:16] for s in ev.observations if not s.integrity_ok]))
    if not stable:
        blockers.append(_blocker("EVIDENCE_UNSTABLE", provider=provider,
                                 note="snapshots covering this ex-date disagree about its rows (an event appeared, "
                                      "disappeared or changed between fetches)"))
    if day and prim.fetched_at[:10] < day.isoformat():
        blockers.append(_blocker("EVIDENCE_PREDATES_EX_DATE", provider=provider, fetched_at=prim.fetched_at,
                                 note="the evidence was fetched before the event took effect"))
    rows = _rows_at_date(provider, prim, day) if day else []
    section = {
        "provider": provider, "available": True,
        "primary_snapshot": {
            "sha256": prim.sha256, "path": prim.path, "fetched_at": prim.fetched_at, "fetch_mode": prim.fetch_mode,
            "purpose": prim.purpose, "query_from": prim.query_from.isoformat() if prim.query_from else None,
            "query_to": prim.query_to.isoformat() if prim.query_to else None, "record_count": prim.record_count,
            "integrity_ok": prim.integrity_ok,
        },
        "observations": [
            {"sha256": s.sha256, "fetched_at": s.fetched_at, "fetch_mode": s.fetch_mode, "purpose": s.purpose,
             "rows_at_ex_date": len(_rows_at_date(provider, s, day)) if day else 0, "integrity_ok": s.integrity_ok}
            for s in sorted(ev.observations, key=lambda s: (s.fetched_at, s.sha256))
        ],
        "observations_identical_at_ex_date": stable,
        "independence_note": (
            "observations of one provider are repeated reads of the same endpoint: they show stability, "
            "not a second source"
        ),
        "rows_at_ex_date": rows,
    }
    if provider == "NSE":
        row_symbols = sorted({str(r.get("symbol")) for r in rows})
        ok = prim.symbol == facts.symbol and all(s == facts.symbol for s in row_symbols)
        section["identity"] = {
            "snapshot_symbol": prim.symbol, "row_symbols": row_symbols, "expected_symbol": facts.symbol,
            "row_isin_informational_only": sorted({str(r.get("isin")) for r in rows}),
            "row_company": sorted({str(r.get("comp")) for r in rows}), "confirmed": ok,
        }
    else:
        scrips = sorted({str(r.get("scrip_code")) for r in rows})
        short = sorted({str(r.get("short_name")) for r in rows})
        longn = sorted({str(r.get("long_name")) for r in rows})
        meta_scrip = str(prim.query_params.get("scripcode")) if prim.query_params else None
        consistent = len(scrips) <= 1 and (meta_scrip is None or all(s == meta_scrip for s in scrips))
        name_ok = all(_norm_name(s) == _norm_name(facts.symbol) for s in short)
        ok = prim.symbol == facts.symbol and consistent and name_ok
        section["identity"] = {
            "snapshot_symbol": prim.symbol, "expected_symbol": facts.symbol, "request_scrip_code": meta_scrip,
            "row_scrip_codes": scrips, "row_short_names": short, "row_long_names": longn,
            "scrip_code_consistent": consistent, "short_name_matches_symbol": name_ok, "confirmed": ok,
            "note": "BSE identity is by scrip code + short name only; BSE carries no ISIN and is never used for ISIN identity",
        }
    if not ok:
        blockers.append(_blocker("EVIDENCE_IDENTITY_MISMATCH", provider=provider, identity=section["identity"]))
    return section


def _sibling_dict(s: SiblingFact) -> dict:
    return {"event_id": s.event_id, "component_id": s.component_id, "class": s.component_type,
            "status": s.component_status, "raw_fragment": s.raw_fragment, "terms": list(s.terms) if s.terms else None}


def _primary_problem(doc: PrimarySourceDoc, terms: ExactTerms) -> str | None:
    if not doc.sha256_verified:
        return "primary-source file missing or its recomputed SHA-256 does not match the supplied hash"
    if not re.fullmatch(r"[0-9a-f]{64}", doc.sha256 or ""):
        return "primary-source sha256 is not a 64-character lowercase hex digest"
    if not (doc.description or "").strip():
        return "primary-source description is empty"
    quote = (doc.quoted_text or "").strip()
    if len(quote) < 20:
        return "primary-source quoted text is too short to establish terms"
    from decimal import Decimal, InvalidOperation

    numbers = set()
    for m in re.finditer(r"\d+(?:\.\d+)?", quote):
        try:
            numbers.add(Decimal(m.group(0)).normalize())
        except InvalidOperation:
            continue
    missing = [t for t in (terms.term_a, terms.term_b) if t.normalize() not in numbers]
    if missing:
        return f"quoted text does not contain the attested terms ({terms.a_text} and {terms.b_text})"
    return None


def build_evidence_sheet(inp: SheetInputs) -> EvidenceSheet:
    f = inp.facts
    blockers: list[dict] = []
    tier_reasons: list[dict] = []
    notes: list[dict] = []
    ctype = f.component_type

    terms: ExactTerms | None = None
    terms_error: str | None = None
    if ctype in SUPPORTED_ADJUSTMENT_TYPES:
        if f.raw_terms is None or len(f.raw_terms) != 2:
            terms_error = "terms absent or incomplete"
        else:
            terms, terms_error = try_normalize(ctype, f.raw_terms[0], f.raw_terms[1])

    active = active_record(inp.records)
    revoked_any = any(r.action == "REVOKED" for r in inp.records)
    if active is not None:
        state = STATE_C2_VERIFIED
    elif f.component_status == MANUALLY_VERIFIED or f.event_status == MANUALLY_VERIFIED:
        state = STATE_LEGACY
    elif revoked_any:
        state = STATE_C2_REVOKED
    else:
        state = STATE_NOT_VERIFIED

    not_candidate_reason: str | None = None
    if ctype in SUPPORTED_ADJUSTMENT_TYPES:
        if active is not None:
            act_terms, _ = try_normalize(active.kind, active.term_a, active.term_b)
            if terms_equal(act_terms, terms):
                not_candidate_reason = "ALREADY_VERIFIED_C2"
            else:
                blockers.append(_blocker("ACTIVE_RECORD_TERMS_MISMATCH", record_id=active.record_id))
    elif ctype == "DIVIDEND":
        not_candidate_reason = "DIVIDEND_OUT_OF_SCOPE"
    elif ctype == "OTHER" and is_benign_meeting_fragment(f.raw_fragment):
        not_candidate_reason = "BENIGN_MEETING_TEXT"
    elif ctype == "OTHER":
        blockers.append(_blocker("UNCLASSIFIED_COMPONENT", fragment=f.raw_fragment,
                                 note="non-benign OTHER text may hide a price-affecting action"))
    else:
        blockers.append(_blocker("UNSUPPORTED_ACTION_TYPE", component_type=ctype,
                                 price_affecting=ctype in PRICE_AFFECTING))

    supported = ctype in SUPPORTED_ADJUSTMENT_TYPES
    sections: dict = {}
    recon_section: dict = {}
    wording: dict = {}
    revision_markers: list[dict] = []

    if supported and not_candidate_reason is None:
        if f.ex_date is None:
            blockers.append(_blocker("MISSING_EX_DATE"))
        if f.parse_revision != f.current_parse_revision:
            blockers.append(_blocker("NOT_CURRENT_PARSE_REVISION", parse_revision=f.parse_revision,
                                     current=f.current_parse_revision,
                                     note="a verification binds to one component; a re-parse creates a new one"))
        if terms_error:
            blockers.append(_blocker("TERMS_INVALID", reason=terms_error))
        if DISPUTED in (f.component_status, f.event_status):
            blockers.append(_blocker("DISPUTED", component_status=f.component_status, event_status=f.event_status))

        twins = [s for s in inp.siblings if s.component_type == ctype and s.component_id != f.component_id]
        if twins:
            blockers.append(_blocker("AMBIGUOUS_SAME_DATE", twins=[_sibling_dict(s) for s in twins],
                                     note="another current component of the same class on this instrument and date"))

        sections["NSE"] = _provider_section("NSE", inp.nse, f, blockers)
        sections["BSE"] = _provider_section("BSE", inp.bse, f, blockers)

        if f.ex_date is not None and sections["NSE"].get("available") and sections["BSE"].get("available"):
            nse_rows = sections["NSE"]["rows_at_ex_date"]
            bse_rows = sections["BSE"]["rows_at_ex_date"]
            nse_c, nse_u = nse_components(nse_rows, "sheet-nse")
            bse_c, bse_u = bse_components(bse_rows, "sheet-bse")
            recon = reconcile_economic_events(nse_c, bse_c, (f.ex_date, f.ex_date), nse_u + bse_u)
            match = next((m for m in recon.matches if m.action_class == ctype and m.ex_date == f.ex_date), None)
            recon_section = {
                "window": [f.ex_date.isoformat(), f.ex_date.isoformat()],
                "summary": recon.summary(),
                "matches_on_date": [m.as_dict() for m in recon.matches],
            }
            if nse_u or bse_u:
                blockers.append(_blocker("UNPARSEABLE_EVIDENCE_ROW", rows=list(nse_u) + list(bse_u)))
            if any(m.status == "UNCLASSIFIED" for m in recon.matches):
                blockers.append(_blocker("SAME_DATE_UNCLASSIFIED_EVIDENCE",
                                         note="unrecognised text on the same date may hide a price-affecting action"))
            if any(d["class"] == ctype for d in recon.duplicates):
                blockers.append(_blocker("DUPLICATE_ECONOMIC_EVENT", class_=ctype))
            if any(d["class"] == ctype for d in recon.ambiguous_same_date):
                blockers.append(_blocker("AMBIGUOUS_SAME_DATE", class_=ctype, source="evidence"))
            if match is None:
                blockers.append(_blocker("NOT_IN_EVIDENCE", class_=ctype,
                                         note="neither provider lists this class on this ex-date"))
            else:
                status_code = {
                    "ONE_SIDED_NSE": "ONE_SIDED_EVENT", "ONE_SIDED_BSE": "ONE_SIDED_EVENT",
                    "CLASS_ONLY": "CLASS_ONLY_AGREEMENT", "TERMS_CONFLICT": "TERMS_CONFLICT",
                    "UNCLASSIFIED": "UNCLASSIFIED_EVIDENCE_TEXT",
                }.get(match.status)
                if status_code:
                    blockers.append(_blocker(status_code, match_status=match.status, nse_terms=_lt(match.nse),
                                             bse_terms=_lt(match.bse)))
                elif match.status == "TERMS_MATCH" and terms is not None:
                    n_terms, _ = try_normalize(ctype, *match.nse.terms)
                    b_terms, _ = try_normalize(ctype, *match.bse.terms)
                    if not (terms_equal(terms, n_terms) and terms_equal(terms, b_terms)):
                        blockers.append(_blocker("COMPONENT_TERMS_DISAGREE_WITH_EVIDENCE",
                                                 component_terms=terms.as_dict(),
                                                 nse_terms=n_terms.as_dict() if n_terms else None,
                                                 bse_terms=b_terms.as_dict() if b_terms else None))
                for prov, comp in (("NSE", match.nse), ("BSE", match.bse)):
                    if comp is None:
                        continue
                    std = wording_is_standard(prov, ctype, comp.fragment)
                    entry = {"fragment": comp.fragment, "normalized": normalize_wording(comp.fragment),
                             "standard_template": std}
                    explicit_ok = False
                    if not std:
                        entry["wording_class"] = "VARIANT"
                        if ctype == "SPLIT" and terms is not None:
                            read = explicit_split_terms(comp.fragment)
                            entry["explicit_reading"] = (
                                {"old_face_value": canonical_decimal(read[0]), "new_face_value": canonical_decimal(read[1])}
                                if read else None)
                            explicit_ok = read is not None and read[0] == terms.term_a and read[1] == terms.term_b
                            if explicit_ok:
                                entry["wording_class"] = "EXPLICIT_NON_TEMPLATE"
                                notes.append({"code": "NON_TEMPLATE_WORDING_READ_EXPLICITLY", "provider": prov,
                                              "fragment": comp.fragment, **entry["explicit_reading"],
                                              "note": "not a standard template, but it explicitly states both face "
                                                      "values and the direction; a parser/template difference only"})
                    wording[prov] = entry
                    if not std and not explicit_ok:
                        tier_reasons.append({"code": "WORDING_VARIANCE", "provider": prov, "fragment": comp.fragment})
                    if comp.revision is not None:
                        marker = {"provider": prov, "kind": comp.revision.kind, "text": comp.revision.matched_text}
                        revision_markers.append(marker)
                        tier_reasons.append({"code": "REVISION_MARKER", **marker})
                if match.status == "TERMS_MATCH" and match.nse and match.bse:
                    nse_rec = _rec_dates("NSE", nse_rows)
                    bse_rec = _rec_dates("BSE", bse_rows)
                    if nse_rec and bse_rec and nse_rec != bse_rec:
                        tier_reasons.append({"code": "PROVIDER_DISAGREEMENT", "field": "record_date",
                                             "nse": sorted(map(str, nse_rec)), "bse": sorted(map(str, bse_rec))})
            ev_rev = _detect(f.event_raw_subject)
            if ev_rev and not any(m["provider"] == "NSE" for m in revision_markers):
                marker = {"provider": "NSE", "kind": ev_rev.kind, "text": ev_rev.matched_text}
                revision_markers.append(marker)
                tier_reasons.append({"code": "REVISION_MARKER", **marker})

        if terms is not None:
            if terms.kind == "BONUS" and terms.term_a != terms.term_b:
                tier_reasons.append({"code": "DIRECTIONAL_BONUS",
                                     "note": f"{terms.a_text}:{terms.b_text} - both exchanges print a:b without saying "
                                             "which side is new; the convention needs a primary source"})
            if terms.kind == "SPLIT" and terms.term_a < terms.term_b:
                tier_reasons.append({"code": "REVERSE_SPLIT",
                                     "note": "old face value below new: a consolidation, unusual, needs a primary source"})

    primary_section = None
    tier_required: int | None = None
    if supported and not_candidate_reason is None:
        tier_required = 2 if tier_reasons else 1
    if inp.primary_source is not None:
        problem = _primary_problem(inp.primary_source, terms) if terms is not None else "terms unavailable"
        primary_section = {
            "description": inp.primary_source.description, "path": inp.primary_source.path,
            "sha256": inp.primary_source.sha256, "sha256_recomputed_and_matched": inp.primary_source.sha256_verified,
            "quoted_text": inp.primary_source.quoted_text, "valid": problem is None, "problem": problem,
        }
        if problem is not None and tier_required == 2:
            blockers.append(_blocker("PRIMARY_SOURCE_INVALID", problem=problem))

    if not_candidate_reason is not None:
        result = NOT_A_CANDIDATE
    elif blockers:
        result = BLOCKED
    elif tier_required == 1:
        result = READY_TIER1
    elif primary_section is not None and primary_section["valid"]:
        result = READY_TIER2
    else:
        result = READY_TIER2_NEEDS_PRIMARY

    readiness = {
        "result": result,
        "blockers": sorted(blockers, key=lambda b: (b["code"], canonical_json(b["detail"]))),
        "not_a_candidate_reason": not_candidate_reason,
        "tier_required": tier_required,
        "tier_reasons": sorted(tier_reasons, key=lambda r: (r["code"], canonical_json(r))),
        "verifiable": result in VERIFIABLE_RESULTS,
    }
    if notes:
        readiness["notes"] = sorted(notes, key=lambda n: (n["code"], n["provider"]))

    content = {
        "schema": SHEET_SCHEMA,
        "pit_status": PIT_STATUS,
        "pit_note": PIT_NOTE,
        "instrument": {"instrument_id": f.instrument_id, "symbol": f.symbol},
        "event": {
            "event_id": f.event_id, "ex_date": f.ex_date.isoformat() if f.ex_date else None,
            "record_date": f.record_date.isoformat() if f.record_date else None, "status": f.event_status,
            "raw_subject": f.event_raw_subject,
        },
        "component": {
            "component_id": f.component_id, "class": ctype, "status": f.component_status,
            "parse_revision": f.parse_revision, "current_parse_revision": f.current_parse_revision,
            "parser_version": f.parser_version, "raw_fragment": f.raw_fragment,
            "raw_terms_as_stored": list(f.raw_terms) if f.raw_terms is not None else None,
        },
        "terms": terms.as_dict() if terms else None,
        "terms_statement": terms.statement() if terms else None,
        "terms_error": terms_error,
        "verification_state": state,
        "active_verification": (
            {"record_id": active.record_id, "kind": active.kind, "tier": active.tier,
             "attested_by": active.attested_by, "attested_at": active.attested_at} if active else None
        ),
        "verification_records": [
            {"record_id": r.record_id, "seq": r.seq, "action": r.action, "kind": r.kind, "tier": r.tier,
             "attested_by": r.attested_by, "attested_at": r.attested_at, "revokes_id": r.revokes_id}
            for r in sorted(inp.records, key=lambda r: r.seq)
        ],
        "prior_verification_history": [
            {"seq": h.seq, "target_kind": h.target_kind, "action": h.action, "from_status": h.from_status,
             "to_status": h.to_status, "actor": h.actor, "logged_at": h.logged_at, "notes": h.notes}
            for h in sorted(inp.history, key=lambda h: h.seq)
        ],
        "same_date_siblings": [_sibling_dict(s) for s in sorted(inp.siblings, key=lambda s: (s.component_type, s.component_id))],
        "evidence": sections,
        "reconciliation": recon_section,
        "wording": wording,
        "revision_markers": revision_markers,
        "primary_source": primary_section,
        "readiness": readiness,
    }
    canonical = canonical_json(content)
    return EvidenceSheet(content=content, sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(), canonical=canonical)


def _lt(comp) -> list | None:
    return list(comp.terms) if comp is not None and comp.terms is not None else None


def _detect(text: str):
    return detect_revision(text or "")
