from __future__ import annotations

import dataclasses
import datetime
import re
from collections import defaultdict
from typing import Sequence

from backend.app.data.normalizers.bse_corporate_actions import parse_bse_purpose
from backend.app.data.normalizers.corporate_actions import parse_subject_components

PRICE_AFFECTING = frozenset({"SPLIT", "BONUS", "RIGHTS", "DEMERGER", "MERGER"})

_BENIGN_MEETING = re.compile(
    r"^(?:annual|extra[\s-]*ordinary)\s+general\s+meeting$|^(?:agm|egm)$", re.IGNORECASE
)


def is_benign_meeting_fragment(fragment: str) -> bool:
    return bool(_BENIGN_MEETING.match(fragment.strip()))


_RECOGNIZED_REVISION_MARKER = re.compile(r"\(\s*purpose\s+revised\s*\)", re.IGNORECASE)
_REVISION_LANGUAGE = re.compile(
    r"\b(?:revis\w*|correct\w*|cancel\w*|withdr\w*|postpon\w*|resched\w*|amend\w*|modif\w*|supersed\w*)\b",
    re.IGNORECASE,
)

NO_PIT_CLAIM = (
    "No announcement/broadcast timing is available from either source (NSE caBroadcastDate is null in "
    "every observed row; BSE has no such field), so this makes NO claim about when the revision "
    "occurred or what was knowable before it."
)


@dataclasses.dataclass(frozen=True)
class RevisionSignal:
    kind: str
    matched_text: str


def detect_revision(text: str) -> RevisionSignal | None:
    other = _REVISION_LANGUAGE.search(_RECOGNIZED_REVISION_MARKER.sub(" ", text))
    if other:
        return RevisionSignal("UNRECOGNIZED_REVISION_LANGUAGE", other.group(0))
    marker = _RECOGNIZED_REVISION_MARKER.search(text)
    if marker:
        return RevisionSignal("PURPOSE_REVISED_MARKER", marker.group(0))
    return None


@dataclasses.dataclass(frozen=True)
class EconomicComponent:
    ex_date: datetime.date
    action_class: str
    terms: tuple | None
    source: str
    fragment: str
    row_ref: str
    revision: RevisionSignal | None = None

    @property
    def price_affecting(self) -> bool:
        return self.action_class in PRICE_AFFECTING

    @property
    def key(self) -> tuple:
        return (self.ex_date, self.action_class, self.terms)


def _nse_date(value) -> datetime.date | None:
    if not value or value == "-":
        return None
    try:
        return datetime.datetime.strptime(value, "%d-%b-%Y").date()
    except ValueError:
        return None


def _bse_date(value) -> datetime.date | None:
    if not value:
        return None
    try:
        return datetime.datetime.strptime(value, "%d %b %Y").date()
    except ValueError:
        return None


def nse_components(
    rows: Sequence[dict], snapshot_ref: str
) -> tuple[list[EconomicComponent], list[dict]]:
    components: list[EconomicComponent] = []
    unparseable: list[dict] = []
    for i, row in enumerate(rows):
        ref = f"{snapshot_ref}#{i}"
        ex_date = _nse_date(row.get("exDate"))
        subject = (row.get("subject") or "").strip()
        if ex_date is None or not subject:
            unparseable.append({"row_ref": ref, "exDate": row.get("exDate"), "subject": row.get("subject")})
            continue
        revision = detect_revision(subject)
        for c in parse_subject_components(subject):
            cls = c.component_type
            if cls == "OTHER":
                if is_benign_meeting_fragment(c.raw_text_fragment):
                    continue
                components.append(EconomicComponent(ex_date, "UNCLASSIFIED", None, "NSE", c.raw_text_fragment, ref, revision))
                continue
            if cls == "SPLIT":
                terms = (c.face_value_from, c.face_value_to) if c.face_value_from and c.face_value_to else None
            elif cls in ("BONUS", "RIGHTS"):
                terms = (c.ratio_new, c.ratio_existing) if c.ratio_new and c.ratio_existing else None
            elif cls == "DIVIDEND":
                terms = (round(c.cash_amount_per_share, 4),) if c.cash_amount_per_share else None
            else:
                terms = None
            components.append(EconomicComponent(ex_date, cls, terms, "NSE", c.raw_text_fragment, ref, revision))
    return components, unparseable


def bse_components(
    rows: Sequence[dict], snapshot_ref: str
) -> tuple[list[EconomicComponent], list[dict]]:
    components: list[EconomicComponent] = []
    unparseable: list[dict] = []
    for i, row in enumerate(rows):
        ref = f"{snapshot_ref}#{i}"
        ex_date = _bse_date(row.get("Ex_date"))
        purpose = row.get("Purpose")
        if ex_date is None or purpose is None:
            unparseable.append({"row_ref": ref, "Ex_date": row.get("Ex_date"), "Purpose": purpose})
            continue
        revision = detect_revision(purpose)
        for c in parse_bse_purpose(purpose):
            components.append(EconomicComponent(ex_date, c.action_class, c.terms, "BSE", c.fragment, ref, revision))
    return components, unparseable


@dataclasses.dataclass(frozen=True)
class EventMatch:
    ex_date: datetime.date
    action_class: str
    status: str
    nse: EconomicComponent | None
    bse: EconomicComponent | None

    def as_dict(self) -> dict:
        return {
            "ex_date": self.ex_date.isoformat(),
            "class": self.action_class,
            "status": self.status,
            "nse_terms": list(self.nse.terms) if self.nse and self.nse.terms else None,
            "bse_terms": list(self.bse.terms) if self.bse and self.bse.terms else None,
            "nse_text": self.nse.fragment if self.nse else None,
            "bse_text": self.bse.fragment.strip() if self.bse else None,
            "nse_ref": self.nse.row_ref if self.nse else None,
            "bse_ref": self.bse.row_ref if self.bse else None,
        }


@dataclasses.dataclass
class ReconciliationResult:
    matches: list[EventMatch]
    duplicates: list[dict]
    ambiguous_same_date: list[dict]
    unparseable_rows: list[dict]

    def count(self, *statuses: str) -> int:
        return sum(1 for m in self.matches if m.status in statuses)

    @property
    def conflict_count(self) -> int:
        return self.count("TERMS_CONFLICT")

    @property
    def one_sided_count(self) -> int:
        return self.count("ONE_SIDED_NSE", "ONE_SIDED_BSE")

    @property
    def unclassified_count(self) -> int:
        return self.count("UNCLASSIFIED")

    @property
    def duplicate_price_affecting_count(self) -> int:
        return sum(1 for d in self.duplicates if d["price_affecting"])

    @property
    def clean(self) -> bool:
        return not (
            self.conflict_count or self.one_sided_count or self.unclassified_count
            or self.duplicate_price_affecting_count or self.ambiguous_same_date or self.unparseable_rows
        )

    def summary(self) -> dict:
        return {
            "term_matches": self.count("TERMS_MATCH"),
            "class_only": self.count("CLASS_ONLY"),
            "conflict_count": self.conflict_count,
            "one_sided_count": self.one_sided_count,
            "unclassified_count": self.unclassified_count,
            "duplicate_price_affecting_count": self.duplicate_price_affecting_count,
            "duplicate_non_price_affecting_count": sum(1 for d in self.duplicates if not d["price_affecting"]),
            "ambiguous_same_date_count": len(self.ambiguous_same_date),
            "unparseable_row_count": len(self.unparseable_rows),
            "matches": [m.as_dict() for m in self.matches],
            "duplicates": self.duplicates,
            "ambiguous_same_date": self.ambiguous_same_date,
            "unparseable_rows": self.unparseable_rows,
        }


def _in_window(components, window):
    return [c for c in components if window[0] <= c.ex_date <= window[1]]


def _dedupe(components: list[EconomicComponent]) -> tuple[list[EconomicComponent], list[dict]]:
    groups: dict[tuple, list[EconomicComponent]] = defaultdict(list)
    for c in components:
        groups[c.key].append(c)
    unique, duplicates = [], []
    for key in sorted(groups, key=lambda k: (k[0], k[1], repr(k[2]))):
        group = sorted(groups[key], key=lambda c: c.row_ref)
        unique.append(group[0])
        if len(group) > 1:
            duplicates.append({
                "ex_date": key[0].isoformat(), "class": key[1], "terms": list(key[2]) if key[2] else None,
                "source": group[0].source, "count": len(group), "row_refs": [c.row_ref for c in group],
                "price_affecting": group[0].price_affecting,
            })
    return unique, duplicates


def _ambiguous_same_date(components: list[EconomicComponent]) -> list[dict]:
    by: dict[tuple, list[EconomicComponent]] = defaultdict(list)
    for c in components:
        if c.price_affecting:
            by[(c.ex_date, c.action_class)].append(c)
    return [
        {"ex_date": d.isoformat(), "class": cls, "source": group[0].source,
         "terms": [list(c.terms) if c.terms else None for c in group], "row_refs": [c.row_ref for c in group]}
        for (d, cls), group in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1]))
        if len(group) > 1
    ]


def reconcile_economic_events(
    nse: Sequence[EconomicComponent],
    bse: Sequence[EconomicComponent],
    window: tuple[datetime.date, datetime.date],
    unparseable_rows: Sequence[dict] = (),
) -> ReconciliationResult:
    nse_unique, nse_dups = _dedupe(_in_window(nse, window))
    bse_unique, bse_dups = _dedupe(_in_window(bse, window))
    ambiguous = _ambiguous_same_date(nse_unique) + _ambiguous_same_date(bse_unique)

    matches: list[EventMatch] = []
    dates = sorted({c.ex_date for c in nse_unique} | {c.ex_date for c in bse_unique})
    for d in dates:
        n = [c for c in nse_unique if c.ex_date == d]
        b = [c for c in bse_unique if c.ex_date == d]

        for c in [x for x in n if x.action_class == "UNCLASSIFIED"]:
            matches.append(EventMatch(d, "UNCLASSIFIED", "UNCLASSIFIED", c, None))
        for c in [x for x in b if x.action_class == "UNCLASSIFIED"]:
            matches.append(EventMatch(d, "UNCLASSIFIED", "UNCLASSIFIED", None, c))
        n = [x for x in n if x.action_class != "UNCLASSIFIED"]
        b = [x for x in b if x.action_class != "UNCLASSIFIED"]

        def take(pred):
            for ni in list(n):
                for bj in list(b):
                    if ni.action_class == bj.action_class and pred(ni, bj):
                        n.remove(ni)
                        b.remove(bj)
                        return ni, bj
            return None

        while (pair := take(lambda x, y: x.terms is not None and x.terms == y.terms)):
            matches.append(EventMatch(d, pair[0].action_class, "TERMS_MATCH", *pair))
        while (pair := take(lambda x, y: x.terms is None or y.terms is None)):
            matches.append(EventMatch(d, pair[0].action_class, "CLASS_ONLY", *pair))
        while (pair := take(lambda x, y: True)):
            matches.append(EventMatch(d, pair[0].action_class, "TERMS_CONFLICT", *pair))
        matches.extend(EventMatch(d, c.action_class, "ONE_SIDED_NSE", c, None) for c in n)
        matches.extend(EventMatch(d, c.action_class, "ONE_SIDED_BSE", None, c) for c in b)

    return ReconciliationResult(matches, nse_dups + bse_dups, ambiguous, list(unparseable_rows))


@dataclasses.dataclass(frozen=True)
class RevisionFinding:
    ex_date: datetime.date
    row_ref: str
    code: str
    blocking: bool
    fragment: str
    note: str

    def as_dict(self) -> dict:
        return {
            "ex_date": self.ex_date.isoformat(), "row_ref": self.row_ref, "code": self.code,
            "blocking": self.blocking, "fragment": self.fragment, "note": self.note,
        }


def classify_revisions(
    nse: Sequence[EconomicComponent], result: ReconciliationResult
) -> list[RevisionFinding]:
    matched_terms = {m.nse.row_ref + repr(m.nse.key) for m in result.matches if m.status == "TERMS_MATCH" and m.nse}
    by_row: dict[str, list[EconomicComponent]] = defaultdict(list)
    for c in nse:
        if c.revision is not None:
            by_row[c.row_ref].append(c)
    findings: list[RevisionFinding] = []
    for row_ref in sorted(by_row):
        comps = by_row[row_ref]
        signal = comps[0].revision
        price = [c for c in comps if c.price_affecting]
        text = " | ".join(c.fragment for c in comps)
        if signal.kind == "UNRECOGNIZED_REVISION_LANGUAGE":
            code, blocking = (
                ("REVISION_UNRECOGNIZED_PRICE_AFFECTING", True) if price
                else ("REVISION_UNRECOGNIZED_NON_PRICE_AFFECTING", False)
            )
        elif not price:
            code, blocking = "REVISED_NON_PRICE_AFFECTING", False
        elif all(c.row_ref + repr(c.key) in matched_terms for c in price):
            code, blocking = "REVISED_PRICE_AFFECTING_CORROBORATED", False
        else:
            code, blocking = "REVISED_PRICE_AFFECTING_UNCORROBORATED", True
        findings.append(RevisionFinding(comps[0].ex_date, row_ref, code, blocking, text,
                                        f"matched {signal.matched_text!r}. " + NO_PIT_CLAIM))
    return findings


@dataclasses.dataclass(frozen=True)
class SnapshotChange:
    ex_date: datetime.date
    action_class: str
    kind: str
    price_affecting: bool
    older_terms: tuple | None
    newer_terms: tuple | None
    older_text: str | None
    newer_text: str | None

    @property
    def blocking(self) -> bool:
        return self.price_affecting and self.kind in ("DISAPPEARED", "TERMS_CHANGED")

    def as_dict(self) -> dict:
        return {
            "ex_date": self.ex_date.isoformat(), "class": self.action_class, "kind": self.kind,
            "price_affecting": self.price_affecting, "blocking": self.blocking,
            "older_terms": list(self.older_terms) if self.older_terms else None,
            "newer_terms": list(self.newer_terms) if self.newer_terms else None,
            "older_text": self.older_text, "newer_text": self.newer_text,
        }


def diff_snapshots(
    older: Sequence[EconomicComponent],
    newer: Sequence[EconomicComponent],
    window: tuple[datetime.date, datetime.date],
) -> list[SnapshotChange]:
    o, _ = _dedupe(_in_window(older, window))
    n, _ = _dedupe(_in_window(newer, window))
    changes: list[SnapshotChange] = []
    for d in sorted({c.ex_date for c in o} | {c.ex_date for c in n}):
        od = [c for c in o if c.ex_date == d]
        nd = [c for c in n if c.ex_date == d]
        for oc in list(od):
            for nc in list(nd):
                if oc.action_class == nc.action_class and oc.terms == nc.terms:
                    od.remove(oc)
                    nd.remove(nc)
                    if oc.fragment.strip() != nc.fragment.strip():
                        changes.append(SnapshotChange(d, oc.action_class, "TEXT_REVISED", oc.price_affecting,
                                                      oc.terms, nc.terms, oc.fragment, nc.fragment))
                    break
        for oc in list(od):
            for nc in list(nd):
                if oc.action_class == nc.action_class:
                    od.remove(oc)
                    nd.remove(nc)
                    changes.append(SnapshotChange(d, oc.action_class, "TERMS_CHANGED", oc.price_affecting,
                                                  oc.terms, nc.terms, oc.fragment, nc.fragment))
                    break
        changes += [SnapshotChange(d, c.action_class, "DISAPPEARED", c.price_affecting, c.terms, None, c.fragment, None) for c in od]
        changes += [SnapshotChange(d, c.action_class, "APPEARED", c.price_affecting, None, c.terms, None, c.fragment) for c in nd]
    return changes
