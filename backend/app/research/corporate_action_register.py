from __future__ import annotations

import datetime
from typing import Mapping, Sequence

from backend.app.research.adjustment_eligibility import SUPPORTED_ADJUSTMENT_TYPES
from backend.app.research.corporate_action_reconciliation import (
    PRICE_AFFECTING,
    ReconciliationResult,
    RevisionFinding,
    SnapshotChange,
)

UNSUPPORTED_FOR_ADJUSTMENT = PRICE_AFFECTING - SUPPORTED_ADJUSTMENT_TYPES
TERMS_BEARING = frozenset({"SPLIT", "BONUS"})


def _terms_equal(a: Sequence | None, b: Sequence | None) -> bool:
    if a is None or b is None or len(a) != len(b):
        return False
    return all(abs(float(x) - float(y)) < 1e-9 for x, y in zip(a, b))


def build_unresolved_register(
    reconciliation: ReconciliationResult,
    revisions: Sequence[RevisionFinding],
    snapshot_diffs: Mapping[str, Sequence[SnapshotChange]],
    jump_scan: dict,
    isin_diagnostic: dict,
    nse_verification: Mapping[tuple[datetime.date, str], Sequence[dict]],
) -> list[dict]:
    items: list[dict] = []

    def add(ex_date, cls, code, severity, detail, event_id=None):
        items.append({
            "ex_date": ex_date, "action_class": cls, "reason_code": code, "severity": severity,
            "detail": detail, "event_id": event_id,
        })

    for m in reconciliation.matches:
        base = m.as_dict()
        if m.status == "TERMS_CONFLICT":
            add(m.ex_date, m.action_class, "TERMS_CONFLICT", "BLOCKS_COMPLETE", base)
            continue
        if m.status in ("ONE_SIDED_NSE", "ONE_SIDED_BSE"):
            add(m.ex_date, m.action_class, m.status, "BLOCKS_COMPLETE", base)
            continue
        if m.status == "UNCLASSIFIED":
            add(m.ex_date, "UNCLASSIFIED", "UNCLASSIFIED_TEXT", "BLOCKS_COMPLETE", base)
            continue

        if m.action_class in UNSUPPORTED_FOR_ADJUSTMENT:
            add(m.ex_date, m.action_class, "UNSUPPORTED_ACTION_TYPE", "BLOCKS_ADJUSTMENT",
                {**base, "note": "price-affecting type the adjustment engine does not support; "
                 f"agreement between sources is {m.status} (class-level only)"})
            continue

        if m.status == "CLASS_ONLY":
            if m.action_class in TERMS_BEARING:
                add(m.ex_date, m.action_class, "CLASS_ONLY_TERMS_UNVERIFIED", "BLOCKS_ADJUSTMENT",
                    {**base, "note": "sources agree the event exists but one states no terms: "
                     "class-only agreement never verifies economic terms"})
            else:
                add(m.ex_date, m.action_class, "CLASS_ONLY_AGREEMENT", "INFO", base)
            continue

        if m.status == "TERMS_MATCH" and m.action_class in TERMS_BEARING:
            candidates = nse_verification.get((m.ex_date, m.action_class), [])
            verified = [c for c in candidates
                        if c["status"] == "MANUALLY_VERIFIED" and _terms_equal(c["terms"], m.nse.terms)
                        and c.get("c2_active") is True]
            if verified:
                continue
            if candidates:
                legacy = any(c["status"] == "MANUALLY_VERIFIED" and _terms_equal(c["terms"], m.nse.terms)
                             and c.get("c2_active") is not True for c in candidates)
                add(m.ex_date, m.action_class,
                    "LEGACY_VERIFICATION_WITHOUT_C2_EVIDENCE" if legacy else "TERMS_NOT_MANUALLY_VERIFIED",
                    "BLOCKS_ADJUSTMENT",
                    {**base, "events_table": [{"event_id": str(c["event_id"]), "status": c["status"],
                                               "terms": c["terms"], "c2_active": bool(c.get("c2_active"))}
                                              for c in candidates]},
                    event_id=candidates[0]["event_id"])
            else:
                add(m.ex_date, m.action_class, "NOT_IN_EVENTS_TABLE", "BLOCKS_ADJUSTMENT", base)

    for d in reconciliation.duplicates:
        add(datetime.date.fromisoformat(d["ex_date"]), d["class"], "DUPLICATE_ECONOMIC_EVENT",
            "BLOCKS_COMPLETE" if d["price_affecting"] else "INFO", d)
    for d in reconciliation.ambiguous_same_date:
        add(datetime.date.fromisoformat(d["ex_date"]), d["class"], "AMBIGUOUS_SAME_DATE", "BLOCKS_COMPLETE", d)
    for row in reconciliation.unparseable_rows:
        add(None, "UNCLASSIFIED", "UNPARSEABLE_ROW", "BLOCKS_COMPLETE", row)

    for r in revisions:
        add(r.ex_date, "REVISION", r.code, "BLOCKS_COMPLETE" if r.blocking else "INFO", r.as_dict())

    for provider, changes in snapshot_diffs.items():
        for c in changes:
            severity = "BLOCKS_COMPLETE" if c.blocking else "INFO"
            add(c.ex_date, c.action_class, f"SNAPSHOT_{c.kind}", severity, {**c.as_dict(), "provider": provider})

    for j in jump_scan["unexplained_blockers"]:
        add(datetime.date.fromisoformat(j["bar_date"]), "DIAGNOSTIC", "UNEXPLAINED_JUMP", "BLOCKS_COMPLETE", j)
    for j in jump_scan["watch_unexplained"]:
        add(datetime.date.fromisoformat(j["bar_date"]), "DIAGNOSTIC", "WATCH_JUMP", "INFO", j)
    for t in isin_diagnostic["transitions"]:
        if not t["corroborated"]:
            add(datetime.date.fromisoformat(t["transition_date"]), "DIAGNOSTIC", "ISIN_TRANSITION_UNCORROBORATED",
                "BLOCKS_COMPLETE", t)
    return items
