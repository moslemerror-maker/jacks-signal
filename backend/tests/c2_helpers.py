from __future__ import annotations

import hashlib
import uuid

from sqlalchemy.dialects.postgresql import insert as pg_insert

from backend.app.db.models import CorporateActionTermVerification as V
from backend.app.research.corporate_actions_coverage import canonical_json

HUMAN = "Dr Ada Lovelace"


def valid_evidence(component_id, kind, a, b, tier: int = 1) -> dict:
    ev = {
        "schema": "corporate-action-term-evidence-sheet/1",
        "pit_status": "NOT_ESTABLISHED",
        "component": {"component_id": str(component_id)},
        "terms": {"kind": kind, "term_a": str(a), "term_b": str(b)},
        "readiness": {
            "result": "READY_TIER1" if tier == 1 else "READY_TIER2", "verifiable": True, "blockers": [], "tier_required": tier,
        },
    }
    if tier == 2:
        ev["primary_source"] = {"valid": True, "sha256": "a" * 64}
    return ev


def sha_of(evidence: dict) -> str:
    return hashlib.sha256(canonical_json(evidence).encode("utf-8")).hexdigest()


def insert_c2_record(db, component_id, kind, a, b, *, tier=1, attested_by=HUMAN, note="reviewed the evidence sheet",
                     action="VERIFIED", revokes_id=None, evidence=None):
    if evidence is None:
        evidence = (valid_evidence(component_id, kind, a, b, tier) if action == "VERIFIED"
                    else {"schema": "corporate-action-term-revocation/1", "revocation": {"reason": "test", "revokes_id": str(revokes_id)}})
    rid = uuid.uuid4()
    db.execute(pg_insert(V).values(
        id=rid, component_id=component_id, action=action, action_kind=kind, term_a=a, term_b=b, tier=tier,
        evidence=evidence, evidence_sha256=sha_of(evidence), attested_by=attested_by, attestation_note=note,
        pit_status="NOT_ESTABLISHED", revokes_id=revokes_id,
    ))
    return rid
