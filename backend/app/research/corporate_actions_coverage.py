from __future__ import annotations

import dataclasses
import datetime

from backend.app.db.models import CorporateActionCoverageStatus


@dataclasses.dataclass(frozen=True)
class SourceCoverageObservation:

    provider: str
    record_count: int
    window_start: datetime.date
    window_end: datetime.date
    known_page_size_cap: int | None = None
    earliest_ex_date: datetime.date | None = None
    latest_ex_date: datetime.date | None = None
    instrument_listing_date: datetime.date | None = None


def assess_single_source_coverage(
    obs: SourceCoverageObservation,
) -> tuple[CorporateActionCoverageStatus, str]:
    effective_start = obs.window_start
    if obs.instrument_listing_date is not None:
        effective_start = max(obs.window_start, obs.instrument_listing_date)

    if obs.known_page_size_cap is not None and obs.record_count >= obs.known_page_size_cap:
        return (
            CorporateActionCoverageStatus.PARTIAL,
            f"{obs.provider}: record_count={obs.record_count} meets/exceeds the known "
            f"page-size cap ({obs.known_page_size_cap}) - consistent with truncation, "
            "not treated as a complete history",
        )

    if obs.record_count == 0:
        return (
            CorporateActionCoverageStatus.UNKNOWN,
            f"{obs.provider}: zero records returned - cannot distinguish 'genuinely no "
            "corporate actions occurred' from 'the source returned nothing usable for "
            "this query'",
        )

    if obs.earliest_ex_date is not None and obs.earliest_ex_date > effective_start:
        return (
            CorporateActionCoverageStatus.UNKNOWN,
            f"{obs.provider}: earliest observed action ({obs.earliest_ex_date}) is after "
            f"the effective window start ({effective_start}) with no page-cap signal - "
            "coverage of the gap before it cannot be confirmed from this source alone",
        )

    return (
        CorporateActionCoverageStatus.UNKNOWN,
        f"{obs.provider}: no truncation signal observed, but a single, "
        "un-cross-validated source is never sufficient for COMPLETE on its own",
    )


def reconcile_coverage(
    single_source_results: list[tuple[SourceCoverageObservation, CorporateActionCoverageStatus, str]],
    cross_source_agreement: bool | None,
) -> tuple[CorporateActionCoverageStatus, str]:
    statuses = [status for _, status, _ in single_source_results]

    if cross_source_agreement is False:
        return (
            CorporateActionCoverageStatus.CONFLICTED,
            "independent sources disagree on an overlapping event - never silently "
            "resolved in either source's favor",
        )

    if CorporateActionCoverageStatus.PARTIAL in statuses:
        reasons = "; ".join(r for _, s, r in single_source_results if s == CorporateActionCoverageStatus.PARTIAL)
        return (
            CorporateActionCoverageStatus.PARTIAL,
            f"at least one source shows a known truncation signal: {reasons}",
        )

    if cross_source_agreement is True and len(single_source_results) >= 2:
        providers = ", ".join(obs.provider for obs, _, _ in single_source_results)
        return (
            CorporateActionCoverageStatus.COMPLETE,
            f"independent sources ({providers}) cross-validated with agreement on "
            "overlapping events and no individual truncation signal",
        )

    return (
        CorporateActionCoverageStatus.UNKNOWN,
        "insufficient independent cross-validation to certify completeness "
        "(fewer than two sources actually compared)",
    )


import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402

DERIVATION_VERSION = "c1-derive-1"
DERIVED_BY = f"coverage-derivation/{DERIVATION_VERSION}"

REQUIRED_CHECKS = (
    "source_integrity_ok",
    "two_independent_providers",
    "window_covered",
    "freshness_ok",
    "boundary_semantics_verified",
    "partition_consistent",
    "truncation_evidence_clear",
    "reconciliation_clean",
    "revisions_resolved",
    "snapshot_diffs_clear",
    "blockers_clear",
)

CHRONOLOGY_STATEMENT = {
    "economic_adjustment_chronology": (
        "ex_date ordering: when an action changes the price basis. Used for reconciliation and windows."
    ),
    "information_chronology": (
        "NOT ESTABLISHED. NSE caBroadcastDate is null in every observed row and BSE has no such field, so "
        "nothing in this evidence claims when any action - or any revision of it - became public. "
        "fetched_at values are OUR retrieval clock only."
    ),
}


@dataclasses.dataclass
class ProviderEvidence:

    provider: str
    whole: object | None
    parts: list
    boundary: object
    partition: dict | None
    cap_observations: dict
    cap_signals: list


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def evidence_sha256(evidence: dict) -> str:
    return hashlib.sha256(canonical_json(evidence).encode("utf-8")).hexdigest()


def derive_coverage_evidence(
    *,
    symbol: str,
    instrument_id: str,
    window: tuple[datetime.date, datetime.date],
    providers: dict[str, ProviderEvidence],
    reconciliation,
    revisions: list,
    snapshot_diffs: dict[str, list],
    jump_scan: dict,
    isin_diagnostic: dict,
    diff_comparisons: list[dict] | None = None,
) -> dict:
    nse, bse = providers.get("NSE"), providers.get("BSE")
    used = [s for p in providers.values() for s in ([p.whole] if p.whole else []) + list(p.parts)]

    checks = {
        "source_integrity_ok": bool(used) and all(s.integrity_ok for s in used),
        "two_independent_providers": bool(nse and nse.whole and bse and bse.whole),
        "window_covered": bool(providers) and all(
            p.whole is not None and p.whole.query_from is not None
            and p.whole.query_from <= window[0] and p.whole.query_to >= window[1]
            for p in providers.values()
        ),
        "freshness_ok": bool(providers) and all(
            p.whole is not None and datetime.date.fromisoformat(p.whole.fetched_at[:10]) >= window[1]
            for p in providers.values()
        ),
        "boundary_semantics_verified": bool(providers) and all(p.boundary.verified for p in providers.values()),
        "partition_consistent": bool(providers) and all(
            p.partition is not None and p.partition.get("consistent") is True for p in providers.values()
        ),
        "truncation_evidence_clear": bool(providers)
        and all(p.cap_signals for p in providers.values())
        and all(
            (not sig["cap_signal"]) and sig["below_observed_max"]
            for p in providers.values() for sig in p.cap_signals
        ),
        "reconciliation_clean": bool(reconciliation.clean),
        "revisions_resolved": not any(r.blocking for r in revisions),
        "snapshot_diffs_clear": not any(c.blocking for changes in snapshot_diffs.values() for c in changes),
        "blockers_clear": jump_scan["blocking_count"] == 0 and isin_diagnostic["uncorroborated_count"] == 0,
    }

    both_reported = checks["two_independent_providers"]
    contradicts = (
        (both_reported and (reconciliation.conflict_count > 0 or reconciliation.one_sided_count > 0))
        or not checks["snapshot_diffs_clear"]
    )
    reported = [p for p in providers.values() if p.whole is not None]

    def _spans(p) -> bool:
        return p.whole.query_from is not None and p.whole.query_from <= window[0] and p.whole.query_to >= window[1]

    partial = any(
        not _spans(p)
        or not (p.cap_signals and all((not s["cap_signal"]) and s["below_observed_max"] for s in p.cap_signals))
        or (p.boundary.verified and not (p.partition is not None and p.partition.get("consistent") is True))
        for p in reported
    )
    if contradicts:
        verdict = "CONFLICTED"
    elif partial:
        verdict = "PARTIAL"
    elif all(checks[k] for k in REQUIRED_CHECKS):
        verdict = "COMPLETE"
    else:
        verdict = "UNKNOWN"

    failed = [k for k in REQUIRED_CHECKS if not checks[k]]
    summary = reconciliation.summary()
    if verdict == "COMPLETE":
        reasoning = (
            f"{symbol}: event set for {window[0]}..{window[1]} supported by two independent sources "
            f"(NSE windowed, BSE) with verified boundary semantics, consistent partitions, no truncation "
            f"signal, clean economic-event reconciliation ({summary['term_matches']} term-matched, "
            f"{summary['class_only']} class-only), resolved revisions, and no diagnostic blockers. "
            "COMPLETE does not mean every event's terms are verified - see the unresolved-event register."
        )
    else:
        detail = []
        if reconciliation.conflict_count:
            detail.append(f"{reconciliation.conflict_count} terms conflict(s)")
        if reconciliation.one_sided_count:
            detail.append(f"{reconciliation.one_sided_count} one-sided event(s)")
        if not checks["snapshot_diffs_clear"]:
            detail.append("a price-affecting action disappeared or changed terms between snapshots")
        reasoning = (
            f"{symbol}: {verdict} for {window[0]}..{window[1]}. Failed checks: {', '.join(failed) or 'none'}"
            + (f". {'; '.join(detail)}" if detail else "")
            + "."
        )

    return copy.deepcopy({
        "schema": "coverage-evidence/1",
        "derivation_version": DERIVATION_VERSION,
        "verdict": verdict,
        "instrument": {"symbol": symbol, "instrument_id": instrument_id},
        "window": {"from": window[0].isoformat(), "to": window[1].isoformat()},
        "chronology": CHRONOLOGY_STATEMENT,
        "sources": [
            {
                "provider": p.provider,
                "whole": p.whole.as_ref_dict() if p.whole else None,
                "parts": [s.as_ref_dict() for s in p.parts],
                "boundary_semantics": p.boundary.as_dict(),
                "partition": p.partition,
                "cap_observations": p.cap_observations,
                "cap_signals": p.cap_signals,
            }
            for p in providers.values()
        ],
        "reconciliation": summary,
        "revisions": {"findings": [r.as_dict() for r in revisions], "blocking_count": sum(r.blocking for r in revisions)},
        "snapshot_diffs": {
            provider: [c.as_dict() for c in changes] for provider, changes in snapshot_diffs.items()
        },
        "snapshot_diff_comparisons": diff_comparisons or [],
        "diagnostics": {"jump_scan": jump_scan, "isin_transitions": isin_diagnostic},
        "checks": checks,
        "reasoning": reasoning[:2000],
    })
