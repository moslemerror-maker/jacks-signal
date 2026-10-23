from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

from sqlalchemy import select

from backend.app.data.instruments.corporate_actions import (
    capture_baseline,
    current_components,
    find_event,
    log_verification_change,
    plan_reparse,
    promote_component_verification_status,
    promote_verification_status,
    reparse_event_components,
)
from backend.app.db.models import (
    AssetType,
    CorporateActionComponent,
    CorporateActionCoverageAssessment,
    CorporateActionEvent,
    CorporateActionVerificationLog,
    CorporateActionVerificationStatus as Status,
    Instrument,
    InstrumentSymbol,
)
from backend.app.db.session import get_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ACTOR = "stage3.9-c0-remediation"
BAJ_BONUS = "Bonus 4:1"
BAJ_SPLIT = "Face Value Split (Sub-Division) - From Rs 2/- Per Share To Re 1/- Per Share"


def _instrument_id(session, symbol: str):
    iid = session.scalar(
        select(InstrumentSymbol.instrument_id)
        .join(Instrument, Instrument.id == InstrumentSymbol.instrument_id)
        .where(
            InstrumentSymbol.symbol == symbol,
            InstrumentSymbol.effective_to.is_(None),
            Instrument.asset_type == AssetType.EQUITY,
        )
    )
    if iid is None:
        raise RuntimeError(f"cannot resolve instrument_id for {symbol}")
    return iid


def _snapshot_ref(directory: str, symbol: str) -> str:
    files = [p for p in Path(directory, symbol).glob("*.json") if not p.name.endswith(".meta.json")]
    if len(files) != 1:
        raise RuntimeError(f"expected exactly one snapshot in {directory}/{symbol}, found {len(files)}")
    meta = json.loads(files[0].with_suffix(files[0].suffix + ".meta.json").read_text())
    return f"{files[0].name} (sha256 {meta['sha256'][:16]}..., fetched {meta['fetched_at'][:10]})"


def _already_baselined(session, target_id) -> bool:
    return session.scalar(
        select(CorporateActionVerificationLog.seq).where(
            CorporateActionVerificationLog.target_id == target_id,
            CorporateActionVerificationLog.action == "BASELINE",
        ).limit(1)
    ) is not None


def step_baseline(session, dry_run: bool) -> None:
    targets: list[tuple[str, object]] = []
    for e in session.scalars(
        select(CorporateActionEvent).where(CorporateActionEvent.verification_status == Status.MANUALLY_VERIFIED)
    ):
        targets.append(("EVENT", e.id))
        targets += [("COMPONENT", c.id) for c in current_components(session, e.id)]
    baj = _instrument_id(session, "BAJFINANCE")
    for subject in (BAJ_BONUS, BAJ_SPLIT):
        e = find_event(session, baj, date(2025, 6, 16), subject)
        targets.append(("EVENT", e.id))
        targets += [("COMPONENT", c.id) for c in current_components(session, e.id)]
    targets += [("COVERAGE_ASSESSMENT", a.id) for a in session.scalars(select(CorporateActionCoverageAssessment))]

    seen, todo = set(), []
    for kind, tid in targets:
        if (kind, tid) not in seen:
            seen.add((kind, tid))
            if not _already_baselined(session, tid):
                todo.append((kind, tid))
    logger.info("baseline: %d target(s) to capture (%d already captured)", len(todo), len(seen) - len(todo))
    if not dry_run:
        for kind, tid in todo:
            capture_baseline(session, kind, tid, actor=ACTOR, note="pre-C0 state")
        session.commit()


def step_bajfinance(session, dry_run: bool) -> None:
    baj = _instrument_id(session, "BAJFINANCE")
    bonus = find_event(session, baj, date(2025, 6, 16), BAJ_BONUS)
    split = find_event(session, baj, date(2025, 6, 16), BAJ_SPLIT)
    if bonus.verification_status != Status.MANUALLY_VERIFIED:
        logger.info("BAJFINANCE 2025-06-16 'Bonus 4:1' already not verified - nothing to revert")
    else:
        notes = (
            "C0 CORRECTION: reverted. Phase B verified this event ('Bonus 4:1') while recording "
            "notes that described the DIFFERENT same-date event (the Face Value Split 2->1): it "
            "selected 'the' event by (instrument, ex_date) and two events share 2025-06-16. Neither "
            "event has yet been verified against a second independent source; both need Phase C1 "
            "evidence. The two same-date actions are separate components that multiply "
            "(bonus 4:1 = 5x shares, split 2->1 = 2x shares). Prior status/notes preserved in the "
            "verification log."
        )
        logger.info("BAJFINANCE 2025-06-16 'Bonus 4:1': MANUALLY_VERIFIED -> UNVERIFIED (correction)")
        if not dry_run:
            promote_verification_status(session, bonus.id, Status.UNVERIFIED, notes, actor=ACTOR, action="CORRECTION")
            for c in current_components(session, bonus.id):
                if c.verification_status == Status.MANUALLY_VERIFIED:
                    promote_component_verification_status(
                        session, c.id, Status.MACHINE_PARSED, notes, actor=ACTOR, action="CORRECTION"
                    )
    already = session.scalar(
        select(CorporateActionVerificationLog.seq).where(
            CorporateActionVerificationLog.target_id == split.id,
            CorporateActionVerificationLog.action == "ANNOTATION",
        ).limit(1)
    )
    if not dry_run and already is None:
        log_verification_change(
            session, "EVENT", split.id, "ANNOTATION", actor=ACTOR,
            to_notes="This is the actual Face Value Split (2->1) event for BAJFINANCE 2025-06-16; "
            "Phase B's 'verification' of this date landed on the 'Bonus 4:1' event instead. "
            "Unverified; needs Phase C1 evidence.",
        )
    if not dry_run:
        session.commit()


def step_reparse(session, dry_run: bool) -> list[tuple[str, date, str, str]]:
    changed = []
    for e in session.scalars(select(CorporateActionEvent).order_by(CorporateActionEvent.ex_date)):
        plan = plan_reparse(session, e.id)
        if plan is None:
            continue
        before = [(c.component_type.value, c.verification_status.value) for c in current_components(session, e.id)]
        after = [(p.component_type, "MACHINE_PARSED" if p.confidently_parsed else "UNVERIFIED") for p in plan]
        changed.append((e.raw_subject_text, e.ex_date, str(before), str(after)))
        if not dry_run:
            reparse_event_components(session, e.id, actor=ACTOR)
    if not dry_run:
        session.commit()
    logger.info("reparse: %d event(s) %s", len(changed), "would change" if dry_run else "given a new parse_revision")
    return changed


def step_jswsteel(session, dry_run: bool) -> None:
    jsw = _instrument_id(session, "JSWSTEEL")
    event = find_event(session, jsw, date(2017, 1, 4))
    old_rev = session.scalars(
        select(CorporateActionComponent).where(
            CorporateActionComponent.event_id == event.id, CorporateActionComponent.parse_revision == 1
        )
    ).all()
    (current,) = current_components(session, event.id)
    if current.component_type.value != "SPLIT":
        raise RuntimeError(f"JSWSTEEL 2017-01-04 current reading is {current.component_type.value}, expected SPLIT - reparse first")
    nse = _snapshot_ref("data/raw/corporate_actions", "JSWSTEEL")
    bse = _snapshot_ref("data/raw/corporate_actions_bse", "JSWSTEEL")
    evidence = (
        "Two independent structured sources agree on ex_date 2017-01-04 and a 10->1 face-value split: "
        f"NSE snapshot {nse} text 'Fv Splt Frm Rs 10 To Re 1'; BSE snapshot {bse} text 'Stock Split "
        "From Rs.10/- to Rs.1/-'. Consistency observation (not a source): the raw close falls ~10x on "
        "the 2017-01-04 bar. Structured terms come from the NSE source text via the C0 parser "
        f"({current.parser_version}, parse_revision {current.parse_revision}). NOTE: the "
        "InstrumentIsin transition is effective 2017-01-05, one session later - ISIN lag varies by event "
        "and is never used as an anchor."
    )
    logger.info("JSWSTEEL 2017-01-04: current reading SPLIT %s->%s (%s)", current.face_value_from, current.face_value_to, current.verification_status.value)
    if dry_run:
        return
    if current.verification_status != Status.MANUALLY_VERIFIED:
        promote_component_verification_status(session, current.id, Status.MANUALLY_VERIFIED, evidence, actor=ACTOR)
    promote_verification_status(
        session, event.id, Status.MANUALLY_VERIFIED,
        "C0 CORRECTION of notes (status unchanged): " + evidence + " Replaces Phase B.1 notes that "
        "framed the split as missing from NSE and asserted a universal ISIN-lag pattern.",
        actor=ACTOR, action="CORRECTION",
    )
    for c in old_rev:
        if c.verification_status == Status.MANUALLY_VERIFIED:
            promote_component_verification_status(
                session, c.id, Status.UNVERIFIED,
                "C0: superseded by parse_revision 2 (SPLIT 10->1). This OTHER reading carried no structured "
                "terms and should never have been MANUALLY_VERIFIED.",
                actor=ACTOR, action="CORRECTION",
            )
    session.commit()


_ASSESSMENT_NOTES = {
    "window_before_bars": (
        "C0 annotation: this window lies almost entirely before the first stored bar (2015-01-01), so its "
        "coverage status gates no price adjustment of stored data. Its PARTIAL/UNKNOWN status remains a valid "
        "statement about NSE's 20-record cap, not about bar adjustability."
    ),
    "jswsteel": (
        "C0 annotation: the reasoning implied NSE lacked JSWSTEEL's 2017 split. NSE does carry it ('Fv Splt Frm "
        "Rs 10 To Re 1'); the v1 parser misclassified it as OTHER. The PARTIAL status remains conservative "
        "because NSE's fetch hit its 20-record cap, but the factual premise in the reasoning text was wrong."
    ),
}


def step_annotate_assessments(session, dry_run: bool) -> None:
    n = 0
    for a in session.scalars(select(CorporateActionCoverageAssessment)):
        already = session.scalar(
            select(CorporateActionVerificationLog.seq).where(
                CorporateActionVerificationLog.target_id == a.id,
                CorporateActionVerificationLog.action == "ANNOTATION",
            ).limit(1)
        )
        if already is not None:
            continue
        symbol = session.scalar(
            select(InstrumentSymbol.symbol).where(
                InstrumentSymbol.instrument_id == a.instrument_id, InstrumentSymbol.effective_to.is_(None)
            )
        )
        key = "jswsteel" if symbol == "JSWSTEEL" else "window_before_bars"
        n += 1
        if not dry_run:
            log_verification_change(
                session, "COVERAGE_ASSESSMENT", a.id, "ANNOTATION", actor=ACTOR,
                to_notes=_ASSESSMENT_NOTES[key], payload={"symbol": symbol, "window": [str(a.window_start), str(a.window_end)]},
            )
    if not dry_run:
        session.commit()
    logger.info("annotated %d coverage assessment(s)", n)


def report(session) -> None:
    logger.info("== final state ==")
    for e in session.scalars(
        select(CorporateActionEvent).where(CorporateActionEvent.verification_status == Status.MANUALLY_VERIFIED)
        .order_by(CorporateActionEvent.instrument_id, CorporateActionEvent.ex_date)
    ):
        sym = session.scalar(select(InstrumentSymbol.symbol).where(InstrumentSymbol.instrument_id == e.instrument_id, InstrumentSymbol.effective_to.is_(None)))
        comps = "; ".join(
            f"{c.component_type.value}(fv {c.face_value_from}->{c.face_value_to} ratio {c.ratio_new}:{c.ratio_existing} {c.verification_status.value} rev{c.parse_revision})"
            for c in current_components(session, e.id)
        )
        logger.info("  VERIFIED %-10s %s %s | %s", sym, e.ex_date, e.raw_subject_text[:38], comps)
    logger.info("  log rows: %d", len(session.scalars(select(CorporateActionVerificationLog)).all()))


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    session = get_session()
    try:
        logger.info("C0 remediation (%s)", "DRY RUN - no writes" if dry_run else "WRITING")
        step_baseline(session, dry_run)
        step_bajfinance(session, dry_run)
        changed = step_reparse(session, dry_run)
        for subject, ex, before, after in changed:
            logger.info("  reparse %s %r: %s -> %s", ex, subject[:44], before, after)
        if dry_run:
            logger.info("dry run stops before the JSWSTEEL step (it needs the reparse to have been applied)")
            return
        step_jswsteel(session, dry_run)
        step_annotate_assessments(session, dry_run)
        report(session)
    finally:
        session.close()


if __name__ == "__main__":
    main()
