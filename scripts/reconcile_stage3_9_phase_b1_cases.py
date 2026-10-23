from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, timezone

from sqlalchemy import select

from backend.app.data.instruments.corporate_actions import (
    EventNotFoundError,
    current_components,
    find_event,
    promote_component_verification_status,
    record_coverage_assessment,
)
from backend.app.db.models import (
    AssetType,
    CorporateActionComponent,
    CorporateActionCoverageStatus,
    CorporateActionEvent,
    CorporateActionVerificationStatus,
    Instrument,
    InstrumentSymbol,
)
from backend.app.db.session import get_session
from backend.app.research.corporate_actions_coverage import (
    SourceCoverageObservation,
    assess_single_source_coverage,
    reconcile_coverage,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NSE_RECORD_COUNTS = {"HDFCBANK": 20, "KOTAKBANK": 20, "JSWSTEEL": 20}
BSE_RECORD_COUNTS = {"HDFCBANK": 8, "KOTAKBANK": 9, "JSWSTEEL": 5}


def _resolve_instrument_id(session, symbol: str):
    return session.scalar(
        select(InstrumentSymbol.instrument_id)
        .join(Instrument, Instrument.id == InstrumentSymbol.instrument_id)
        .where(
            InstrumentSymbol.symbol == symbol,
            InstrumentSymbol.effective_to.is_(None),
            Instrument.asset_type == AssetType.EQUITY,
        )
    )


CASES = [
    (
        "HDFCBANK", date(2011, 7, 14), "Stock  Split From Rs.10/- to Rs.2/-", 10.0, 2.0,
        "2011-07-14 (BSE ex_date) precedes Phase 1's 2015-01-01 backfill window - no "
        "direct bhavcopy confirmation possible from this project's own ingested data. "
        "Cross-checked only against NSE's OWN corporate-actions source, which shows "
        "the identical ex_date and ratio (see Stage 3.9 Phase B report) - two "
        "independent structured sources (BSE + NSE) agree exactly on date and ratio.",
    ),
    (
        "KOTAKBANK", date(2010, 9, 13), "Stock  Split From Rs.10/- to Rs.5/-", 10.0, 5.0,
        "2010-09-13 (BSE ex_date) precedes Phase 1's 2015-01-01 backfill window - no "
        "direct bhavcopy confirmation possible from this project's own ingested data. "
        "Cross-checked only against NSE's OWN corporate-actions source, which shows "
        "the identical ex_date and ratio (see Stage 3.9 Phase B report) - two "
        "independent structured sources (BSE + NSE) agree exactly on date and ratio.",
    ),
    (
        "JSWSTEEL", date(2017, 1, 4), "Stock  Split From Rs.10/- to Rs.1/-", 10.0, 1.0,
        "NSE and BSE independently agree on ex_date 2017-01-04 and on a 10->1 face-value "
        "split (NSE text 'Fv Splt Frm Rs 10 To Re 1', BSE text 'Stock Split From Rs.10/- to "
        "Rs.1/-'). The raw close falls ~10x on that ex_date bar. (This project's "
        "bhavcopy-derived InstrumentIsin transition is effective 2017-01-05, one session "
        "later - the ISIN lag relative to ex_date varies by event and is never used as an "
        "anchor.)",
    ),
]


def main() -> None:
    session = get_session()
    try:
        for symbol, ex_date, purpose, fv_from, fv_to, evidence_note in CASES:
            instrument_id = _resolve_instrument_id(session, symbol)
            if instrument_id is None:
                raise RuntimeError(f"could not resolve instrument_id for {symbol}")

            try:
                existing = find_event(session, instrument_id, ex_date)
            except EventNotFoundError:
                existing = None
            if existing is not None:
                logger.info("%s %s: event already recorded, skipping insert", symbol, ex_date)
                event_id = existing.id
            else:
                event_id = uuid.uuid4()
                session.add(
                    CorporateActionEvent(
                        id=event_id,
                        instrument_id=instrument_id,
                        source_symbol=symbol,
                        ex_date=ex_date,
                        record_date=None,
                        source_isin=None,
                        raw_subject_text=purpose,
                        raw_face_val=None,
                        source_provider="BSE (bseindia.com, unofficial JSON API)",
                        source_url="https://api.bseindia.com/BseIndiaAPI/api/DefaultData/w",
                        source_snapshot_path=f"data/raw/corporate_actions_bse/{symbol}/",
                        source_snapshot_sha256="see raw snapshot .meta.json",
                        source_fetched_at=datetime.now(timezone.utc),
                        announcement_date=None,
                        verification_status=CorporateActionVerificationStatus.UNVERIFIED,
                        created_by="stage3.9-phase-b1",
                    )
                )
                session.flush()
                session.add(
                    CorporateActionComponent(
                        id=uuid.uuid4(),
                        event_id=event_id,
                        component_order=1,
                        component_type="SPLIT",
                        raw_text_fragment=purpose,
                        face_value_from=fv_from,
                        face_value_to=fv_to,
                        verification_status=CorporateActionVerificationStatus.UNVERIFIED,
                    )
                )
                session.flush()

            from backend.app.data.instruments.corporate_actions import promote_verification_status

            promote_verification_status(
                session, event_id, CorporateActionVerificationStatus.MANUALLY_VERIFIED, evidence_note,
                actor="stage3.9-phase-b1-script",
            )
            for component in current_components(session, event_id):
                promote_component_verification_status(
                    session, component.id, CorporateActionVerificationStatus.MANUALLY_VERIFIED,
                    evidence_note, actor="stage3.9-phase-b1-script",
                )
            session.commit()
            logger.info("%s %s: event MANUALLY_VERIFIED (%s -> %s split)", symbol, ex_date, fv_from, fv_to)

            window_start = date(2009, 1, 1) if symbol != "JSWSTEEL" else date(2015, 1, 1)
            window_end = date(2015, 12, 31) if symbol != "JSWSTEEL" else date(2018, 12, 31)

            nse_obs = SourceCoverageObservation(
                provider="NSE", record_count=NSE_RECORD_COUNTS[symbol],
                known_page_size_cap=20, window_start=window_start, window_end=window_end,
            )
            bse_obs = SourceCoverageObservation(
                provider="BSE", record_count=BSE_RECORD_COUNTS[symbol],
                known_page_size_cap=None, window_start=window_start, window_end=window_end,
            )
            nse_status, nse_reason = assess_single_source_coverage(nse_obs)
            bse_status, bse_reason = assess_single_source_coverage(bse_obs)
            overall_status, overall_reason = reconcile_coverage(
                [(nse_obs, nse_status, nse_reason), (bse_obs, bse_status, bse_reason)],
                cross_source_agreement=True,
            )
            record_coverage_assessment(
                session, instrument_id, window_start, window_end,
                overall_status,
                f"{symbol}: {overall_reason}. Specific known event ({ex_date}) is "
                f"separately MANUALLY_VERIFIED regardless of this window-level status "
                f"- see the event's own verification_notes.",
                [
                    {"provider": "NSE", "record_count": nse_obs.record_count, "known_page_size_cap": 20, "status": nse_status.value},
                    {"provider": "BSE", "record_count": bse_obs.record_count, "known_page_size_cap": None, "status": bse_status.value},
                ],
            )
            session.commit()
            logger.info("%s %s window coverage: %s (%s)", symbol, [window_start, window_end], overall_status.value, overall_reason)

        n_verified = len(
            session.scalars(
                select(CorporateActionEvent).where(
                    CorporateActionEvent.verification_status
                    == CorporateActionVerificationStatus.MANUALLY_VERIFIED
                )
            ).all()
        )
        logger.info("total MANUALLY_VERIFIED events across the database: %d", n_verified)
    finally:
        session.close()


if __name__ == "__main__":
    main()
