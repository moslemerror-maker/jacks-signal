from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import select

from backend.app.data.instruments.corporate_actions import (
    current_components,
    find_event,
    normalize_snapshot,
    promote_component_verification_status,
    promote_verification_status,
)
from backend.app.db.models import (
    AssetType,
    CorporateActionComponent,
    CorporateActionEvent,
    CorporateActionVerificationStatus,
    Instrument,
    InstrumentSymbol,
)
from backend.app.db.session import get_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/corporate_actions")
SAMPLE_SYMBOLS = ["RELIANCE", "KOTAKBANK", "BAJFINANCE", "HDFCBANK", "JSWSTEEL"]


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


def _latest_snapshot(symbol: str) -> Path:
    files = [p for p in (RAW_DIR / symbol).glob("*.json") if not p.name.endswith(".meta.json")]
    if not files:
        raise RuntimeError(f"no raw snapshot found for {symbol} - run the Phase A fetch script first")
    return max(files, key=lambda p: p.stat().st_mtime)


def _promote(session, instrument_id, ex_date, raw_subject_text: str, notes: str) -> None:
    event = find_event(session, instrument_id, ex_date, raw_subject_text)
    promote_verification_status(
        session, event.id, CorporateActionVerificationStatus.MANUALLY_VERIFIED, notes,
        actor="stage3.9-phase-b-script",
    )
    for component in current_components(session, event.id):
        promote_component_verification_status(
            session, component.id, CorporateActionVerificationStatus.MANUALLY_VERIFIED, notes,
            actor="stage3.9-phase-b-script",
        )
    logger.info("promoted event %s (ex_date=%s) to MANUALLY_VERIFIED", event.id, ex_date)


def main() -> None:
    session = get_session()
    try:
        instrument_ids = {}
        for symbol in SAMPLE_SYMBOLS:
            iid = _resolve_instrument_id(session, symbol)
            if iid is None:
                raise RuntimeError(f"could not resolve instrument_id for {symbol}")
            instrument_ids[symbol] = iid
            path = _latest_snapshot(symbol)
            written = normalize_snapshot(session, path, iid)
            session.commit()
            logger.info("%s: normalized %d new events from %s", symbol, written, path.name)

        from datetime import date

        _promote(
            session, instrument_ids["KOTAKBANK"], date(2026, 1, 14),
            "Face Value Split (Sub-Division) - From Rs 5/- Per Share To Re 1/- Per Share",
            "KOTAKBANK 2026-01-14: authoritative source text 'Face Value "
            "Split (Sub-Division) - From Rs 5/- Per Share To Re 1/- Per "
            "Share' = exact 1:5 split. Confirms and CORRECTS isin_links.py's "
            "price-inferred '~1:4.7' (which must never be used as an "
            "adjustment factor). InstrumentIsin history confirms "
            "INE237A01028->INE237A01036 transition exactly at this ex_date.",
        )
        _promote(
            session, instrument_ids["BAJFINANCE"], date(2016, 9, 8),
            "Bonus 1:1/Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share",
            "BAJFINANCE 2016-09-08: compound event, source text "
            "'Bonus 1:1/Face Value Split (Sub-Division) - From Rs 10/- Per "
            "Share To Rs 2/- Per Share' = BONUS 1:1 + SPLIT 10->2, two "
            "distinct components (not flattened). InstrumentIsin history "
            "confirms INE296A01016 effective_to=2016-09-08, matching this "
            "event's source_isin exactly (the ISIN active ON the ex_date, "
            "changing the next day) - full reconciliation.",
        )
        _promote(
            session, instrument_ids["HDFCBANK"], date(2019, 9, 19),
            "Face Value Split (Sub-Division) - From Rs 2 Per Share To Rs 1 Per Share",
            "HDFCBANK 2019-09-19 (source exDate; InstrumentIsin records the "
            "transition as effective 2019-09-20, i.e. the next trading "
            "session - consistent, not a discrepancy): source text 'Face "
            "Value Split (Sub-Division) - From Rs 2 Per Share To Rs 1 Per "
            "Share' = exact 2:1 split. InstrumentIsin confirms "
            "INE040A01026->INE040A01034 transition. A second, EARLIER real "
            "split is also visible in this same raw snapshot (exDate "
            "2011-07-14, 'Face Value Split From Rs.10/- To Rs.2/-', a 5:1 "
            "split) - predates Phase 1's 2015 backfill window, plausibly "
            "explaining why this endpoint's source_isin "
            "(INE040A01018) matches NEITHER of isin_links.py's two "
            "recorded ISINs: a pre-2015 ISIN transition likely exists that "
            "isin_links.py's 2015-2026 bhavcopy scan could never have "
            "observed. NOT added to isin_links.py - inference only, not "
            "confirmed by direct pre-2015 bhavcopy evidence (none ingested).",
        )

        logger.info(
            "JSWSTEEL: NOTE - Phase B originally logged that its 2017 split was absent from "
            "NSE's window. That was wrong: NSE carries it as 'Fv Splt Frm Rs 10 To Re 1', which "
            "the v1 parser misclassified as OTHER (fixed in Phase C0)."
        )
        session.commit()

        total_events = session.scalar(select(CorporateActionEvent.id).limit(1))
        n_events = len(session.scalars(select(CorporateActionEvent)).all())
        n_components = len(session.scalars(select(CorporateActionComponent)).all())
        n_verified = len(
            session.scalars(
                select(CorporateActionEvent).where(
                    CorporateActionEvent.verification_status
                    == CorporateActionVerificationStatus.MANUALLY_VERIFIED
                )
            ).all()
        )
        logger.info(
            "totals: %d events, %d components, %d MANUALLY_VERIFIED events",
            n_events, n_components, n_verified,
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
