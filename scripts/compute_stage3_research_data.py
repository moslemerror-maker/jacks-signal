from __future__ import annotations

import logging
import uuid

from sqlalchemy import select

from backend.app.data.instruments.universe import EQUITY_BASKET
from backend.app.db.models import AssetType, Instrument, InstrumentSymbol
from backend.app.db.session import get_session
from backend.app.research.features.compute import compute_and_store_features
from backend.app.research.features.definitions import register_feature_definitions
from backend.app.research.labels.forward_return import (
    compute_and_store_labels,
    register_label_definition,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _resolve_basket_instrument_ids(session) -> dict[str, uuid.UUID]:
    rows = session.execute(
        select(InstrumentSymbol.symbol, InstrumentSymbol.instrument_id)
        .join(Instrument, Instrument.id == InstrumentSymbol.instrument_id)
        .where(
            InstrumentSymbol.symbol.in_(EQUITY_BASKET),
            InstrumentSymbol.effective_to.is_(None),
            Instrument.asset_type == AssetType.EQUITY,
        )
    ).all()
    result = {symbol: instrument_id for symbol, instrument_id in rows}
    missing = set(EQUITY_BASKET) - result.keys()
    if missing:
        raise RuntimeError(
            f"could not resolve current instrument for basket symbols: {sorted(missing)}"
        )
    return result


def main() -> None:
    session = get_session()
    try:
        feature_definitions = register_feature_definitions(session)
        label_definition = register_label_definition(session)
        session.commit()

        instrument_ids = _resolve_basket_instrument_ids(session)
        totals = {
            "return_5d": 0,
            "close_vs_sma20": 0,
            "atr_14": 0,
            "forward_return_5d": 0,
        }
        for symbol, instrument_id in sorted(instrument_ids.items()):
            written = compute_and_store_features(session, instrument_id, feature_definitions)
            label_written = compute_and_store_labels(session, instrument_id, label_definition)
            for name, count in written.items():
                totals[name] += count
            totals["forward_return_5d"] += label_written
            session.commit()
            logger.info(
                "%s: return_5d=+%d close_vs_sma20=+%d atr_14=+%d forward_return_5d=+%d",
                symbol,
                written["return_5d"],
                written["close_vs_sma20"],
                written["atr_14"],
                label_written,
            )
        logger.info("done. new rows written across basket: %s", totals)
    finally:
        session.close()


if __name__ == "__main__":
    main()
