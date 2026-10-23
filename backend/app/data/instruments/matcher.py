from __future__ import annotations

import datetime
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.data.canonical.schema import INDEX_SERIES, CanonicalBar
from backend.app.data.instruments.index_symbol_links import INDEX_SYMBOL_SUCCESSOR_LINKS
from backend.app.data.instruments.isin_links import all_linked_isins
from backend.app.db.models import AssetType, Instrument, InstrumentIsin, InstrumentSymbol


def get_or_create_instrument_for_bar(session: Session, bar: CanonicalBar) -> Instrument:
    if bar.isin:
        return get_or_create_equity_instrument(session, bar)
    if bar.series == INDEX_SERIES:
        return get_or_create_index_instrument(session, bar)
    raise ValueError(
        f"{bar.symbol}: cannot match/create an instrument - no ISIN and series "
        f"is not {INDEX_SERIES!r} (got {bar.series!r})"
    )


def get_or_create_equity_instrument(session: Session, bar: CanonicalBar) -> Instrument:
    if not bar.isin:
        raise ValueError(
            f"{bar.symbol}: cannot match/create an equity instrument without an ISIN"
        )

    instrument = _resolve_instrument_by_isin(session, bar.isin)
    if instrument is None:
        instrument = Instrument(exchange="NSE", asset_type=AssetType.EQUITY)
        session.add(instrument)
        session.flush()

    _ensure_period(session, InstrumentIsin, "isin", instrument.id, bar.isin, bar.trade_date)
    _ensure_period(
        session, InstrumentSymbol, "symbol", instrument.id, bar.symbol, bar.trade_date
    )
    return instrument


def _resolve_instrument_by_isin(session: Session, isin: str) -> Instrument | None:
    existing = session.scalar(select(InstrumentIsin).where(InstrumentIsin.isin == isin))
    if existing is not None:
        return existing.instrument

    for linked_isin in all_linked_isins(isin) - {isin}:
        existing = session.scalar(
            select(InstrumentIsin).where(InstrumentIsin.isin == linked_isin)
        )
        if existing is not None:
            return existing.instrument

    return None


def get_or_create_index_instrument(session: Session, bar: CanonicalBar) -> Instrument:
    instrument = _resolve_index_instrument_by_symbol(session, bar.symbol)
    if instrument is None:
        instrument = Instrument(exchange="NSE", asset_type=AssetType.INDEX)
        session.add(instrument)
        session.flush()

    _ensure_period(
        session, InstrumentSymbol, "symbol", instrument.id, bar.symbol, bar.trade_date
    )
    return instrument


def _resolve_index_instrument_by_symbol(session: Session, symbol: str) -> Instrument | None:
    instrument = (
        session.query(Instrument)
        .join(InstrumentSymbol, InstrumentSymbol.instrument_id == Instrument.id)
        .filter(
            Instrument.asset_type == AssetType.INDEX,
            InstrumentSymbol.symbol == symbol,
        )
        .first()
    )
    if instrument is not None:
        return instrument

    if symbol in INDEX_SYMBOL_SUCCESSOR_LINKS:
        predecessor_symbol, _reason = INDEX_SYMBOL_SUCCESSOR_LINKS[symbol]
        return _resolve_index_instrument_by_symbol(session, predecessor_symbol)

    return None


def _ensure_period(
    session: Session,
    period_model: type,
    value_field: str,
    instrument_id: uuid.UUID,
    value: str,
    trade_date: datetime.date,
) -> None:
    periods = list(
        session.scalars(
            select(period_model)
            .where(period_model.instrument_id == instrument_id)
            .order_by(period_model.effective_from)
        )
    )

    if not periods:
        session.add(
            period_model(
                instrument_id=instrument_id,
                effective_from=trade_date,
                effective_to=None,
                **{value_field: value},
            )
        )
        return

    earliest = periods[0]
    latest = periods[-1]
    assert latest.effective_to is None, "the most recent period must be open-ended"

    if trade_date < earliest.effective_from:
        earliest_value = getattr(earliest, value_field)
        if earliest_value == value:
            earliest.effective_from = trade_date
        else:
            session.add(
                period_model(
                    instrument_id=instrument_id,
                    effective_from=trade_date,
                    effective_to=earliest.effective_from - datetime.timedelta(days=1),
                    **{value_field: value},
                )
            )
        return

    if trade_date >= latest.effective_from:
        latest_value = getattr(latest, value_field)
        if latest_value == value:
            return
        latest.effective_to = trade_date - datetime.timedelta(days=1)
        session.add(
            period_model(
                instrument_id=instrument_id,
                effective_from=trade_date,
                effective_to=None,
                **{value_field: value},
            )
        )
        return

    for period in periods[:-1]:
        period_value = getattr(period, value_field)
        if period.effective_from <= trade_date and (
            period.effective_to is None or trade_date <= period.effective_to
        ):
            if period_value == value:
                return
            raise ValueError(
                f"instrument {instrument_id}: bar for {trade_date} falls within "
                f"an existing {value_field} period ({period_value} from "
                f"{period.effective_from} to {period.effective_to}) but has a "
                f"different {value_field} ({value}) - ambiguous, needs manual "
                "resolution"
            )

    raise ValueError(
        f"instrument {instrument_id}: bar for {trade_date} (value {value!r}) "
        f"falls in a gap between existing {value_field} periods - ambiguous, "
        "needs manual resolution"
    )
