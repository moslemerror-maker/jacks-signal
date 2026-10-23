from datetime import date

import pytest

from backend.app.data.canonical.schema import INDEX_SERIES, CanonicalBar, SourceFormat
from backend.app.data.instruments.index_symbol_links import INDEX_SYMBOL_SUCCESSOR_LINKS
from backend.app.data.instruments.isin_links import ISIN_SUCCESSOR_LINKS
from backend.app.data.instruments.matcher import (
    get_or_create_equity_instrument,
    get_or_create_index_instrument,
    get_or_create_instrument_for_bar,
)
from backend.app.db.models import AssetType, Instrument, InstrumentIsin, InstrumentSymbol


def _current_isin(session, instrument) -> str | None:
    row = (
        session.query(InstrumentIsin)
        .filter_by(instrument_id=instrument.id, effective_to=None)
        .one_or_none()
    )
    return row.isin if row else None


def _bar(**overrides) -> CanonicalBar:
    fields = dict(
        symbol="RELIANCE",
        series="EQ",
        isin="INE002A01018",
        trade_date=date(2024, 7, 5),
        open=100.0,
        high=110.0,
        low=95.0,
        close=105.0,
        source_format=SourceFormat.NSE_LEGACY_CSV,
        source_file="test.zip",
    )
    fields.update(overrides)
    return CanonicalBar(**fields)


def _index_bar(**overrides) -> CanonicalBar:
    fields = dict(
        symbol="NIFTY 50",
        series=INDEX_SERIES,
        isin=None,
        trade_date=date(2024, 7, 5),
        open=24000.0,
        high=24100.0,
        low=23950.0,
        close=24050.0,
        source_format=SourceFormat.NSE_INDEX_HISTORY,
        source_file="test.json",
    )
    fields.update(overrides)
    return CanonicalBar(**fields)


class TestEquityMatching:
    def test_creates_instrument_and_symbol_period_on_first_bar(self, db_session):
        instrument = get_or_create_equity_instrument(db_session, _bar())

        assert _current_isin(db_session, instrument) == "INE002A01018"
        assert instrument.exchange == "NSE"
        assert instrument.asset_type is AssetType.EQUITY

        periods = (
            db_session.query(InstrumentSymbol).filter_by(instrument_id=instrument.id).all()
        )
        assert len(periods) == 1
        assert periods[0].symbol == "RELIANCE"
        assert periods[0].effective_from == date(2024, 7, 5)
        assert periods[0].effective_to is None

    def test_same_isin_returns_same_instrument(self, db_session):
        first = get_or_create_equity_instrument(db_session, _bar(trade_date=date(2024, 7, 5)))
        second = get_or_create_equity_instrument(
            db_session, _bar(trade_date=date(2024, 7, 8))
        )

        assert first.id == second.id
        assert db_session.query(InstrumentIsin).filter_by(isin="INE002A01018").count() == 1

    def test_symbol_rename_opens_new_period_and_closes_old(self, db_session):
        get_or_create_equity_instrument(
            db_session, _bar(symbol="OLDNAME", trade_date=date(2024, 1, 1))
        )
        instrument = get_or_create_equity_instrument(
            db_session, _bar(symbol="NEWNAME", trade_date=date(2024, 6, 1))
        )

        periods = (
            db_session.query(InstrumentSymbol)
            .filter_by(instrument_id=instrument.id)
            .order_by(InstrumentSymbol.effective_from)
            .all()
        )
        assert len(periods) == 2
        assert periods[0].symbol == "OLDNAME"
        assert periods[0].effective_from == date(2024, 1, 1)
        assert periods[0].effective_to == date(2024, 5, 31)
        assert periods[1].symbol == "NEWNAME"
        assert periods[1].effective_from == date(2024, 6, 1)
        assert periods[1].effective_to is None

    def test_bar_predating_earliest_period_backfills_an_earlier_period(self, db_session):
        instrument = get_or_create_equity_instrument(
            db_session, _bar(symbol="NEWNAME", trade_date=date(2024, 6, 1))
        )
        get_or_create_equity_instrument(
            db_session, _bar(symbol="OLDNAME", trade_date=date(2024, 1, 1))
        )

        periods = (
            db_session.query(InstrumentSymbol)
            .filter_by(instrument_id=instrument.id)
            .order_by(InstrumentSymbol.effective_from)
            .all()
        )
        assert len(periods) == 2
        assert periods[0].symbol == "OLDNAME"
        assert periods[0].effective_from == date(2024, 1, 1)
        assert periods[0].effective_to == date(2024, 5, 31)
        assert periods[1].symbol == "NEWNAME"
        assert periods[1].effective_from == date(2024, 6, 1)
        assert periods[1].effective_to is None

    def test_backfilling_earlier_period_extends_backward_across_multiple_bars(
        self, db_session
    ):
        instrument = get_or_create_equity_instrument(
            db_session, _bar(symbol="NEWNAME", trade_date=date(2024, 6, 1))
        )
        get_or_create_equity_instrument(
            db_session, _bar(symbol="OLDNAME", trade_date=date(2024, 3, 1))
        )
        get_or_create_equity_instrument(
            db_session, _bar(symbol="OLDNAME", trade_date=date(2024, 1, 1))
        )

        periods = (
            db_session.query(InstrumentSymbol)
            .filter_by(instrument_id=instrument.id)
            .order_by(InstrumentSymbol.effective_from)
            .all()
        )
        assert len(periods) == 2
        assert periods[0].symbol == "OLDNAME"
        assert periods[0].effective_from == date(2024, 1, 1)
        assert periods[0].effective_to == date(2024, 5, 31)

    def test_bar_matching_earliest_period_value_just_extends_it(self, db_session):
        instrument = get_or_create_equity_instrument(
            db_session, _bar(symbol="RELIANCE", trade_date=date(2024, 6, 1))
        )
        get_or_create_equity_instrument(
            db_session, _bar(symbol="RELIANCE", trade_date=date(2024, 1, 1))
        )

        periods = (
            db_session.query(InstrumentSymbol).filter_by(instrument_id=instrument.id).all()
        )
        assert len(periods) == 1
        assert periods[0].effective_from == date(2024, 1, 1)

    def test_reprocessing_bar_within_existing_closed_period_is_a_no_op(self, db_session):
        instrument = get_or_create_equity_instrument(
            db_session, _bar(symbol="OLDNAME", trade_date=date(2024, 1, 1))
        )
        get_or_create_equity_instrument(
            db_session, _bar(symbol="NEWNAME", trade_date=date(2024, 6, 1))
        )
        get_or_create_equity_instrument(
            db_session, _bar(symbol="OLDNAME", trade_date=date(2024, 3, 15))
        )

        periods = (
            db_session.query(InstrumentSymbol)
            .filter_by(instrument_id=instrument.id)
            .order_by(InstrumentSymbol.effective_from)
            .all()
        )
        assert len(periods) == 2

    def test_bar_conflicting_with_existing_closed_period_still_raises(self, db_session):
        get_or_create_equity_instrument(
            db_session, _bar(symbol="OLDNAME", trade_date=date(2024, 1, 1))
        )
        get_or_create_equity_instrument(
            db_session, _bar(symbol="NEWNAME", trade_date=date(2024, 6, 1))
        )
        with pytest.raises(ValueError, match="ambiguous"):
            get_or_create_equity_instrument(
                db_session, _bar(symbol="SOMETHINGELSE", trade_date=date(2024, 3, 15))
            )

    def test_bar_without_isin_raises(self, db_session):
        with pytest.raises(ValueError, match="ISIN"):
            get_or_create_equity_instrument(db_session, _bar(isin=None))

    def test_concurrent_series_same_day_is_not_a_rename(self, db_session):
        eq_bar = _bar(series="EQ", trade_date=date(2025, 9, 15))
        bl_bar = _bar(series="BL", trade_date=date(2025, 9, 15))

        instrument_from_eq = get_or_create_equity_instrument(db_session, eq_bar)
        instrument_from_bl = get_or_create_equity_instrument(db_session, bl_bar)

        assert instrument_from_eq.id == instrument_from_bl.id
        periods = (
            db_session.query(InstrumentSymbol)
            .filter_by(instrument_id=instrument_from_eq.id)
            .all()
        )
        assert len(periods) == 1
        assert periods[0].effective_to is None

    def test_same_symbol_text_different_isin_is_not_a_conflict(self, db_session):
        equity = get_or_create_equity_instrument(
            db_session,
            _bar(
                symbol="M&MFIN", series="EQ", isin="INE774D01024", trade_date=date(2024, 7, 5)
            ),
        )
        bond = get_or_create_equity_instrument(
            db_session,
            _bar(
                symbol="M&MFIN", series="N3", isin="INE774D08MG3", trade_date=date(2024, 7, 5)
            ),
        )

        assert equity.id != bond.id
        assert _current_isin(db_session, equity) != _current_isin(db_session, bond)


class TestIsinSuccessorLinking:

    OLD_ISIN, _REASON = ISIN_SUCCESSOR_LINKS["INE296A01032"]
    NEW_ISIN = "INE296A01032"

    def test_new_isin_continues_the_same_instrument(self, db_session):
        old = get_or_create_equity_instrument(
            db_session,
            _bar(symbol="BAJFINANCE", isin=self.OLD_ISIN, trade_date=date(2024, 7, 5)),
        )
        new = get_or_create_equity_instrument(
            db_session,
            _bar(symbol="BAJFINANCE", isin=self.NEW_ISIN, trade_date=date(2025, 9, 15)),
        )

        assert old.id == new.id

    def test_isin_history_shows_both_periods(self, db_session):
        instrument = get_or_create_equity_instrument(
            db_session,
            _bar(symbol="BAJFINANCE", isin=self.OLD_ISIN, trade_date=date(2024, 7, 5)),
        )
        get_or_create_equity_instrument(
            db_session,
            _bar(symbol="BAJFINANCE", isin=self.NEW_ISIN, trade_date=date(2025, 9, 15)),
        )

        periods = (
            db_session.query(InstrumentIsin)
            .filter_by(instrument_id=instrument.id)
            .order_by(InstrumentIsin.effective_from)
            .all()
        )
        assert len(periods) == 2
        assert periods[0].isin == self.OLD_ISIN
        assert periods[0].effective_from == date(2024, 7, 5)
        assert periods[0].effective_to == date(2025, 9, 14)
        assert periods[1].isin == self.NEW_ISIN
        assert periods[1].effective_from == date(2025, 9, 15)
        assert periods[1].effective_to is None

    def test_symbol_history_is_unaffected_by_the_isin_change(self, db_session):
        instrument = get_or_create_equity_instrument(
            db_session,
            _bar(symbol="BAJFINANCE", isin=self.OLD_ISIN, trade_date=date(2024, 7, 5)),
        )
        get_or_create_equity_instrument(
            db_session,
            _bar(symbol="BAJFINANCE", isin=self.NEW_ISIN, trade_date=date(2025, 9, 15)),
        )

        periods = (
            db_session.query(InstrumentSymbol).filter_by(instrument_id=instrument.id).all()
        )
        assert len(periods) == 1
        assert periods[0].symbol == "BAJFINANCE"
        assert periods[0].effective_to is None

    def test_oldest_isin_arriving_after_newer_ones_reuses_the_same_instrument(
        self, db_session
    ):
        newer = get_or_create_equity_instrument(
            db_session,
            _bar(symbol="BAJFINANCE", isin=self.NEW_ISIN, trade_date=date(2025, 9, 15)),
        )
        oldest_isin, _reason = ISIN_SUCCESSOR_LINKS[self.OLD_ISIN]
        older = get_or_create_equity_instrument(
            db_session,
            _bar(symbol="BAJFINANCE", isin=oldest_isin, trade_date=date(2015, 1, 1)),
        )

        assert older.id == newer.id
        assert db_session.query(Instrument).count() == 1


class TestIndexMatching:
    def test_creates_index_instrument_on_first_bar(self, db_session):
        instrument = get_or_create_index_instrument(db_session, _index_bar())

        assert db_session.query(InstrumentIsin).filter_by(instrument_id=instrument.id).count() == 0
        assert instrument.asset_type is AssetType.INDEX

        periods = (
            db_session.query(InstrumentSymbol).filter_by(instrument_id=instrument.id).all()
        )
        assert len(periods) == 1
        assert periods[0].symbol == "NIFTY 50"

    def test_same_index_symbol_returns_same_instrument(self, db_session):
        first = get_or_create_index_instrument(
            db_session, _index_bar(trade_date=date(2024, 7, 5))
        )
        second = get_or_create_index_instrument(
            db_session, _index_bar(trade_date=date(2024, 7, 8))
        )

        assert first.id == second.id

    def test_different_index_symbols_are_different_instruments(self, db_session):
        nifty = get_or_create_index_instrument(db_session, _index_bar(symbol="NIFTY 50"))
        banknifty = get_or_create_index_instrument(
            db_session, _index_bar(symbol="NIFTY BANK")
        )

        assert nifty.id != banknifty.id


class TestIndexSymbolSuccessorLinking:

    OLD_SYMBOL, _REASON = INDEX_SYMBOL_SUCCESSOR_LINKS["NIFTY 50"]
    NEW_SYMBOL = "NIFTY 50"

    def test_new_symbol_continues_the_same_instrument(self, db_session):
        old = get_or_create_index_instrument(
            db_session, _index_bar(symbol=self.OLD_SYMBOL, trade_date=date(2015, 6, 1))
        )
        new = get_or_create_index_instrument(
            db_session, _index_bar(symbol=self.NEW_SYMBOL, trade_date=date(2015, 11, 9))
        )

        assert old.id == new.id

    def test_symbol_history_shows_both_periods(self, db_session):
        instrument = get_or_create_index_instrument(
            db_session, _index_bar(symbol=self.OLD_SYMBOL, trade_date=date(2015, 1, 1))
        )
        get_or_create_index_instrument(
            db_session, _index_bar(symbol=self.NEW_SYMBOL, trade_date=date(2015, 11, 9))
        )

        periods = (
            db_session.query(InstrumentSymbol)
            .filter_by(instrument_id=instrument.id)
            .order_by(InstrumentSymbol.effective_from)
            .all()
        )
        assert len(periods) == 2
        assert periods[0].symbol == self.OLD_SYMBOL
        assert periods[0].effective_to == date(2015, 11, 8)
        assert periods[1].symbol == self.NEW_SYMBOL
        assert periods[1].effective_to is None


class TestDispatch:
    def test_dispatches_isin_bar_to_equity_matcher(self, db_session):
        instrument = get_or_create_instrument_for_bar(db_session, _bar())
        assert instrument.asset_type is AssetType.EQUITY

    def test_dispatches_index_series_bar_to_index_matcher(self, db_session):
        instrument = get_or_create_instrument_for_bar(db_session, _index_bar())
        assert instrument.asset_type is AssetType.INDEX

    def test_bar_with_no_isin_and_non_index_series_raises(self, db_session):
        with pytest.raises(ValueError, match="cannot match/create"):
            get_or_create_instrument_for_bar(db_session, _bar(isin=None, series="EQ"))
