from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import text

from backend.app.data.live.watermark import missing_trading_dates, per_symbol_watermark

D = dt.date


def _insert_bars(db_session, *, security_id: int, trading_date: dt.date, n: int):
    base = dt.datetime(trading_date.year, trading_date.month, trading_date.day, 3, 45, tzinfo=dt.timezone.utc)
    for i in range(n):
        db_session.execute(
            text(
                """
                INSERT INTO intraday_equity_bars
                (id, provider, exchange_segment, security_id, trading_symbol, timeframe, bar_start_utc, trading_date,
                 raw_timestamp_epoch, open, high, low, close, volume, source_provider, source_dataset, source_key, source_sha256, normalizer_version)
                VALUES (gen_random_uuid(), 'dhan', 'NSE_EQ', :sid, 'TEST', '1m', :ts, :td, :epoch, 100, 101, 99, 100.5, 1000,
                        'dhan', 'nse_eq_intraday_1m', 'TEST-KEY', :sha, 'test/1')
                """
            ),
            {"sid": security_id, "ts": base + dt.timedelta(minutes=i), "td": trading_date, "epoch": int((base + dt.timedelta(minutes=i)).timestamp()), "sha": "x" * 64},
        )


class TestPerSymbolWatermark:
    def test_symbol_with_a_complete_session_reports_that_date(self, db_session):
        _insert_bars(db_session, security_id=1, trading_date=D(2026, 9, 25), n=360)
        wm = per_symbol_watermark(db_session, security_ids=[1], expected_bars=360)
        assert wm[1] == D(2026, 9, 25)

    def test_symbol_with_only_a_partial_session_reports_none(self, db_session):
        _insert_bars(db_session, security_id=2, trading_date=D(2026, 9, 25), n=100)
        wm = per_symbol_watermark(db_session, security_ids=[2], expected_bars=360)
        assert wm[2] is None

    def test_symbol_with_no_rows_at_all_reports_none(self, db_session):
        wm = per_symbol_watermark(db_session, security_ids=[999999], expected_bars=360)
        assert wm[999999] is None

    def test_reports_the_latest_of_multiple_complete_sessions(self, db_session):
        _insert_bars(db_session, security_id=3, trading_date=D(2026, 9, 24), n=360)
        _insert_bars(db_session, security_id=3, trading_date=D(2026, 9, 25), n=360)
        wm = per_symbol_watermark(db_session, security_ids=[3], expected_bars=360)
        assert wm[3] == D(2026, 9, 25)

    def test_a_partial_later_session_does_not_override_an_earlier_complete_one(self, db_session):
        _insert_bars(db_session, security_id=4, trading_date=D(2026, 9, 24), n=360)
        _insert_bars(db_session, security_id=4, trading_date=D(2026, 9, 25), n=50)
        wm = per_symbol_watermark(db_session, security_ids=[4], expected_bars=360)
        assert wm[4] == D(2026, 9, 24)

    def test_handles_multiple_symbols_independently_in_one_query(self, db_session):
        _insert_bars(db_session, security_id=5, trading_date=D(2026, 9, 25), n=360)
        wm = per_symbol_watermark(db_session, security_ids=[5, 6], expected_bars=360)
        assert wm[5] == D(2026, 9, 25)
        assert wm[6] is None

    def test_empty_security_id_list_returns_empty_dict(self, db_session):
        assert per_symbol_watermark(db_session, security_ids=[], expected_bars=360) == {}


HOLIDAYS_2026_SEP = frozenset()


class TestMissingTradingDates:
    def test_no_gap_when_watermark_is_yesterday(self):
        assert missing_trading_dates(D(2026, 9, 24), today=D(2026, 9, 25), holiday_dates=HOLIDAYS_2026_SEP) == []

    def test_weekend_is_never_reported_as_missing(self):
        missing = missing_trading_dates(D(2026, 9, 25), today=D(2026, 9, 28), holiday_dates=HOLIDAYS_2026_SEP)
        assert missing == []

    def test_a_genuinely_missing_weekday_is_reported(self):
        missing = missing_trading_dates(D(2026, 9, 25), today=D(2026, 9, 29), holiday_dates=HOLIDAYS_2026_SEP)
        assert missing == [D(2026, 9, 28)]

    def test_a_holiday_in_the_gap_is_excluded(self):
        holidays = frozenset({"28-Sep-2026"})
        missing = missing_trading_dates(D(2026, 9, 25), today=D(2026, 9, 29), holiday_dates=holidays)
        assert missing == []

    def test_today_itself_is_never_in_the_missing_list(self):
        missing = missing_trading_dates(D(2026, 9, 28), today=D(2026, 9, 29), holiday_dates=HOLIDAYS_2026_SEP)
        assert D(2026, 9, 29) not in missing

    def test_none_watermark_without_bootstrap_raises(self):
        with pytest.raises(ValueError, match="bootstrap_from"):
            missing_trading_dates(None, today=D(2026, 9, 29), holiday_dates=HOLIDAYS_2026_SEP)

    def test_none_watermark_with_bootstrap_starts_from_the_given_date(self):
        missing = missing_trading_dates(None, today=D(2026, 9, 29), holiday_dates=HOLIDAYS_2026_SEP, bootstrap_from=D(2026, 9, 28))
        assert missing == [D(2026, 9, 28)]
