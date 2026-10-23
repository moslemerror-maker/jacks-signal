from __future__ import annotations

import datetime as dt

import pytest

from backend.app.data.canonical.intraday_equity_bar import CanonicalIntradayEquityBar, IntradayProvider
from backend.app.data.canonical.intraday_loader import upsert_intraday_bars
from backend.app.research.intraday_point_in_time import latest_bar_as_of, select_bars_as_of

UTC = dt.timezone.utc
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def _bar_at(hour: int, minute: int, *, day: dt.date = dt.date(2026, 8, 26), close: float = 100.0, volume: float = 1000.0) -> CanonicalIntradayEquityBar:
    bar_start_ist = dt.datetime(day.year, day.month, day.day, hour, minute, tzinfo=IST)
    bar_start_utc = bar_start_ist.astimezone(UTC)
    return CanonicalIntradayEquityBar(
        provider=IntradayProvider.DHAN, exchange_segment="NSE_EQ", security_id=2885, trading_symbol="RELIANCE",
        raw_timestamp_epoch=int(bar_start_utc.timestamp()), bar_start_utc=bar_start_utc, bar_start_ist=bar_start_ist,
        trading_date=day, interval_minutes=1, open=close, high=close + 1, low=close - 1, close=close, volume=volume,
        source_provider="dhan", source_dataset="nse_eq_intraday_1m", source_key="NSE_EQ-2885-1m-test", source_sha256="a" * 64,
        normalizer_version="dhan_intraday_equity/1",
    )


@pytest.fixture
def full_day_seeded(db_session):
    bars = [_bar_at(9 + (15 + i) // 60, (15 + i) % 60, close=100.0 + i * 0.1, volume=10.0) for i in range(360)]
    bars.append(_bar_at(15, 14, day=dt.date(2026, 8, 25), close=999.0, volume=5000.0))
    upsert_intraday_bars(db_session, bars)
    db_session.commit()
    return bars


class TestFutureRowExclusion:
    def test_a_bar_at_10_01_is_not_visible_at_as_of_10_00(self, db_session):
        upsert_intraday_bars(db_session, [_bar_at(10, 0), _bar_at(10, 1), _bar_at(10, 5), _bar_at(11, 0)])
        db_session.commit()
        as_of = dt.datetime(2026, 8, 26, 10, 0, tzinfo=IST).astimezone(UTC)
        bars = select_bars_as_of(db_session, security_id=2885, timeframe="1m", as_of_utc=as_of)
        assert [b.bar_start_utc for b in bars] == [dt.datetime(2026, 8, 26, 10, 0, tzinfo=IST).astimezone(UTC)]

    def test_deliberately_inserted_future_rows_are_never_selected(self, db_session):
        upsert_intraday_bars(db_session, [
            _bar_at(10, 0), _bar_at(10, 1), _bar_at(10, 5), _bar_at(11, 0), _bar_at(15, 14, close=500.0),
        ])
        db_session.commit()
        as_of = dt.datetime(2026, 8, 26, 10, 0, tzinfo=IST).astimezone(UTC)
        bars = select_bars_as_of(db_session, security_id=2885, timeframe="1m", as_of_utc=as_of)
        assert len(bars) == 1
        assert all(b.bar_start_utc <= as_of for b in bars)
        assert 500.0 not in [b.close for b in bars]

    def test_bars_are_included_up_to_and_including_as_of_exactly(self, db_session):
        upsert_intraday_bars(db_session, [_bar_at(10, 0), _bar_at(10, 1)])
        db_session.commit()
        as_of = dt.datetime(2026, 8, 26, 10, 0, tzinfo=IST).astimezone(UTC)
        bars = select_bars_as_of(db_session, security_id=2885, timeframe="1m", as_of_utc=as_of)
        assert len(bars) == 1 and bars[0].bar_start_utc == as_of

    def test_naive_as_of_is_rejected(self, db_session):
        with pytest.raises(ValueError, match="timezone-aware"):
            select_bars_as_of(db_session, security_id=2885, timeframe="1m", as_of_utc=dt.datetime(2026, 8, 26, 10, 0))


class TestSameDayInformation:
    def test_todays_close_is_not_visible_before_it_happens(self, db_session, full_day_seeded):
        as_of = dt.datetime(2026, 8, 26, 10, 0, tzinfo=IST).astimezone(UTC)
        bars = select_bars_as_of(db_session, security_id=2885, timeframe="1m", as_of_utc=as_of, start_utc=dt.datetime(2026, 8, 26, 0, 0, tzinfo=UTC))
        assert 100.0 + 359 * 0.1 not in [b.close for b in bars]

    def test_previous_trading_days_close_is_available_via_the_same_primitive(self, db_session, full_day_seeded):
        as_of = dt.datetime(2026, 8, 26, 10, 0, tzinfo=IST).astimezone(UTC)
        prev_day_bars = select_bars_as_of(
            db_session, security_id=2885, timeframe="1m", as_of_utc=as_of,
            start_utc=dt.datetime(2026, 8, 25, 0, 0, tzinfo=UTC),
        )
        prev_day_last = [b for b in prev_day_bars if b.trading_date == dt.date(2026, 8, 25)]
        assert prev_day_last and prev_day_last[-1].close == 999.0

    def test_partial_day_volume_through_10am_differs_materially_from_full_day_volume(self, db_session, full_day_seeded):
        as_of = dt.datetime(2026, 8, 26, 10, 0, tzinfo=IST).astimezone(UTC)
        partial = select_bars_as_of(db_session, security_id=2885, timeframe="1m", as_of_utc=as_of, start_utc=dt.datetime(2026, 8, 26, 0, 0, tzinfo=UTC))
        full_day = select_bars_as_of(db_session, security_id=2885, timeframe="1m", as_of_utc=dt.datetime(2026, 8, 26, 23, 59, tzinfo=UTC), start_utc=dt.datetime(2026, 8, 26, 0, 0, tzinfo=UTC))
        partial_volume = sum(b.volume for b in partial)
        full_volume = sum(b.volume for b in full_day)
        assert partial_volume < full_volume
        assert len(partial) == 46 and len(full_day) == 360


class TestLatestBarAsOf:
    def test_returns_none_before_any_bar_exists(self, db_session):
        upsert_intraday_bars(db_session, [_bar_at(10, 0)])
        db_session.commit()
        assert latest_bar_as_of(db_session, security_id=2885, timeframe="1m", as_of_utc=dt.datetime(2026, 8, 26, 9, 0, tzinfo=IST).astimezone(UTC)) is None

    def test_returns_the_most_recent_bar_not_a_future_one(self, db_session):
        upsert_intraday_bars(db_session, [_bar_at(10, 0, close=50.0), _bar_at(10, 5, close=60.0), _bar_at(10, 10, close=70.0)])
        db_session.commit()
        latest = latest_bar_as_of(db_session, security_id=2885, timeframe="1m", as_of_utc=dt.datetime(2026, 8, 26, 10, 6, tzinfo=IST).astimezone(UTC))
        assert latest.close == 60.0
