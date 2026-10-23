from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from backend.app.data.canonical.intraday_equity_bar import CanonicalIntradayEquityBar, IntradayProvider
from backend.app.data.canonical.intraday_loader import upsert_intraday_bars, upsert_intraday_bars_batched
from backend.app.db.models import IntradayEquityBar, MarketBar

UTC = dt.timezone.utc
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def _bar(**overrides) -> CanonicalIntradayEquityBar:
    fields = dict(
        provider=IntradayProvider.DHAN, exchange_segment="NSE_EQ", security_id=2885, trading_symbol="RELIANCE",
        raw_timestamp_epoch=1787715900, bar_start_utc=dt.datetime(2026, 8, 26, 3, 45, tzinfo=UTC),
        bar_start_ist=dt.datetime(2026, 8, 26, 9, 15, tzinfo=IST), trading_date=dt.date(2026, 8, 26), interval_minutes=1,
        open=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0,
        source_provider="dhan", source_dataset="nse_eq_intraday_1m", source_key="NSE_EQ-2885-1m-20260826-20260925",
        source_sha256="a" * 64, normalizer_version="dhan_intraday_equity/1",
    )
    fields.update(overrides)
    return CanonicalIntradayEquityBar(**fields)


class TestSchema:
    def test_insert_one_bar(self, db_session):
        upserted = upsert_intraday_bars(db_session, [_bar()])
        db_session.commit()
        assert upserted == 1
        assert db_session.query(IntradayEquityBar).count() == 1

    def test_multiple_bars_same_day_persist_as_distinct_rows(self, db_session):
        bars = [
            _bar(
                raw_timestamp_epoch=1787715900 + i * 60,
                bar_start_utc=dt.datetime(2026, 8, 26, 3, 45, tzinfo=UTC) + dt.timedelta(minutes=i),
                bar_start_ist=dt.datetime(2026, 8, 26, 9, 15, tzinfo=IST) + dt.timedelta(minutes=i),
            )
            for i in range(5)
        ]
        upsert_intraday_bars(db_session, bars)
        db_session.commit()
        assert db_session.query(IntradayEquityBar).count() == 5


class TestUniqueness:
    def test_exact_timestamp_is_the_identity_database_level_enforced(self, db_session):
        upsert_intraday_bars(db_session, [_bar()])
        db_session.commit()
        with pytest.raises(IntegrityError):
            db_session.execute(text(
                "INSERT INTO intraday_equity_bars (id, provider, exchange_segment, security_id, trading_symbol, timeframe, "
                "bar_start_utc, trading_date, raw_timestamp_epoch, open, high, low, close, volume, "
                "source_provider, source_dataset, source_key, source_sha256, normalizer_version) "
                "VALUES (gen_random_uuid(), 'DHAN', 'NSE_EQ', 2885, 'RELIANCE', '1m', "
                "'2026-08-26T03:45:00+00', '2026-08-26', 1787715900, 999, 999, 999, 999, 1, "
                "'dhan', 'nse_eq_intraday_1m', 'x', '" + "b" * 64 + "', 'v1')"
            ))
            db_session.commit()

    def test_different_security_id_same_timestamp_is_not_a_conflict(self, db_session):
        upsert_intraday_bars(db_session, [_bar(security_id=2885), _bar(security_id=11536, trading_symbol="TCS")])
        db_session.commit()
        assert db_session.query(IntradayEquityBar).count() == 2

    _INSERT_SQL = (
        "INSERT INTO intraday_equity_bars (id, provider, exchange_segment, security_id, trading_symbol, timeframe, "
        "bar_start_utc, trading_date, raw_timestamp_epoch, open, high, low, close, volume, "
        "source_provider, source_dataset, source_key, source_sha256, normalizer_version) "
        "VALUES (gen_random_uuid(), 'DHAN', 'NSE_EQ', 2885, 'RELIANCE', '1m', "
        "'2026-08-26T03:45:00+00', '2026-08-26', 1787715900, :open, :high, :low, :close, :volume, "
        "'dhan', 'nse_eq_intraday_1m', 'x', '" + "b" * 64 + "', 'v1')"
    )

    @pytest.mark.parametrize("open_, high, low, close", [(100.0, 90.0, 99.0, 100.5), (100.0, 101.0, 110.0, 100.5)])
    def test_check_constraints_reject_bad_ohlc_at_db_level(self, db_session, open_, high, low, close):
        with pytest.raises(IntegrityError):
            db_session.execute(text(self._INSERT_SQL), {"open": open_, "high": high, "low": low, "close": close, "volume": 1})
            db_session.commit()

    def test_check_constraint_rejects_negative_volume_at_db_level(self, db_session):
        with pytest.raises(IntegrityError):
            db_session.execute(text(self._INSERT_SQL), {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": -1})
            db_session.commit()


class TestIdempotency:
    def test_running_the_same_upsert_twice_does_not_duplicate(self, db_session):
        bars = [_bar()]
        upsert_intraday_bars(db_session, bars)
        db_session.commit()
        first_count = db_session.query(IntradayEquityBar).count()

        upsert_intraday_bars(db_session, bars)
        db_session.commit()
        second_count = db_session.query(IntradayEquityBar).count()

        assert first_count == second_count == 1

    def test_reupsert_updates_changed_fields_in_place(self, db_session):
        upsert_intraday_bars(db_session, [_bar(close=100.5)])
        db_session.commit()
        upsert_intraday_bars(db_session, [_bar(close=102.0, high=103.0)])
        db_session.commit()

        rows = db_session.query(IntradayEquityBar).all()
        assert len(rows) == 1 and rows[0].close == 102.0

    def test_deterministic_across_three_runs(self, db_session):
        bars = [_bar()]
        for _ in range(3):
            upsert_intraday_bars(db_session, bars)
            db_session.commit()
        assert db_session.query(IntradayEquityBar).count() == 1


class TestProvenance:
    def test_every_field_needed_to_trace_back_to_the_raw_artifact_is_persisted(self, db_session):
        upsert_intraday_bars(db_session, [_bar()])
        db_session.commit()
        row = db_session.query(IntradayEquityBar).one()
        assert row.source_provider == "dhan" and row.source_dataset == "nse_eq_intraday_1m"
        assert row.source_key == "NSE_EQ-2885-1m-20260826-20260925" and row.source_sha256 == "a" * 64
        assert row.normalizer_version == "dhan_intraday_equity/1"
        assert row.raw_timestamp_epoch == 1787715900


class TestEodIsolation:

    def test_intraday_persistence_creates_no_market_bar_rows(self, db_session):
        upsert_intraday_bars(db_session, [_bar()])
        db_session.commit()
        assert db_session.query(MarketBar).count() == 0

    def test_intraday_and_eod_tables_are_fully_independent(self, db_session):
        from backend.app.data.canonical.schema import CanonicalBar, SourceFormat
        from backend.app.data.instruments.loader import upsert_bars

        eod_bar = CanonicalBar(
            symbol="RELIANCE", series="EQ", isin="INE002A01018", trade_date=dt.date(2026, 8, 26),
            open=100.0, high=110.0, low=95.0, close=105.0, source_format=SourceFormat.NSE_LEGACY_CSV, source_file="test.zip",
        )
        upsert_bars(db_session, [eod_bar])
        upsert_intraday_bars(db_session, [_bar()])
        db_session.commit()
        assert db_session.query(MarketBar).count() == 1
        assert db_session.query(IntradayEquityBar).count() == 1


def _many_bars(n: int, *, security_id: int = 2885, symbol: str = "RELIANCE", start_epoch: int = 1787715900) -> list[CanonicalIntradayEquityBar]:
    out = []
    for i in range(n):
        bs_utc = dt.datetime.fromtimestamp(start_epoch + i * 60, tz=UTC)
        bs_ist = bs_utc.astimezone(IST)
        out.append(_bar(
            security_id=security_id, trading_symbol=symbol, raw_timestamp_epoch=start_epoch + i * 60,
            bar_start_utc=bs_utc, bar_start_ist=bs_ist, trading_date=bs_ist.date(),
        ))
    return out


class TestBatchedLoaderEquivalence:

    def test_batched_and_per_row_loaders_produce_identical_rows(self, db_session):
        bars_a = _many_bars(50, security_id=2885, symbol="RELIANCE")
        bars_b = _many_bars(50, security_id=11536, symbol="TCS")

        upsert_intraday_bars(db_session, bars_a)
        upsert_intraday_bars_batched(db_session, bars_b, batch_size=7)
        db_session.commit()

        rows_a = db_session.query(IntradayEquityBar).filter_by(security_id=2885).count()
        rows_b = db_session.query(IntradayEquityBar).filter_by(security_id=11536).count()
        assert rows_a == rows_b == 50

    def test_batched_upsert_is_idempotent_across_two_full_invocations(self, db_session):
        bars = _many_bars(120)
        upsert_intraday_bars_batched(db_session, bars, batch_size=25)
        db_session.commit()
        first_count = db_session.query(IntradayEquityBar).count()

        upsert_intraday_bars_batched(db_session, bars, batch_size=25)
        db_session.commit()
        second_count = db_session.query(IntradayEquityBar).count()

        assert first_count == second_count == 120

    def test_batched_upsert_updates_changed_fields_in_place(self, db_session):
        bars = _many_bars(10)
        upsert_intraday_bars_batched(db_session, bars, batch_size=4)
        db_session.commit()

        changed = list(bars)
        changed[3] = _bar(
            security_id=2885, trading_symbol="RELIANCE", raw_timestamp_epoch=bars[3].raw_timestamp_epoch,
            bar_start_utc=bars[3].bar_start_utc, bar_start_ist=bars[3].bar_start_ist, trading_date=bars[3].trading_date,
            close=999.0, high=999.5,
        )
        upsert_intraday_bars_batched(db_session, changed, batch_size=4)
        db_session.commit()

        assert db_session.query(IntradayEquityBar).count() == 10
        row = db_session.query(IntradayEquityBar).filter_by(raw_timestamp_epoch=bars[3].raw_timestamp_epoch).one()
        assert row.close == 999.0

    def test_duplicate_identity_within_the_same_batch_does_not_error_or_duplicate(self, db_session):
        bars = _many_bars(3)
        with_duplicate = bars + [bars[1]]
        upsert_intraday_bars_batched(db_session, with_duplicate, batch_size=10)
        db_session.commit()
        assert db_session.query(IntradayEquityBar).count() == 3

    def test_batch_size_boundary_exact_multiple_and_remainder(self, db_session):
        bars = _many_bars(23)
        upserted = upsert_intraday_bars_batched(db_session, bars, batch_size=10)
        db_session.commit()
        assert upserted == 23
        assert db_session.query(IntradayEquityBar).count() == 23

    def test_empty_input_is_a_safe_noop(self, db_session):
        upserted = upsert_intraday_bars_batched(db_session, [], batch_size=100)
        db_session.commit()
        assert upserted == 0 and db_session.query(IntradayEquityBar).count() == 0

    def test_batched_loader_creates_no_market_bar_rows(self, db_session):
        upsert_intraday_bars_batched(db_session, _many_bars(5), batch_size=2)
        db_session.commit()
        assert db_session.query(MarketBar).count() == 0

    def test_batch_size_beyond_the_postgres_parameter_limit_is_rejected_before_reaching_the_db(self, db_session):
        from backend.app.data.canonical.intraday_loader import _MAX_SAFE_BATCH_SIZE
        with pytest.raises(ValueError, match="65,535"):
            upsert_intraday_bars_batched(db_session, _many_bars(5), batch_size=_MAX_SAFE_BATCH_SIZE + 1)
        assert db_session.query(IntradayEquityBar).count() == 0

    def test_max_safe_batch_size_itself_is_accepted(self, db_session):
        from backend.app.data.canonical.intraday_loader import _MAX_SAFE_BATCH_SIZE
        upsert_intraday_bars_batched(db_session, _many_bars(3), batch_size=_MAX_SAFE_BATCH_SIZE)
        db_session.commit()
        assert db_session.query(IntradayEquityBar).count() == 3
