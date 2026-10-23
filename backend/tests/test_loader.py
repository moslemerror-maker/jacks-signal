from datetime import date
from pathlib import Path

from backend.app.data.canonical.schema import CanonicalBar, SourceFormat
from backend.app.data.instruments.loader import upsert_bars
from backend.app.data.normalizers.index_history import normalize_index_history
from backend.app.data.normalizers.legacy_csv import normalize_legacy_csv
from backend.app.data.normalizers.udiff import normalize_udiff
from backend.app.db.models import Instrument, InstrumentIsin, MarketBar

FIXTURES = Path(__file__).parent / "fixtures" / "bhavcopy"
LEGACY_FILE = FIXTURES / "cm05JUL2024bhav.csv.zip"
UDIFF_FILE = FIXTURES / "BhavCopy_NSE_CM_0_0_0_20250915_F_0000.csv.zip"

INDEX_FIXTURES = Path(__file__).parent / "fixtures" / "index_history"
NIFTY50_FILE = INDEX_FIXTURES / "NIFTY_50_2025-09-01_2025-09-15.json"


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


def test_upsert_creates_one_row_per_bar(db_session):
    upserted = upsert_bars(db_session, [_bar()])
    db_session.commit()

    assert upserted == 1
    assert db_session.query(MarketBar).count() == 1


def test_reupsert_same_bar_updates_in_place_not_duplicate(db_session):
    upsert_bars(db_session, [_bar(close=105.0)])
    db_session.commit()

    upsert_bars(db_session, [_bar(close=107.5)])
    db_session.commit()

    rows = db_session.query(MarketBar).all()
    assert len(rows) == 1
    assert rows[0].close == 107.5


def test_reupsert_is_deterministic_across_three_runs(db_session):
    for _ in range(3):
        upsert_bars(db_session, [_bar()])
        db_session.commit()

    assert db_session.query(MarketBar).count() == 1


class TestEndToEndRealFixtures:

    def test_legacy_file_loads_completely(self, db_session):
        bars, rejected = normalize_legacy_csv(LEGACY_FILE)
        assert rejected == []

        upserted = upsert_bars(db_session, bars)
        db_session.commit()

        assert upserted == len(bars)
        assert db_session.query(MarketBar).count() == len(bars)

    def test_udiff_file_loads_completely(self, db_session):
        bars, rejected = normalize_udiff(UDIFF_FILE)
        assert rejected == []

        upserted = upsert_bars(db_session, bars)
        db_session.commit()

        assert upserted == len(bars)
        assert db_session.query(MarketBar).count() == len(bars)

    def test_reloading_legacy_file_is_idempotent(self, db_session):
        bars, _ = normalize_legacy_csv(LEGACY_FILE)

        upsert_bars(db_session, bars)
        db_session.commit()
        first_count = db_session.query(MarketBar).count()

        upsert_bars(db_session, bars)
        db_session.commit()
        second_count = db_session.query(MarketBar).count()

        assert first_count == second_count == len(bars)

    def test_index_file_loads_as_single_instrument(self, db_session):
        bars, rejected = normalize_index_history(NIFTY50_FILE)
        assert rejected == []

        upserted = upsert_bars(db_session, bars)
        db_session.commit()

        assert upserted == len(bars)
        assert db_session.query(MarketBar).count() == len(bars)
        assert db_session.query(Instrument).count() == 1
        assert db_session.query(InstrumentIsin).count() == 0

    def test_reloading_index_file_is_idempotent(self, db_session):
        bars, _ = normalize_index_history(NIFTY50_FILE)

        upsert_bars(db_session, bars)
        db_session.commit()
        first_count = db_session.query(MarketBar).count()

        upsert_bars(db_session, bars)
        db_session.commit()
        second_count = db_session.query(MarketBar).count()

        assert first_count == second_count == len(bars)
