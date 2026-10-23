from datetime import date
from pathlib import Path

from backend.app.data.canonical.schema import INDEX_SERIES, SourceFormat
from backend.app.data.normalizers.index_history import normalize_index_history

FIXTURES = Path(__file__).parent / "fixtures" / "index_history"
NIFTY50_FILE = FIXTURES / "NIFTY_50_2025-09-01_2025-09-15.json"
BANKNIFTY_FILE = FIXTURES / "NIFTY_BANK_2025-09-01_2025-09-15.json"


def test_parses_all_rows_with_no_rejections():
    bars, rejected = normalize_index_history(NIFTY50_FILE)
    assert len(bars) == 11
    assert rejected == []


def test_bar_values_match_known_row():
    bars, _ = normalize_index_history(NIFTY50_FILE)
    bar = next(b for b in bars if b.trade_date == date(2025, 9, 15))

    assert bar.symbol == "NIFTY 50"
    assert bar.series == INDEX_SERIES
    assert bar.isin is None
    assert bar.open == 25118.9
    assert bar.high == 25138.45
    assert bar.low == 25048.75
    assert bar.close == 25069.2
    assert bar.source_format is SourceFormat.NSE_INDEX_HISTORY


def test_banknifty_file_parses_and_uses_correct_symbol():
    bars, rejected = normalize_index_history(BANKNIFTY_FILE)
    assert rejected == []
    assert all(b.symbol == "NIFTY BANK" for b in bars)


def test_deterministic_across_runs():
    bars1, _ = normalize_index_history(NIFTY50_FILE)
    bars2, _ = normalize_index_history(NIFTY50_FILE)
    assert [b.model_dump() for b in bars1] == [b.model_dump() for b in bars2]


def test_bars_are_sorted_ascending_by_trade_date():
    bars, _ = normalize_index_history(NIFTY50_FILE)
    trade_dates = [b.trade_date for b in bars]
    assert trade_dates == sorted(trade_dates)
    assert trade_dates[0] == date(2025, 9, 1)
    assert trade_dates[-1] == date(2025, 9, 15)
