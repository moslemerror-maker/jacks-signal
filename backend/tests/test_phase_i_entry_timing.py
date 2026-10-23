from __future__ import annotations

import datetime as dt

from backend.app.research.intraday_features.bar_math import Bar
from scripts.run_phase_i_entry_timing import next_bar_entry_label

UTC = dt.timezone.utc


def _bar(day: dt.date, minute: int, open_: float, close: float) -> Bar:
    ts = dt.datetime(day.year, day.month, day.day, 9, 15, tzinfo=UTC) + dt.timedelta(minutes=minute)
    hi, lo = max(open_, close), min(open_, close)
    return Bar(ts, day, open_, hi, lo, close, 100.0)


class TestNextBarEntryLabel:
    def test_entry_price_is_the_open_of_the_bar_immediately_after_t_not_ts_own_close(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, 0, 100.0, 101.0), _bar(d, 1, 105.0, 106.0), _bar(d, 2, 106.0, 107.0), _bar(d, 3, 107.0, 108.0)]
        result = next_bar_entry_label(bars, t_index=0, horizon_bars=2, vol_normalizer=1.0, threshold=0.5)
        assert result.entry_price == 105.0

    def test_outcome_window_spans_horizon_bars_starting_at_the_entry_bar(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, i, 100.0 + i, 100.0 + i) for i in range(6)]
        result = next_bar_entry_label(bars, t_index=0, horizon_bars=3, vol_normalizer=1.0, threshold=0.5)
        assert result.exit_price == bars[3].close

    def test_undefined_when_entry_bar_crosses_into_the_next_session(self):
        d1, d2 = dt.date(2026, 1, 5), dt.date(2026, 1, 6)
        bars = [_bar(d1, 0, 100.0, 100.0), _bar(d2, 0, 101.0, 101.0), _bar(d2, 1, 102.0, 102.0)]
        result = next_bar_entry_label(bars, t_index=0, horizon_bars=1, vol_normalizer=1.0, threshold=0.5)
        assert result is None

    def test_undefined_when_outcome_window_crosses_into_the_next_session(self):
        d1, d2 = dt.date(2026, 1, 5), dt.date(2026, 1, 6)
        bars = [_bar(d1, 0, 100.0, 100.0), _bar(d1, 1, 101.0, 101.0), _bar(d2, 0, 102.0, 102.0)]
        result = next_bar_entry_label(bars, t_index=0, horizon_bars=2, vol_normalizer=1.0, threshold=0.5)
        assert result is None

    def test_undefined_with_no_vol_normalizer(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, i, 100.0, 100.0) for i in range(3)]
        assert next_bar_entry_label(bars, t_index=0, horizon_bars=1, vol_normalizer=None, threshold=0.5) is None
        assert next_bar_entry_label(bars, t_index=0, horizon_bars=1, vol_normalizer=0.0, threshold=0.5) is None

    def test_undefined_when_not_enough_bars_remain(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, i, 100.0, 100.0) for i in range(3)]
        assert next_bar_entry_label(bars, t_index=1, horizon_bars=5, vol_normalizer=1.0, threshold=0.5) is None

    def test_label_class_reflects_normalized_move_against_threshold(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, 0, 100.0, 100.0), _bar(d, 1, 100.0, 100.0), _bar(d, 2, 110.0, 110.0)]
        result = next_bar_entry_label(bars, t_index=0, horizon_bars=2, vol_normalizer=0.05, threshold=0.5)
        assert result.label_class == "UP"

    def test_same_bar_close_is_never_used_as_the_entry_price(self):
        d = dt.date(2026, 1, 5)
        bars = [_bar(d, 0, 100.0, 999.0), _bar(d, 1, 555.0, 555.0), _bar(d, 2, 556.0, 556.0)]
        result = next_bar_entry_label(bars, t_index=0, horizon_bars=1, vol_normalizer=1.0, threshold=0.5)
        assert result.entry_price != 999.0
        assert result.entry_price == 555.0
