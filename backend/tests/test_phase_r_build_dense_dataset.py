from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from backend.app.research.intraday_features import mover_mechanics, price_action, volatility_volume, vwap_levels
from backend.app.research.intraday_features.bar_math import Bar
from scripts.run_phase_k_entry_and_path import entry_and_path
from scripts.run_phase_r_build_dense_dataset import DIRECTION_THRESHOLD, compute_for_symbol, subsample_to_grid

UTC = dt.timezone.utc
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def _synth_session(trading_date: dt.date, n: int = 60, base_price: float = 100.0, seed: int = 0) -> list[Bar]:
    bars = []
    price = base_price
    open_ist = dt.datetime(trading_date.year, trading_date.month, trading_date.day, 9, 15, tzinfo=IST)
    state = seed
    for i in range(n):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        drift = ((state % 200) - 100) / 5000.0
        o = price
        c = price * (1 + drift)
        h, l = max(o, c) * 1.002, min(o, c) * 0.998
        vol = 1000 + (state % 5000)
        ts = open_ist + dt.timedelta(minutes=i)
        bars.append(Bar(ts.astimezone(UTC), trading_date, o, h, l, c, float(vol)))
        price = c
    return bars


def _to_df(bars: list[Bar]) -> pd.DataFrame:
    return pd.DataFrame([
        {"bar_start_utc": b.bar_start_utc, "trading_date": b.trading_date, "open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume}
        for b in bars
    ])


@pytest.fixture(scope="module")
def three_day_bars() -> list[Bar]:
    d1, d2, d3 = dt.date(2026, 1, 5), dt.date(2026, 1, 6), dt.date(2026, 1, 7)
    return _synth_session(d1, 60, seed=1) + _synth_session(d2, 60, seed=2) + _synth_session(d3, 60, seed=3)


@pytest.fixture(scope="module")
def three_day_computed(three_day_bars) -> pd.DataFrame:
    return compute_for_symbol(_to_df(three_day_bars))


class TestEquivalenceToPureFunctions:

    def test_return_windows(self, three_day_bars, three_day_computed):
        for n in (1, 3, 5, 15, 30, 60):
            for i in range(len(three_day_bars)):
                pure = price_action.return_over_bars(three_day_bars[: i + 1], n)
                vec = three_day_computed[f"return_{n}m"].iloc[i]
                vec = None if pd.isna(vec) else float(vec)
                assert (pure is None and vec is None) or (pure is not None and vec is not None and abs(pure - vec) < 1e-9), f"return_{n}m@{i}: {pure} vs {vec}"

    def test_relative_volume_20(self, three_day_bars, three_day_computed):
        for i in range(len(three_day_bars)):
            pure = volatility_volume.relative_volume(three_day_bars[: i + 1], 20)
            vec = three_day_computed["relative_volume_20"].iloc[i]
            vec = None if pd.isna(vec) else float(vec)
            assert (pure is None and vec is None) or (pure is not None and vec is not None and abs(pure - vec) < 1e-9)

    def test_overnight_gap(self, three_day_bars, three_day_computed):
        for i in range(len(three_day_bars)):
            pure = price_action.overnight_gap(three_day_bars[: i + 1])
            vec = three_day_computed["overnight_gap"].iloc[i]
            vec = None if pd.isna(vec) else float(vec)
            assert (pure is None and vec is None) or (pure is not None and vec is not None and abs(pure - vec) < 1e-9)

    def test_distance_from_vwap(self, three_day_bars, three_day_computed):
        for i in range(len(three_day_bars)):
            pure = vwap_levels.distance_from_vwap(three_day_bars[: i + 1])
            vec = three_day_computed["distance_from_vwap"].iloc[i]
            vec = None if pd.isna(vec) else float(vec)
            assert (pure is None and vec is None) or (pure is not None and vec is not None and abs(pure - vec) < 1e-9)

    def test_opening_range_and_breakouts(self, three_day_bars, three_day_computed):
        for i in range(len(three_day_bars)):
            pure_or = vwap_levels.opening_range_high_low(three_day_bars[: i + 1])
            vec_high = three_day_computed["opening_range_high"].iloc[i]
            vec_high = None if pd.isna(vec_high) else float(vec_high)
            if pure_or is None:
                assert vec_high is None
                continue
            assert vec_high is not None and abs(pure_or[0] - vec_high) < 1e-9

            pure_up = mover_mechanics.recent_breakout_up(three_day_bars[: i + 1], pure_or[0], within_bars=5)
            vec_up = three_day_computed["recent_breakout_up_opening_range"].iloc[i]
            vec_up = None if pd.isna(vec_up) else bool(vec_up)
            assert pure_up == vec_up

            pure_down = mover_mechanics.recent_breakout_down(three_day_bars[: i + 1], pure_or[1], within_bars=5)
            vec_down = three_day_computed["recent_breakout_down_opening_range"].iloc[i]
            vec_down = None if pd.isna(vec_down) else bool(vec_down)
            assert pure_down == vec_down

    def test_gross_return_and_direction_label_match_entry_and_path(self, three_day_bars, three_day_computed):
        for h in (5, 15, 30, 60):
            for i in range(len(three_day_bars)):
                pure = entry_and_path(three_day_bars, i, horizon_bars=h)
                pure_gross = pure.gross_return if pure else None
                vec_gross = three_day_computed[f"gross_return_{h}m"].iloc[i]
                vec_gross = None if pd.isna(vec_gross) else float(vec_gross)
                assert (pure_gross is None and vec_gross is None) or (pure_gross is not None and vec_gross is not None and abs(pure_gross - vec_gross) < 1e-9)

                vec_label = three_day_computed[f"direction_label_{h}m"].iloc[i]
                if pure_gross is None:
                    assert vec_label is None or pd.isna(vec_label)
                else:
                    expected = "UP" if pure_gross >= DIRECTION_THRESHOLD else ("DOWN" if pure_gross <= -DIRECTION_THRESHOLD else "AMBIGUOUS")
                    assert vec_label == expected


class TestAdversarialFutureAppendOnTheVectorizedFunction:
    def test_appending_two_future_sessions_never_changes_an_earlier_rows_values(self):
        d1, d2, d3, d4 = dt.date(2026, 1, 5), dt.date(2026, 1, 6), dt.date(2026, 1, 7), dt.date(2026, 1, 8)
        short_bars = _synth_session(d1, 60, seed=1) + _synth_session(d2, 60, seed=2)
        long_bars = short_bars + _synth_session(d3, 60, seed=3) + _synth_session(d4, 60, seed=4)

        short_computed = compute_for_symbol(_to_df(short_bars))
        long_computed = compute_for_symbol(_to_df(long_bars))

        cols = [c for c in short_computed.columns if c not in ("bar_start_utc", "trading_date", "open", "high", "low", "close", "volume")]
        for i in range(len(short_bars)):
            for c in cols:
                a, b = short_computed[c].iloc[i], long_computed[c].iloc[i]
                a_na = pd.isna(a) if not isinstance(a, str) else False
                b_na = pd.isna(b) if not isinstance(b, str) else False
                if a_na and b_na:
                    continue
                assert a_na == b_na, f"{c}@{i}: NA-mismatch short={a} long={b}"
                if isinstance(a, (str, bool)):
                    assert a == b, f"{c}@{i}: short={a} long={b}"
                else:
                    assert abs(float(a) - float(b)) < 1e-9, f"{c}@{i}: short={a} long={b}"


class TestSubsampleToGrid:
    def test_keeps_every_5th_bar_within_each_day(self, three_day_bars):
        df = compute_for_symbol(_to_df(three_day_bars))
        dense = subsample_to_grid(df)
        assert len(dense) == 36

    def test_a_gap_in_one_symbols_day_does_not_shift_its_sample_grid_out_of_phase(self):
        d1 = dt.date(2026, 1, 5)
        bars = _synth_session(d1, n=60, seed=1)
        bars_with_gap = [b for i, b in enumerate(bars) if i != 5]

        df = compute_for_symbol(_to_df(bars_with_gap))
        dense = subsample_to_grid(df)
        kept_minutes = dense["bar_start_utc"].dt.tz_convert(IST).dt.strftime("%H:%M").tolist()

        assert "09:20" not in kept_minutes
        assert "09:21" not in kept_minutes and "09:26" not in kept_minutes
        for m in kept_minutes:
            minute = int(m.split(":")[1])
            assert minute % 5 == 0, f"{m} is off the true wall-clock 5-minute grid"

    def test_first_kept_row_of_each_day_is_the_session_open_bar(self, three_day_bars):
        df = compute_for_symbol(_to_df(three_day_bars))
        dense = subsample_to_grid(df)
        first_per_day = dense.groupby("trading_date")["bar_start_utc"].min()
        for day, ts in first_per_day.items():
            session_open = dt.datetime(day.year, day.month, day.day, 9, 15, tzinfo=IST).astimezone(UTC)
            assert ts == session_open
