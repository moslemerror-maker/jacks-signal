from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from backend.app.research.intraday_features import momentum_oscillators as mo
from backend.app.research.intraday_features import price_action, volatility_volume, vwap_levels
from backend.app.research.intraday_features.bar_math import Bar
from scripts.run_phase_s_extend_dense_dataset import _nearest_level, compute_extension_for_symbol

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
    return compute_extension_for_symbol(_to_df(three_day_bars))


class TestEquivalenceToPureFunctions:
    def test_rsi_and_slope(self, three_day_bars, three_day_computed):
        for i in range(len(three_day_bars)):
            bars_up_to_t = three_day_bars[: i + 1]
            for period, col in ((9, "rsi_9"), (14, "rsi_14")):
                pure = mo.relative_strength_index(bars_up_to_t, period)
                vec = three_day_computed[col].iloc[i]
                vec = None if pd.isna(vec) else float(vec)
                assert (pure is None and vec is None) or (pure is not None and vec is not None and abs(pure - vec) < 1e-9)
            pure_slope = mo.rsi_slope(bars_up_to_t, 14, 5)
            vec_slope = three_day_computed["rsi_14_slope_5"].iloc[i]
            vec_slope = None if pd.isna(vec_slope) else float(vec_slope)
            assert (pure_slope is None and vec_slope is None) or (pure_slope is not None and vec_slope is not None and abs(pure_slope - vec_slope) < 1e-9)

    def test_macd(self, three_day_bars, three_day_computed):
        for i in range(len(three_day_bars)):
            pure = mo.macd_histogram(three_day_bars[: i + 1], 12, 26, 9)
            vec = three_day_computed["macd_histogram_12_26_9"].iloc[i]
            vec = None if pd.isna(vec) else float(vec)
            assert (pure is None and vec is None) or (pure is not None and vec is not None and abs(pure - vec) < 1e-6)

    def test_ema_9_20_50(self, three_day_bars, three_day_computed):
        for span in (9, 20, 50):
            for i in range(len(three_day_bars)):
                pure = mo.ema(three_day_bars[: i + 1], span)
                vec = three_day_computed[f"ema_{span}"].iloc[i]
                vec = None if pd.isna(vec) else float(vec)
                assert (pure is None and vec is None) or (pure is not None and vec is not None and abs(pure - vec) < 1e-6)

    def test_realized_volatility_20(self, three_day_bars, three_day_computed):
        for i in range(len(three_day_bars)):
            pure = volatility_volume.realized_volatility(three_day_bars[: i + 1], 20)
            vec = three_day_computed["realized_volatility_20"].iloc[i]
            vec = None if pd.isna(vec) else float(vec)
            assert (pure is None and vec is None) or (pure is not None and vec is not None and abs(pure - vec) < 1e-9)

    def test_previous_day_high_low_close(self, three_day_bars, three_day_computed):
        for i in range(len(three_day_bars)):
            pure = vwap_levels.previous_day_high_low_close(three_day_bars[: i + 1])
            vec_h = three_day_computed["previous_day_high"].iloc[i]
            vec_h = None if pd.isna(vec_h) else float(vec_h)
            if pure is None:
                assert vec_h is None
            else:
                assert vec_h is not None and abs(pure[0] - vec_h) < 1e-9

    def test_session_high_low_so_far(self, three_day_bars, three_day_computed):
        for i in range(len(three_day_bars)):
            pure = vwap_levels.session_high_low_so_far(three_day_bars[: i + 1])
            vec_h = three_day_computed["session_high_so_far"].iloc[i]
            vec_h = None if pd.isna(vec_h) else float(vec_h)
            assert pure is not None and vec_h is not None and abs(pure[0] - vec_h) < 1e-9

    def test_distance_from_session_open(self, three_day_bars, three_day_computed):
        for i in range(len(three_day_bars)):
            pure = price_action.distance_from_session_open(three_day_bars[: i + 1])
            vec = three_day_computed["distance_from_session_open"].iloc[i]
            vec = None if pd.isna(vec) else float(vec)
            assert (pure is None and vec is None) or (pure is not None and vec is not None and abs(pure - vec) < 1e-9)

    def test_candle_direction_and_consecutive_count(self, three_day_bars, three_day_computed):
        for i in range(len(three_day_bars)):
            bars_up_to_t = three_day_bars[: i + 1]
            pure_cd = price_action.candle_direction(bars_up_to_t[-1])
            vec_cd = int(three_day_computed["candle_direction"].iloc[i])
            assert pure_cd == vec_cd

            pure_csdc = price_action.consecutive_same_direction_count(bars_up_to_t)
            vec_csdc = int(three_day_computed["consecutive_same_direction_count"].iloc[i])
            assert pure_csdc == vec_csdc


class TestNearestLevel:
    def test_selects_the_closest_candidate_on_the_correct_side(self):
        current = pd.Series([100.0, 100.0])
        below = [pd.Series([95.0, 90.0]), pd.Series([90.0, 85.0]), pd.Series([98.0, 99.0])]
        above = [pd.Series([105.0, 110.0]), pd.Series([110.0, 108.0]), pd.Series([102.0, 101.0])]
        assert _nearest_level(current, below, above=False).tolist() == [98.0, 99.0]
        assert _nearest_level(current, above, above=True).tolist() == [102.0, 101.0]

    def test_a_level_exactly_at_current_price_is_never_selected_on_either_side(self):
        current = pd.Series([100.0])
        assert pd.isna(_nearest_level(current, [pd.Series([100.0])], above=True).iloc[0])
        assert pd.isna(_nearest_level(current, [pd.Series([100.0])], above=False).iloc[0])


class TestAdversarialFutureAppend:
    def test_appending_two_future_sessions_never_changes_an_earlier_rows_values(self):
        d1, d2, d3, d4 = dt.date(2026, 1, 5), dt.date(2026, 1, 6), dt.date(2026, 1, 7), dt.date(2026, 1, 8)
        short_bars = _synth_session(d1, 60, seed=1) + _synth_session(d2, 60, seed=2)
        long_bars = short_bars + _synth_session(d3, 60, seed=3) + _synth_session(d4, 60, seed=4)

        short_computed = compute_extension_for_symbol(_to_df(short_bars))
        long_computed = compute_extension_for_symbol(_to_df(long_bars))

        cols = [c for c in short_computed.columns if c != "bar_start_utc"]
        for i in range(len(short_bars)):
            for c in cols:
                a, b = short_computed[c].iloc[i], long_computed[c].iloc[i]
                a_na, b_na = pd.isna(a), pd.isna(b)
                if a_na and b_na:
                    continue
                assert a_na == b_na, f"{c}@{i}: NA-mismatch short={a} long={b}"
                assert abs(float(a) - float(b)) < 1e-6, f"{c}@{i}: short={a} long={b}"
