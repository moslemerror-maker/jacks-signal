from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from backend.app.research.intraday_features.bar_math import Bar
from backend.app.research.intraday_features.labels import forward_movement_label
from backend.app.research.intraday_features.mover_mechanics import (
    compression_then_expansion,
    directional_pressure,
    recent_breakout_down,
    recent_breakout_up,
)
from backend.app.research.intraday_features.price_action import candle_body_ratio, minutes_since_open
from backend.app.research.intraday_features.volatility_volume import realized_volatility, volatility_expansion_ratio
from backend.app.research.intraday_features.vwap_levels import opening_range_high_low, previous_day_high_low_close
from scripts.run_phase_v3_build_liquid_mechanism_dataset import FORWARD_LABEL_THRESHOLD, HORIZONS, compute_new_columns_for_symbol

UTC = dt.timezone.utc
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def _synthetic_symbol_frame(n_sessions: int = 4, bars_per_session: int = 90, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sessions):
        trading_date = dt.date(2026, 1, 5) + dt.timedelta(days=s)
        open_ist = dt.datetime(trading_date.year, trading_date.month, trading_date.day, 9, 15, tzinfo=IST)
        price = 100.0 + s * 3
        for i in range(bars_per_session):
            o = price
            c = price * (1 + rng.normal(scale=0.004))
            h = max(o, c) * (1 + abs(rng.normal(scale=0.0015)))
            l = min(o, c) * (1 - abs(rng.normal(scale=0.0015)))
            vol = 1000 + rng.integers(0, 5000)
            ts = (open_ist + dt.timedelta(minutes=i)).astimezone(UTC)
            rows.append({"bar_start_utc": ts, "trading_date": trading_date, "open": o, "high": h, "low": l, "close": c, "volume": float(vol)})
            price = c
    return pd.DataFrame(rows)


def _to_bars(df: pd.DataFrame) -> list[Bar]:
    return [Bar(r.bar_start_utc, r.trading_date, r.open, r.high, r.low, r.close, r.volume) for r in df.itertuples()]


class TestVolatilityExpansionEquivalence:
    def test_rv10_rv60_ratio_and_compression_expansion_flag_match_pure_functions(self):
        df = _synthetic_symbol_frame()
        vectorized = compute_new_columns_for_symbol(df)
        bars = _to_bars(df)
        mismatches = []
        for i in range(0, len(df), 3):
            bars_up_to_t = bars[: i + 1]
            ref_rv10 = realized_volatility(bars_up_to_t, 10)
            ref_rv60 = realized_volatility(bars_up_to_t, 60)
            ref_ratio = volatility_expansion_ratio(bars_up_to_t, 10, 60)
            ref_flag = compression_then_expansion(bars_up_to_t, lookback_bars=10, short_n=10, long_n=60, compression_threshold=0.7, expansion_threshold=1.3)
            row = vectorized.iloc[i]
            for ref_val, col in [(ref_rv10, "realized_volatility_10"), (ref_rv60, "realized_volatility_60"), (ref_ratio, "volatility_expansion_ratio")]:
                vec_val = row[col]
                if ref_val is None:
                    if not pd.isna(vec_val):
                        mismatches.append((i, col, "expected None", vec_val))
                elif pd.isna(vec_val) or not np.isclose(vec_val, ref_val, rtol=1e-9):
                    mismatches.append((i, col, ref_val, vec_val))
            vec_flag = row["compression_then_expansion_flag"]
            vec_flag_bool = None if pd.isna(vec_flag) else bool(vec_flag)
            if vec_flag_bool != ref_flag:
                mismatches.append((i, "compression_then_expansion_flag", ref_flag, vec_flag_bool))
        assert mismatches == [], f"{len(mismatches)} mismatches, first few: {mismatches[:10]}"


class TestCandleAndPressureEquivalence:
    def test_candle_body_ratio_matches(self):
        df = _synthetic_symbol_frame()
        vectorized = compute_new_columns_for_symbol(df)
        bars = _to_bars(df)
        for i in range(len(df)):
            assert vectorized.iloc[i]["candle_body_ratio"] == pytest.approx(candle_body_ratio(bars[i]), abs=1e-9)

    def test_directional_pressure_10_matches(self):
        df = _synthetic_symbol_frame()
        vectorized = compute_new_columns_for_symbol(df)
        bars = _to_bars(df)
        mismatches = []
        for i in range(0, len(df), 2):
            ref = directional_pressure(bars[: i + 1], lookback_bars=10)
            vec = vectorized.iloc[i]["directional_pressure_10"]
            if ref is None:
                if not pd.isna(vec):
                    mismatches.append((i, "expected None", vec))
            elif not np.isclose(vec, ref, rtol=1e-9):
                mismatches.append((i, ref, vec))
        assert mismatches == []

    def test_minutes_since_open_matches(self):
        df = _synthetic_symbol_frame()
        vectorized = compute_new_columns_for_symbol(df)
        bars = _to_bars(df)
        for i in range(0, len(df), 5):
            ref = minutes_since_open(bars[: i + 1])
            vec = vectorized.iloc[i]["minutes_since_open"]
            assert vec == ref


class TestMeaningfulBreakoutEquivalence:
    def test_matches_recent_breakout_up_down_with_the_combined_level(self):
        df = _synthetic_symbol_frame()
        vectorized = compute_new_columns_for_symbol(df)
        bars = _to_bars(df)
        mismatches = []
        for i in range(0, len(df), 3):
            bars_up_to_t = bars[: i + 1]
            prev_levels = previous_day_high_low_close(bars_up_to_t)
            or_levels = opening_range_high_low(bars_up_to_t, 15)
            ups = [v for v in (prev_levels[0] if prev_levels else None, or_levels[0] if or_levels else None) if v is not None]
            downs = [v for v in (prev_levels[1] if prev_levels else None, or_levels[1] if or_levels else None) if v is not None]
            level_up = max(ups) if ups else None
            level_down = min(downs) if downs else None
            ref_up = recent_breakout_up(bars_up_to_t, level_up, within_bars=5) if level_up is not None else None
            ref_down = recent_breakout_down(bars_up_to_t, level_down, within_bars=5) if level_down is not None else None
            row = vectorized.iloc[i]
            vec_up = None if pd.isna(row["breakout_up_meaningful"]) else bool(row["breakout_up_meaningful"])
            vec_down = None if pd.isna(row["breakout_down_meaningful"]) else bool(row["breakout_down_meaningful"])
            if vec_up != ref_up:
                mismatches.append((i, "up", ref_up, vec_up))
            if vec_down != ref_down:
                mismatches.append((i, "down", ref_down, vec_down))
        assert mismatches == [], f"{len(mismatches)} mismatches, first few: {mismatches[:10]}"


class TestForwardMovementLabelEquivalence:
    def test_normalized_move_and_direction_label_match_pure_function(self):
        df = _synthetic_symbol_frame()
        vectorized = compute_new_columns_for_symbol(df)
        bars = _to_bars(df)
        mismatches = []
        for i in range(0, len(df) - 60, 4):
            bars_up_to_t = bars[: i + 1]
            future_bars = bars[i + 1 :]
            rv20 = realized_volatility(bars_up_to_t, 20)
            for h in HORIZONS:
                ref = forward_movement_label(bars_up_to_t, future_bars, horizon_bars=h, vol_normalizer=rv20, threshold=FORWARD_LABEL_THRESHOLD) if rv20 else None
                row = vectorized.iloc[i]
                vec_move = row[f"normalized_move_{h}"]
                vec_label = row[f"direction_label_pit_{h}"]
                if ref is None:
                    if not pd.isna(vec_move):
                        mismatches.append((i, h, "expected None move", vec_move))
                    continue
                if pd.isna(vec_move) or not np.isclose(vec_move, ref.normalized_move, rtol=1e-9):
                    mismatches.append((i, h, "move", ref.normalized_move, vec_move))
                if vec_label != ref.label_class:
                    mismatches.append((i, h, "label", ref.label_class, vec_label))
        assert mismatches == [], f"{len(mismatches)} mismatches, first few: {mismatches[:10]}"
