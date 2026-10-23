from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from backend.app.research.intraday_features.bar_math import Bar
from backend.app.research.intraday_features.price_action import (
    candle_lower_wick_ratio,
    candle_upper_wick_ratio,
    is_bearish_engulfing,
    is_bullish_engulfing,
)
from backend.app.research.intraday_features.vwap_levels import vwap_cross
from scripts.run_phase_v4_strategy_discovery import (
    ALL_HORIZONS,
    DISCOVERY_HORIZONS,
    SINGLE_FEATURES,
    TOP_N_VALUES,
    add_feature_columns,
    chronological_splits,
    final_verdicts,
    pick_survivors,
    run_fade_discovery,
    run_named_hypothesis_discovery,
    run_single_feature_discovery,
    run_top_n_discovery,
)
from scripts.run_phase_v4_strategy_discovery import test_budget_discovery as compute_test_budget

UTC = dt.timezone.utc
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def _synthetic_dataset(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_symbols = 10
    data: dict[str, np.ndarray] = {
        "symbol": np.array([f"S{i % n_symbols}" for i in range(n)]),
        "security_id": np.array([i % n_symbols for i in range(n)]),
        "trading_date": pd.date_range("2026-01-01", periods=n, freq="5min").date,
        "bar_start_utc": pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC"),
        "open": 100 + rng.normal(size=n).cumsum() * 0.01,
        "minutes_since_open": rng.integers(0, 375, n),
        "distance_from_session_open": rng.normal(scale=0.01, size=n),
        "distance_from_vwap": rng.normal(scale=0.004, size=n),
        "opening_range_high": 101 + rng.random(n), "opening_range_low": 99 - rng.random(n),
        "previous_day_high": 101 + rng.random(n), "previous_day_low": 99 - rng.random(n),
        "distance_to_nearest_support": rng.random(n) * 0.02, "distance_to_nearest_resistance": rng.random(n) * 0.02,
        "breakout_up_meaningful": rng.random(n) < 0.03, "breakout_down_meaningful": rng.random(n) < 0.03,
        "relative_volume_20": rng.random(n) * 4, "directional_pressure_10": rng.normal(size=n) * 1000,
        "volatility_expansion_ratio": rng.random(n) * 2, "candle_body_ratio": rng.random(n),
        "overnight_gap": rng.normal(scale=0.006, size=n),
        "consecutive_same_direction_count": rng.integers(0, 10, n),
        "candle_direction": rng.choice([-1, 0, 1], size=n),
        "compression_then_expansion_flag": rng.random(n) < 0.05,
        "rolling_liquidity_group": rng.choice(["mid_liquidity", "high_liquidity"], size=n),
        "rolling_median_daily_traded_value_20d": rng.random(n) * 1e7,
        "return_5m": rng.normal(scale=0.004, size=n), "return_15m": rng.normal(scale=0.006, size=n),
    }
    close = data["open"] + rng.normal(scale=0.3, size=n)
    high = np.maximum(data["open"], close) + rng.random(n) * 0.3
    low = np.minimum(data["open"], close) - rng.random(n) * 0.3
    data["close"], data["high"], data["low"] = close, high, low
    for h in ALL_HORIZONS:
        up = rng.random(n) < 0.33
        down = ~up & (rng.random(n) < 0.5)
        data[f"direction_label_pit_{h}"] = np.where(up, "UP", np.where(down, "DOWN", "NEUTRAL"))
        data[f"gross_return_{h}m"] = rng.normal(scale=0.005, size=n)
    return pd.DataFrame(data)


def _to_bar(row) -> Bar:
    return Bar(row.bar_start_utc, row.trading_date, row.open, row.high, row.low, row.close, 1000.0)


class TestWickRatioEquivalence:
    def test_upper_lower_wick_ratio_match_pure_functions(self):
        df = _synthetic_dataset(n=500)
        out = add_feature_columns(df.copy())
        for i in range(0, 500, 7):
            bar = _to_bar(df.iloc[i])
            assert out.iloc[i]["candle_upper_wick_ratio"] == pytest.approx(candle_upper_wick_ratio(bar), abs=1e-9)
            assert out.iloc[i]["candle_lower_wick_ratio"] == pytest.approx(candle_lower_wick_ratio(bar), abs=1e-9)


class TestEngulfingEquivalence:
    def test_matches_pure_function_at_5min_grid_resolution(self):
        df = _synthetic_dataset(n=500)
        out = add_feature_columns(df.copy())
        mismatches = []
        for sid in df["security_id"].unique():
            sym_df = df[df["security_id"] == sid].sort_values("bar_start_utc").reset_index(drop=True)
            sym_out = out[out["security_id"] == sid].sort_values("bar_start_utc").reset_index(drop=True)
            for i in range(1, len(sym_df)):
                bars_up_to_t = [_to_bar(sym_df.iloc[i - 1]), _to_bar(sym_df.iloc[i])]
                ref_bull = is_bullish_engulfing(bars_up_to_t)
                ref_bear = is_bearish_engulfing(bars_up_to_t)
                vec_bull = sym_out.iloc[i]["two_bar_engulfing_bullish"]
                vec_bear = sym_out.iloc[i]["two_bar_engulfing_bearish"]
                if bool(vec_bull) != bool(ref_bull):
                    mismatches.append((sid, i, "bull", ref_bull, vec_bull))
                if bool(vec_bear) != bool(ref_bear):
                    mismatches.append((sid, i, "bear", ref_bear, vec_bear))
        assert mismatches == [], f"{len(mismatches)} mismatches, first few: {mismatches[:10]}"


class TestVwapCrossEquivalence:
    def test_reclaim_and_rejection_match_sign_flip_of_distance_from_vwap(self):
        day = dt.date(2026, 1, 5)
        base_ts = dt.datetime(2026, 1, 5, 9, 15, tzinfo=IST).astimezone(UTC)
        df = pd.DataFrame({
            "symbol": ["S"] * 4, "security_id": [1] * 4, "trading_date": [day] * 4,
            "bar_start_utc": [base_ts + dt.timedelta(minutes=5 * i) for i in range(4)],
            "distance_from_vwap": [-0.002, 0.001, 0.0015, -0.0005],
        })
        for col in ("open", "high", "low", "close", "minutes_since_open", "opening_range_high", "opening_range_low",
                    "previous_day_high", "previous_day_low",
                    "distance_from_session_open", "distance_to_nearest_support", "distance_to_nearest_resistance",
                    "breakout_up_meaningful", "breakout_down_meaningful"):
            df[col] = 0.0
        df["breakout_up_meaningful"] = False
        df["breakout_down_meaningful"] = False
        out = add_feature_columns(df.copy())
        assert bool(out.iloc[1]["vwap_reclaim"]) is True
        assert bool(out.iloc[1]["vwap_rejection"]) is False
        assert bool(out.iloc[2]["vwap_reclaim"]) is False
        assert bool(out.iloc[3]["vwap_rejection"]) is True


class TestTestBudget:
    def test_matches_frozen_totals(self):
        budget = compute_test_budget()
        assert budget["single_feature"] == len(SINGLE_FEATURES) * 2 * len(DISCOVERY_HORIZONS) == 12 * 2 * 2
        assert budget["named_hypothesis"] == 22
        assert budget["H_FADE"] == 4 * 2 * len(DISCOVERY_HORIZONS) == 16
        assert budget["top_n_ranking"] == len(TOP_N_VALUES) * len(DISCOVERY_HORIZONS) == 6
        assert sum(budget.values()) == 92

    def test_row_counts_match_declared_budget_on_synthetic_data(self):
        df = add_feature_columns(_synthetic_dataset())
        returns_5m_mean = df.groupby("bar_start_utc")["return_5m"].transform("mean")
        df["relative_strength_vs_universe"] = df["return_5m"] - returns_5m_mean
        df["universe_breadth"] = df.groupby("bar_start_utc")["return_5m"].transform(lambda s: (s > 0).mean())
        df["cs_rank_15m"] = df.groupby("bar_start_utc")["return_15m"].rank(pct=True)
        alpha = 0.05 / sum(compute_test_budget().values())
        assert len(run_single_feature_discovery(df, alpha)) == compute_test_budget()["single_feature"]
        assert len(run_named_hypothesis_discovery(df, alpha)) == compute_test_budget()["named_hypothesis"]
        assert len(run_fade_discovery(df, alpha)) == compute_test_budget()["H_FADE"]
        assert len(run_top_n_discovery(df, alpha)) == compute_test_budget()["top_n_ranking"]


class TestPickSurvivorsDirectionAware:
    def test_excludes_significant_but_lift_below_one(self):
        results = [{"family": "single_feature", "feature": "x", "cell": "top_implies_up", "horizon_min": 15, "significant": True, "lift": 0.9}]
        assert pick_survivors(results) == []

    def test_includes_significant_with_lift_above_one(self):
        results = [{"family": "single_feature", "feature": "x", "cell": "top_implies_up", "horizon_min": 15, "significant": True, "lift": 1.1}]
        assert len(pick_survivors(results)) == 1

    def test_excludes_top_n_ranking_family(self):
        results = [{"family": "top_n_ranking", "n": 10, "horizon_min": 15, "significant": True, "lift": 1.5}]
        assert pick_survivors(results) == []


class TestFinalVerdicts:
    def test_verdict_e_with_no_survivors(self):
        v = final_verdicts([], [], [], [], [])
        assert v["verdict"] == "E"


class TestChronologicalSplits:
    def test_splits_are_disjoint_and_cover_all_dates(self):
        df = _synthetic_dataset(n=2000)
        splits = chronological_splits(df)
        all_dates = set(df["trading_date"].unique())
        union = splits["discovery"] | splits["validation"] | splits["held_out"]
        assert union == all_dates
        assert splits["discovery"] & splits["validation"] == set()
        assert splits["validation"] & splits["held_out"] == set()
        assert splits["discovery"] & splits["held_out"] == set()

    def test_discovery_is_roughly_half(self):
        df = _synthetic_dataset(n=20000)
        splits = chronological_splits(df)
        n_total = len(splits["discovery"]) + len(splits["validation"]) + len(splits["held_out"])
        assert abs(len(splits["discovery"]) / n_total - 0.5) < 0.05
