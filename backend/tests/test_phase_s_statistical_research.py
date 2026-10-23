from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_phase_s_statistical_research import (
    CANDIDATES,
    HORIZONS,
    IS_LONG_BUCKET,
    core_bucket_masks,
    final_verdict,
    pick_survivors,
    run_family_a,
    run_family_b,
)
from scripts.run_phase_s_statistical_research import test_budget as compute_test_budget

UTC = "UTC"


def _synthetic_dataset(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "symbol": [f"S{i % 12}" for i in range(n)],
        "bar_start_utc": pd.date_range("2026-01-01", periods=n, freq="5min", tz=UTC),
        "trading_date": pd.date_range("2026-01-01", periods=n, freq="5min", tz=UTC).date,
        "return_1m": rng.normal(scale=0.002, size=n), "return_3m": rng.normal(scale=0.003, size=n),
        "return_5m": rng.normal(scale=0.004, size=n), "return_15m": rng.normal(scale=0.007, size=n),
        "return_30m": rng.normal(scale=0.01, size=n),
        "relative_volume_20": rng.random(n) * 3, "distance_from_vwap": rng.normal(scale=0.003, size=n),
        "close": 100 + rng.normal(size=n).cumsum() * 0.01,
        "ema_9": 100 + rng.normal(size=n), "ema_20": 100 + rng.normal(size=n), "ema_50": 100 + rng.normal(size=n),
        "ema_20_slope_3": rng.normal(size=n), "distance_from_ema20_normalized": rng.normal(size=n),
        "rsi_14": rng.random(n) * 100, "rsi_14_slope_5": rng.normal(size=n),
        "macd_histogram_12_26_9": rng.normal(size=n), "consecutive_same_direction_count": rng.integers(0, 10, n),
        "distance_to_nearest_support": rng.random(n) * 0.02, "distance_to_nearest_resistance": rng.random(n) * 0.02,
        "session_low_so_far": 95 + rng.random(n), "session_high_so_far": 105 + rng.random(n),
        "opening_range_low": 96 + rng.random(n), "opening_range_high": 104 + rng.random(n),
        "recent_breakout_up_opening_range": rng.random(n) < 0.1, "recent_breakout_down_opening_range": rng.random(n) < 0.1,
        "distance_from_session_open": rng.normal(scale=0.005, size=n),
        "relative_strength_vs_universe": rng.normal(scale=0.003, size=n),
    })
    for h in HORIZONS:
        up = rng.random(n) < 0.4
        down = ~up & (rng.random(n) < 0.6)
        df[f"direction_label_{h}m"] = np.where(up, "UP", np.where(down, "DOWN", "AMBIGUOUS"))
        df[f"gross_return_{h}m"] = rng.normal(scale=0.005, size=n)
    df["cs_rank_15m"] = df.groupby("bar_start_utc")["return_15m"].rank(pct=True)
    return df


class TestCandidateCount:
    def test_exactly_24_frozen_candidates(self):
        assert len(CANDIDATES) == 24

    def test_exactly_4_core_buckets_with_correct_implied_sides(self):
        assert IS_LONG_BUCKET == {
            "bottom_1pct_rank15m": True, "bottom_2pct_rank15m": True,
            "top_1pct_rank15m": False, "top_2pct_rank15m": False,
        }


class TestTestBudget:
    def test_matches_the_frozen_design_totals(self):
        budget = compute_test_budget()
        assert budget["A_standalone"] == 4 * 24 * 4 == 384
        assert budget["B_incremental_value"] == 4 * 24 * 2 * 2 == 384
        assert sum(budget.values()) == 768

    def test_row_count_produced_by_family_a_matches_the_budgeted_count(self):
        df = _synthetic_dataset()
        bucket_masks = core_bucket_masks(df)
        alpha = 0.05 / sum(compute_test_budget().values())
        family_a = run_family_a(df, bucket_masks, alpha)
        assert len(family_a) == compute_test_budget()["A_standalone"]

    def test_row_count_produced_by_family_b_matches_the_budgeted_count(self):
        df = _synthetic_dataset()
        bucket_masks = core_bucket_masks(df)
        alpha = 0.05 / sum(compute_test_budget().values())
        family_b = run_family_b(df, bucket_masks, alpha)
        assert len(family_b) == compute_test_budget()["B_incremental_value"]


class TestDirectionalMirroring:

    NON_DIRECTIONAL_BY_DESIGN = {
        "A6_stretched_from_ema20", "B5_exhaustion_sequence", "C1_elevated_volume",
        "D4_near_vwap", "E2_thirty_min_decelerating", "E3_trend_disagreement",
    }
    MAY_LEGITIMATELY_OVERLAP = {
        "D1_near_nearest_support_resistance", "D2_near_session_high_low_so_far",
        "D3_near_opening_range_boundary", "D5_breakout_then_rejection",
    }

    def test_no_directional_candidate_is_true_for_both_long_and_short_on_a_flat_row(self):
        df = _synthetic_dataset(n=50)
        flat_cols = ["ema_20_slope_3", "rsi_14_slope_5", "macd_histogram_12_26_9", "return_1m", "return_3m",
                     "return_30m", "distance_from_session_open", "distance_from_vwap"]
        for c in flat_cols:
            df.loc[0, c] = 0.0
        bucket_mask = pd.Series([True] + [False] * (len(df) - 1))
        for name, fn in CANDIDATES.items():
            if name in self.NON_DIRECTIONAL_BY_DESIGN or name in self.MAY_LEGITIMATELY_OVERLAP:
                continue
            long_result = fn(df, bucket_mask, True).iloc[0]
            short_result = fn(df, bucket_mask, False).iloc[0]
            if pd.isna(long_result) or pd.isna(short_result):
                continue
            assert not (bool(long_result) and bool(short_result)), f"{name} is True on BOTH sides for a flat row"

    def test_the_non_directional_candidates_are_confirmed_identical_on_both_sides(self):
        df = _synthetic_dataset(n=50)
        bucket_mask = pd.Series([True] * len(df))
        for name in self.NON_DIRECTIONAL_BY_DESIGN:
            fn = CANDIDATES[name]
            long_result = fn(df, bucket_mask, True)
            short_result = fn(df, bucket_mask, False)
            pd.testing.assert_series_equal(long_result.reset_index(drop=True), short_result.reset_index(drop=True), check_names=False)


class TestPickSurvivors:
    def test_requires_significance_in_family_a_and_all_four_family_b_cells(self):
        family_a = [{"bucket": "bottom_1pct_rank15m", "candidate": "B1_rsi_extreme_aligned", "significant_after_correction": True}]
        family_b_all_sig = [
            {"bucket": "bottom_1pct_rank15m", "candidate": "B1_rsi_extreme_aligned", "control": "return_5m", "controlled_group": "below", "significant_after_correction": True},
            {"bucket": "bottom_1pct_rank15m", "candidate": "B1_rsi_extreme_aligned", "control": "return_5m", "controlled_group": "above", "significant_after_correction": True},
            {"bucket": "bottom_1pct_rank15m", "candidate": "B1_rsi_extreme_aligned", "control": "relative_strength_vs_universe", "controlled_group": "below", "significant_after_correction": True},
            {"bucket": "bottom_1pct_rank15m", "candidate": "B1_rsi_extreme_aligned", "control": "relative_strength_vs_universe", "controlled_group": "above", "significant_after_correction": True},
        ]
        assert len(pick_survivors(family_a, family_b_all_sig)) == 1

    def test_excludes_when_only_three_of_four_cells_significant(self):
        family_a = [{"bucket": "bottom_1pct_rank15m", "candidate": "B1_rsi_extreme_aligned", "significant_after_correction": True}]
        family_b = [
            {"bucket": "bottom_1pct_rank15m", "candidate": "B1_rsi_extreme_aligned", "control": "return_5m", "controlled_group": "below", "significant_after_correction": True},
            {"bucket": "bottom_1pct_rank15m", "candidate": "B1_rsi_extreme_aligned", "control": "return_5m", "controlled_group": "above", "significant_after_correction": True},
            {"bucket": "bottom_1pct_rank15m", "candidate": "B1_rsi_extreme_aligned", "control": "relative_strength_vs_universe", "controlled_group": "below", "significant_after_correction": True},
            {"bucket": "bottom_1pct_rank15m", "candidate": "B1_rsi_extreme_aligned", "control": "relative_strength_vs_universe", "controlled_group": "above", "significant_after_correction": False},
        ]
        assert pick_survivors(family_a, family_b) == []


class TestFinalVerdict:
    def test_verdict_d_with_no_survivors(self):
        v = final_verdict([], {}, {}, {})
        assert v["verdict"] == "D"
        assert v["n_survivors"] == 0
