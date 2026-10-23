from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_phase_t_build_event_paths import compute_paths_for_symbol
from scripts.run_phase_v6_statistical_research import TOP_N_VALUES
from scripts.run_phase_v7a_opportunity_intensity_tod import TOD_BUCKETS
from scripts.run_phase_v7b_long_horizon_opportunity import (
    LONG_HORIZONS,
    compute_long_horizon_paths_for_symbol,
    final_classification,
    pick_survivors,
    run_discovery,
    target_before_stop_note,
    test_budget_v7b as compute_test_budget_v7b,
)


def _one_session_bars(n_minutes: int, day: str, seed: int = 0, start: str = "03:45") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range(f"{day} {start}", periods=n_minutes, freq="min", tz="UTC")
    close = 100.0 + np.cumsum(rng.normal(scale=0.3, size=n_minutes))
    high = close + rng.random(n_minutes) * 0.5
    low = close - rng.random(n_minutes) * 0.5
    open_ = close - rng.normal(scale=0.1, size=n_minutes)
    return pd.DataFrame({
        "bar_start_utc": times, "trading_date": pd.Timestamp(day).date(),
        "open": open_, "high": high, "low": low, "close": close, "volume": rng.integers(1000, 5000, n_minutes).astype(float),
    })


class TestEquivalenceToPhaseT:
    def test_h60_entry_offset_1_is_bit_identical_to_phase_t(self):
        bars = _one_session_bars(375, "2026-01-05")
        reference = compute_paths_for_symbol(bars)
        mine = compute_long_horizon_paths_for_symbol(bars, (60,))
        for metric in ("gross_return", "mfe", "mae"):
            ref_col, my_col = reference[f"{metric}_o1_h60"], mine[f"{metric}_o1_h60"]
            assert (ref_col.isna() == my_col.isna()).all(), f"{metric}: eligibility (NaN pattern) differs"
            both = ref_col.notna()
            np.testing.assert_allclose(ref_col[both], my_col[both], atol=1e-12)


class TestSessionBoundaryRule:
    def test_candidate_too_close_to_session_end_is_excluded_not_truncated(self):
        bars = _one_session_bars(50, "2026-01-05")
        out = compute_long_horizon_paths_for_symbol(bars, (300,))
        assert out["mfe_o1_h300"].isna().all()
        assert out["gross_return_o1_h300"].isna().all()

    def test_candidate_with_exactly_enough_remaining_session_is_eligible(self):
        h = 60
        bars = _one_session_bars(375, "2026-01-05")
        out = compute_long_horizon_paths_for_symbol(bars, (h,))
        last_eligible_idx = 375 - 1 - h - 1
        assert pd.notna(out[f"mfe_o1_h{h}"].iloc[last_eligible_idx])
        assert pd.isna(out[f"mfe_o1_h{h}"].iloc[last_eligible_idx + 1])

    def test_never_crosses_into_a_second_trading_date(self):
        day1 = _one_session_bars(375, "2026-01-05")
        day2 = _one_session_bars(375, "2026-01-06", seed=1)
        day2["close"] += 10_000.0
        both_days = pd.concat([day1, day2], ignore_index=True)
        out = compute_long_horizon_paths_for_symbol(both_days, (60,))
        near_end_of_day1 = 370
        assert pd.isna(out["mfe_o1_h60"].iloc[near_end_of_day1])
        assert out.loc[near_end_of_day1, "gross_return_o1_h60"] != out.loc[near_end_of_day1, "gross_return_o1_h60"]


class TestBudget:
    def test_discovery_budget_is_45(self):
        assert compute_test_budget_v7b()["discovery"] == len(LONG_HORIZONS) * len(TOD_BUCKETS) * len(TOP_N_VALUES) == 45


class TestPickSurvivors:
    def test_excludes_significant_but_below_baseline(self):
        results = [{"time_of_day": "late", "top_n": 20, "horizon_min": 120, "significant": True, "realized_mean": 0.0001, "baseline_mean": 0.0002}]
        assert pick_survivors(results) == []

    def test_includes_significant_above_baseline(self):
        results = [{"time_of_day": "late", "top_n": 20, "horizon_min": 120, "significant": True, "realized_mean": 0.0002, "baseline_mean": 0.0001}]
        assert len(pick_survivors(results)) == 1


class TestTargetBeforeStopNote:
    def test_applicable_within_race_checkpoint_range(self):
        assert target_before_stop_note(60)["applicable"] is True

    def test_not_applicable_beyond_frozen_checkpoint_grid(self):
        note = target_before_stop_note(120)
        assert note["applicable"] is False
        assert "120" in note["note"]


class TestFinalClassification:
    def test_a_when_no_survivors(self):
        assert final_classification([], {})["classification"] == "A"

    def test_b_when_gross_mean_negative(self):
        survivors = [{"time_of_day": "late", "top_n": 20, "horizon_min": 120}]
        checks = {("late", 20, 120): {
            "validation": {"significant": True, "realized_mean": 0.0003, "baseline_mean": 0.0001},
            "held_out": {"significant": True, "realized_mean": 0.0003, "baseline_mean": 0.0001},
            "cross_stock": {"significant": True, "realized_mean": 0.0003, "baseline_mean": 0.0001},
            "economic": {"gross_mean": -0.0001, "scenarios": {"A_optimistic": {"mean_net": -0.0002}, "C_conservative": {"mean_net": -0.0003}}},
        }}
        result = final_classification(survivors, checks)
        assert result["classification"] == "B"

    def test_d_when_conservative_bar_clears(self):
        survivors = [{"time_of_day": "late", "top_n": 20, "horizon_min": 120}]
        checks = {("late", 20, 120): {
            "validation": {"significant": True, "realized_mean": 0.0003, "baseline_mean": 0.0001},
            "held_out": {"significant": True, "realized_mean": 0.0003, "baseline_mean": 0.0001},
            "cross_stock": {"significant": True, "realized_mean": 0.0003, "baseline_mean": 0.0001},
            "economic": {"gross_mean": 0.0008, "scenarios": {"A_optimistic": {"mean_net": 0.0006}, "C_conservative": {"mean_net": 0.0001}}},
        }}
        result = final_classification(survivors, checks)
        assert result["classification"] == "D"


class TestRunDiscoveryRowCount:
    def test_produces_exactly_the_budgeted_45_rows(self):
        rng = np.random.default_rng(0)
        n = 6000
        rows = {
            "bar_start_utc": pd.date_range("2026-01-05 03:45", periods=n, freq="min", tz="UTC"),
            "relative_volume_20": rng.random(n) * 3,
            "volatility_expansion_ratio": rng.random(n) * 2,
            "time_of_day_bucket": rng.choice(list(TOD_BUCKETS), size=n),
        }
        for h in LONG_HORIZONS:
            rows[f"mfe_o1_h{h}"] = np.abs(rng.normal(scale=0.001, size=n))
            rows[f"mae_o1_h{h}"] = np.abs(rng.normal(scale=0.001, size=n))
        df = pd.DataFrame(rows)
        alpha = 0.05 / compute_test_budget_v7b()["discovery"]
        results = run_discovery(df, alpha)
        assert len(results) == 45
